import os
import sys
sys.path.append(os.path.join(os.path.dirname(f"{__file__}"), '../..'))


import socket
print(f"Running on: {socket.gethostname()}")
mnt_args = "/mnt" if "KW" in socket.gethostname() else "" 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import glob
import timm

import numpy as np
import pyarrow.parquet as pq
import tifffile
import zarr
from zarr.core import Array as ZArray
import h5py
import random
from collections import defaultdict
import argparse
import json
import math
from pathlib import Path
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from dataset import CLIPDataset, PASTDataset, CombinedDataset
from modules import GraphEncoder
from models import PASTModel
import config_files.config_train as CFG
import config_files.config_past_train as CFGP
from utils.constants import DefaultPaths
from utils.util import AvgMeter, get_lr, get_subdirs, get_subfiles, parse_token_folder_name, set_seed, check_dirs, worker_init_fn
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser(description='PAST Training')
    
    parser.add_argument('--exp_name', type=str, default='past_ds1', help='')
    parser.add_argument('--batch_size', type=int, default=96, help='')
    parser.add_argument('--epochs', type=int, default=30, help='')
    parser.add_argument('--num_workers', type=int, default=4, help='')
    parser.add_argument('--cell_path', type=str, default='/ibex/project/c2277/data/Pathomics/tokenized/pretrain', help='')
    parser.add_argument('--image_path', type=str, default='/ibex/project/c2277/data/Pathomics/raw', help='')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.add_argument('--saving_dir', default='./result/task1')


    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


    parser.add_argument('--train_mode', type=str, default='same_tissue', 
                        choices=['same_tissue'],
                        help='Train on the same tissue, excluding held-out slides')
    parser.add_argument('--target_tissue', type=str, default=None,
                        help='Target tissue name')
    parser.add_argument('--target_sample', type=str, nargs='+', default=None,
                        help='One or more same-tissue slides to hold out for evaluation (required)')
    parser.add_argument('--feature_mode', type=str, default='union',
                        choices=['union', 'intersection', 'union_test_intersection'],
                        help='Gene selection strategy')
    parser.add_argument('--dist_eval', action='store_true', 
                        help='Enable distributed validation')
    

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='lower lr bound for schedulers')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='epochs to warmup LR')
    

    parser.add_argument('--output_dir', default='./result/task1', help='path where to save')
    parser.add_argument('--log_dir', default='./result/task1', help='path where to tensorboard log')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--save_interval', type=int, default=2000, help='Save checkpoint every x steps')
    parser.add_argument('--accum_iter', type=int, default=3, help='Gradient accumulation steps')
    
    return parser


def group_samples_by_tissue(align_name_list):


    tissue_groups = defaultdict(list)
    
    for sample_name in align_name_list:
        specie, tissue, doner, sample = parse_token_folder_name(sample_name)
        tissue_groups[tissue].append(sample_name)
    
    return tissue_groups


def select_samples_for_training(align_name_list, args):

    groups = group_samples_by_tissue(align_name_list)
    if args.target_tissue not in groups:
        raise ValueError(f'Unknown tissue: {args.target_tissue}')
    if not args.target_sample:
        raise ValueError('same_tissue requires --target_sample (held-out slide ID)')
    held_out = list(dict.fromkeys(args.target_sample))
    tissue_samples = groups[args.target_tissue]
    invalid = sorted(set(held_out) - set(tissue_samples))
    if invalid:
        raise ValueError(f'Held-out slides not found in {args.target_tissue}: {invalid}')
    training = [sample for sample in tissue_samples if sample not in held_out]
    if not training:
        raise ValueError('No same-tissue training slides remain')
    if misc.is_main_process():
        print('Training mode: same_tissue')
        print('Target tissue:', args.target_tissue)
        print('Held-out slides:', held_out)
        print('Training slides:', training)
    return training


def get_data_indices(train_samples_pool, args):


    all_indices_dict = {}
    

    for sample_name in train_samples_pool:
        index_path = os.path.join("./data_idx", f"{sample_name}.npy")
        if os.path.exists(index_path):
            indices = np.load(index_path, allow_pickle=True).tolist()
            all_indices_dict[sample_name] = set(indices)
        else:
            if misc.is_main_process():
                print(f"Warning: feature index file not found for sample {sample_name}")
    

    # Original same-tissue behavior: test feature sets are not used to select targets.
    test_indices_sets = []

    if args.feature_mode == 'union':

        union_set = set()
        for indices in all_indices_dict.values():
            union_set.update(indices)
        data_idx = list(union_set)
    
    elif args.feature_mode == 'intersection':

        if not all_indices_dict:
            data_idx = []
        else:

            sets_list = list(all_indices_dict.values())
            intersection_set = sets_list[0]
            for indices in sets_list[1:]:
                intersection_set &= indices
            data_idx = list(intersection_set)
    
    elif args.feature_mode == 'union_test_intersection':


        train_union = set()
        for indices in all_indices_dict.values():
            train_union.update(indices)
        

        if not test_indices_sets:
            test_intersection = set()
        else:
            test_intersection = test_indices_sets[0]
            for indices in test_indices_sets[1:]:
                test_intersection &= indices
        

        data_idx = list(train_union & test_intersection) if test_intersection else list(train_union)
    

    data_idx.sort()
    
    if misc.is_main_process():
        print(f"Feature selection mode: {args.feature_mode}")
        print(f"Training samples: {len(all_indices_dict)}/{len(train_samples_pool)}")
        print(f"Selected feature count: {len(data_idx)}")
    
    return data_idx


def build_datasets(args, transform, train_samples_pool, data_idx):

    if misc.is_main_process():
        print("Building datasets...")
    
    dataset_name = "Xenium"
    dataset_list = []
    

    name_files = get_subfiles(os.path.join(args.cell_path, dataset_name))
    

    for file_name in name_files:
        file_name_without_ext = os.path.splitext(os.path.basename(file_name))[0]
        
        if file_name_without_ext in train_samples_pool:
            specie, tissue, doner, sample = parse_token_folder_name(file_name_without_ext)
            img_path = os.path.join(args.image_path, dataset_name, doner, sample)
            
            dataset = NewPASTDataset(
                image_path=img_path, 
                data_idx=data_idx,
                cell_path=os.path.join(args.cell_path, dataset_name, file_name),
                padding=CFGP.image_size / 2, 
                img_transform=transform,
            )
            dataset_list.append(dataset)
    
    if len(dataset_list) != len(train_samples_pool):
        raise ValueError(f'Only found {len(dataset_list)}/{len(train_samples_pool)} training slides')


    combined_dataset = CombinedDataset(dataset_list) if dataset_list else None
    
    if combined_dataset is None or len(combined_dataset) == 0:
        raise ValueError("No valid datasets")
    

    train_size = int(0.9 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    if not train_size or not val_size:
        raise ValueError('Need >=2 cells to create train and validation splits')
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    if misc.is_main_process():
        print(f"Training set: {len(train_dataset)} cells")
        print(f"Validation set: {len(val_dataset)} cells")
    
    return train_dataset, val_dataset


class NewPASTDataset(PASTDataset):
    def __init__(self, image_path, cell_path, data_idx,
                 padding=112, neighbor_radius=224, img_transform=None):
        self.image_path = image_path
        self.cell_path = cell_path
        self.h5_cell_data = h5py.File(cell_path)
        self.idx_ls = data_idx

        self.total_rows = len(self.h5_cell_data["idx"])

        self.padding = padding
        self.neighbor_radius = neighbor_radius
        self.img_transform = img_transform

        self.all_indices = np.array(self.h5_cell_data["idx"][:self.total_rows])

        tif_files = self.get_he_tif()
        assert len(tif_files) == 1, "More than one TIFF file found in the specified directory."

        page = tifffile.imread(os.path.join(image_path, tif_files[0]), is_ome=True, level=0, aszarr=True)
        self.whole_image = zarr.open(page, mode='r')
        self.pixel_size = self.get_pixel_size(self.whole_image)
        
        zeros_array = np.zeros(len(self.idx_ls), dtype=np.int32)
        self.default_item = {
            'X': zeros_array,
            'valid': False  
        }

    def get_padded_image(self, x, y):
        h, w = self.pixel_size
        y1 = max(0, int(y - self.padding))
        y2 = min(h, int(y + self.padding))
        x1 = max(0, int(x - self.padding))
        x2 = min(w, int(x + self.padding))

        if self.channel_dim == 0:
            sub_image = self.whole_image[:, y1:y2, x1:x2]
            sub_image = np.transpose(sub_image, (1, 2, 0))
        elif self.channel_dim == 2:
            sub_image = self.whole_image[y1:y2, x1:x2, :]
        else:
            raise ValueError("The image channels should be on the first or last dimension.")

        pad_y1 = int((self.padding * 2 - (y2 - y1)) // 2)
        pad_y2 = int(self.padding * 2 - (y2 - y1) - pad_y1)
        pad_x1 = int((self.padding * 2 - (x2 - x1)) // 2)
        pad_x2 = int(self.padding * 2 - (x2 - x1) - pad_x1)

        padded_image = np.pad(sub_image, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), mode='constant',
                              constant_values=0)

        return padded_image

    def __len__(self):
        return self.total_rows

    def extract_sparse_features_vectorized(self, indices, values, idx_ls):
        
        result = np.zeros(len(idx_ls), dtype=np.int32)
        
        if indices is None or len(indices) == 0:
            return result
        
        mask = np.isin(indices, idx_ls)
        if not np.any(mask):
            return result
        
        common_indices = indices[mask]
        common_values = values[mask]
        
        positions = np.searchsorted(idx_ls, common_indices)
        
        result[positions] = common_values
        
        return result

    def __getitem__(self, idx):
        if idx >= self.total_rows:
            raise IndexError(f"Index {idx} out of range for dataset with {self.total_rows} items")
            
        row_idx = self.all_indices[idx]
        
        try:
            x = self.h5_cell_data["image_x"][row_idx]
            y = self.h5_cell_data["image_y"][row_idx]
            
            X_indices = self.h5_cell_data["X_model_indices"][row_idx]
            X_values = self.h5_cell_data["X_model_values"][row_idx]

            cluster = self.h5_cell_data["cluster"][row_idx]

            if not np.isnan(cluster):
                tensor_cluster = torch.tensor(cluster, dtype=torch.int32)
            else:
                tensor_cluster = torch.tensor(-1, dtype=torch.int32)
            
            if X_indices is None or len(X_indices) == 0:
                padded_image = self.get_padded_image(x, y)
                
                if self.img_transform:
                    padded_image = self.img_transform(padded_image)
                else:
                    padded_image = torch.tensor(padded_image, dtype=torch.float32)
                
                if padded_image.shape != (3, 224, 224):
                    padded_image = torch.zeros(3, 224, 224, dtype=torch.float32)
                
                item = {
                    'image': padded_image,
                    'X': torch.tensor(np.zeros(len(self.idx_ls)), dtype=torch.int32),
                    'cluster': tensor_cluster,
                    'relative_coords': torch.tensor([self.padding, self.padding], dtype=torch.float32),
                    'valid': False  
                }
                return item
            
            features = self.extract_sparse_features_vectorized(X_indices, X_values, self.idx_ls)
            
            padded_image = self.get_padded_image(x, y)
            
            if self.img_transform:
                padded_image = self.img_transform(padded_image)
            else:
                padded_image = torch.tensor(padded_image, dtype=torch.float32)
            
            if padded_image.shape != (3, 224, 224):

                padded_image = torch.zeros(3, 224, 224, dtype=torch.float32)
                return {
                    'image': padded_image,
                    'X': torch.tensor(features, dtype=torch.int32),
                    'cluster': tensor_cluster,
                    'relative_coords': torch.tensor([self.padding, self.padding], dtype=torch.float32),
                    'valid': False
                }
            
            return {
                'image': padded_image,
                'X': torch.tensor(features, dtype=torch.int32),
                'cluster': tensor_cluster,
                'relative_coords': torch.tensor([self.padding, self.padding], dtype=torch.float32),
                'valid': True
            }
            
        except Exception as e:
            if misc.get_rank() == 0:
                print(f"Error loading data at index {idx} (row_idx {row_idx}): {e}")
            return {
                'image': torch.zeros(3, 224, 224, dtype=torch.float32),
                'X': torch.tensor(np.zeros(len(self.idx_ls)), dtype=torch.int32),
                'cluster': torch.tensor(0, dtype=torch.int32),
                'relative_coords': torch.tensor([self.padding, self.padding], dtype=torch.float32),
                'valid': False
            }

    def get_pixel_size(self, whole_image):
        array_shape = whole_image.shape
        assert len(array_shape) == 3, "The image should have 3 dimensions."
        if array_shape[0] == 3:
            self.channel_dim = 0
            return array_shape[1:]
        elif array_shape[2] == 3:
            self.channel_dim = 2
            return array_shape[:2]
        else:
            raise ValueError("The image channels should be on the first or last dimension.")


def library_size_normalize(x: torch.Tensor, scale=1e4) -> torch.Tensor:


    if x.dim() == 1:
        row_sums = torch.sum(x, dim=0, dtype=torch.float32)
    else:
        row_sums = torch.sum(x, dim=1, keepdim=True, dtype=torch.float32)


    row_sums = torch.clamp(row_sums, min=1e-10)
    normalized = x / row_sums * scale
    return normalized

def log_transform(x: torch.Tensor) -> torch.Tensor:


    return torch.log10(x + 1)


def train_epoch(model, train_loader, optimizer, device, epoch, loss_scaler, log_writer=None, args=None):

    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None and misc.get_rank() == 0:
        print('log_dir: {}'.format(log_writer.log_dir))


    for data_iter_step, batch in enumerate(metric_logger.log_every(train_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr = adjust_learning_rate(optimizer, data_iter_step / len(train_loader) + epoch, args)


        valid_mask = batch.get('valid', torch.ones(batch['image'].size(0), dtype=torch.bool))
        

        if not valid_mask.any():
            continue
        

        valid_images = batch['image'][valid_mask]
        valid_X = batch['X'][valid_mask]
        

        current_batch_size = valid_images.size(0)
        if current_batch_size == 0:
            continue
        

        target = log_transform(library_size_normalize(valid_X))
        target = target.to(device, non_blocking=True)
        

        valid_images = valid_images.to(device, non_blocking=True)
        

        with torch.cuda.amp.autocast():
            output = model(valid_images)
            loss = nn.functional.mse_loss(output, target)
            loss = loss / accum_iter
            

        loss_value = loss.item()
        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            

        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
                    
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            
        torch.cuda.synchronize()
        

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        metric_logger.update(loss=loss_value_reduce)
        metric_logger.update(lr=lr)
        

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0 and misc.get_rank() == 0:
            epoch_1000x = int((data_iter_step / len(train_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def adjust_learning_rate(optimizer, epoch, args):

    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def val_epoch(model, test_loader, device):

    model.eval()
    loss_meter = AvgMeter()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), 
                          disable=misc.get_rank() != 0):

            valid_mask = batch.get('valid', torch.ones(batch['image'].size(0), dtype=torch.bool))
            if not valid_mask.any():
                continue
                
            valid_images = batch['image'][valid_mask]
            valid_X = batch['X'][valid_mask]
            

            current_batch_size = valid_images.size(0)
            if current_batch_size == 0:
                continue
            

            target = log_transform(library_size_normalize(valid_X))
            target = target.to(device, non_blocking=True)
            valid_images = valid_images.to(device, non_blocking=True)
            

            prediction = model(valid_images)
            loss = nn.functional.mse_loss(prediction, target)
            

            loss_meter.update(loss.item(), current_batch_size)
    

    return loss_meter.avg


def main(args):

    misc.init_distributed_mode(args)
    
    device = torch.device(args.device)
    

    seed = args.seed + misc.get_rank()
    set_seed(seed)
    

    output_parts = [args.train_mode]
    if args.target_tissue:
        output_parts.append(args.target_tissue)
    if args.target_sample and len(args.target_sample) == 1:

        output_parts.append(args.target_sample[0])
    output_parts.append(args.feature_mode)
    
    output_dir_name = "_".join(output_parts)
    args.output_dir = os.path.join(args.output_dir, output_dir_name)
    args.log_dir = os.path.join(args.log_dir, output_dir_name) if args.log_dir else None
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
    

    args.cell_path = mnt_args + args.cell_path
    args.image_path = mnt_args + args.image_path
    

    train_samples_pool = select_samples_for_training(align_name, args)
    

    data_idx = get_data_indices(train_samples_pool, args)
    X_channel = len(data_idx)
    
    if len(data_idx) == 0:
        if misc.is_main_process():
            print("Error: no selected genes. Check feature metadata.")
        return
    

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    

    train_dataset, val_dataset = build_datasets(args, transform, train_samples_pool, data_idx)
    

    if misc.is_main_process():

        np.save(os.path.join(args.output_dir, "feature_indices.npy"), np.array(data_idx))
        

        config_info = {
            'train_mode': args.train_mode,
            'target_tissue': args.target_tissue,
            'target_sample': args.target_sample,
            'feature_mode': args.feature_mode,
            'train_samples_count': len(train_samples_pool),
            'train_samples': train_samples_pool,
            'test_samples': list(args.target_sample),
            'feature_count': len(data_idx),
            'seed': args.seed
        }
        
        with open(os.path.join(args.output_dir, "train_config.json"), 'w') as f:
            json.dump(config_info, f, indent=2)
    

    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, 
        num_replicas=misc.get_world_size(),
        rank=misc.get_rank(),
        shuffle=True
    )
    
    sampler_val = torch.utils.data.DistributedSampler(
        val_dataset,
        num_replicas=misc.get_world_size(),
        rank=misc.get_rank(),
        shuffle=False
    )
    

    train_loader = DataLoader(
        train_dataset, 
        sampler=sampler_train,
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        sampler=sampler_val,
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    

    model = timm.create_model(
        "vit_large_patch16_224", 
        img_size=224, 
        patch_size=16, 
        init_values=1e-5, 
        num_classes=X_channel,
        dynamic_img_size=True
    )
    

    if misc.is_main_process():
        print("Loading pre-trained weights...")

    checkpoint_dir = f"{mnt_args}{DefaultPaths.SPATIAL}/checkpoint"
    state_dict = torch.load(
        os.path.join(checkpoint_dir, "past_latest_2.pth"), map_location="cpu"
    )["model"]
    state_dict = {
        k.replace("image_encoder.model.", ""): v
        for k, v in state_dict.items()
        if "image_encoder.model." in k
    }
    model.load_state_dict(state_dict, strict=False)


    model.to(device)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu]
        )
    

    if misc.is_main_process() and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=CFGP.weight_decay,
        betas=(0.9, 0.95)
    )
    

    loss_scaler = NativeScaler()
    

    best_val_loss = float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
            sampler_val.set_epoch(epoch)
        

        train_stats = train_epoch(
            model=model, 
            train_loader=train_loader, 
            optimizer=optimizer, 
            device=device, 
            epoch=epoch, 
            loss_scaler=loss_scaler,
            log_writer=log_writer,
            args=args
        )
        

        val_loss = val_epoch(model, val_loader, device)
        

        if misc.is_dist_avail_and_initialized():
            val_loss = misc.all_reduce_mean(val_loss)
        

        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        

        if misc.is_main_process():
            print(f"Epoch {epoch}: train_loss={train_stats['loss']:.4f}, val_loss={val_loss:.4f}")
            
            if is_best:
                save_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save({
                    'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_loss': val_loss,
                    'data_idx': data_idx,
                }, save_path)
                print(f"Saved best model, validation loss: {best_val_loss:.4f}")
        

        if misc.is_dist_avail_and_initialized():
            dist.barrier()
    
    if misc.is_main_process():
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")


align_name = [
    'human_bone_marrow_HBABMDWCAP_ALL_Bone_Marrow',
    'human_bone_marrow_HBABMDWCAP_Nondiseased_Bone_Marrow',
    'human_bone_marrow_HBABMDWCAP_Nondiseased_Femur_Bone',
    'human_brain_FFPEHBCDWHIOPPACA_Glioblastoma_Multiforme',
    'human_breast_FFPEHBUTESA_Replicate_1',
    'human_breast_FFPEHBUTESA_Replicate_2',
    'human_breast_FFPEHBWCAP_Tissue_sample_1',
    'human_breast_FFPEHBWCAP_Tissue_sample_2',
    'human_breast_FFPEHBWPP_Tissue_sample_1',
    'human_breast_FFPEHBWPP_Tissue_sample_2',
    'human_breast_HRMOTBCTMUISCSAISAOFFPET_Sample_1_Replicate_1',
    'human_breast_HRMOTBCTMUISCSAISAOFFPET_Sample_1_Replicate_2',
    'human_breast_XFFPEHBWCAP_Tissue_sample_1',
    'human_colon_COICPITTMOCCUHDSP_Sample_P1',
    'human_colon_COICPITTMOCCUHDSP_Sample_P2',
    'human_colon_COICPITTMOCCUHDSP_Sample_P5',
    'human_colon_FFPEHCCDWHIOPPACA_Colorectal_Cancer',
    'human_colon_HCPDXHCGEP_CancerPAP',
    'human_colon_HCPDXHCGEP_NondiseasedPP',
    'human_heart_HHDWXHMTACP_Nondiseased',
    'human_kidney_HKPDXHMTACP_Nondiseased',
    'human_kidney_HKPDXHMTACP_PRCC',
    'human_liver_HLDWXHMTACP_LiverCancer',
    'human_liver_HLDWXHMTACP_Nondiseased',
    'human_lung_FFPEHLCDWHIOPPACA_Lung_Cancer',
    'human_lung_PDFFPEHLCWXMCS_Adenocarcinoma',
    'human_ovary_FFPEHOCDWHIOPPACA_Ovarian_Cancer',
    'human_pancreas_FFPEHPDADWHIOPP_Ductal_Adenocarcinoma',
    'human_pancreas_FFPEHPWXMCS_Pancreas',
    'human_pancreas_PCWXHMTACP_Adenocarcinoma',
    'human_skin_HSDWXHMTACP_Sample_1',
    'human_skin_HSDWXHMTACP_Sample_2',
    'human_skin_HSPDXHSGEP_MelanomaPP',
    'human_skin_HSPDXHSGEPWCA_MelanomaPPA',
    'human_tonsil_HTDWXHMTACP_Follicular_Lymphoid_Hyperplasia',
    'human_tonsil_HTDWXHMTACP_Reactive_Follicular_Hyperplasia',
]


if __name__ == "__main__":
    args = get_args_parser().parse_args()


    main(args)
