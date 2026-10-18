import os
import sys
sys.path.append(os.path.join(os.path.dirname(f"{__file__}"), '../..'))

# Detect the runtime environment
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

    # Distributed training arguments
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Training parameters
    # Sample selection arguments
    parser.add_argument('--train_mode', type=str, default='leave_one_out', 
                        choices=['leave_one_out'],
                        help='Training mode: leave_one_out (exclude the target tissue)')
    parser.add_argument('--target_tissue', type=str, default=None,
                        help='Target tissue type to exclude from training')
    parser.add_argument('--target_sample', type=str, nargs='+', default=None,
                        help='Optional training samples (must not belong to the excluded target tissue)')
    parser.add_argument('--feature_mode', type=str, default='union',
                        choices=['union', 'intersection', 'union_test_intersection'],
                        help='Feature selection: union, intersection, or intersection of training union and test intersection')
    parser.add_argument('--dist_eval', action='store_true', 
                        help='Enable distributed evaluation (parallel evaluation across processes)')
    
    # Optimizer arguments
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='lower lr bound for schedulers')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='epochs to warmup LR')
    
    # Other arguments
    parser.add_argument('--output_dir', default='./result/task1', help='path where to save')
    parser.add_argument('--log_dir', default='./result/task1', help='path where to tensorboard log')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--save_interval', type=int, default=2000, help='Save checkpoint every x steps')
    parser.add_argument('--accum_iter', type=int, default=3, help='Gradient accumulation steps')
    
    return parser


def group_samples_by_tissue(align_name_list):
    """
    Group samples by tissue type.
    Returns: {tissue_type: [sample_names, ...]}
    """
    tissue_groups = defaultdict(list)
    
    for sample_name in align_name_list:
        specie, tissue, doner, sample = parse_token_folder_name(sample_name)
        tissue_groups[tissue].append(sample_name)
    
    return tissue_groups


def select_samples_for_training(align_name_list, args):
    """
    Select training samples using leave-one-tissue-out selection.
    Returns: list of training sample names.
    """
    # Group samples by tissue type.
    tissue_groups = group_samples_by_tissue(align_name_list)

    # Initialize the training sample pool and the excluded samples.
    train_samples_pool = []
    excluded_samples = []

    # Leave-one-out mode: exclude all samples from the target tissue.
    if args.target_tissue is None:
        raise ValueError("leave_one_out mode requires the target_tissue argument")

    if args.target_tissue not in tissue_groups:
        raise ValueError(f"Target tissue type not found: {args.target_tissue}")

    # Exclude all samples belonging to the target tissue.
    excluded_samples = tissue_groups[args.target_tissue]

    if args.target_sample:
        # Use only specified training samples, excluding the target tissue.
        valid_samples = []
        for sample in args.target_sample:
            if sample in align_name_list:
                _, sample_tissue, _, _ = parse_token_folder_name(sample)
                if sample_tissue != args.target_tissue:
                    valid_samples.append(sample)
                else:
                    if misc.is_main_process():
                        print(f"Warning: skipping training sample {sample} because it belongs to the target test tissue {args.target_tissue}")

        if not valid_samples:
            raise ValueError("No valid training samples")
        train_samples_pool = valid_samples
    else:
        # Otherwise, use all samples except those from the target tissue.
        for tissue, samples in tissue_groups.items():
            if tissue != args.target_tissue:
                train_samples_pool.extend(samples)

    # Print information about the selected training samples.
    if misc.is_main_process():
        print(f"Training mode: {args.train_mode}")
        if args.target_tissue:
            print(f"Target tissue: {args.target_tissue}")

        if excluded_samples:
            if len(excluded_samples) == 1:
                print(f"Excluded sample: {excluded_samples[0]}")
            else:
                print(f"Excluded samples: {len(excluded_samples)} samples")

        print(f"Training sample pool size: {len(train_samples_pool)}")
        if train_samples_pool:
            tissue_counts = {}
            for sample_name in train_samples_pool:
                tissue = parse_token_folder_name(sample_name)[1]
                tissue_counts[tissue] = tissue_counts.get(tissue, 0) + 1
            tissue_summary = [f"{tissue}({count})" for tissue, count in sorted(tissue_counts.items())]
            print(f"Training tissue distribution: {', '.join(tissue_summary)}")

    return train_samples_pool


def get_data_indices(train_samples_pool, args):
    """
    Generate feature indices according to the selected feature mode
    """
    all_indices_dict = {}
    
    # Load feature indices for the training samples
    for sample_name in train_samples_pool:
        index_path = os.path.join("./data_idx", f"{sample_name}.npy")
        if os.path.exists(index_path):
            indices = np.load(index_path, allow_pickle=True).tolist()
            all_indices_dict[sample_name] = set(indices)
        else:
            if misc.is_main_process():
                print(f"Warning: feature index file not found for sample {sample_name}")
    
    # Load the target tissue feature indices for feature_mode='union_test_intersection'
    test_indices_sets = []
    if args.target_tissue and args.feature_mode == 'union_test_intersection':
        tissue_groups = group_samples_by_tissue(align_name)
        if args.target_tissue in tissue_groups:
            for sample_name in tissue_groups[args.target_tissue]:
                index_path = os.path.join("./data_idx", f"{sample_name}.npy")
                if os.path.exists(index_path):
                    indices = np.load(index_path, allow_pickle=True).tolist()
                    test_indices_sets.append(set(indices))
    
    # Generate final feature indices according to the selected mode
    if args.feature_mode == 'union':
        # Union mode: union of features across all training samples
        union_set = set()
        for indices in all_indices_dict.values():
            union_set.update(indices)
        data_idx = list(union_set)
    
    elif args.feature_mode == 'intersection':
        # Intersection mode: features shared by all training samples
        if not all_indices_dict:
            data_idx = []
        else:
            # Start with the first set and intersect with subsequent sets
            sets_list = list(all_indices_dict.values())
            intersection_set = sets_list[0]
            for indices in sets_list[1:]:
                intersection_set &= indices
            data_idx = list(intersection_set)
    
    elif args.feature_mode == 'union_test_intersection':
        # Intersection of training union and test intersection
        # First, compute the union of training features
        train_union = set()
        for indices in all_indices_dict.values():
            train_union.update(indices)
        
        # Next, compute the intersection of test features
        if not test_indices_sets:
            test_intersection = set()
        else:
            test_intersection = test_indices_sets[0]
            for indices in test_indices_sets[1:]:
                test_intersection &= indices
        
        # Finally, intersect the two feature sets
        data_idx = list(train_union & test_intersection) if test_intersection else list(train_union)
    
    # Sort to ensure consistent ordering
    data_idx.sort()
    
    if misc.is_main_process():
        print(f"Feature selection mode: {args.feature_mode}")
        print(f"Training sample count: {len(all_indices_dict)}/{len(train_samples_pool)}")
        print(f"Final feature count: {len(data_idx)}")
    
    return data_idx


def build_datasets(args, transform, train_samples_pool, data_idx):
    """Build the training and validation datasets"""
    if misc.is_main_process():
        print("Building datasets...")
    
    dataset_name = "Xenium"
    dataset_list = []
    
    # Find all available data files
    name_files = get_subfiles(os.path.join(args.cell_path, dataset_name))
    
    # Process training samples
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
    
    # Create the combined dataset
    combined_dataset = CombinedDataset(dataset_list) if dataset_list else None
    
    if combined_dataset is None or len(combined_dataset) == 0:
        raise ValueError("No valid datasets")
    
    # Split into training and validation datasets with a 90/10 ratio
    train_size = int(0.9 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    if misc.is_main_process():
        print(f"Training set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


# Dataset: reuse the existing NewPASTDataset implementation
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

        return padded_image  # H x W x C

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
            # print(cluster)
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
                # Invalid shape; return an invalid item
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
            if misc.get_rank() == 0:  # Print errors only in the main process
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


# Data processing functions
def library_size_normalize(x: torch.Tensor, scale=1e4) -> torch.Tensor:
    """
    Perform library size normalization on the tensor.
    Each row is divided by the sum of its elements and then scaled to 10,000.
    """
    # x = torch.abs(x)
    
    if x.dim() == 1:
        row_sums = torch.sum(x, dim=0, dtype=torch.float32)
    else:
        row_sums = torch.sum(x, dim=1, keepdim=True, dtype=torch.float32)

    # Avoid division by zero
    row_sums = torch.clamp(row_sums, min=1e-10)
    normalized = x / row_sums * scale
    return normalized

def log_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Apply a log transformation to the tensor.
    Adds 1 to avoid log(0) issues.
    """
    return torch.log10(x + 1)


# Distributed training with gradient accumulation
def train_epoch(model, train_loader, optimizer, device, epoch, loss_scaler, log_writer=None, args=None):
    """Train one epoch with gradient accumulation and valid-sample filtering"""
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    
    # Gradient accumulation settings
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None and misc.get_rank() == 0:
        print('log_dir: {}'.format(log_writer.log_dir))

    # Iterate through the data loader
    for data_iter_step, batch in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
        # Adjust the learning rate
        if data_iter_step % accum_iter == 0:
            lr = adjust_learning_rate(optimizer, data_iter_step / len(train_loader) + epoch, args)

        # Filter valid samples
        valid_mask = batch.get('valid', torch.ones(batch['image'].size(0), dtype=torch.bool))
        
        # Skip batches without valid samples
        if not valid_mask.any():
            continue
        
        # Extract valid samples
        valid_images = batch['image'][valid_mask]
        valid_X = batch['X'][valid_mask]
        
        # Check the number of valid samples again
        current_batch_size = valid_images.size(0)
        if current_batch_size == 0:
            continue
        
        # Prepare target values
        target = log_transform(library_size_normalize(valid_X))
        target = target.to(device, non_blocking=True)
        
        # Move valid images to GPU
        valid_images = valid_images.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            output = model(valid_images)
            loss = nn.functional.mse_loss(output, target)
            loss = loss / accum_iter  # Scale loss for gradient accumulation
            
        # Record the loss
        loss_value = loss.item()
        
        # Check whether the loss is finite
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        # Backward pass and gradient accumulation
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
                    
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            
        torch.cuda.synchronize()
        
        # Update metrics
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        metric_logger.update(loss=loss_value_reduce)
        metric_logger.update(lr=lr)
        
        # TensorBoard logging
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0 and misc.get_rank() == 0:
            epoch_1000x = int((data_iter_step / len(train_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            
        # # Periodically save checkpoints
        # if (data_iter_step + 1) % args.save_interval == 0 and misc.get_rank() == 0:
        #     save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch}_step{data_iter_step + 1}.pth')
        #     misc.save_on_master({
        #         'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'epoch': epoch,
        #         'args': args,
        #         'scaler': loss_scaler.state_dict(),
        #     }, save_path)
        #     print(f"Saved checkpoint at {save_path}")
    
    # Aggregate statistics across processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def adjust_learning_rate(optimizer, epoch, args):
    """Use half-cycle cosine LR decay after warmup"""
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
    """Validation with distributed or sequential evaluation"""
    model.eval()
    loss_meter = AvgMeter()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), 
                          disable=misc.get_rank() != 0):  # Show progress bar only on the main process
            # Filter valid samples
            valid_mask = batch.get('valid', torch.ones(batch['image'].size(0), dtype=torch.bool))
            if not valid_mask.any():
                continue
                
            valid_images = batch['image'][valid_mask]
            valid_X = batch['X'][valid_mask]
            
            # Check batch size
            current_batch_size = valid_images.size(0)
            if current_batch_size == 0:
                continue
            
            # Process data
            target = log_transform(library_size_normalize(valid_X))
            target = target.to(device, non_blocking=True)
            valid_images = valid_images.to(device, non_blocking=True)
            
            # Run model predictions
            prediction = model(valid_images)
            loss = nn.functional.mse_loss(prediction, target)
            
            # Update loss metrics
            loss_meter.update(loss.item(), current_batch_size)
    
    # Return local loss; main synchronizes it using misc.all_reduce_mean
    return loss_meter.avg


def main(args):
    # Initialize the distributed environment
    misc.init_distributed_mode(args)
    
    device = torch.device(args.device)
    
    # Set the random seed
    seed = args.seed + misc.get_rank()
    set_seed(seed)
    
    # Construct the output directory name
    output_parts = [args.train_mode]
    if args.target_tissue:
        output_parts.append(args.target_tissue)
    if args.target_sample and len(args.target_sample) == 1:
        # sample_short = args.target_sample[0].split('_')[-3] + '_' + args.target_sample[0].split('_')[-1]
        output_parts.append(args.target_sample[0])
    output_parts.append(args.feature_mode)
    
    output_dir_name = "_".join(output_parts)
    args.output_dir = os.path.join(args.output_dir, output_dir_name)
    args.log_dir = os.path.join(args.log_dir, output_dir_name) if args.log_dir else None
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
    
    # Configure data paths
    args.cell_path = mnt_args + args.cell_path
    args.image_path = mnt_args + args.image_path
    
    # Select the training sample pool
    train_samples_pool = select_samples_for_training(align_name, args)
    
    # Generate feature indices
    data_idx = get_data_indices(train_samples_pool, args)
    X_channel = len(data_idx)
    
    if len(data_idx) == 0:
        if misc.is_main_process():
            print("Error: no valid features. Check the feature selection arguments or data.")
        return
    
    # Image transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    # Build and split the training/validation datasets
    train_dataset, val_dataset = build_datasets(args, transform, train_samples_pool, data_idx)
    
    # Save training configuration and feature indices
    if misc.is_main_process():
        # Save feature indices
        np.save(os.path.join(args.output_dir, "feature_indices.npy"), np.array(data_idx))
        
        # Save training configuration
        config_info = {
            'train_mode': args.train_mode,
            'target_tissue': args.target_tissue,
            'target_sample': args.target_sample,
            'feature_mode': args.feature_mode,
            'train_samples_count': len(train_samples_pool),
            'feature_count': len(data_idx),
            'seed': args.seed
        }
        
        with open(os.path.join(args.output_dir, "train_config.json"), 'w') as f:
            json.dump(config_info, f, indent=2)
    
    # Configure data samplers for distributed training
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
    
    # Create data loaders
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
    
    # Create the model
    model = timm.create_model(
        "vit_large_patch16_224", 
        img_size=224, 
        patch_size=16, 
        init_values=1e-5, 
        num_classes=X_channel,
        dynamic_img_size=True
    )
    
    # Load pretrained weights
    if misc.is_main_process():
        print("Loading pre-trained weights...")
    checkpoint_dir = f"{mnt_args}{DefaultPaths.SPATIAL}/checkpoint"
    state_dict = torch.load(os.path.join(checkpoint_dir, "past_latest_2.pth"), map_location="cpu")["model"]
    state_dict = {k.replace("image_encoder.model.", ""): v for k, v in state_dict.items() 
                  if "image_encoder.model." in k}
    model.load_state_dict(state_dict, strict=False)
    
    # Move the model to GPU and configure distributed training
    model.to(device)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu]
        )
    
    # Set up TensorBoard
    if misc.is_main_process() and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    
    # Configure the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=CFGP.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Set up mixed precision training
    loss_scaler = NativeScaler()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
            sampler_val.set_epoch(epoch)
        
        # Training phase
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
        
        # Validation phase
        val_loss = val_epoch(model, val_loader, device)
        
        # Synchronize validation loss
        if misc.is_dist_avail_and_initialized():
            val_loss = misc.all_reduce_mean(val_loss)
        
        # Update the best model
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        # Save the model only on the main process
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
        
        # Synchronize distributed processes
        if misc.is_dist_avail_and_initialized():
            dist.barrier()
    
    if misc.is_main_process():
        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")


# Define the sample ID list
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

    # args.distributed = False  # for debug
    #

    # args.train_mode = 'leave_one_out'
    # args.target_tissue = 'liver'
    # args.target_sample = ['human_heart_HHDWXHMTACP_Nondiseased', 'human_skin_HSDWXHMTACP_Sample_1']
    # args.feature_mode = 'union_test_intersection'

    main(args)
