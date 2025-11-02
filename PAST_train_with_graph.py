import os
from tqdm import tqdm

import torch
from torch import nn
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.data import DataLoader, ConcatDataset

from torch.utils.tensorboard import SummaryWriter
import json

import config_files.config_train as CFG
import config_files.config_past_train as CFGP
from dataset import PASTDataset_v2, PASTDataset_v3, CombinedDataset, graph_collate_fn
from models import PASTModel, PASTModelGraph
from utils.util import (
    AvgMeter, get_lr, get_subdirs, parse_token_folder_name, 
    set_seed, worker_init_fn, complete_masking, preprocess_cell_data, process_graph_batch
)
import argparse
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import math
import sys
from typing import Iterable
from pathlib import Path


def get_args_parser():
    # ------------------------ Argument Parser ------------------------
    parser = argparse.ArgumentParser(description='PAST Training')

    parser.add_argument('--exp_name', type=str, default='past', help='Name of the experiment/checkpoint folder')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=20, help='Max number of epochs')
    parser.add_argument('--num_workers', type=int, default=3, help='Number of workers for DataLoader')
    parser.add_argument('--cell_path', type=str, default='/ibex/project/c2277/data/Pathomics/tokenized/pretrain', help='Path to cell token data')
    parser.add_argument('--image_path', type=str, default='/ibex/project/c2277/data/Pathomics/raw', help='Path to raw image data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
    parser.add_argument('--pin_mem', action='store_true',
                            help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--use_graph', default=True, type=bool)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')

    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                            help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                            help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                            help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                            help='epochs to warmup LR')

    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='init_method for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='backend for distributed training')

    # distributed training parameters
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--model', type=str, default='past', help='Model name')
    parser.add_argument('--save_interval', type=int, default=2000, help='Steps to save checkpoint')
    parser.add_argument('--accum_iter', type=int, default=5, help='Gradient accumulation steps')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    return parser


# ------------------------ DataLoader Builder ------------------------
def build_loaders(args):
    print("Building loaders...")
    assay_dirs = get_subdirs(args.cell_path)
    dataset_list = []

    for assay_dir in assay_dirs:
        if assay_dir != 'Xenium':  # Customize your dataset filter here
            continue

        # datasetv2 many files
        # name_dirs = sorted(get_subdirs(os.path.join(args.cell_path, assay_dir)))
        # for name_dir in name_dirs:
        #     specie, tissue, doner, sample = parse_token_folder_name(name_dir)
        #     img_path = os.path.join(args.image_path, assay_dir, doner, sample)
        #     if not os.path.isdir(img_path):
        #         continue
        #
        #     dataset = PASTDataset_v2(
        #         image_path=img_path,
        #         cell_path=os.path.join(args.cell_path, assay_dir, name_dir),
        #         context_length=CFGP.cell_context_length,
        #         padding=CFGP.image_size / 2,
        #         model_transform=CFGP.image_model_name
        #     )
        #     dataset_list.append(dataset)

        # datasetv3 single file
        name_files = sorted(os.listdir(os.path.join(args.cell_path, assay_dir)))
        for name_file in name_files:
            specie, tissue, doner, sample = parse_token_folder_name(Path(name_file).stem)
            img_path = os.path.join(args.image_path, assay_dir, doner, sample)
            if not os.path.isdir(img_path):
                continue

            dataset = PASTDataset_v3(
                image_path=img_path,
                cell_file=os.path.join(args.cell_path, assay_dir, name_file),
                context_length=CFGP.cell_context_length,
                padding=CFGP.image_size / 2,
                model_transform=CFGP.image_model_name
            )
            dataset_list.append(dataset)

    dataset_train = ConcatDataset(dataset_list)
    return dataset_train
    # dataset = CombinedDataset(dataset_list)
    # train_size = int(0.9999 * len(dataset))
    # test_size = len(dataset) - train_size

    # train_dataset, test_dataset = torch.utils.data.random_split(
    #     dataset,
    #     [train_size, test_size],
    #     generator=torch.Generator()
    # )

    # print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")



# ------------------------ Cleanup ------------------------
def cleanup():
    dist.destroy_process_group()

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
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



def train_epoch(model: torch.nn.Module, model_without_ddp, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                device: torch.device, epoch: int, loss_scaler, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (batch, graph_batch) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
   
        # process cell data
        batch = preprocess_cell_data(batch, CFGP)
        batch = complete_masking(batch, CFGP.cell_masking_p, CFGP.cell_n_tokens + 5)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if
                 k in ["image", "masked_indices", "attention_mask"]}

        # process graph data
        graph_batch = process_graph_batch(graph_batch, CFGP)
        graph_batch = graph_batch.to(device)

        # Forward pass and gradient accumulation
        with torch.cuda.amp.autocast():
            loss, temp = model(
                image=batch["image"],
                masked_indices=batch["masked_indices"],
                attention_mask=batch["attention_mask"],
                graph_batch=graph_batch
            )
            loss = loss / args.accum_iter

        loss_value = loss.item()
        temp_reduce = misc.all_reduce_mean(temp)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(temperature=temp_reduce)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('temperature', temp_reduce, epoch_1000x)

        if (data_iter_step + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch}_step{data_iter_step + 1}.pth')
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            misc.save_on_master(to_save, save_path)
            print(f"Saved checkpoint at {save_path}")
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# ------------------------ Main Training Function ------------------------
def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    set_seed(seed)

    # Build Dataset
    dataset_train = build_loaders(args)

    # Distributed sampler for training
    if True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=graph_collate_fn,
        # worker_init_fn=worker_init_fn,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    # Define model and wrap with DDP
    model = PASTModelGraph()
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    args.lr = CFGP.lr
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # ---------------------- 分组参数设置示例 (如需区分 lr) ----------------------
    vit_params = []
    other_params = []
    for name, param in model_without_ddp.named_parameters():
        if 'image_encoder' in name:
            vit_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": vit_params, "lr": args.lr * 0.1, "weight_decay": CFGP.weight_decay},  # ViT
        {"params": other_params, "lr": args.lr,      "weight_decay": CFGP.weight_decay}, # 其他
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # Resume from checkpoint
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)


    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)
        
        if args.output_dir and (epoch % 1 == 0 or epoch + 1 == args.epochs):
            misc.save_model_checkpoints(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


        # val_loss = val_epoch(model, test_loader)

        # if val_loss.avg < best_loss and dist.get_rank() == 0:
        #     best_loss = val_loss.avg
        #     save_path = f'result/{args.exp_name}/best.pt'
        #     torch.save({
        #         'epoch': epoch,
        #         'global_step': global_step,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'best_loss': best_loss
        #     }, save_path)
        #     print(f"Saved best model at epoch {epoch}, loss: {best_loss}")

    cleanup()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    args.distributed = False  # for debug
    main(args)
