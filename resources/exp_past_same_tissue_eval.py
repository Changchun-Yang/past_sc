                                           
import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import timm
import h5py
import anndata as ad
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import scanpy as sc
import mygene

from exp_past_same_tissue_train_original_aligned import (
    group_samples_by_tissue,
    parse_token_folder_name,
    get_subfiles,
    CombinedDataset,
    NewPASTDataset,
    set_seed,
    align_name,
    CFGP,
    mnt_args,
    library_size_normalize,
    log_transform
)

def get_args_parser():
    parser = argparse.ArgumentParser(description='PAST Same-Tissue Evaluation')

    parser.add_argument('--model_dirs', type=str, nargs='+', required=True,
                        help='Model directories; multiple directories can be evaluated')

    parser.add_argument('--test_sample', type=str, default=None,
                        help='Optional sample ID for leave-one-out evaluation')

    parser.add_argument('--output_file', type=str, default='test_results.json',
                        help='Output filename for aggregated evaluation results')
                        
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory for evaluation results and h5ad files')

    parser.add_argument('--cell_path', type=str, default='/ibex/project/c2277/data/Pathomics/tokenized/pretrain',
                        help='Path to tokenized cell features')

    parser.add_argument('--image_path', type=str, default='/ibex/project/c2277/data/Pathomics/raw',
                        help='Path to histology images')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Evaluation batch size')

    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (match training seed)')
                        
    parser.add_argument('--metadata', action='store_true',
                        help='Save additional metadata to h5ad')

    return parser


def get_sample_feature_indices(cell_path):

    with h5py.File(cell_path, 'r') as h5f:
        indices_data = h5f["X_model_indices"][:]
        feature_indices_set = set()
        for indices in indices_data:
            if indices is not None and len(indices) > 0:
                feature_indices_set.update(indices)
    return feature_indices_set

def get_spatial_coordinates(cell_path):

    with h5py.File(cell_path, 'r') as h5f:
        x_coords = h5f["image_x"][:]
        y_coords = h5f["image_y"][:]
        coordinates = np.column_stack((x_coords, y_coords))
    return coordinates

def get_cell_types(cell_path):

    cell_types = None
    try:
        with h5py.File(cell_path, 'r') as h5f:
            if "cell_type" in h5f:
                cell_types = h5f["cell_type"][:]
    except:
        pass
    return cell_types

def prepare_test_data(target_tissue, test_sample, data_idx, args):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dataset_name = "Xenium"


    test_samples = [args.test_sample] if args.test_sample else (
        [test_sample] if isinstance(test_sample, str) else list(test_sample or []))
    if not test_samples:
        raise ValueError('Same-tissue evaluation requires held-out target_sample in train_config.json')
    tissue_groups = group_samples_by_tissue(align_name)
    allowed = set(tissue_groups.get(target_tissue, []))
    if not set(test_samples).issubset(allowed):
        raise ValueError(f'Test slides not in target tissue {target_tissue}: {test_samples}')

    dataset_list = []
    sample_feature_indices = set()
    test_sample_path = None
    spatial_coords = None
    cell_types = None
    sample_info = {"sample_name": None}
    coord_parts, type_parts = [], []

    name_files = get_subfiles(os.path.join(args.cell_path, dataset_name))
    for file_name in name_files:
        file_name_without_ext = os.path.splitext(os.path.basename(file_name))[0]
        if file_name_without_ext in test_samples:
            specie, tissue, doner, sample = parse_token_folder_name(file_name_without_ext)
            img_path = os.path.join(args.image_path, dataset_name, doner, sample)
            sample_path = os.path.join(args.cell_path, dataset_name, file_name)
            test_sample_path = sample_path
            sample_info["sample_name"] = file_name_without_ext
            sample_info["tissue"] = tissue

            sample_indices = get_sample_feature_indices(sample_path)
            sample_feature_indices.update(sample_indices)
            coord_parts.append(get_spatial_coordinates(sample_path))
            type_parts.append(get_cell_types(sample_path))

            dataset = NewPASTDataset(
                image_path=img_path,
                data_idx=data_idx,
                cell_path=sample_path,
                padding=CFGP.image_size / 2,
                img_transform=transform,
            )
            dataset_list.append(dataset)

    test_dataset = CombinedDataset(dataset_list) if dataset_list else None
    if coord_parts:
        spatial_coords = np.concatenate(coord_parts, axis=0)
    if type_parts and all(ct is not None for ct in type_parts):
        cell_types = np.concatenate(type_parts, axis=0)
    test_sample_path = test_samples if len(test_samples) > 1 else test_sample_path
    sample_info['sample_name'] = test_samples if len(test_samples) > 1 else sample_info['sample_name']
    test_sample_feature_idx = sample_feature_indices
    print(f"Same-tissue evaluation samples: {test_samples}")
    print(f"Number of cells: {len(test_dataset) if test_dataset else 0}")
    if test_dataset is None or len(test_dataset) == 0:
        raise ValueError("Test dataset is empty")

    return test_dataset, test_sample_feature_idx, test_sample_path, spatial_coords, cell_types, sample_info

def get_R(data1, data2, dim=1, func=pearsonr):


    adata1 = data1 if isinstance(data1, np.ndarray) else data1.X
    adata2 = data2 if isinstance(data2, np.ndarray) else data2.X
    
    r1, p1 = [], []

    total = data1.shape[dim]
    with tqdm(total=total, desc=f"Calculating {'cell' if dim==0 else 'gene'} correlations") as pbar:
        for g in range(total):
            try:
                if dim == 1:
                    r, pv = func(adata1[:, g], adata2[:, g])
                elif dim == 0:
                    r, pv = func(adata1[g, :], adata2[g, :])
                r1.append(r)
                p1.append(pv)
            except:
                r1.append(np.nan)
                p1.append(np.nan)
            pbar.update(1)
    
    r1 = np.array(r1)
    p1 = np.array(p1)
    return r1, p1

def calculate_metrics(all_preds, all_targets, data_idx, sample_feature_idx=None):

    preds_np = all_preds.numpy() if isinstance(all_preds, torch.Tensor) else all_preds
    targets_np = all_targets.numpy() if isinstance(all_targets, torch.Tensor) else all_targets


    if sample_feature_idx is not None:
        position_indices = [i for i, gene in enumerate(data_idx) if gene in sample_feature_idx]
    else:
        position_indices = list(range(len(data_idx)))

    if not position_indices:
        raise ValueError("No shared genes between model outputs and the test sample")

    intersection_indices = [data_idx[i] for i in position_indices]
    final_preds = preds_np[:, position_indices]
    final_targets = targets_np[:, position_indices]

    mse = float(((final_preds - final_targets) ** 2).mean())

    gene_pccs, gene_pvalues = get_R(final_preds, final_targets, dim=1)
    valid_gene_pccs = gene_pccs[~np.isnan(gene_pccs)]
    avg_gene_pcc = float(np.mean(valid_gene_pccs)) if len(valid_gene_pccs) > 0 else 0.0
    median_gene_pcc = float(np.median(valid_gene_pccs)) if len(valid_gene_pccs) > 0 else 0.0

    cell_pccs, cell_pvalues = get_R(final_preds, final_targets, dim=0)
    valid_cell_pccs = cell_pccs[~np.isnan(cell_pccs)]
    avg_cell_pcc = float(np.mean(valid_cell_pccs)) if len(valid_cell_pccs) > 0 else 0.0
    median_cell_pcc = float(np.median(valid_cell_pccs)) if len(valid_cell_pccs) > 0 else 0.0

    preds_flat = final_preds.flatten()
    targets_flat = final_targets.flatten()

    try:
        global_pcc, global_pvalue = pearsonr(preds_flat, targets_flat)
        if np.isnan(global_pcc):
            global_pcc, global_pvalue = 0.0, 1.0
    except Exception:
        global_pcc, global_pvalue = 0.0, 1.0

    try:
        global_spearman, global_spearman_pvalue = spearmanr(preds_flat, targets_flat)
        if np.isnan(global_spearman):
            global_spearman, global_spearman_pvalue = 0.0, 1.0
    except Exception:
        global_spearman, global_spearman_pvalue = 0.0, 1.0

    return {
        'mse': mse,
        'global_pcc': global_pcc,
        'global_pcc_pvalue': global_pvalue,
        'global_spearman': global_spearman,
        'global_spearman_pvalue': global_spearman_pvalue,
        'gene_pcc_mean': avg_gene_pcc,
        'gene_pcc_median': median_gene_pcc,
        'cell_pcc_mean': avg_cell_pcc,
        'cell_pcc_median': median_cell_pcc,
        'intersection_count': len(intersection_indices),
        'data_idx_count': len(data_idx),
        'sample_feature_count': len(sample_feature_idx) if sample_feature_idx is not None else len(data_idx),
        'intersection_indices': intersection_indices,
        'position_indices': intersection_indices,
        'data': {'preds': final_preds, 'targets': final_targets},
        'gene_corrs': {'pccs': gene_pccs, 'pvalues': gene_pvalues},
        'cell_corrs': {'pccs': cell_pccs, 'pvalues': cell_pvalues},
    }


def save_anndata_results(preds, targets, data_idx, position_indices, spatial_coords, cell_types, sample_info, output_file):


    BASE_PATH = '/ibex/project/c2277/data/Pathomics/preprocessed'

    ori_model = sc.read_h5ad(
        f"{BASE_PATH}/model_means/model.h5ad"
    )


    selected_indices = position_indices if position_indices is not None else data_idx
    ens_ids = [ori_model.var_names[idx] for idx in selected_indices]

    adata = ad.AnnData(X=targets)

    adata.layers['predicted'] = preds

    adata.uns['sample_info'] = sample_info

    adata.uns['feature_ids'] = ens_ids
    ori_gene_name = [ori_model.var_names[idx] for idx in data_idx]

    adata.uns['original_data_idx'] = ori_gene_name
    if position_indices is not None:
        adata.uns['position_indices'] = ens_ids

    adata.var_names = [f"Gene_{id}_1" if id in ori_gene_name else f"Gene_{id}_0" for id in ens_ids]

    if spatial_coords is not None:
        adata.obsm['spatial'] = spatial_coords

    if cell_types is not None:
        adata.obs['cell_type'] = cell_types

    gene_pccs, gene_pvals = get_R(preds, targets, dim=1)
    adata.var['pcc'] = gene_pccs
    adata.var['pcc_pvalue'] = gene_pvals

    cell_pccs, cell_pvals = get_R(preds, targets, dim=0)
    adata.obs['pcc'] = cell_pccs
    adata.obs['pcc_pvalue'] = cell_pvals

    cell_mse = np.mean((targets - preds) ** 2, axis=1)
    adata.obs['mse'] = cell_mse

    adata.write_h5ad(output_file)
    print(f"Saved AnnData results to: {output_file}")
    
    return adata

def test_model(model_dir, args):

    print(f"\n===== Evaluating model: {model_dir} =====")
    
    device = torch.device(args.device)

    model_output_dir = os.path.join(args.output_dir, os.path.basename(model_dir))

    if args.test_sample is not None:
        model_output_dir = os.path.join(model_output_dir, args.test_sample)

    os.makedirs(model_output_dir, exist_ok=True)

    try:
                               
        config_path = os.path.join(model_dir, "train_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                train_config = json.load(f)
            
            train_mode = train_config.get('train_mode', 'same_tissue')
            target_tissue = train_config.get('target_tissue')
            target_sample = train_config.get('test_samples', train_config.get('target_sample'))
            feature_mode = train_config.get('feature_mode', 'union')
        else:
            raise FileNotFoundError(f'Missing train_config.json: {config_path}')

        if train_mode != 'same_tissue':
            raise ValueError(f'Expected same_tissue checkpoint, got {train_mode}')
        selected_tests = [args.test_sample] if args.test_sample else (
            [target_sample] if isinstance(target_sample, str) else list(target_sample or []))
        held_out_slides = set([target_sample] if isinstance(target_sample, str) else (target_sample or []))
        if not set(selected_tests).issubset(held_out_slides):
            raise ValueError(f'Test slides not declared held out during training: {selected_tests}')
        training_slides = set(train_config.get('train_samples', []))
        if training_slides.intersection(selected_tests):
            raise ValueError(f'Test/training slide overlap: {sorted(training_slides.intersection(selected_tests))}')
        print(f"Training mode: {train_mode}")
        print(f"Target tissue: {target_tissue}")
        print(f"Feature mode: {feature_mode}")

        model_path = os.path.join(model_dir, "best_model.pt")
        checkpoint = torch.load(model_path, map_location=device)
        model_state = checkpoint['model']
        data_idx = checkpoint['data_idx']
        val_loss = checkpoint.get('val_loss', 'N/A')

        num_features = len(data_idx)
        model = timm.create_model(
            "vit_large_patch16_224", 
            img_size=224, 
            patch_size=16, 
            init_values=1e-5, 
            num_classes=num_features,
            dynamic_img_size=True
        )
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        
        print(f"Model loaded, feature count: {num_features}")

        test_dataset, test_sample_feature_idx, test_sample_path, spatial_coords, cell_types, sample_info = prepare_test_data(
            target_tissue, target_sample, data_idx, args
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        print("Starting evaluation...")
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation progress"):
                imgs = batch['image'].to(device, non_blocking=True)
                labels = log_transform(library_size_normalize(batch['X']))
                labels = labels.to(device, non_blocking=True)
                outputs = model(imgs)
                all_preds.append(outputs.cpu())
                all_targets.append(labels.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        print("Computing metrics...")
        metrics = calculate_metrics(
            all_preds, all_targets, data_idx, test_sample_feature_idx
        )

        preds_data = metrics['data']['preds']
        targets_data = metrics['data']['targets']
        position_indices = metrics['position_indices']

        print("Saving h5ad output...")
        h5ad_file = os.path.join(model_output_dir, 'expression_data.h5ad')
        save_anndata_results(
            preds_data, 
            targets_data, 
            data_idx, 
            position_indices, 
            spatial_coords, 
            cell_types, 
            sample_info, 
            h5ad_file
        )

        del metrics['data']
        del metrics['gene_corrs']['pccs']
        del metrics['gene_corrs']['pvalues']
        del metrics['cell_corrs']['pccs']
        del metrics['cell_corrs']['pvalues']

        with open(os.path.join(model_output_dir, 'detailed_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)

        result = {
            "model_dir": model_dir,
            "train_mode": train_mode,
            "target_tissue": target_tissue,
            "target_sample": selected_tests,
            "feature_mode": feature_mode,
            "test_samples_count": len(test_dataset),
            "test_sample_path": test_sample_path,
            "feature_count": num_features,
            "h5ad_file": h5ad_file,
            "mse": metrics['mse'],
            "global_pcc": metrics['global_pcc'],
            "global_spearman": metrics['global_spearman'],
            "gene_pcc_mean": metrics['gene_pcc_mean'],
            "gene_pcc_median": metrics['gene_pcc_median'],
            "cell_pcc_mean": metrics['cell_pcc_mean'],
            "cell_pcc_median": metrics['cell_pcc_median'],
            "intersection_count": metrics['intersection_count'],
            "data_idx_count": metrics['data_idx_count'],
            "sample_feature_count": metrics['sample_feature_count'],
            "training_val_loss": float(val_loss) if isinstance(val_loss, (int, float, np.number)) else None,
        }

        print(f"\nEvaluation summary:")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"Global PCC: {metrics['global_pcc']:.6f}")
        print(f"Global Spearman: {metrics['global_spearman']:.6f}")
        print(f"Mean gene-level PCC: {metrics['gene_pcc_mean']:.6f}")
        print(f"Mean cell-level PCC: {metrics['cell_pcc_mean']:.6f}")
        print(f"Feature intersection size: {metrics['intersection_count']}/{metrics['data_idx_count']} (test sample: {metrics['sample_feature_count']} features)")
        
        return result
    
    except Exception as e:
        print(f"Model evaluation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "model_dir": model_dir,
            "error": str(e)
        }

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def main():
             
    args = get_args_parser().parse_args()

    args.cell_path = mnt_args + args.cell_path
    args.image_path = mnt_args + args.image_path

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    results = []
    for model_dir in args.model_dirs:
        if not os.path.exists(model_dir):
            print(f"Warning: directory does not exist {model_dir}")
            continue
        
        result = test_model(model_dir, args)
        if result:
            results.append(result)

    if results:
        with open(os.path.join(args.output_dir, args.output_file), 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        print(f"\nAll evaluation results saved to: {os.path.join(args.output_dir, args.output_file)}")

        print("\nEvaluation summary:")
        for result in results:
            model_name = os.path.basename(result["model_dir"])
            if "error" in result:
                print(f"{model_name}: error - {result['error']}")
            else:
                print(f"{model_name}:")
                print(f"  MSE={result['mse']:.6f}")
                print(f"  Global PCC={result['global_pcc']:.6f}")
                print(f"  Gene PCC={result['gene_pcc_mean']:.6f}")
                print(f"  Cell PCC={result['cell_pcc_mean']:.6f}")
                print(f"  Feature intersection: {result['intersection_count']}/{result['data_idx_count']}")
    else:
        print("No evaluation results generated")

if __name__ == "__main__":
    main()

