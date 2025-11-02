import functools
import json
import logging
import os
from pathlib import Path
import random
import subprocess
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import pandas as pd
from anndata import AnnData
import scib
from matplotlib import pyplot as plt
from matplotlib import axes
from IPython import get_ipython

from .logger import logger


def complete_masking(batch, p, n_tokens):
    
    padding_token = 1
    cls_token = 3

    indices = batch['X']

    indices = torch.where(indices == 0, torch.tensor(padding_token), indices) # 0 is originally the padding token, we change it to 1
    batch['X'] = indices

    mask = 1 - torch.bernoulli(torch.ones_like(indices), p) # mask indices with probability p
    
    masked_indices = indices * mask # masked_indices 
    masked_indices = torch.where(indices != padding_token, masked_indices, indices) # we just mask non-padding indices
    mask = torch.where(indices == padding_token, torch.tensor(padding_token), mask) # in the model we evaluate the loss of mask position 0
    # so we make the mask of all PAD tokens to be 1 so that it's not taken into account in the loss computation
    
    # Notice for the following 2 lines that masked_indices has already not a single padding token masked
    masked_indices = torch.where(indices != cls_token, masked_indices, indices) # same with CLS, no CLS token can be masked
    mask = torch.where(indices == cls_token, torch.tensor(padding_token), mask) # we change the mask so that it doesn't mask any CLS token

    # 80% of masked indices are masked
    # 10% of masked indices are a random token
    # 10% of masked indices are the real token

    random_tokens = torch.randint(10, n_tokens, size=masked_indices.shape, device=masked_indices.device)
    random_tokens = random_tokens * torch.bernoulli(torch.ones_like(random_tokens)*0.1).type(torch.int64) 

    masked_indices = torch.where(masked_indices == 0, random_tokens, masked_indices) # put random tokens just in the previously masked tokens

    same_tokens = indices.clone()
    same_tokens = same_tokens * torch.bernoulli(torch.ones_like(same_tokens) * 0.1).type(torch.int64)

    masked_indices = torch.where(masked_indices == 0, same_tokens, masked_indices) # put same tokens just in the previously masked tokens

    batch['masked_indices'] = masked_indices
    batch['mask'] = mask
    
    attention_mask = (masked_indices == padding_token)
    batch['attention_mask'] = attention_mask.type(torch.bool)

    return batch


def preprocess_cell_data(batch, config_file):
    """
    Preprocess the cell data by adding auxiliary tokens or finetuneing task.
    """
    x = batch['X']  # Original input sequence: batch_size x seq_length

    # Add auxiliary tokens
    tokens_to_add = []
    if config_file.cell_modality:
        modality = batch['modality'].unsqueeze(1)  # batch_size x 1
        tokens_to_add.append(modality)

    if config_file.cell_assay:
        assay = batch['assay'].unsqueeze(1)  # batch_size x 1
        tokens_to_add.append(assay)

    if config_file.cell_specie:
        specie = batch['specie'].unsqueeze(1)  # batch_size x 1
        tokens_to_add.append(specie)

    if tokens_to_add:
        aux_tokens = torch.cat(tokens_to_add, dim=1)  # batch_size x num_aux_tokens
        x = torch.cat((aux_tokens, x), dim=1)  # batch_size x (seq_length + num_aux_tokens)

    # # Add label to predict
    # if config_file.label in batch.keys():
    #     batch['label'] = batch[config_file.label].to(torch.float32)
    # elif not config_file.predict:
    #     raise NotImplementedError("Label specified not existent in parquet or model.")
    # else:
    #     batch['label'] = batch['specie']  # whatever to label, it's not used
    #
    # if config_file.pool == 'cls':  # Add cls token at the beginning of the set
    #     CLS_TOKEN = 2
    #     cls = torch.ones((x.shape[0], 1), dtype=torch.int32, device=x.device) * CLS_TOKEN  # CLS token is index 2
    #     x = torch.cat((cls, x), dim=1)  # add CLS

    # Truncate or pad the sequence to the context length
    x = x[:, :config_file.cell_context_length]  # Ensure the sequence is not longer than context_length

    batch['X'] = x

    return batch



def process_graph_batch(graph_batch, config):
    """为图批次处理原始特征（不进行掩码）"""
    # 在这里，我们只需确保图批次的特征与模型兼容
    # 例如，添加模态信息，但不进行掩码
    
    # 获取节点特征
    x = graph_batch.x
    
    # 这里可以添加任何必要的预处理，但不进行掩码
    # 例如，可以将特征截断到适当的大小
    if hasattr(graph_batch, 'modality') and config.cell_modality:
        modality = graph_batch.modality.unsqueeze(1)
        x = torch.cat([modality, x], dim=1)
    
    if hasattr(graph_batch, 'assay') and config.cell_assay:
        assay = graph_batch.assay.unsqueeze(1)
        x = torch.cat([assay, x], dim=1)
        
    if hasattr(graph_batch, 'specie') and config.cell_specie:
        specie = graph_batch.specie.unsqueeze(1)
        x = torch.cat([specie, x], dim=1)
    
    # 截断到上下文长度
    x = x[:, :config.cell_context_length]
    
    # 更新图批次的节点特征
    graph_batch.x = x
    
    return graph_batch


def gene_vocabulary():
    """
    Generate the gene name2id and id2name dictionaries.
    """
    pass


def set_seed(seed):
    """
    Sets the seed for all libraries used.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_file_handler(logger: logging.Logger, log_file_path: Path):
    """
    Add a file handler to the logger.
    """
    h = logging.FileHandler(log_file_path)

    # format showing time, name, function, and message
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    h.setFormatter(formatter)
    h.setLevel(logger.level)
    logger.addHandler(h)


def category_str2int(category_strs: List[str]) -> List[int]:
    set_category_strs = set(category_strs)
    name2id = {name: i for i, name in enumerate(set_category_strs)}
    return [name2id[name] for name in category_strs]


def isnotebook() -> bool:
    """check whether excuting in jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return True  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def get_free_gpu():
    import subprocess
    import sys
    from io import StringIO
    import pandas as pd

    gpu_stats = subprocess.check_output(
        [
            "nvidia-smi",
            "--format=csv",
            "--query-gpu=memory.used,memory.free",
        ]
    ).decode("utf-8")
    gpu_df = pd.read_csv(
        StringIO(gpu_stats), names=["memory.used", "memory.free"], skiprows=1
    )
    print("GPU usage:\n{}".format(gpu_df))
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    idx = gpu_df["memory.free"].idxmax()
    print(
        "Find free GPU{} with {} free MiB".format(idx, gpu_df.iloc[idx]["memory.free"])
    )

    return idx


def get_git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def histogram(
    *data: List[np.ndarray],
    label: List[str] = ["train", "valid"],
    color: List[str] = ["blue", "red"],
    figsize: Tuple[int, int] = (9, 4),
    title: Optional[str] = None,
    show: bool = False,
    save: Optional[str] = None,
) -> axes.Axes:
    """
    Plot histogram of the data.

    Args:
        data (List[np.ndarray]): The data to plot.
        label (List[str]): The label of the data.
        color (List[str]): The color of the data.
        figsize (Tuple[int, int]): The size of the figure.
        title (Optional[str]): The title of the figure.
        show (bool): Whether to show the figure.
        save (Optional[str]): The path to save the figure.

    Returns:
        axes.Axes: The axes of the figure.
    """
    # show histogram of the clipped values
    assert len(data) == len(label), "The number of data and labels must be equal."

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
    max_value = max(np.max(data) for data in data)
    ax.hist(
        [d.flatten() for d in data],
        bins=np.arange(0, max_value + 1, 1) + 0.5 if max_value < 60 else 60,
        label=label,
        density=True,
        histtype="bar",
        linewidth=2,
        rwidth=0.85,
        color=color,
    )
    ax.legend()
    ax.set_xlabel("counts")
    ax.set_ylabel("density")

    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    return ax


def _indicate_col_name(adata: AnnData, promt_str: str) -> Optional[str]:
    """
    Indicate the column name of the data.

    Args:
        adata (AnnData): The AnnData object.
        promt_str (str): The prompt string.

    Returns:
        Optional[str]: The column name.
    """
    while True:
        col_name = input(promt_str)
        if col_name == "":
            col_name = None
            break
        elif col_name in adata.var.columns:
            break
        elif col_name in adata.obs.columns:
            break
        else:
            print(f"The column {col_name} is not in the data. " f"Please input again.")

    return col_name


def find_required_colums(
    adata: AnnData,
    id: str,
    configs_dir: Union[str, Path],
    update: bool = False,
) -> List[Optional[str]]:
    """
    Find the required columns in AnnData, including celltype column, str_celltype
    column, the gene name column, and the experimental batch key.

    This function asks the user to input the required column names if the first
    time loading the data. The names are saved in the config file and will be
    automatically loaded next time.

    Args:
        adata (AnnData): The AnnData object.
        id (str): The id of the AnnData object, will be used as the file name for
            saving the config file.
        configs_dir (Union[str, Path]): The directory of saved config files.
        update (bool): Whether to update the config file.

    Returns:
        List[Optional[str]]: The required columns, including celltype_col, str_celltype_col,
            gene_col, and batch_col.
    """
    if isinstance(configs_dir, str):
        configs_dir = Path(configs_dir)

    if not configs_dir.exists():
        configs_dir.mkdir()

    config_file = configs_dir / f"{id}.json"

    if not config_file.exists() or update:
        print(
            "The config file does not exist, this may be the first time "
            "loading the data. \nPlease input the required column names."
        )
        print(adata)
        celltype_col = _indicate_col_name(
            adata,
            "Please input the celltype column name (skip if not applicable): ",
        )
        str_celltype_col = _indicate_col_name(
            adata, "Please input the str_celltype column name: "
        )
        gene_col = _indicate_col_name(adata, "Please input the gene column name: ")
        batch_col = _indicate_col_name(adata, "Please input the batch column name: ")

        config = {
            "celltype_col": celltype_col,
            "str_celltype_col": str_celltype_col,
            "gene_col": gene_col,
            "batch_col": batch_col,
        }

        with open(config_file, "w") as f:
            json.dump(config, f)

    else:
        with open(config_file, "r") as f:
            config = json.load(f)

    return [
        config["celltype_col"],
        config["str_celltype_col"],
        config["gene_col"],
        config["batch_col"],
    ]


def tensorlist2tensor(tensorlist, pad_value):
    max_len = max(len(t) for t in tensorlist)
    dtype = tensorlist[0].dtype
    device = tensorlist[0].device
    tensor = torch.zeros(len(tensorlist), max_len, dtype=dtype, device=device)
    tensor.fill_(pad_value)
    for i, t in enumerate(tensorlist):
        tensor[i, : len(t)] = t
    return tensor


def map_raw_id_to_vocab_id(
    raw_ids: Union[np.ndarray, torch.Tensor],
    gene_ids: np.ndarray,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Map some raw ids which are indices of the raw gene names to the indices of the

    Args:
        raw_ids: the raw ids to map
        gene_ids: the gene ids to map to
    """
    if isinstance(raw_ids, torch.Tensor):
        device = raw_ids.device
        dtype = raw_ids.dtype
        return_pt = True
        raw_ids = raw_ids.cpu().numpy()
    elif isinstance(raw_ids, np.ndarray):
        return_pt = False
        dtype = raw_ids.dtype
    else:
        raise ValueError(f"raw_ids must be either torch.Tensor or np.ndarray.")

    if raw_ids.ndim != 1:
        raise ValueError(f"raw_ids must be 1d, got {raw_ids.ndim}d.")

    if gene_ids.ndim != 1:
        raise ValueError(f"gene_ids must be 1d, got {gene_ids.ndim}d.")

    mapped_ids: np.ndarray = gene_ids[raw_ids]
    assert mapped_ids.shape == raw_ids.shape
    if return_pt:
        return torch.from_numpy(mapped_ids).type(dtype).to(device)
    return mapped_ids.astype(dtype)


def load_pretrained(
    model: torch.nn.Module,
    pretrained_params: Mapping[str, torch.Tensor],
    strict: bool = False,
    prefix: Optional[List[str]] = None,
    verbose: bool = True,
) -> torch.nn.Module:
    """
    Load pretrained weights to the model.

    Args:
        model (torch.nn.Module): The model to load weights to.
        pretrained_params (Mapping[str, torch.Tensor]): The pretrained parameters.
        strict (bool): Whether to strictly enforce that the keys in :attr:`pretrained_params`
            match the keys returned by this module's :meth:`Module.state_dict`. Default to False.
        prefix (List[str]): The list of prefix strings to match with the keys in
            :attr:`pretrained_params`. The matched keys will be loaded. Default to None.

    Returns:
        torch.nn.Module: The model with pretrained weights.
    """

    use_flash_attn = getattr(model, "use_fast_transformer", True)
    if not use_flash_attn:
        pretrained_params = {
            k.replace("Wqkv.", "in_proj_"): v for k, v in pretrained_params.items()
        }

    if prefix is not None and len(prefix) > 0:
        if isinstance(prefix, str):
            prefix = [prefix]
        pretrained_params = {
            k: v
            for k, v in pretrained_params.items()
            if any(k.startswith(p) for p in prefix)
        }

    model_dict = model.state_dict()
    if strict:
        if verbose:
            for k, v in pretrained_params.items():
                logger.info(f"Loading parameter {k} with shape {v.shape}")
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)
    else:
        if verbose:
            for k, v in pretrained_params.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    logger.info(f"Loading parameter {k} with shape {v.shape}")
        pretrained_params = {
            k: v
            for k, v in pretrained_params.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)

    return model


# Wrapper for all scib metrics, we leave out some metrics like hvg_score, cell_cyvle,
# trajectory_conservation, because we only evaluate the latent embeddings here and
# these metrics are evaluating the reconstructed gene expressions or pseudotimes.
def eval_scib_metrics(
    adata: AnnData,
    batch_key: str = "str_batch",
    label_key: str = "celltype",
    notes: Optional[str] = None,
) -> Dict:
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed="X_scGPT",
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
    )

    if notes is not None:
        logger.info(f"{notes}")

    logger.info(f"{results}")

    result_dict = results[0].to_dict()
    logger.info(
        "Biological Conservation Metrics: \n"
        f"ASW (cell-type): {result_dict['ASW_label']:.4f}, graph cLISI: {result_dict['cLISI']:.4f}, "
        f"isolated label silhouette: {result_dict['isolated_label_silhouette']:.4f}, \n"
        "Batch Effect Removal Metrics: \n"
        f"PCR_batch: {result_dict['PCR_batch']:.4f}, ASW (batch): {result_dict['ASW_label/batch']:.4f}, "
        f"graph connectivity: {result_dict['graph_conn']:.4f}, graph iLISI: {result_dict['iLISI']:.4f}"
    )

    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ARI_cluster/label"],
            result_dict["ASW_label"],
        ]
    )

    # remove nan value in result_dict
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}

    return result_dict


# wrapper to make sure all methods are called only on the main process
def main_process_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if os.environ.get("LOCAL_RANK", "0") == "0":
            return func(*args, **kwargs)

    return wrapper


# class wrapper to make sure all methods are called only on the main process
class MainProcessOnly:
    def __init__(self, obj):
        self.obj = obj

    def __getattr__(self, name):
        attr = getattr(self.obj, name)

        if callable(attr):
            attr = main_process_only(attr)

        return attr


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_subdirs(path):
    #
    # Get all the subdirectories in the given path
    #
    subdirs = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
    return subdirs


def get_subfiles(directory):
    """
    获取指定目录下的所有文件（不包括子目录）。
    
    Args:
        directory (str): 目标目录路径。

    Returns:
        list: 该目录下所有文件的文件名列表（不包含目录）。
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def check_dirs(directory):
    """
    检查目录是否存在，如果不存在，则创建该目录。

    Args:
        directory (str): 需要检查的目录路径。
    """
    os.makedirs(directory, exist_ok=True)


def parse_token_folder_name(folder_name):
    #
    # Parse the folder name into specie, tissue, doner, and sample
    # The folder name should be in the format of specie_tissue_doner_sample
    # but tissue and sample maybe contain more than one part, such as
    # human_bone_marrow_HBABMDWCAP_Nondiseased_Bone_Marrow.h5ad
    #
    parts = folder_name.split('_')
    
    if len(parts) < 4:
        raise ValueError("format is wrong, should be specie_tissue_doner_sample")
    
    specie = parts[0]

    donor_index = next((i for i, part in enumerate(parts) if part[0].isupper()), None)

    if donor_index is None:
        raise ValueError("Unable to identify donor in the folder name.")

    # everything between species and donor is tissue
    tissue = '_'.join(parts[1:donor_index])

    doner = parts[donor_index]
    
    # Everything after donor is sample
    sample = '_'.join(parts[donor_index+1:])

    # sample = sample.split('.')[0]
    
    return specie, tissue, doner, sample

