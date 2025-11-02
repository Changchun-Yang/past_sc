from .constants import DefaultPaths, ObsConstants, UnsConstants, VarConstants, AssayOntologyTermId, SexOntologyTermId, OrganismOntologyTermId, TissueOntologyTermId, SuspensionTypeId
from .validate import validate
from scipy.sparse import csr_matrix
import json
import pandas as pd
import anndata as ad
import numpy as np
import numba
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs
import scanpy as sc
import os
import xml
from scipy.spatial import cKDTree


# functing for process the original adata so that can be used for tokenization
def process_adata(adata, assay_="Xenium", sex="unknown", specie="human", tissue="lung", 
                  donor_info="FFPE_Human_Breast_with_Pre-designed_Panel", condition_info="unknown",
                  cell_boundaries_dict=None, nucleus_boundaries_dict=None, 
                  library_key="section", split="train", niche="nan", region="nan", cluster="nan"):
    adata.var = adata.var.reset_index().rename(columns={'index': 'gene_name'}).set_index('gene_ids')
    adata.var.index.name = None

    # setting MERFISH for now until an official ontology term is released, because now xenium can not be validated
    if assay_ == "Xenium":
        assay = str(AssayOntologyTermId.MERFISH_SPATIAL.value)
        # assay = str(AssayOntologyTermId.XENIUM_SPATIAL.value)

        suspension_type = str(SuspensionTypeId.SPATIAL.value)
    else:
        raise ValueError(f"Invalid assay value: {assay_}. Expected 'Xenium'.")

    if sex == "unknown":
        sex = str(SexOntologyTermId.UNKNOWN.value)
    elif sex == "male":
        sex = str(SexOntologyTermId.MALE.value)
    elif sex == "female":
        sex = str(SexOntologyTermId.FEMALE.value)
    else:
        raise ValueError(f"Invalid sex value: {sex}. Expected 'unknown', 'male', or 'female'.")

    if specie == "human":
        organism = str(OrganismOntologyTermId.HUMAN.value)
    elif specie == "mouse":
        organism = str(OrganismOntologyTermId.MOUSE.value)
    else:
        raise ValueError(f"Invalid specie value: {specie}. Expected 'human' or 'mouse'.")

    organism_validator = specie

    if tissue == "blood":
        tissue = str(TissueOntologyTermId.BLOOD.value)
    elif tissue == "bone":
        tissue = str(TissueOntologyTermId.BONE.value)
    elif tissue == "bone_marrow":
        tissue = str(TissueOntologyTermId.BONE_MARROW.value)
    elif tissue == "brain":
        tissue = str(TissueOntologyTermId.BRAIN.value)
    elif tissue == "breast":
        tissue = str(TissueOntologyTermId.BREAST.value)
    elif tissue == "cervix":
        tissue = str(TissueOntologyTermId.CERVIX.value)
    elif tissue == "colon":
        tissue = str(TissueOntologyTermId.COLON.value)
    elif tissue == "heart":
        tissue = str(TissueOntologyTermId.HEART.value)
    elif tissue == "intestine":
        tissue = str(TissueOntologyTermId.INTESTINE.value)
    elif tissue == "kidney":
        tissue = str(TissueOntologyTermId.KIDNEY.value)
    elif tissue == "liver":
        tissue = str(TissueOntologyTermId.LIVER.value)
    elif tissue == "lung":
        tissue = str(TissueOntologyTermId.LUNG.value)
    elif tissue == "lymph" or tissue == "lymph_node":
        tissue = str(TissueOntologyTermId.LYMPH.value)
    elif tissue == "ovary":
        tissue = str(TissueOntologyTermId.OVARY.value)
    elif tissue == "pancreas":
        tissue = str(TissueOntologyTermId.PANCREAS.value)
    elif tissue == "prostate":
        tissue = str(TissueOntologyTermId.PROSTATE.value)
    elif tissue == "skin":
        tissue = str(TissueOntologyTermId.SKIN.value)
    elif tissue == "tonsil":
        tissue = str(TissueOntologyTermId.TONSIL.value)
    elif tissue == "whole_organism" or tissue == "whole":
        tissue = str(TissueOntologyTermId.WHOLE_ORGANISM.value)
    else:
        raise ValueError(f"Invalid tissue value: {tissue}. Expected 'blood', 'bone', 'bone_marrow', 'brain', 'breast', 'cervix', 'colon', 'heart', 'intestine', 'kidney', 'liver', 'lung', 'lymph', 'ovary', 'pancreas', 'prostate', 'skin', 'tonsil', or 'whole_organism'.")

    tissue_type = "tissue"

    adata.X = csr_matrix(adata.X)
    adata.obs[ObsConstants.SPATIAL_X] = adata.obs['x_centroid']
    adata.obs[ObsConstants.SPATIAL_Y] = adata.obs['y_centroid']

    # to change list to string for save, when loading, need to use json to convert to list
    # such as for getting cell 0: cell_vertex_x = json.loads(adata.obs[ObsConstants.CELL_VERTEX_X][0])
    # adata.obs[ObsConstants.CELL_VERTEX_X] = adata.obs.index.map(lambda x: cell_boundaries_dict.get(x, {}).get('vertex_x', []))
    # adata.obs[ObsConstants.CELL_VERTEX_Y] = adata.obs.index.map(lambda x: cell_boundaries_dict.get(x, {}).get('vertex_y', []))
    # adata.obs[ObsConstants.NUCLEUS_VERTEX_X] = adata.obs.index.map(lambda x: nucleus_boundaries_dict.get(x, {}).get('vertex_x', []))
    # adata.obs[ObsConstants.NUCLEUS_VERTEX_Y] = adata.obs.index.map(lambda x: nucleus_boundaries_dict.get(x, {}).get('vertex_y', []))
    adata.obs[ObsConstants.CELL_VERTEX_X] = adata.obs.index.map(lambda x: json.dumps(cell_boundaries_dict.get(x, {}).get('vertex_x', [])))
    adata.obs[ObsConstants.CELL_VERTEX_Y] = adata.obs.index.map(lambda x: json.dumps(cell_boundaries_dict.get(x, {}).get('vertex_y', [])))
    adata.obs[ObsConstants.NUCLEUS_VERTEX_X] = adata.obs.index.map(lambda x: json.dumps(nucleus_boundaries_dict.get(x, {}).get('vertex_x', [])))
    adata.obs[ObsConstants.NUCLEUS_VERTEX_Y] = adata.obs.index.map(lambda x: json.dumps(nucleus_boundaries_dict.get(x, {}).get('vertex_y', [])))

    adata.obs[ObsConstants.ASSAY_ONTOLOGY_TERM_ID] = pd.Categorical([assay for i in range(len(adata))])
    adata.obs[ObsConstants.SEX_ONTOLOGY_TERM_ID] = pd.Categorical([sex for i in range(len(adata))])
    adata.obs[ObsConstants.ORGANISM_ONTOLOGY_TERM_ID] = pd.Categorical([organism for i in range(len(adata))])
    adata.obs[ObsConstants.TISSUE_ONTOLOGY_TERM_ID] = pd.Categorical([tissue for i in range(len(adata))])
    adata.obs[ObsConstants.SUSPENSION_TYPE] = pd.Categorical([suspension_type for i in range(len(adata))])

    adata.obs[ObsConstants.DONOR_ID] = pd.Categorical([donor_info for i in range(len(adata))])
    adata.obs[ObsConstants.CONDITION_ID] = pd.Categorical([condition_info for i in range(len(adata))])
    adata.obs[ObsConstants.TISSUE_TYPE] = pd.Categorical([tissue_type for i in range(len(adata))])


    adata.uns[UnsConstants.TITLE] = donor_info
    
    adata.var[VarConstants.FEATURE_IS_FILTERED] = False

    adata.obs[ObsConstants.LIBRARY_KEY] = pd.Categorical([library_key for i in range(len(adata))])

    adata_output, valid, errors, is_seurat_convertible = validate(adata, organism=organism_validator)

    adata_output.obs[ObsConstants.ASSAY] = pd.Categorical([assay_ for i in range(len(adata_output))])
    # because now xenium can not be validated, we set no yet defined
    adata_output.obs[ObsConstants.ASSAY_ONTOLOGY_TERM_ID] = pd.Categorical(['no yet defined' for i in range(len(adata_output))])

    adata_output.obs[ObsConstants.DATASET] = adata_output.uns['title']

    # if need to define test data set for the pretraining, change here
    adata_output.obs[ObsConstants.SPLIT] = split
    adata_output.obs[ObsConstants.NICHE] = niche
    adata_output.obs[ObsConstants.REGION] = region

    # if cluster == "nan":
    #     adata_output.obs[ObsConstants.CLUSTER] = pd.Categorical([cluster for i in range(len(adata_output))])
    # else:
    adata_output.obs[ObsConstants.CLUSTER] = cluster

    return adata_output


# function for normalize data before tokenizing the data
## all cells are normalized so that each of them has 10,000 counts
def sf_normalize(X):
    X = X.copy()
    counts = np.array(X.sum(axis=1))
    # avoid zero devision error
    counts += counts == 0.
    # normalize to 10000. counts
    scaling_factor = 10000. / counts

    if issparse(X):
        sparsefuncs.inplace_row_scale(X, scaling_factor)
    else:
        np.multiply(X, scaling_factor.reshape((-1, 1)), out=X)

    return X


# a function t hat returns t he index o f gene i n a previously d efined v ocabulary o f genes
# returns descending order after the genes are ranked 
# @numba.jit(nopython=True, nogil=True)
def _sub_tokenize_data(x: np.array, max_seq_len: int = -1, aux_tokens: int = 30):
    scores_final = np.empty((x.shape[0], max_seq_len if max_seq_len > 0 else x.shape[1]))
    # for i, cell in enumerate(x):
    #modified, for csr matrix
    for i in range(x.shape[0]):
        cell = x.getrow(i).toarray()[0]
        nonzero_mask = np.nonzero(cell)[0]    
        sorted_indices = nonzero_mask[np.argsort(-cell[nonzero_mask])][:max_seq_len] 
        sorted_indices = sorted_indices + aux_tokens # we reserve some tokens for padding etc (just in case)
        if max_seq_len:
            scores = np.zeros(max_seq_len, dtype=np.int32)
        else:
            scores = np.zeros_like(cell, dtype=np.int32)
        scores[:len(sorted_indices)] = sorted_indices.astype(np.int32)
        
        scores_final[i, :] = scores
        
    return scores_final


# function for tokenizing the data
## 1.firstly all cells are normalized so that each of them has 10,000 counts
## 2.Secondly normalize the expression of each cell using the corresponding technology-specific mean expression
## vector to obtain the expression of each gene in each cell relative to the whole training corpusfor 
## To account for technological variations, we then compute a technology-specific gene
## expression non-zero mean vector i.e., the mean expression value of each gene, without considering the zero counts
## 3.Finally, the genes are ranked in descending order, from most expressed to lowest expressed
def tokenize_data(x: np.array, median_counts_per_gene: np.array, max_seq_len: int = 4096, aux_tokens: int = 30):
    """Tokenize the input gene vector to a vector of 32-bit integers."""
    x = np.nan_to_num(x) # is NaN values, fill with 0s
    x = sf_normalize(x)
    
    median_counts_per_gene += median_counts_per_gene == 0
    out = x / median_counts_per_gene.reshape((1, -1))

    scores_final = _sub_tokenize_data(out, max_seq_len, aux_tokens)

    return scores_final.astype('i4'), out


## used for data statistic
def adata_umap(adata):
    adata_figs = adata.copy()
    adata_figs.layers['counts'] = adata_figs.X
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.settings.set_figure_params(dpi=300, facecolor='white')
    sc.pl.umap(adata, color='condition_id', save='umap_adata_test.pdf')


# for 5k gene data
def align_genes_with_model(model, xenium):
    
    model_genes = model.var_names

    xenium_genes = xenium.var_names

    common_genes = xenium_genes.isin(model_genes)
    removed_genes = xenium_genes[~common_genes].tolist()

    xenium_aligned = xenium[:, common_genes].copy()
    # xenium_aligned = xenium_aligned[:, model_genes[model_genes.isin(xenium_aligned.var_names)]].copy()

    return xenium_aligned, removed_genes

def find_pixel_size(metadata):
    try:
        root = xml.etree.ElementTree.fromstring(metadata)
        namespaces = {
            'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'
        }

        pixels = root.find('.//ome:Pixels', namespaces)

        if pixels is not None:
            physical_size_x = pixels.get('PhysicalSizeX')
            physical_size_y = pixels.get('PhysicalSizeY')

            if physical_size_x is not None and physical_size_y is not None:
                return [float(physical_size_x), float(physical_size_y)]

    except xml.etree.ElementTree.ParseError as e:
        print(f"XML parsing error: {e}, setting default pixel size to 0.2125 um.")
    except Exception as e:
        print(f"An error occurred: {e}, setting default pixel size to 0.2125 um.")

    return [0.2125, 0.2125]

def get_he_image_file(image_path):
    tif_files = []
    alignment_matrix = None
    alignment_csv_file = None

    for item in os.listdir(image_path):
        if item.lower().endswith(".tif") or item.lower().endswith(".tiff"):
            if "he" in item.lower():
                tif_files.append(item)

    if not tif_files:
        raise FileNotFoundError("No TIFF files found in the specified directory.")

    if not any("he" in file.lower() for file in tif_files):
        raise ValueError("No TIFF file contains 'he' in its name.")

    for item in os.listdir(image_path):
        if item.lower().endswith(".csv") and "he_imagealignment" in item.lower():
            alignment_csv_file = os.path.join(image_path, item)
            # print(f"Found alignment CSV file: {alignment_csv_file}")
            # # 读取 CSV 文件中的仿射变换矩阵
            alignment_matrix = np.loadtxt(alignment_csv_file, delimiter=',')   # from he to dapi
            break

    if alignment_matrix is not None:
        alignment_matrix = np.linalg.inv(alignment_matrix)

    return tif_files, alignment_csv_file, alignment_matrix                # from dapi to he

def find_neighbors(coords, window_size=224):
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=window_size/2)
    return neighbors

