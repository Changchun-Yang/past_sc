import os
import cv2
import h5py
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
# from scipy.sparse import csr_matrix
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
import math
from heapq import nsmallest
from PIL import Image
import glob
import pyarrow.parquet as pq
import tifffile
import zarr
from zarr.core import Array as ZArray
import xml
from collections import defaultdict
from dask.dataframe import DataFrame as DDF
import dask.dataframe as dd
import dask
dask.config.set(scheduler='synchronous')
from torch.nn.utils.rnn import pad_sequence
import torch_geometric
from torch_geometric.data import Batch

from functools import lru_cache


# PARQUET_SCHEMA = {
#     'X': int32,
#     'X_niche_0': int32,
#     'X_niche_1': int32,
#     'X_niche_2': int32,
#     'X_niche_3': int32,
#     'X_niche_4': int32,
#     'density_0': float32,
#     'density_1': float32,
#     'density_2': float32,
#     'density_3': float32,
#     'density_4': float32,
#     'niche': int64,
#     'author_cell_type': int64,
#     'region': int64,
#     'soma_joinid': int64,
#     'is_primary_data': boolean,
#     'dataset_id': int64,
#     'donor_id': int64,
#     'assay': int64,
#     'cell_type': int64,
#     'development_stage': int64,
#     'disease': int64,
#     'tissue': int64,
#     'tissue_general': int64,
#     'tech_sample': int64,
#     'idx': int64,
#     'specie': int64,
#     'modality': int64,
#     'organism': int64,
#     'measured_genes': int32,
# }

def custom_collate_fn(batch):

    batch_dict = {
        'image': [],
        'X': [],
        'modality': [],
        'assay': [],
        'specie': [],
        'spatial_coords': [],
        'relative_coords': []
    }

    for item in batch:
        image = item['image']
        batch_dict['image'].append(image)

        batch_dict['X'].append(item['X'])
        batch_dict['modality'].append(item['modality'])
        batch_dict['assay'].append(item['assay'])
        batch_dict['specie'].append(item['specie'])
        batch_dict['spatial_coords'].append(item['spatial_coords'])
        batch_dict['relative_coords'].append(item['relative_coords'])

    return {
        'image': torch.stack(batch_dict['image']),
        'X': torch.stack(batch_dict['X']),
        'modality': torch.stack(batch_dict['modality']),
        'assay': torch.stack(batch_dict['assay']),
        'specie': torch.stack(batch_dict['specie']),
        'spatial_coords': torch.stack(batch_dict['spatial_coords']),
        'relative_coords': torch.stack(batch_dict['relative_coords'])
    }


def graph_collate_fn(batch: dict):
    # Separate graphs and non-graph data
    graph_data = [item['graph'] for item in batch]
    non_graph_data = {key: [] for key in batch[0] if key != 'graph'}

    # Batch non-graph data
    for item in batch:
        for key, value in item.items():
            if key != 'graph':
                non_graph_data[key].append(value)

    # Stack non-graph data (is instance of torch.tensor)
    for key in non_graph_data:
        non_graph_data[key] = torch.stack(non_graph_data[key], dim=0)

    # Batch graph data from torch_geometric
    graph_batch = Batch.from_data_list(graph_data)

    return non_graph_data, graph_batch


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.dataset_lengths = [len(dataset) for dataset in dataset_list]
        self.cumulative_lengths = np.cumsum(self.dataset_lengths)

    def __len__(self):
        return sum(self.dataset_lengths)

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        if dataset_idx > 0:
            idx_within_dataset = idx - self.cumulative_lengths[dataset_idx - 1]
        else:
            idx_within_dataset = idx
        return self.dataset_list[dataset_idx][idx_within_dataset]


class PASTDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, cell_path, context_length=1500, padding=112, cache_size=50, neighbor_radius=224,
                 model_transform='uni', is_toy=False):
        self.image_path = image_path
        self.cell_path = cell_path
        self.context_length = context_length

        self.parquet_files = glob.glob(os.path.join(cell_path, '**', '*.parquet'), recursive=True)
        if is_toy:
            self.parquet_files = self.parquet_files[0:1]

        if model_transform == 'uni':
            self.model_transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        self.total_rows = 0
        self.file_row_counts = []
        self.padding = padding
        self.neighbor_radius = neighbor_radius

        tif_files = self.get_he_tif()
        assert len(tif_files) == 1, "More than one TIFF file found in the specified directory."

        with tifffile.TiffFile(os.path.join(image_path, tif_files[0])) as tif:
            for tag in tif.pages[0].tags.values():
                if tag.name == 'ImageDescription':
                    self.pixel_size = self.find_pixel_size(tag.value)
                    break
            self.whole_image = tif.series[0].levels[0].asarray()
            self.whole_image = self.adjust_channel_dimension(self.whole_image)

        # Load all cell data
        self.cell_data = defaultdict(list)
        for file in self.parquet_files:
            df = pq.read_table(file).to_pandas()
            rows = len(df)
            self.file_row_counts.append(rows)
            self.total_rows += rows

            for idx, row in df.iterrows():

                x, y = row['x'] / self.pixel_size[0], row['y'] / self.pixel_size[1]

                self.cell_data[file].append({
                    'idx': idx,
                    'x': x,
                    'y': y,
                    'X': row['X'],
                    'modality': row['modality'],
                    'assay': row['assay'],
                    'specie': row['specie']
                })

        self.file_weights = np.array(self.file_row_counts) / self.total_rows

        self.cache_size = cache_size
        self.cache = {}
        
        print("Finished loading all files")

    def adjust_channel_dimension(self, image):
        if len(image.shape) == 2: 
            return np.expand_dims(image, axis=0)
        elif len(image.shape) == 3:
            if image.shape[-1] in [1, 3, 4]: 
                return np.transpose(image, (2, 0, 1))
            elif image.shape[0] in [1, 3, 4]:  
                return image
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
        else:
            raise ValueError(f"Unexpected number of dimensions: {len(image.shape)}")

    def truncate_X(self, X):
        if len(X) > self.context_length:
            return X[:self.context_length]
        elif len(X) < self.context_length:
            return np.pad(X, (0, self.context_length - len(X)), 'constant')
        else:
            return X
    
    def find_pixel_size(self, metadata):
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
    
    def get_he_tif(self):
        tif_files = []
        for item in os.listdir(self.image_path):
            if item.lower().endswith(".tif") or item.lower().endswith(".tiff"):
                if "he" in item.lower():
                    tif_files.append(item)
        
        if not tif_files:
            raise FileNotFoundError("No TIFF files found in the specified directory.")
        
        if not any("he" in file.lower() for file in tif_files):
            raise ValueError("No TIFF file contains 'he' in its name.")
        
        return tif_files

    def aug_transform(self, image):
        image = Image.fromarray(image)
        # Random flipping and rotations
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.choice([180, 90, 0, -90])
        image = TF.rotate(image, angle)
        return np.asarray(image)

    def get_neighbors(self, file, x, y, k=8, excluding_self=False, get_distances=False):
        neighbors = []

        # return neighbors
        for cell in self.cell_data[file].itertuples():
            # cell is named tuple
            cell = cell._asdict()
            nx, ny = cell['image_x'], cell['image_y']
            distance = math.sqrt((nx - x) ** 2 + (ny - y) ** 2)

            # Exclude the cell itself
            if excluding_self and nx == x and ny == y:
                continue

            neighbors.append((distance, cell))

        # Sort by distance and select the k nearest neighbors
        k_neighbors_with_distances = nsmallest(k, neighbors, key=lambda item: item[0])

        # Separate distances and cells
        distances = [dist for dist, _ in k_neighbors_with_distances]
        cells = [cell for _, cell in k_neighbors_with_distances]

        return cells, distances if get_distances else cells

    def get_neighbors_from_indices(self, pdf, current_cell, x, y, k=8, excluding_self=True, get_distances=False):

        neighbor_indices = current_cell['neighbor_indices']

        neighbors_df = pdf.loc[neighbor_indices]

        neighbors = neighbors_df.to_dict(orient='records')

        neighbors_with_distances = []

        for neighbor in neighbors:
            nx, ny = neighbor['image_x'], neighbor['image_y']
            distance = math.sqrt((nx - x) ** 2 + (ny - y) ** 2)

            if excluding_self and nx == x and ny == y:
                continue

            neighbors_with_distances.append((distance, neighbor))

        neighbors_with_distances = sorted(neighbors_with_distances, key=lambda item: item[0])

        closest_neighbors = neighbors_with_distances[:k]

        filtered_neighbors = [neighbor for _, neighbor in closest_neighbors]
        distances = [distance for distance, _ in closest_neighbors]

        return (filtered_neighbors, distances) if get_distances else filtered_neighbors
    
    def get_padded_image(self, x, y):
        # Critical bug: the image is not transposed, so the x and y shoud not swapped
        # original: h, w = self.whole_image.shape[1:], but should be:
        # 有时候通道在0维，有时候在2维，所以这里需要判断
        h, w = self.whole_image.shape[:2]
        y1 = max(0, int(y - self.padding))
        y2 = min(h, int(y + self.padding))
        x1 = max(0, int(x - self.padding))
        x2 = min(w, int(x + self.padding))

        # original: sub_image = self.whole_image[:, y1:y2, x1:x2], but should be:
        sub_image = self.whole_image[x1:x2, y1:y2, :]

        pad_y1 = int(max(0, self.padding - int(y)))
        pad_y2 = int(max(0, int(y + self.padding) - h))
        pad_x1 = int(max(0, self.padding - int(x)))
        pad_x2 = int(max(0, int(x + self.padding) - w))

        padded_image = np.pad(sub_image, ((0, 0), (pad_y1, pad_y2), (pad_x1, pad_x2)), mode='constant', constant_values=0)

        padded_image = np.transpose(padded_image, (1, 2, 0))
        padded_image = Image.fromarray(padded_image.astype(np.uint8))
        padded_image = self.model_transform(padded_image)

        return padded_image
    

    def build_graph(self, neighbors, distances:list = None):
        num_nodes = len(neighbors)
        
        # create node features
        x = torch.stack([torch.tensor(self.truncate_X(n['X']), dtype=torch.float32) for n in neighbors])
        
        # for simplicity, we will create a fully connected graph
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected graph
        
        # create node positions
        pos = torch.tensor([[n['x'], n['y']] for n in neighbors], dtype=torch.float32)

        edge_attr = None
        if distances is not None:
            # use distances as edge weights
            # egde_attr: [num_edges, num_edge_features] = [num_nodes, 1]
            edge_attr = torch.tensor(distances * 2, dtype=torch.float32)  # multiply by 2 for undirected edges
        
        return torch_geometric.data.Data(x=x, edge_index=edge_index, pos=pos, edge_attr=edge_attr)
    

    def transform_coordinates(self, coords, center_coords):
        x, y = coords
        cx, cy = center_coords
        new_x = x - cx + self.padding
        new_y = y - cy + self.padding
        return [new_x, new_y]
    
    def __getitem__(self, idx):
        file_idx = np.random.choice(len(self.parquet_files), p=self.file_weights)
        file = self.parquet_files[file_idx]

        row_idx = random.randint(0, self.file_row_counts[file_idx] - 1)

        cache_key = (file, row_idx)
        if cache_key in self.cache:
            return self.cache[cache_key]

        cell_data = self.cell_data[file][row_idx]

        # 'assay', 'specie', 'modality', 'idx', 'dataset', 'condition_id', 'x', 'y',
        # 'cell_vertex_x', 'cell_vertex_y', 'nucleus_vertex_x', 'nucleus_vertex_y'
        # to change list to string for save, when loading, need to use json to convert to list
        # such as for getting cell 0: cell_vertex_x = json.loads(adata.obs[ObsConstants.CELL_VERTEX_X][0])

        x, y = cell_data['x'], cell_data['y']

        padded_image = self.get_padded_image(x, y)

        ##
        # there is one problem needs to be solve, that is neighbors is not always same, they have different numbers of cell
        ##
        neighbors, distances = self.get_neighbors(file, x, y, get_distances=True)
        #
        # relative_coords = [self.padding, self.padding]
        # neighbor_relative_coords = [self.transform_coordinates((neighbor['x'], neighbor['y']), (x, y)) for neighbor in neighbors]
        #
        # item = {
        #     'image': torch.tensor(padded_image, dtype=torch.float32),
        #     'X': torch.tensor(self.truncate_X(cell_data['X']), dtype=torch.int32),  # Changed to int32
        #     'spatial_coords': torch.tensor([x, y], dtype=torch.float32),
        #     'relative_coords': torch.tensor(relative_coords, dtype=torch.float32),
        #     # 'graph': self.build_graph(neighbors),
        #     'neighbors': {
        #     'idx': torch.tensor([n['idx'] for n in neighbors], dtype=torch.long),
        #     'X': torch.stack([torch.tensor(self.truncate_X(n['X']), dtype=torch.int32) for n in neighbors]),  # Changed to int32
        #     'spatial_coords': torch.tensor([[n['x'], n['y']] for n in neighbors], dtype=torch.float32),
        #     'relative_coords': torch.tensor(neighbor_relative_coords, dtype=torch.float32) }
        # }

        relative_coords = [self.padding, self.padding]

        item = {
            'image': torch.tensor(padded_image, dtype=torch.float32),
            'X': torch.tensor(self.truncate_X(cell_data['X']), dtype=torch.int32),  # Changed to int32
            'modality': torch.tensor(cell_data['modality'], dtype=torch.int32),
            'assay': torch.tensor(cell_data['assay'], dtype=torch.int32),
            'specie': torch.tensor(cell_data['specie'], dtype=torch.int32),
            'spatial_coords': torch.tensor([x, y], dtype=torch.float32),
            'relative_coords': torch.tensor(relative_coords, dtype=torch.float32),
            # 'graph': self.build_graph(neighbors, distances)
        }

        self.cache[cache_key] = item
        if len(self.cache) > self.cache_size:
            keys_to_remove = random.sample(list(self.cache.keys()), len(self.cache) // 2)
            for key in keys_to_remove:
                del self.cache[key]

        return item


    def __len__(self):
        return self.total_rows


# for data format parquet
class PASTDataset_v2(torch.utils.data.Dataset):
    def __init__(self, image_path, cell_path, context_length=1500, padding=112, cache_size=50, neighbor_radius=224,
                 model_transform='uni', is_toy=False):
        self.image_path = image_path
        self.cell_path = cell_path
        self.context_length = context_length

        self.parquet_files = glob.glob(os.path.join(cell_path, '**', '*.parquet'), recursive=True)
        if is_toy:
            self.parquet_files = self.parquet_files[0:1]

        if model_transform == 'uni':
            self.model_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        self.total_rows = 0
        self.file_row_counts = []
        self.padding = padding
        self.neighbor_radius = neighbor_radius

        tif_files = self.get_he_tif()
        assert len(tif_files) == 1, "More than one TIFF file found in the specified directory."

        # read tif image to zarr format
        # this is a lazy loading
        page = tifffile.imread(os.path.join(image_path, tif_files[0]), is_ome=True, level=0, aszarr=True)
        self.whole_image: ZArray = zarr.open(page, mode='r')
        self.pixel_size = self.get_pixel_size(self.whole_image)

        # Dead lock bug could happen if the scheduler is not synchronous
        # source: https://discuss.pytorch.org/t/dataloader-parallelization-synchronization-with-zarr-xarray-dask/176149
        assert dask.config.get('scheduler') == 'synchronous', "Dask scheduler must be synchronous"

        self.cell_data = defaultdict(DDF)
        self.file_row_counts = []
        self.total_rows = 0

        for file in self.parquet_files:
            ddf: DDF = dd.read_parquet(file)
            self.cell_data[file] = ddf
            self.file_row_counts.append(ddf.shape[0])

        self.total_rows = sum(row_count.compute() for row_count in self.file_row_counts)
        self.file_weights = np.array([row_count.compute() for row_count in self.file_row_counts]) / self.total_rows

        # print("Finished loading all files")

    def get_pixel_size(self, whole_image: ZArray):
        array_shape = whole_image.shape
        assert len(array_shape) == 3, "The image should have 3 channels."
        if array_shape[0] == 3:
            return array_shape[1:]
        elif array_shape[2] == 3:
            return array_shape[:2]
        else:
            raise ValueError("The image channels should be on the first or last dimension.")

    def truncate_X(self, X):
        if len(X) > self.context_length:
            return X[:self.context_length]
        elif len(X) < self.context_length:
            return np.pad(X, (0, self.context_length - len(X)), 'constant')
        else:
            return X

    def get_he_tif(self):
        tif_files = []
        for item in os.listdir(self.image_path):
            if item.lower().endswith(".tif") or item.lower().endswith(".tiff"):
                if "he" in item.lower():
                    tif_files.append(item)

        if not tif_files:
            raise FileNotFoundError("No TIFF files found in the specified directory.")

        if not any("he" in file.lower() for file in tif_files):
            raise ValueError("No TIFF file contains 'he' in its name.")

        return tif_files

    def get_neighbors_from_indices(self, pdf, current_cell, x, y, k=8, excluding_self=True, get_distances=False):
        neighbor_indices = current_cell['neighbor_indices']

        neighbors_df = pdf.loc[neighbor_indices]
        coords = neighbors_df[['image_x', 'image_y']].values
        current_coords = np.array([x, y])

        distances = np.sqrt(np.sum((coords - current_coords) ** 2, axis=1))

        if excluding_self:
            mask = (coords != current_coords).any(axis=1)
            neighbors_df = neighbors_df[mask]
            distances = distances[mask]

        idx = np.argsort(distances)[:k]
        filtered_neighbors = neighbors_df.iloc[idx].to_dict('records')
        filtered_distances = distances[idx]

        return (filtered_neighbors, filtered_distances.tolist()) if get_distances else filtered_neighbors

    def get_padded_image(self, x, y):
        h, w = self.pixel_size
        y1 = max(0, int(y - self.padding))
        y2 = min(h, int(y + self.padding))
        x1 = max(0, int(x - self.padding))
        x2 = min(w, int(x + self.padding))

        # original: sub_image = self.whole_image[:, y1:y2, x1:x2], but should be:
        if self.whole_image.shape[2] == 3:
            sub_image = self.whole_image[y1:y2, x1:x2, :]
        else:
            sub_image = self.whole_image[:, y1:y2, x1:x2]
            sub_image = np.transpose(sub_image, (1, 2, 0))

        # pad_y1 = int(max(0, self.padding - int(y)))
        # pad_y2 = int(max(0, int(y + self.padding) - h))
        # pad_x1 = int(max(0, self.padding - int(x)))
        # pad_x2 = int(max(0, int(x + self.padding) - w))

        # padded_image = np.pad(sub_image, ((0, 0), (pad_y1, pad_y2), (pad_x1, pad_x2)), mode='constant',
        #                       constant_values=0)

        pad_y1 = int((self.padding * 2 - (y2 - y1)) // 2)
        pad_y2 = int(self.padding * 2 - (y2 - y1) - pad_y1)
        pad_x1 = int((self.padding * 2 - (x2 - x1)) // 2)
        pad_x2 = int(self.padding * 2 - (x2 - x1) - pad_x1)

        padded_image = np.pad(sub_image, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), mode='constant',
                            constant_values=0)

        # padded_image = np.transpose(padded_image, (1, 2, 0))
        padded_image = Image.fromarray(padded_image.astype(np.uint8))
        padded_image = self.model_transform(padded_image)

        return padded_image

    def build_graph(self, neighbors, distances: list = None):
        num_nodes = len(neighbors)

        # create node features
        x = torch.stack([torch.tensor(self.truncate_X(n['X']), dtype=torch.float32) for n in neighbors])

        # for simplicity, we will create a fully connected graph
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected graph

        # create node positions
        pos = torch.tensor([[n['image_x'], n['image_y']] for n in neighbors], dtype=torch.float32)

        edge_attr = None
        if distances is not None:
            # use distances as edge weights
            # egde_attr: [num_edges, num_edge_features] = [num_nodes, 1]
            edge_attr = torch.tensor(distances * 2, dtype=torch.float32)  # multiply by 2 for undirected edges

        return torch_geometric.data.Data(x=x, edge_index=edge_index, pos=pos, edge_attr=edge_attr)

    @lru_cache(maxsize=128)
    def get_cached_data(self, file):
        """
        get Pandas DataFrame（cache）。
        """
        return self.cell_data[file].compute()

    def __getitem__(self, idx):
        file_idx = np.random.choice(len(self.parquet_files), p=self.file_weights)
        file = self.parquet_files[file_idx]

        # pdf = self.get_cached_data(file)

        pdf = self.cell_data[file].compute()
        # ddf = self.cell_data[file]
        # partitions = ddf.npartitions
        # partition_idx = random.randint(0, partitions - 1)
        # pdf = ddf.get_partition(partition_idx).compute()

        idx_values = pdf["idx"]
        # self.assert_continuous_and_ascending(idx_values)

        # get the first and last index of the file
        min_idx, max_idx = idx_values.iloc[0], idx_values.iloc[-1]
        # print(f"min_idx: {min_idx}, max_idx: {max_idx}")

        row_idx = random.randint(min_idx, max_idx)

        cell_data = pdf.loc[row_idx].to_dict()

        # cell_data = cell_data.to_dict(orient='list')  # .iloc[row_idx]
        # for key, value in cell_data.items():
        #     cell_data[key] = value[0]
        # # print(cell_data)

        # 'assay', 'specie', 'modality', 'idx', 'dataset', 'condition_id', 'x', 'y',
        # 'cell_vertex_x', 'cell_vertex_y', 'nucleus_vertex_x', 'nucleus_vertex_y'
        x, y = cell_data['image_x'], cell_data['image_y']

        padded_image = self.get_padded_image(x, y)

        neighbors, distances = self.get_neighbors_from_indices(pdf, cell_data, x, y, get_distances=True)

        neighbors.append(cell_data)
        distances.append(0.0)

        relative_coords = [self.padding, self.padding]

        item = {
            'image': padded_image,
            'X': torch.tensor(self.truncate_X(cell_data['X']), dtype=torch.int32),  # Changed to int32
            'modality': torch.tensor(cell_data['modality'], dtype=torch.int32),
            'assay': torch.tensor(cell_data['assay'], dtype=torch.int32),
            'specie': torch.tensor(cell_data['specie'], dtype=torch.int32),
            'spatial_coords': torch.tensor([x, y], dtype=torch.float32),
            'relative_coords': torch.tensor(relative_coords, dtype=torch.float32),
            'graph': self.build_graph(neighbors, distances)
        }

        return item

    def __len__(self):
        return self.total_rows

    @staticmethod
    def assert_continuous_and_ascending(idx_values):
        # Check if idx_values is a pandas Series
        assert isinstance(idx_values, pd.Series), "idx_values must be a pandas Series"

        # Check if the series is sorted
        is_sorted = idx_values.is_monotonic_increasing
        assert is_sorted, "idx_values must be sorted in increasing order"

        # Calculate the difference between consecutive values and check if all are equal to 1
        differences = idx_values.diff().iloc[1:]
        assert (differences == 1).all(), "idx_values must contain continuous integers"

        print("Assertion passed: idx_values is a series of continuous integers.")


# for data format hdf5
class PASTDataset_v3(torch.utils.data.Dataset):
    def __init__(self, image_path, cell_file, context_length=1500, padding=112, cache_size=50, neighbor_radius=224,
                 model_transform='uni', is_toy=False):
        self.image_path = image_path
        self.cell_file = cell_file
        self.context_length = context_length


        if model_transform == 'uni':
            self.model_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        self.padding = padding
        self.neighbor_radius = neighbor_radius

        tif_files = self.get_he_tif()
        assert len(tif_files) == 1, "More than one TIFF file found in the specified directory."

        # read tif image to zarr format
        # this is a lazy loading
        page = tifffile.imread(os.path.join(image_path, tif_files[0]), is_ome=True, level=0, aszarr=True)
        self.whole_image: ZArray = zarr.open(page, mode='r')
        self.pixel_size = self.get_pixel_size(self.whole_image)

        # Dead lock bug could happen if the scheduler is not synchronous
        # source: https://discuss.pytorch.org/t/dataloader-parallelization-synchronization-with-zarr-xarray-dask/176149
        assert dask.config.get('scheduler') == 'synchronous', "Dask scheduler must be synchronous"

        self.hdf_data = h5py.File(self.cell_file, 'r')

        self.num_rows = self.hdf_data['idx'].shape[0]

        # print("Finished loading all files")

    def get_pixel_size(self, whole_image: ZArray):
        array_shape = whole_image.shape
        assert len(array_shape) == 3, "The image should have 3 channels."
        if array_shape[0] == 3:
            return array_shape[1:]
        elif array_shape[2] == 3:
            return array_shape[:2]
        else:
            raise ValueError("The image channels should be on the first or last dimension.")

    def truncate_X(self, X):
        if len(X) > self.context_length:
            return X[:self.context_length]
        elif len(X) < self.context_length:
            return np.pad(X, (0, self.context_length - len(X)), 'constant')
        else:
            return X

    def get_he_tif(self):
        tif_files = []
        for item in os.listdir(self.image_path):
            if item.lower().endswith(".tif") or item.lower().endswith(".tiff"):
                if "he" in item.lower():
                    tif_files.append(item)

        if not tif_files:
            raise FileNotFoundError("No TIFF files found in the specified directory.")

        if not any("he" in file.lower() for file in tif_files):
            raise ValueError("No TIFF file contains 'he' in its name.")

        return tif_files

    def get_neighbors_from_indices(self, row_idx, cell_data, x, y, k=8, excluding_self=True, get_distances=False):

        neighbor_indices = np.array(cell_data['neighbor_indices'], dtype=np.int32)

        coords_x = self.hdf_data['image_x'][neighbor_indices]
        coords_y = self.hdf_data['image_y'][neighbor_indices]
        coords = np.vstack((coords_x, coords_y)).T  

        current_coords = np.array([x, y])
        distances = np.sqrt(np.sum((coords - current_coords) ** 2, axis=1))

        if excluding_self:
            mask = (coords != current_coords).any(axis=1)
            neighbor_indices = neighbor_indices[mask]
            distances = distances[mask]

        idx = np.argsort(distances)[:k]
        filtered_neighbor_indices = neighbor_indices[idx]
        filtered_distances = distances[idx]

        filtered_neighbors = []
        for neighbor_idx in filtered_neighbor_indices:
            neighbor_data = {key: self.hdf_data[key][neighbor_idx] for key in self.hdf_data.keys()}

            for key in neighbor_data:
                if isinstance(neighbor_data[key], np.generic):
                    neighbor_data[key] = neighbor_data[key].item()

            filtered_neighbors.append(neighbor_data)

        return (filtered_neighbors, filtered_distances.tolist()) if get_distances else filtered_neighbors

    def get_padded_image(self, x, y):
        h, w = self.pixel_size
        y1 = max(0, int(y - self.padding))
        y2 = min(h, int(y + self.padding))
        x1 = max(0, int(x - self.padding))
        x2 = min(w, int(x + self.padding))

        # original: sub_image = self.whole_image[:, y1:y2, x1:x2], but should be:
        if self.whole_image.shape[2] == 3:
            sub_image = self.whole_image[y1:y2, x1:x2, :]
        else:
            sub_image = self.whole_image[:, y1:y2, x1:x2]
            sub_image = np.transpose(sub_image, (1, 2, 0))

        # pad_y1 = int(max(0, self.padding - int(y)))
        # pad_y2 = int(max(0, int(y + self.padding) - h))
        # pad_x1 = int(max(0, self.padding - int(x)))
        # pad_x2 = int(max(0, int(x + self.padding) - w))

        # padded_image = np.pad(sub_image, ((0, 0), (pad_y1, pad_y2), (pad_x1, pad_x2)), mode='constant',
        #                       constant_values=0)

        pad_y1 = int((self.padding * 2 - (y2 - y1)) // 2)
        pad_y2 = int(self.padding * 2 - (y2 - y1) - pad_y1)
        pad_x1 = int((self.padding * 2 - (x2 - x1)) // 2)
        pad_x2 = int(self.padding * 2 - (x2 - x1) - pad_x1)

        padded_image = np.pad(sub_image, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), mode='constant',
                            constant_values=0)

        # padded_image = np.transpose(padded_image, (1, 2, 0))
        padded_image = Image.fromarray(padded_image.astype(np.uint8))
        padded_image = self.model_transform(padded_image)

        return padded_image

    # def build_graph(self, neighbors, distances: list = None):
    #     num_nodes = len(neighbors)

    #     # create node features
    #     x = torch.stack([torch.tensor(self.truncate_X(n['X']), dtype=torch.int32) for n in neighbors])

    #     # for simplicity, we will create a fully connected graph
    #     edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
    #     edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected graph

    #     # create node positions
    #     pos = torch.tensor([[n['image_x'], n['image_y']] for n in neighbors], dtype=torch.float32)

    #     edge_attr = None
    #     if distances is not None:
    #         # use distances as edge weights
    #         # egde_attr: [num_edges, num_edge_features] = [num_nodes, 1]
    #         edge_attr = torch.tensor(distances * 2, dtype=torch.float32)  # multiply by 2 for undirected edges

    #     return torch_geometric.data.Data(x=x, edge_index=edge_index, pos=pos, edge_attr=edge_attr)

    def build_graph(self, neighbors, distances: list = None):
        num_nodes = len(neighbors)

        # 创建节点特征
        x = torch.stack([torch.tensor(self.truncate_X(n['X']), dtype=torch.float32) for n in neighbors])

        # 创建全连接图
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
        
        # 处理边属性（如果有）
        edge_attr = None
        if distances is not None:
            # 构建完整的边属性向量，与edge_index匹配
            # 首先为无向图的正向边创建属性
            forward_edge_attr = []
            for i, j in zip(edge_index[0], edge_index[1]):
                i, j = i.item(), j.item()
                # 找到对应节点的距离
                if i == num_nodes - 1:  # 如果i是中心节点
                    forward_edge_attr.append(distances[j])
                elif j == num_nodes - 1:  # 如果j是中心节点
                    forward_edge_attr.append(distances[i])
                else:
                    # 对于两个非中心节点之间的边，可以使用两个节点到中心的距离的某种组合
                    # 例如，平均值
                    forward_edge_attr.append((distances[i] + distances[j]) / 2)
            
            forward_edge_attr = torch.tensor(forward_edge_attr, dtype=torch.float32)
            
            # 现在创建无向图的双向边
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_attr = torch.cat([forward_edge_attr, forward_edge_attr], dim=0)

        # 创建节点位置
        pos = torch.tensor([[n['image_x'], n['image_y']] for n in neighbors], dtype=torch.float32)

        return torch_geometric.data.Data(x=x, edge_index=edge_index, pos=pos, edge_attr=edge_attr)

    def __getitem__(self, idx):

        if not hasattr(self, 'hdf_data') or self.hdf_data is None:
            self.hdf_data = h5py.File(self.cell_file, 'r')

        row_idx = random.randint(0, self.num_rows - 1)

        cell_data = {key: self.hdf_data[key][row_idx] for key in self.hdf_data.keys()}

        for key in cell_data:
            if isinstance(cell_data[key], np.ndarray) and cell_data[key].shape == ():
                cell_data[key] = cell_data[key].item()  # convert numpy scalar to Python scalar

        # 'assay', 'specie', 'modality', 'idx', 'dataset', 'condition_id', 'x', 'y',
        # 'cell_vertex_x', 'cell_vertex_y', 'nucleus_vertex_x', 'nucleus_vertex_y'
        x, y = cell_data['image_x'], cell_data['image_y']

        padded_image = self.get_padded_image(x, y)

        neighbors, distances = self.get_neighbors_from_indices(row_idx, cell_data, x, y, get_distances=True)

        neighbors.append(cell_data)
        distances.append(0.0)

        relative_coords = [self.padding, self.padding]

        item = {
            'image': padded_image,
            'X': torch.tensor(self.truncate_X(cell_data['X']), dtype=torch.int32),  # Changed to int32
            'modality': torch.tensor(cell_data['modality'], dtype=torch.int32),
            'assay': torch.tensor(cell_data['assay'], dtype=torch.int32),
            'specie': torch.tensor(cell_data['specie'], dtype=torch.int32),
            'spatial_coords': torch.tensor([x, y], dtype=torch.float32),
            'relative_coords': torch.tensor(relative_coords, dtype=torch.float32),
            'graph': self.build_graph(neighbors, distances)
        }

        return item

    def __len__(self):
        return self.num_rows

    @staticmethod
    def assert_continuous_and_ascending(idx_values):
        # Check if idx_values is a pandas Series
        assert isinstance(idx_values, pd.Series), "idx_values must be a pandas Series"

        # Check if the series is sorted
        is_sorted = idx_values.is_monotonic_increasing
        assert is_sorted, "idx_values must be sorted in increasing order"

        # Calculate the difference between consecutive values and check if all are equal to 1
        differences = idx_values.diff().iloc[1:]
        assert (differences == 1).all(), "idx_values must contain continuous integers"

        print("Assertion passed: idx_values is a series of continuous integers.")
    
    

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, spatial_pos_path, barcode_path, reduced_mtx_path):
        #image_path is the path of an entire slice of visium h&e stained image (~2.5GB)
        
        #spatial_pos_csv
            #barcode name
            #detected tissue boolean
            #x spot index
            #y spot index
            #x spot position (px)
            #y spot position (px)
        
        #expression_mtx
            #feature x spot (alphabetical barcode order)
    
        #barcode_tsv
            #spot barcodes - alphabetical order

        self.whole_image = cv2.imread(image_path)
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header = None) 
        # self.expression_mtx = csr_matrix(sio.mmread(expression_mtx_path)).toarray()
        self.barcode_tsv = pd.read_csv(barcode_path, sep="\t", header = None) 
        self.reduced_matrix = np.load(reduced_mtx_path).T  #cell x features
        
        print("Finished loading all files")

    def transform(self, image):
        image = Image.fromarray(image)
        # Random flipping and rotations
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.choice([180, 90, 0, -90])
        image = TF.rotate(image, angle)
        return np.asarray(image)

    def __getitem__(self, idx):
        item = {}
        barcode = self.barcode_tsv.values[idx,0]
        v1 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode,4].values[0]
        v2 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode,5].values[0]
        image = self.whole_image[(v1-112):(v1+112),(v2-112):(v2+112)]
        image = self.transform(image)
        
        item['image'] = torch.tensor(image).permute(2, 0, 1).float() #color channel first, then XY
        item['reduced_expression'] = torch.tensor(self.reduced_matrix[idx,:]).float()  #cell x features (3467)
        item['barcode'] = barcode
        item['spatial_coords'] = [v1,v2]

        return item


    def __len__(self):
        return len(self.barcode_tsv)