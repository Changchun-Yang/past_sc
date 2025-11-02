import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, GCNConv, MessagePassing, GINEConv, TransformerConv
from torch_geometric.data import Batch

import config_files.config_past_train as CFGP
import config_files.config_train as CFG
from modules import ImageEncoder, UniEncoder, UniEncoder_Random, ProjectionHead, ImageEncoder_ViT, ImageEncoder_ViT_L, ImageEncoder_CLIP, ImageEncoder_resnet101, ImageEncoder_resnet152
from modules import SpotEncoder, CellEncoder


from utils.util import (
    complete_masking, preprocess_cell_data
)

class PASTModel(nn.Module):
    def __init__(
        self,
        temperature=CFGP.temperature,
        image_embedding=CFGP.image_embedding,

        cell_embedding=CFGP.cell_dim_model,
        cell_dim_model=CFGP.cell_dim_model,
        cell_nheads=CFGP.cell_nheads,
        cell_dim_feedforward=CFGP.cell_dim_feedforward,
        cell_nlayers=CFGP.cell_nlayers,
        cell_dropout=CFGP.cell_dropout,
        cell_batch_first=CFGP.cell_batch_first,
        cell_masking_p=CFGP.cell_masking_p,
        cell_n_tokens=CFGP.cell_n_tokens,
        cell_context_length=CFGP.cell_context_length,
        cell_autoregressive=CFGP.cell_autoregressive,
        cell_pool=CFGP.cell_pool,
        cell_supervised_task=CFGP.cell_supervised_task,
        cell_learnable_pe=CFGP.cell_learnable_pe,
        cell_specie=CFGP.cell_specie,
        cell_assay=CFGP.cell_assay,
        cell_modality=CFGP.cell_modality,
        cell_contrastive=CFGP.cell_contrastive,
        cell_pretrained_path=CFGP.cell_pretrained_path,
        cell_freeze=CFGP.cell_freeze,
        cell_reinit_layers=CFGP.cell_reinit_layers,
        cell_extract_layers=CFGP.cell_extract_layers,
        cell_function_layers=CFGP.cell_function_layers,
        cell_without_context=CFGP.cell_without_context,
    ):
        super().__init__()
        self.image_encoder = UniEncoder()
        self.cell_encoder = CellEncoder(dim_model=cell_dim_model, nheads=cell_nheads, dim_feedforward=cell_dim_feedforward,
                                        nlayers=cell_nlayers, dropout=cell_dropout, batch_first=cell_batch_first,
                                        masking_p=cell_masking_p, n_tokens=cell_n_tokens, context_length=cell_context_length,
                                        autoregressive=cell_autoregressive, pool=cell_pool, supervised_task=cell_supervised_task,
                                        learnable_pe=cell_learnable_pe, specie=cell_specie, assay=cell_assay, modality=cell_modality,
                                        contrastive=cell_contrastive, pretrained_path=cell_pretrained_path, freeze=cell_freeze,
                                        reinit_layers=cell_reinit_layers, extract_layers=cell_extract_layers,
                                        function_layers=cell_function_layers, without_context=cell_without_context)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.cell_projection = ProjectionHead(embedding_dim=cell_embedding) #nicheformer embedding size
    #     self.temperature = temperature
    #
    # def forward(self, image, masked_indices, attention_mask):
    #     # Getting Image and cell Features
    #     image_features = self.image_encoder(image)
    #     cell_features = self.cell_encoder(masked_indices, attention_mask)
    #     # different nebirhoods numbers cells
    #     # cell_features2 = self.cell_encoder(masked_indices2, attention_mask2)
    #     # cell_features3 = self.cell_encoder(masked_indices3, attention_mask3)
    #
    #     # Getting Image and Cell Embeddings (with same dimension)
    #     image_embeddings = self.image_projection(image_features)
    #     cell_embeddings = self.cell_projection(cell_features)
    #
    #     # Calculating the Loss
    #     logits = (cell_embeddings @ image_embeddings.T) / self.temperature
    #     images_similarity = image_embeddings @ image_embeddings.T
    #     cells_similarity = cell_embeddings @ cell_embeddings.T
    #     targets = F.softmax(
    #         ((images_similarity + cells_similarity) / 2) / self.temperature, dim=-1
    #     )
    #     cells_loss = cross_entropy(logits, targets, reduction='none')
    #     images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    #     loss = (images_loss + cells_loss) / 2.0 # shape: (batch_size)
    #     return loss.mean()


        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.temperature_min = 0.01
        self.temperature_max = 1
        

    def forward(self, image, masked_indices, attention_mask):

        temperature = torch.clamp(
            self.temperature, 
            self.temperature_min, 
            self.temperature_max
        )

        image_features = self.image_encoder(image)
        cell_features = self.cell_encoder(masked_indices, attention_mask)
        
        # 投影和归一化
        image_embeddings = F.normalize(self.image_projection(image_features), dim=-1)
        cell_embeddings = F.normalize(self.cell_projection(cell_features), dim=-1)
        
        # 计算相似度
        logits = (cell_embeddings @ image_embeddings.T) / temperature
        # logits = torch.clamp(logits, -100, 100)  # 数值稳定性
        
        # 计算self-similarity
        images_similarity = image_embeddings @ image_embeddings.T
        cells_similarity = cell_embeddings @ cell_embeddings.T
        
        # 计算targets (使用softmax而不是log_softmax)
        targets = F.softmax(
            ((images_similarity + cells_similarity) / 2) / temperature,
            dim=-1
        )

        # # 使用cross_entropy
        # cells_loss = cross_entropy(logits, targets, reduction='mean')
        # images_loss = cross_entropy(logits.T, targets.T, reduction='mean')
        #
        # return (cells_loss + images_loss) / 2.0

        cells_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + cells_loss) / 2.0 # shape: (batch_size)
        return loss.mean(), self.temperature.item()


class PASTModel_Random(nn.Module):
    def __init__(
            self,
            temperature=CFGP.temperature,
            image_embedding=CFGP.image_embedding,

            cell_embedding=CFGP.cell_dim_model,
            cell_dim_model=CFGP.cell_dim_model,
            cell_nheads=CFGP.cell_nheads,
            cell_dim_feedforward=CFGP.cell_dim_feedforward,
            cell_nlayers=CFGP.cell_nlayers,
            cell_dropout=CFGP.cell_dropout,
            cell_batch_first=CFGP.cell_batch_first,
            cell_masking_p=CFGP.cell_masking_p,
            cell_n_tokens=CFGP.cell_n_tokens,
            cell_context_length=CFGP.cell_context_length,
            cell_autoregressive=CFGP.cell_autoregressive,
            cell_pool=CFGP.cell_pool,
            cell_supervised_task=CFGP.cell_supervised_task,
            cell_learnable_pe=CFGP.cell_learnable_pe,
            cell_specie=CFGP.cell_specie,
            cell_assay=CFGP.cell_assay,
            cell_modality=CFGP.cell_modality,
            cell_contrastive=CFGP.cell_contrastive,
            cell_pretrained_path=CFGP.cell_pretrained_path,
            cell_freeze=CFGP.cell_freeze,
            cell_reinit_layers=CFGP.cell_reinit_layers,
            cell_extract_layers=CFGP.cell_extract_layers,
            cell_function_layers=CFGP.cell_function_layers,
            cell_without_context=CFGP.cell_without_context,
    ):
        super().__init__()
        self.image_encoder = UniEncoder_Random()
        self.cell_encoder = CellEncoder(dim_model=cell_dim_model, nheads=cell_nheads,
                                        dim_feedforward=cell_dim_feedforward,
                                        nlayers=cell_nlayers, dropout=cell_dropout, batch_first=cell_batch_first,
                                        masking_p=cell_masking_p, n_tokens=cell_n_tokens,
                                        context_length=cell_context_length,
                                        autoregressive=cell_autoregressive, pool=cell_pool,
                                        supervised_task=cell_supervised_task,
                                        learnable_pe=cell_learnable_pe, specie=cell_specie, assay=cell_assay,
                                        modality=cell_modality,
                                        contrastive=cell_contrastive, pretrained_path=cell_pretrained_path,
                                        freeze=cell_freeze,
                                        reinit_layers=cell_reinit_layers, extract_layers=cell_extract_layers,
                                        function_layers=cell_function_layers, without_context=cell_without_context)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)  # aka the input dim, 2048 for resnet50
        self.cell_projection = ProjectionHead(embedding_dim=cell_embedding)  # nicheformer embedding size
        #     self.temperature = temperature
        #
        # def forward(self, image, masked_indices, attention_mask):
        #     # Getting Image and cell Features
        #     image_features = self.image_encoder(image)
        #     cell_features = self.cell_encoder(masked_indices, attention_mask)
        #     # different nebirhoods numbers cells
        #     # cell_features2 = self.cell_encoder(masked_indices2, attention_mask2)
        #     # cell_features3 = self.cell_encoder(masked_indices3, attention_mask3)
        #
        #     # Getting Image and Cell Embeddings (with same dimension)
        #     image_embeddings = self.image_projection(image_features)
        #     cell_embeddings = self.cell_projection(cell_features)
        #
        #     # Calculating the Loss
        #     logits = (cell_embeddings @ image_embeddings.T) / self.temperature
        #     images_similarity = image_embeddings @ image_embeddings.T
        #     cells_similarity = cell_embeddings @ cell_embeddings.T
        #     targets = F.softmax(
        #         ((images_similarity + cells_similarity) / 2) / self.temperature, dim=-1
        #     )
        #     cells_loss = cross_entropy(logits, targets, reduction='none')
        #     images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        #     loss = (images_loss + cells_loss) / 2.0 # shape: (batch_size)
        #     return loss.mean()

        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.temperature_min = 0.01
        self.temperature_max = 1

    def forward(self, image, masked_indices, attention_mask):
        temperature = torch.clamp(
            self.temperature,
            self.temperature_min,
            self.temperature_max
        )

        image_features = self.image_encoder(image)
        cell_features = self.cell_encoder(masked_indices, attention_mask)

        # 投影和归一化
        image_embeddings = F.normalize(self.image_projection(image_features), dim=-1)
        cell_embeddings = F.normalize(self.cell_projection(cell_features), dim=-1)

        # 计算相似度
        logits = (cell_embeddings @ image_embeddings.T) / temperature
        # logits = torch.clamp(logits, -100, 100)  # 数值稳定性

        # 计算self-similarity
        images_similarity = image_embeddings @ image_embeddings.T
        cells_similarity = cell_embeddings @ cell_embeddings.T

        # 计算targets (使用softmax而不是log_softmax)
        targets = F.softmax(
            ((images_similarity + cells_similarity) / 2) / temperature,
            dim=-1
        )

        # # 使用cross_entropy
        # cells_loss = cross_entropy(logits, targets, reduction='mean')
        # images_loss = cross_entropy(logits.T, targets.T, reduction='mean')
        #
        # return (cells_loss + images_loss) / 2.0

        cells_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + cells_loss) / 2.0  # shape: (batch_size)
        return loss.mean(), self.temperature.item()


class GraphEncoder(nn.Module):
    """利用距离信息作为边权重的图编码器"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, conv_type='gcn'):
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.conv_type = conv_type.lower()
        
        # 选择卷积类型
        if self.conv_type == 'gcn':
            ConvLayer = GCNConv
        elif self.conv_type == 'gat':
            ConvLayer = GATConv
        else:
            raise ValueError(f"不支持的卷积类型: {conv_type}")
        
        # 第一层
        self.convs.append(ConvLayer(in_channels, hidden_channels))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(ConvLayer(hidden_channels, hidden_channels))
        
        # 最后一层
        if num_layers > 1:
            self.convs.append(ConvLayer(hidden_channels, out_channels))
        
        # 归一化层
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers - 1)
        ])
        if num_layers > 0:
            self.norms.append(nn.LayerNorm(out_channels))
        
        self.dropout = nn.Dropout(0.1)
        
        # 距离变换函数
        self.distance_transform = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()  # 将距离映射到0-1之间的权重
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        使用距离作为边权重进行消息传递
        """
        # 处理边权重
        edge_weight = None
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                # 1. 简单的倒数转换: 距离越小，权重越大
                # edge_weight = 1.0 / (edge_attr + 1e-6)  # 防止除零
                
                # 2. 或使用高斯核: exp(-d²/σ²)
                # sigma = 1.0  # 可调参数
                # edge_weight = torch.exp(-(edge_attr**2) / (2 * sigma**2))
                
                # 3. 使用可学习的变换函数
                edge_weight = self.distance_transform(edge_attr.unsqueeze(1)).squeeze()
                
                # 4. 距离归一化
                edge_weight = F.normalize(edge_weight, p=1, dim=0)
        
        # 图卷积处理
        for i, conv in enumerate(self.convs):
            # 对于GCN，传入edge_weight
            if self.conv_type == 'gcn':
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:  # GAT等其他不使用edge_weight的卷积
                x = conv(x, edge_index)
            
            # 归一化、激活和dropout
            x = self.norms[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x
    

class GraphEncoder_v2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4):
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # 第一层，注意输出维度需要考虑多头
        self.convs.append(TransformerConv(
            in_channels=in_channels, 
            out_channels=hidden_channels // heads,  # 除以头数
            heads=heads,
            edge_dim=1,  # 假设边属性是1维的，如距离
            dropout=0.1
        ))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(
                in_channels=hidden_channels, 
                out_channels=hidden_channels // heads,
                heads=heads,
                edge_dim=1,
                dropout=0.1
            ))
        
        # 最后一层，通常使用1个头以获得确定的输出维度
        if num_layers > 1:
            self.convs.append(TransformerConv(
                in_channels=hidden_channels, 
                out_channels=out_channels,
                heads=1,  # 单头
                edge_dim=1,
                dropout=0.1
            ))
        
        # 归一化层
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers - 1)
        ])
        if num_layers > 0:
            self.norms.append(nn.LayerNorm(out_channels))
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)  # [num_edges, 1]
        
        for i, conv in enumerate(self.convs):
            # TransformerConv直接支持边属性
            x = conv(x, edge_index, edge_attr)
            
            # 归一化、激活和dropout
            x = self.norms[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x
    

class PASTModelGraph(nn.Module):
    def __init__(
        self,
        temperature=CFGP.temperature,
        image_embedding=CFGP.image_embedding,

        cell_embedding=CFGP.cell_dim_model,
        cell_dim_model=CFGP.cell_dim_model,
        cell_nheads=CFGP.cell_nheads,
        cell_dim_feedforward=CFGP.cell_dim_feedforward,
        cell_nlayers=CFGP.cell_nlayers,
        cell_dropout=CFGP.cell_dropout,
        cell_batch_first=CFGP.cell_batch_first,
        cell_masking_p=CFGP.cell_masking_p,
        cell_n_tokens=CFGP.cell_n_tokens,
        cell_context_length=CFGP.cell_context_length,
        cell_autoregressive=CFGP.cell_autoregressive,
        cell_pool=CFGP.cell_pool,
        cell_supervised_task=CFGP.cell_supervised_task,
        cell_learnable_pe=CFGP.cell_learnable_pe,
        cell_specie=CFGP.cell_specie,
        cell_assay=CFGP.cell_assay,
        cell_modality=CFGP.cell_modality,
        cell_contrastive=CFGP.cell_contrastive,
        cell_pretrained_path=CFGP.cell_pretrained_path,
        cell_freeze=CFGP.cell_freeze,
        cell_reinit_layers=CFGP.cell_reinit_layers,
        cell_extract_layers=CFGP.cell_extract_layers,
        cell_function_layers=CFGP.cell_function_layers,
        cell_without_context=CFGP.cell_without_context,

        # for graph
        graph_hidden_dim=256,
        graph_num_layers=2,
        # fusion type
        fusion_type='attention',  # 'concat', 'add', 'attention'
    ):
        super().__init__()
        self.image_encoder = UniEncoder()
        self.cell_encoder = CellEncoder(dim_model=cell_dim_model, nheads=cell_nheads, dim_feedforward=cell_dim_feedforward,
                                        nlayers=cell_nlayers, dropout=cell_dropout, batch_first=cell_batch_first,
                                        masking_p=cell_masking_p, n_tokens=cell_n_tokens, context_length=cell_context_length,
                                        autoregressive=cell_autoregressive, pool=cell_pool, supervised_task=cell_supervised_task,
                                        learnable_pe=cell_learnable_pe, specie=cell_specie, assay=cell_assay, modality=cell_modality,
                                        contrastive=cell_contrastive, pretrained_path=cell_pretrained_path, freeze=cell_freeze,
                                        reinit_layers=cell_reinit_layers, extract_layers=cell_extract_layers,
                                        function_layers=cell_function_layers, without_context=cell_without_context)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.cell_projection = ProjectionHead(embedding_dim=cell_embedding) #nicheformer embedding size

        self.fusion_type = fusion_type
        self.graph_encoder = GraphEncoder(
            in_channels=cell_context_length,  # 输入特征维度
            hidden_channels=graph_hidden_dim,
            out_channels=cell_embedding,      # 输出与细胞特征相同维度便于融合
            num_layers=graph_num_layers,
            conv_type='gat'  # 'gcn' or 'gat'
        )

        if fusion_type == 'weighted_add':
            # 加权相加方式，通过参数控制Transformer特征比重
            self.alpha = nn.Parameter(torch.tensor(0.7))  # 初始设置Transformer特征占70%
    
        elif fusion_type == 'gated':
            # 门控融合机制 - 动态学习最佳融合比例
            self.gate_net = nn.Sequential(
                nn.Linear(cell_embedding * 2, cell_embedding),
                nn.ReLU(),
                nn.Linear(cell_embedding, 1),
                nn.Sigmoid()
            )
            # 初始化偏向Transformer特征
            for m in self.gate_net.modules():
                if isinstance(m, nn.Linear):
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.5)  # 偏置初始化使门控偏向Transformer特征
            
        elif fusion_type == 'film':
            # 特征线性调制 - 用Transformer特征调制图特征
            self.film_generator = nn.Linear(cell_embedding, 2 * cell_embedding)
            self.film_norm = nn.LayerNorm(cell_embedding)
            
        elif fusion_type == 'concat':
            # 加权连接 - 初始化权重使Transformer特征更重要
            self.fusion_layer = nn.Sequential(
                nn.Linear(cell_embedding * 2, cell_embedding),
                nn.LayerNorm(cell_embedding),
                nn.ReLU()
            )
            # 初始化使Transformer特征权重更大
            with torch.no_grad():
                self.fusion_layer[0].weight[:, :cell_embedding] *= 1.5
                self.fusion_layer[0].weight[:, cell_embedding:] *= 0.5
            
        elif fusion_type == 'attention':
            # 带偏好的注意力融合
            self.fusion_layer = nn.MultiheadAttention(
                embed_dim=cell_embedding,
                num_heads=4,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(cell_embedding)
            # 使用2D掩码，PyTorch会自动处理批次和头部维度
            self.register_buffer('attn_bias', torch.tensor([[0.0, -1.0]]))


    #     self.temperature = temperature
    #
    # def forward(self, image, masked_indices, attention_mask):
    #     # Getting Image and cell Features
    #     image_features = self.image_encoder(image)
    #     cell_features = self.cell_encoder(masked_indices, attention_mask)
    #     # different nebirhoods numbers cells
    #     # cell_features2 = self.cell_encoder(masked_indices2, attention_mask2)
    #     # cell_features3 = self.cell_encoder(masked_indices3, attention_mask3)
    #
    #     # Getting Image and Cell Embeddings (with same dimension)
    #     image_embeddings = self.image_projection(image_features)
    #     cell_embeddings = self.cell_projection(cell_features)
    #
    #     # Calculating the Loss
    #     logits = (cell_embeddings @ image_embeddings.T) / self.temperature
    #     images_similarity = image_embeddings @ image_embeddings.T
    #     cells_similarity = cell_embeddings @ cell_embeddings.T
    #     targets = F.softmax(
    #         ((images_similarity + cells_similarity) / 2) / self.temperature, dim=-1
    #     )
    #     cells_loss = cross_entropy(logits, targets, reduction='none')
    #     images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    #     loss = (images_loss + cells_loss) / 2.0 # shape: (batch_size)
    #     return loss.mean()


        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.temperature_min = 0.01
        self.temperature_max = 1

    
    # def process_node_features(self, batch, graph_batch):
    #     """
    #     为图中的每个节点处理特征，使用你现有的预处理函数
    #     """
    #     device = batch['image'].device
    #     batch_size = len(batch['image'])
        
    #     # 处理每个图中的节点
    #     all_node_embeddings = []
    #     start_idx = 0
        
    #     for graph_idx in range(batch_size):
    #         # 找出当前图的所有节点索引
    #         if graph_idx < batch_size - 1:
    #             num_nodes = graph_batch.ptr[graph_idx + 1] - graph_batch.ptr[graph_idx]
    #         else:
    #             num_nodes = graph_batch.num_nodes - graph_batch.ptr[graph_idx]
            
    #         # 处理当前图的每个节点
    #         for node_idx in range(start_idx, start_idx + num_nodes):
    #             # 获取节点特征
    #             node_X = graph_batch.x[node_idx]
                
    #             # 构建单个节点的batch
    #             node_batch = {
    #                 'X': node_X.unsqueeze(0),  # 添加batch维度
    #                 'modality': batch['modality'][graph_idx].unsqueeze(0),
    #                 'assay': batch['assay'][graph_idx].unsqueeze(0),
    #                 'specie': batch['specie'][graph_idx].unsqueeze(0),
    #             }
                
    #             # 应用你的预处理函数
    #             processed_node = preprocess_cell_data(node_batch, CFGP)
    #             masked_node = complete_masking(processed_node, CFGP.cell_masking_p, CFGP.cell_n_tokens + 5)
                
    #             # 准备输入
    #             masked_indices = masked_node['masked_indices'].to(device)
    #             attention_mask = masked_node['attention_mask'].to(device)
                
    #             # 使用cell_encoder获取节点嵌入
    #             # with torch.no_grad():  # 可选的，如果你想冻结cell_encoder
    #             node_embedding = self.cell_encoder(masked_indices, attention_mask)
                
    #             all_node_embeddings.append(node_embedding)
            
    #         start_idx += num_nodes
        
    #     # 将所有节点的embeddings堆叠成一个tensor
    #     all_node_embeddings = torch.cat(all_node_embeddings, dim=0)
        
    #     return all_node_embeddings
        

    # def apply_gnn(self, node_embeddings, graph_batch):
    #     """
    #     应用图神经网络处理节点embeddings
    #     """
    #     x = node_embeddings
    #     edge_index = graph_batch.edge_index
    #     edge_attr = graph_batch.edge_attr if hasattr(graph_batch, 'edge_attr') else None
        
    #     # 应用GNN层
    #     for gnn_layer in self.gnn_layers:
    #         if edge_attr is not None:
    #             x = gnn_layer(x, edge_index, edge_attr)
    #         else:
    #             x = gnn_layer(x, edge_index)
    #         x = F.relu(x)
    #         x = self.dropout(x)
        
    #     # 最终投影
    #     x = self.graph_projection(x)
        
    #     # 获取中心节点嵌入
    #     # 根据你的build_graph函数，中心细胞应该是每个图的最后一个节点
    #     center_embeddings = []
    #     start_idx = 0
        
    #     for i in range(graph_batch.num_graphs):
    #         if i < graph_batch.num_graphs - 1:
    #             num_nodes = graph_batch.ptr[i + 1] - graph_batch.ptr[i]
    #         else:
    #             num_nodes = graph_batch.num_nodes - graph_batch.ptr[i]
            
    #         # 获取中心节点（最后一个节点）
    #         center_idx = start_idx + num_nodes - 1
    #         center_embeddings.append(x[center_idx])
            
    #         start_idx += num_nodes
        
    #     center_embeddings = torch.stack(center_embeddings)
        
    #     return x, center_embeddings
    
    
    # def process_center_nodes_only(self, batch, graph_batch):
    #     """
    #     仅处理每个图的中心节点
    #     """
    #     device = batch['image'].device
    #     batch_size = len(batch['image'])
        
    #     # 获取每个图的中心节点（最后一个节点）
    #     center_embeddings = []
    #     start_idx = 0
        
    #     for graph_idx in range(batch_size):
    #         if graph_idx < batch_size - 1:
    #             num_nodes = graph_batch.ptr[graph_idx + 1] - graph_batch.ptr[graph_idx]
    #         else:
    #             num_nodes = graph_batch.num_nodes - graph_batch.ptr[graph_idx]
            
    #         # 中心节点是最后一个节点
    #         center_idx = start_idx + num_nodes - 1
    #         center_X = graph_batch.x[center_idx]
            
    #         # 构建单个节点的batch
    #         center_batch = {
    #             'X': center_X.unsqueeze(0),  # 添加batch维度
    #             'modality': batch['modality'][graph_idx].unsqueeze(0),
    #             'assay': batch['assay'][graph_idx].unsqueeze(0),
    #             'specie': batch['specie'][graph_idx].unsqueeze(0),
    #         }
            
    #         # 应用你的预处理函数
    #         processed_center = preprocess_cell_data(center_batch, CFGP)
    #         masked_center = complete_masking(processed_center, CFGP.cell_masking_p, CFGP.cell_n_tokens + 5)
            
    #         # 准备输入
    #         masked_indices = masked_center['masked_indices'].to(device)
    #         attention_mask = masked_center['attention_mask'].to(device)
            
    #         # 使用cell_encoder获取节点嵌入
    #         center_embedding = self.cell_encoder(masked_indices, attention_mask)
    #         center_embeddings.append(center_embedding)
            
    #         start_idx += num_nodes
        
    #     center_embeddings = torch.cat(center_embeddings, dim=0)
        
    #     return center_embeddings
    
    def forward(self, image, masked_indices, attention_mask, graph_batch):
        """
        Args:
            image: 病理图像 [batch_size, channels, height, width]
            masked_indices: 掩码后的中心细胞索引 [batch_size, seq_length]
            attention_mask: 注意力掩码 [batch_size, seq_length]
            graph_batch: 包含中心节点和邻居的图批次
        """
        batch_size = image.size(0)
        temperature = torch.clamp(self.temperature, self.temperature_min, self.temperature_max)
        
        # 1. 处理图像
        image_features = self.image_encoder(image)
        image_embeddings = F.normalize(self.image_projection(image_features), dim=-1)
        
        # 2. 处理中心细胞（使用掩码和预训练的Transformer）
        center_cell_features = self.cell_encoder(masked_indices, attention_mask)
        
        # 3. 处理图数据（包括中心节点和邻居）
        # 注意：这里假设graph_batch.x已经过预处理但未掩码
        graph_node_features = self.graph_encoder(
            x=graph_batch.x,
            edge_index=graph_batch.edge_index,
            edge_attr=graph_batch.edge_attr if hasattr(graph_batch, 'edge_attr') else None
        )
        
        # 4. 提取每个图的中心节点特征
        center_graph_features = []
        
        # 获取每个图在批次中的节点索引区间
        ptr = graph_batch.ptr.cpu()
        
        for i in range(batch_size):
            # 中心节点是每个图的最后一个节点
            center_idx = ptr[i+1] - 1
            center_graph_features.append(graph_node_features[center_idx])
        
        # 转换为张量 [batch_size, embedding_dim]
        center_graph_features = torch.stack(center_graph_features)
        
        # 5. 融合中心细胞特征和图特征
        if self.fusion_type == 'add':
            # 传统的加法融合但略微偏向Transformer
            fused_features = 0.6 * center_cell_features + 0.4 * center_graph_features

        elif self.fusion_type == 'weighted_add':
            # 可学习权重的加权融合
            alpha = torch.sigmoid(self.alpha)  # 将参数映射到0-1之间
            fused_features = alpha * center_cell_features + (1-alpha) * center_graph_features
            
        elif self.fusion_type == 'gated':
            # 门控融合 - 动态决定每个特征的权重
            combined = torch.cat([center_cell_features, center_graph_features], dim=1)
            gate = self.gate_net(combined)
            fused_features = gate * center_cell_features + (1 - gate) * center_graph_features
            
        elif self.fusion_type == 'film':
            # 用Transformer特征调制图特征
            film_params = self.film_generator(center_cell_features)
            gamma, beta = torch.chunk(film_params, 2, dim=1)
            modulated_graph = gamma * center_graph_features + beta
            # 主要使用Transformer特征，辅以调制后的图特征
            fused_features = center_cell_features + 0.3 * modulated_graph
            fused_features = self.film_norm(fused_features)
            
        elif self.fusion_type == 'concat':
            # 带权重初始化的连接融合
            concat_features = torch.cat([center_cell_features, center_graph_features], dim=1)
            fused_features = self.fusion_layer(concat_features)
            
        elif self.fusion_type == 'attention':
            # 带偏好的注意力融合
            query = center_cell_features.unsqueeze(1)
            key_value = torch.stack([center_cell_features, center_graph_features], dim=1)
            
            # 使用2D掩码
            attn_output, _ = self.fusion_layer(
                query, key_value, key_value,
                need_weights=False,
                attn_mask=self.attn_bias
            )
            
            # 加强Transformer特征影响
            fused_features = self.fusion_norm(attn_output.squeeze(1) + center_cell_features)
        
        # 6. 投影融合特征
        cell_embeddings = F.normalize(self.cell_projection(fused_features), dim=-1)
        
        # 7. 计算对比损失
        logits = (cell_embeddings @ image_embeddings.T) / temperature
        
        # 自相似度
        images_similarity = image_embeddings @ image_embeddings.T
        cells_similarity = cell_embeddings @ cell_embeddings.T
        
        # 计算目标
        targets = F.softmax(((images_similarity + cells_similarity) / 2) / temperature, dim=-1)
        
        # 计算损失
        cells_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + cells_loss) / 2.0
        
        return loss.mean(), self.temperature.item()



class PAST_Finetune_Model(nn.Module):
    def __init__(
            self,
            temperature=CFGP.temperature,
            image_embedding=CFGP.image_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()

        self.image_projection = ProjectionHead(embedding_dim=image_embedding)  # aka the input dim, 2048 for resnet50

    def forward(self, image, masked_indices, attention_mask):
        # Getting Image and cell Features
        image_features = self.image_encoder(image)
        cell_features = self.cell_encoder(masked_indices, attention_mask)
        # different nebirhoods numbers cells
        # cell_features2 = self.cell_encoder(masked_indices2, attention_mask2)
        # cell_features3 = self.cell_encoder(masked_indices3, attention_mask3)

        # Getting Image and Cell Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        cell_embeddings = self.cell_projection(cell_features)

        # Calculating the Loss
        logits = (cell_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        cells_similarity = cell_embeddings @ cell_embeddings.T
        targets = F.softmax(
            ((images_similarity + cells_similarity) / 2) / self.temperature, dim=-1
        )
        cells_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + cells_loss) / 2.0  # shape: (batch_size)
        return loss.mean()


class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            ((images_similarity + spots_similarity) / 2) / self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    
class CLIPModel_ViT(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=768,
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_ViT()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


class CLIPModel_CLIP(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=768,
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_CLIP()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

class CLIPModel_ViT_L(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=1024,
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_ViT_L()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


class CLIPModel_resnet101(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=2048,
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_resnet101()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

class CLIPModel_resnet152(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=2048,
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_resnet152()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print("")