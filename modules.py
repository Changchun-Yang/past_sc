import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import config_files.config_train as CFG
import config_files.config_past_train as CFGP
import config_files.config_past_finetune as CFGPF
import math
import torch.nn.init as init
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Batch
from functools import partial
from typing import Union, List, Callable, Tuple, OrderedDict
from torchvision import transforms


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)
        self.register_buffer('encoding', encoding, persistent=False)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

MASK_TOKEN = 0
CLS_TOKEN = 2


class CellEncoder(nn.Module):
    def __init__(self,
                 dim_model: int,
                 nheads: int,
                 dim_feedforward: int,
                 nlayers: int,
                 dropout: float,
                 batch_first: bool,
                 masking_p: float,
                 n_tokens: int,
                 context_length: int,
                 autoregressive: bool,
                 pool: str = 'mean',  # Added pooling parameter
                 cls_classes: int = 164,
                 supervised_task: int = None,
                 learnable_pe: bool = True,
                 specie: bool = False,
                 assay: bool = False,
                 modality: bool = False,
                 contrastive: bool = False,
                 pretrained_path: str = None,
                 freeze: bool = False,  # Added freeze parameter
                 reinit_layers: list = None,  # Added reinit_layers parameter
                 extract_layers: list = None,  # For extracting representations
                 function_layers: str = 'mean',  # For combining hidden representations
                 without_context: bool = True,  # if contextual tokens will be used or not
                 ):
        """
        Args:
            dim_model (int): Dimensionality of the model
            nheads (int): Number of attention heads
            dim_feedforward (int): Dimensionality of MLPs of attention blocks
            nlayers (int): Number of TransformerEncoder layers
            dropout (float): Dropout rate
            batch_first (bool): If True, batch dimension comes first
            masking_p (float): Probability of masking tokens
            n_tokens (int): Number of tokens (excluding auxiliary tokens)
            context_length (int): Length of the context (sequence length)
            autoregressive (bool): If True, uses causal attention
            pool (str): Pooling method ('mean', 'cls', or None)
            cls_classes (int): Number of classes for classification head
            supervised_task (int): Supervised task identifier
            learnable_pe (bool): If True, uses learnable positional embeddings
            specie (bool): If True, adds specie token
            assay (bool): If True, adds assay token
            modality (bool): If True, adds modality token
            contrastive (bool): If True, uses contrastive loss
            pretrained_path (str): Path to the pretrained checkpoint file (PyTorch  .ckpt)
            freeze (bool): If True, freezes the encoder parameters.
            reinit_layers (list): List of layer indices to re-initialize.
            extract_layers (list): Layers from which to extract representations.
            function_layers (str): How to combine representations ('mean', 'sum', 'concat').
            without_context: bool: if contextual tokens (modality, assay, specie) will be used or not
        """
        super().__init__()

        self.dim_model = dim_model
        self.n_tokens = n_tokens
        self.context_length = context_length
        self.pool = pool
        self.learnable_pe = learnable_pe
        self.autoregressive = autoregressive
        self.extract_layers = extract_layers if extract_layers is not None else [nlayers - 1]
        self.function_layers = function_layers
        self.without_context = without_context

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=nheads,
                                                        dim_feedforward=dim_feedforward, dropout=dropout,
                                                        batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=nlayers, norm=None)

        # Token embedding learnable weights
        self.embeddings = nn.Embedding(num_embeddings=n_tokens + 5, embedding_dim=dim_model, padding_idx=1)

        if pool == 'cls':
            self.context_length += 1  # Adjust context length for CLS token

        if not learnable_pe:
            self.positional_embedding = PositionalEncoding(d_model=dim_model, max_seq_len=self.context_length)
        else:
            # Uses learnable positional embeddings
            self.positional_embedding = nn.Embedding(num_embeddings=self.context_length, embedding_dim=dim_model)
            self.dropout = nn.Dropout(p=dropout)
            self.pos = torch.arange(0, self.context_length, dtype=torch.long)

        self.initialize_weights()

        # self.masking_p = masking_p

        # Load pretrained weights if a path is provided
        if pretrained_path is not None:
            self.load_pretrained_weights(pretrained_path)

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        # Re-initialize specified layers
        if reinit_layers is not None:
            for layer_idx in reinit_layers:
                if 0 <= layer_idx < nlayers:
                    layer = self.encoder.layers[layer_idx]
                    print(f"Re-initializing layer {layer_idx}")
                    self.initialize_weights(layer)
                else:
                    print(f"Layer index {layer_idx} is out of bounds, skipping.")

    def forward(self, x, attention_mask):
        # x: batch_size x seq_length
        # attention_mask: batch_size x seq_length

        # Get token embeddings
        token_embedding = self.embeddings(x)  # batch_size x seq_length x dim_model

        # Add positional embeddings
        if self.learnable_pe:
            positions = self.pos.unsqueeze(0).expand(x.size(0), -1).to(x.device)  # batch_size x seq_length
            pos_embedding = self.positional_embedding(positions)  # batch_size x seq_length x dim_model
            embeddings = self.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.positional_embedding(token_embedding)

        # # Apply Transformer encoder
        # transformer_output = self.encoder(embeddings, src_key_padding_mask=attention_mask)

        hidden_repr = []
        output = embeddings

        for i in range(len(self.encoder.layers)):
            layer = self.encoder.layers[i]
            output = layer(output, src_key_padding_mask=attention_mask)
            if i in self.extract_layers:
                hidden_repr.append(output)

        # Combine hidden representations
        if len(hidden_repr) > 1:
            if self.function_layers == 'mean':
                combined_tensor = torch.stack(hidden_repr, dim=-1)
                transformer_output = combined_tensor.mean(dim=-1)
            elif self.function_layers == 'sum':
                combined_tensor = torch.stack(hidden_repr, dim=-1)
                transformer_output = combined_tensor.sum(dim=-1)
            elif self.function_layers == 'concat':
                transformer_output = torch.cat(hidden_repr, dim=-1)
            else:
                transformer_output = hidden_repr[-1]
        else:
            transformer_output = hidden_repr[0]

        # Pooling
        if self.pool == 'mean':
            if self.without_context:
                # Exclude auxiliary tokens if any (e.g., first three tokens)
                cell_features = transformer_output[:, 3:, :].mean(dim=1)
            else:
                cell_features = transformer_output.mean(dim=1)
        elif self.pool == 'cls':
            # Take the representation of the first token (assumed to be [CLS])
            cell_features = transformer_output[:, 0, :]
        else:
            # Return the sequence of feature vectors without pooling
            cell_features = transformer_output  # batch_size x seq_length x dim_model

        return cell_features

    def initialize_weights(self, module=None):
        if module is None:
            module = self
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize embeddings
        if isinstance(self.embeddings, nn.Embedding):
            nn.init.normal_(self.embeddings.weight, mean=0, std=0.02)
        if self.learnable_pe and isinstance(self.positional_embedding, nn.Embedding):
            nn.init.normal_(self.positional_embedding.weight, mean=0, std=0.02)

    def load_pretrained_weights(self, pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        # Process state_dict keys
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove prefixes like 'backbone.', if present
            if k.startswith('backbone.'):
                new_key = k[len('backbone.'):]
            elif k.startswith('model.'):
                new_key = k[len('model.'):]
            elif k.startswith('module.'):
                new_key = k[len('module.'):]
            else:
                new_key = k
            new_state_dict[new_key] = v
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys when loading pretrained weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in pretrained weights: {unexpected_keys}")


class SpotEncoder(nn.Module):
    
    def __init__(self, 
                 dim_model: int = CFG.dim_model, 
                 nheads: int = CFG.nheads, 
                 dim_feedforward: int = CFG.dim_feedforward,
                 nlayers: int = CFG.nlayers, 
                 dropout: float = CFG.dropout,
                 batch_first: bool = CFG.batch_first, 
                 masking_p: float = CFG.masking_p, 
                 n_tokens: int = CFG.n_tokens,
                 context_length: int = CFG.context_length,
                 lr: float = CFG.lr, 
                 warmup: int = CFG.warmup, 
                 batch_size: int = CFG.batch_size, 
                 max_epochs: int = CFG.max_epochs,
                 autoregressive: bool = CFG.autoregressive,
                 pool: str = None,
                 cls_classes: int = 164,
                 supervised_task: int = None,
                 learnable_pe: bool = True,
                 specie: bool = False,
                 assay: bool = False,
                 modality: bool = False,
                 contrastive: bool = False
                 ):
        """
        Args:
            dim_model (int): Dimensionality of the model
            nheads (int): Number of attention heads
            dim_feedforward (int): Dimensionality of MLPs of attention blocks
            batch_first (int): batch first dimension
            masking_p (float): p value of Bernoulli for masking
            n_tokens (int): total number of tokens (WITHOUT auxiliar tokens)
            context_length (int): length of the context, who would have guessed... 
            lr (float): learning rate
            warmup (int): number of steps that the warmup takes
            max_epochs (int): number of steps until the learning rate reaches 0
            autoregressive (bool): if True, implements autoregressive training
            pool (str): could be None, 'cls' or 'mean'. CLS adds a token at the beginning, mean just averages all tokens. If not supervised task during training, is ignored
            cls_classes (int): number of classes to classify
            supervised_task (str): None, 'classification' or 'regression'
            learnable_pe (bool): if True, positional embeddings are learnable embeddings, otherwise are derived from trigonometric functions
            specie (bool): if True, add a token to identify the specie of the observation (human or mouse)
            assay (bool): if True, add a token to identify the assay of the observations 
            modality (bool): if True, add a token to identify the modality of the observations (spatial or dissociated)
            contrastive (bool): if True, uses contrastive loss
        """
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=nheads, dim_feedforward=dim_feedforward, batch_first=batch_first, dropout=dropout, layer_norm_eps=1e-12)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=nlayers, enable_nested_tensor=False)
        
        # As in HuggingFace
        self.classifier_head = nn.Linear(dim_model, n_tokens, bias=False)
        bias = nn.Parameter(torch.zeros(n_tokens)) # each token has its own bias
        self.classifier_head.bias = bias
            
        # As in HuggingFace
        self.pooler_head = nn.Linear(dim_model, dim_model)
        self.activation = nn.Tanh()
        self.cls_head = nn.Linear(dim_model, cls_classes)

        # Token embedding learnable weights
        self.embeddings = nn.Embedding(num_embeddings=n_tokens+5, embedding_dim=dim_model, padding_idx=1)
        
        if pool == 'cls':
            context_length += 1
            
        if not learnable_pe:
            self.positional_embedding = PositionalEncoding(d_model=dim_model, max_seq_len=context_length)
        else:
            # uses learnable weights as positional embeddings
            self.positional_embedding = nn.Embedding(num_embeddings=context_length, embedding_dim=dim_model) 
            self.dropout = nn.Dropout(p=dropout)
            self.pos = torch.arange(0, context_length, dtype=torch.long)
        
        # MLM loss
        self.loss = nn.CrossEntropyLoss()
       
        if supervised_task is not None and not False:
            self.cls_loss = nn.CrossEntropyLoss()
            
        self.autoregressive = autoregressive

            
    def forward(self, x, attention_mask):
                
        # x -> size: batch x (context_length) x 1
        token_embedding = self.embeddings(x) # batch x (n_tokens) x dim_model
        
        if self.hparams.learnable_pe:
            pos_embedding = self.positional_embedding(self.pos.to(token_embedding.device)) # batch x (n_tokens) x dim_model        
            embeddings = self.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.positional_embedding(token_embedding)
        
        transformer_output = self.encoder(embeddings, is_causal=self.autoregressive, src_key_padding_mask=attention_mask) # batch x (n_tokens) x dim_model

        # MLM prediction
        prediction = self.classifier_head(transformer_output)
            
        return {'mlm_prediction': prediction,
                'transformer_output': transformer_output}

class GNNBase(nn.Module):
    def __init__(self):
        super(GNNBase, self).__init__()

    def _argsparse(self, *args, **kwargs):
        r""" Parse the possible input types.
        If the x and edge_index are in args, follow the args.
        In other case, find them in kwargs.
        """
        if args:
            if len(args) == 1:
                data = args[0]
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, 'batch'):
                    batch = data.batch
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)

            elif len(args) == 2:
                x, edge_index = args[0], args[1]
                batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)

            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]

            else:
                raise ValueError(f"forward's args should take 1, 2 or 3 arguments but got {len(args)}")
        else:
            data: Batch = kwargs.get('data')
            if not data:
                x = kwargs.get('x')
                edge_index = kwargs.get('edge_index')
                assert x is not None, "forward's args is empty and required node features x is not in kwargs"
                assert edge_index is not None, "forward's args is empty and required edge_index is not in kwargs"
                batch = kwargs.get('batch')
                if not batch:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
            else:
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, 'batch'):
                    batch = data.batch
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        return x, edge_index, batch

    # def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor],
    #                     strict: bool = True):
    #     new_state_dict = OrderedDict()
    #     for key in state_dict.keys():
    #         if key in self.state_dict().keys():
    #             new_state_dict[key] = state_dict[key]
    #
    #     super(GNNBase, self).load_state_dict(new_state_dict)

class GraphEncoder(GNNBase):
    """
    Encode the target graph node to a fixed size vector
    """

    @staticmethod
    def get_nonlinear(nonlinear: str):
        nonlinear_func_dict = {
            "relu": F.relu,
            "leakyrelu": partial(F.leaky_relu, negative_slope=0.2),
            "sigmoid": F.sigmoid,
            "elu": F.elu
        }
        return nonlinear_func_dict[nonlinear]
    
    @staticmethod
    def get_gnn_layers(model_name: str, in_channels: int, out_channels: int, **kargs):
        GNN_DICT = {
            "gcn": gnn.GCNConv,
            "gat": gnn.GATConv,
            "graph": gnn.GraphConv,
            "sage": gnn.SAGEConv,
            "cheb": gnn.ChebConv,
            "gin": gnn.GINConv,
        }
        if model_name.lower() not in GNN_DICT:
            raise ValueError(f"Model {model_name} not implemented")
        return GNN_DICT[model_name.lower()](in_channels, out_channels, **kargs)
    
    @staticmethod
    def identity(x: torch.Tensor, batch: torch.Tensor):
        return x

    @staticmethod
    def cat_max_sum(x: torch.Tensor, batch: torch.Tensor):
        node_dim = x.shape[-1]
        num_node = 25
        x = x.reshape(-1, num_node, node_dim)
        return torch.cat([x.max(dim=1)[0], x.sum(dim=1)], dim=-1)
    
    @staticmethod
    def get_readout_layers(readout: str):
        readout_func_dict = {
            "mean": gnn.global_mean_pool,
            "sum": gnn.global_add_pool,
            "max": gnn.global_max_pool,
            'identity': GraphEncoder.identity,
            "cat_max_sum": GraphEncoder.cat_max_sum,
        }
        readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
        return readout_func_dict[readout.lower()]
    
    @staticmethod
    def GNNPool(readout: str):
        return GraphEncoder.get_readout_layers(readout)
    
    def __init__(self,
                 gnn_name: str,
                 input_dim: int,
                 output_dim: int,
                 gnn_latent_dim: Union[List[int]],
                 gnn_dropout: float = 0.0,
                 gnn_emb_normalization: bool = False,
                 gcn_adj_normalization: bool = True,
                 add_self_loop: bool = True,
                 gnn_nonlinear: str = 'relu',
                 readout: str = 'mean',
                 concate: bool = False,
                 fc_latent_dim: Union[List[int]] = [],
                 fc_dropout: float = 0.0,
                 fc_nonlinear: str = 'relu',
                 ):
        super(GraphEncoder, self).__init__()
        self.gnn_name = gnn_name
        # first and last layer - dim_features and classes
        self.input_dim = input_dim
        self.output_dim = output_dim
        # GNN part
        self.gnn_latent_dim = gnn_latent_dim
        self.gnn_dropout = gnn_dropout
        self.num_gnn_layers = len(self.gnn_latent_dim)
        self.add_self_loop = add_self_loop
        self.gnn_emb_normalization = gnn_emb_normalization
        self.gcn_adj_normalization = gcn_adj_normalization
        self.gnn_nonlinear = self.get_nonlinear(gnn_nonlinear)
        self.concate = concate
        # readout
        self.readout_layer = self.GNNPool(readout)
        # FC part
        self.fc_latent_dim = fc_latent_dim
        self.fc_dropout = fc_dropout
        self.num_mlp_layers = len(self.fc_latent_dim) + 1
        self.fc_nonlinear = self.get_nonlinear(fc_nonlinear)

        if self.concate:
            self.emb_dim = sum(self.gnn_latent_dim)
        else:
            self.emb_dim = self.gnn_latent_dim[-1]

        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(
            self.get_gnn_layers(self.gnn_name, input_dim, self.gnn_latent_dim[0],
                                add_self_loops=self.add_self_loop,
                                normalize=self.gcn_adj_normalization)
        )
        for i in range(1, self.num_gnn_layers):
            self.convs.append(
                self.get_gnn_layers(self.gnn_name, self.gnn_latent_dim[i - 1], self.gnn_latent_dim[i],
                                    add_self_loops=self.add_self_loop,
                                    normalize=self.gcn_adj_normalization)
            )
        # FC layers
        self.mlps = nn.ModuleList()
        if self.num_mlp_layers > 1:
            self.mlps.append(nn.Linear(self.emb_dim, self.fc_latent_dim[0]))

            for i in range(1, self.num_mlp_layers - 1):
                self.mlps.append(nn.Linear(self.fc_latent_dim[i - 1], self.fc_latent_dim[1]))
            self.mlps.append(nn.Linear(self.fc_latent_dim[-1], self.output_dim))
        else:
            self.mlps.append(nn.Linear(self.emb_dim, self.output_dim))

    def device(self):
        return self.convs[0].weight.device

    def get_emb(self, *args, **kwargs):
        #  node embedding for GNN
        x, edge_index, _ = self._argsparse(*args, **kwargs)
        xs = []
        for i in range(self.num_gnn_layers):
            x = self.convs[i](x, edge_index)
            if self.gnn_emb_normalization:
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_nonlinear(x)
            x = F.dropout(x, self.gnn_dropout)
            xs.append(x)

        if self.concate:
            return torch.cat(xs, dim=1)
        else:
            return x

    def forward(self, *args, **kwargs):
        _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        emb = self.get_emb(*args, **kwargs)
        # pooling process
        x = self.readout_layer(emb, batch)

        for i in range(self.num_mlp_layers - 1):
            x = self.mlps[i](x)
            x = self.fc_nonlinear(x)
            x = F.dropout(x, p=self.fc_dropout)

        logits = self.mlps[-1](x)
        return logits, emb       # (prediction, embedding)
                

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFGP.image_model_name, pretrained=CFGP.image_pretrained, trainable=CFGP.image_trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class UniEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFGPF.image_model_name, pretrained=CFGPF.image_pretrained,
            trainable=CFGPF.image_trainable, num_classes=CFGPF.num_classes
    ):
        super().__init__()
        self.model = timm.create_model(
            'vit_large_patch16_224', img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )

        self.model.load_state_dict(torch.load("clip/pytorch_model.bin", map_location="cpu"), strict=True)

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class UniEncoder_Random(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFGPF.image_model_name, pretrained=CFGPF.image_pretrained,
            trainable=CFGPF.image_trainable, num_classes=CFGPF.num_classes
    ):
        super().__init__()
        self.model = timm.create_model(
            'vit_large_patch16_224', img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )

        # self.model.load_state_dict(torch.load("clip/pytorch_model.bin", map_location="cpu"), strict=True)

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class ImageEncoder_resnet50(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_resnet101(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet101', pretrained=True, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_resnet152(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet152', pretrained=True, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_ViT(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name="vit_base_patch32_224", pretrained=False, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    

class ImageEncoder_CLIP(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name="vit_base_patch32_224_clip_laion2b", pretrained=True, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class ImageEncoder_ViT_L(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name="vit_large_patch32_224_in21k", pretrained=False, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    

#  'vit_base_patch32_224',
#  'vit_base_patch32_224_clip_laion2b',
#  'vit_base_patch32_224_in21k',
#  'vit_base_patch32_224_sam',
    

# class SpotEncoder(nn.Module):
#     #to change...
#     def __init__(self, model_name=CFG.spot_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
#         super().__init__()
#         if pretrained:
#             self.model = DistilBertModel.from_pretrained(model_name)
#         else:
#             self.model = DistilBertModel(config=DistilBertConfig())
            
#         for p in self.model.parameters():
#             p.requires_grad = trainable

#         # we are using the CLS token hidden representation as the sentence's embedding
#         self.target_token_idx = 0

#     def forward(self, input_ids, attention_mask):
#         output = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_state = output.last_hidden_state
#         return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
