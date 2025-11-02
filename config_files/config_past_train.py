
# config parameter for pa model:

#commented out variables are handled by argparse in main.py
debug = True
# batch_size = 128
# num_workers = 0
lr = 5e-4
weight_decay = 2e-2
patience = 2
factor = 0.5
# epochs = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_model_name = 'uni'
image_embedding = 1024
# spot_embedding = 3467 #number of shared hvgs (change for each dataset)
cell_embedding = 20340 # number of genes (change for each dataset)

image_pretrained = True
image_trainable = True
temperature = 1.0

# image size
image_size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1


#  'vit_base_patch32_224',
#  'vit_base_patch32_224_clip_laion2b',
#  'vit_base_patch32_224_in21k',
#  'vit_base_patch32_224_sam',


# config parameter for st model:
# cell_pretrained_path = "/ibex/user/yangc0h/nicheformer/nicheformer.ckpt"
# cell_context_length = 1500

# do not use nicheformer
cell_pretrained_path = None
cell_context_length = 300

# retake_training = False
cell_dim_feedforward = 1024
cell_nheads = 16
cell_masking_p = 0.
cell_nlayers = 12
cell_dropout = 0.

cell_dim_model = 512
cell_batch_first = True
cell_n_tokens = 20340
# batch_size = 2


# warmup = 100000
# max_epochs = 30661140
cell_autoregressive = False
cell_pool = 'mean'
cell_supervised_task = False
cell_learnable_pe = True

cell_specie = True
cell_assay = True
cell_modality = True
cell_contrastive = False

cell_freeze = False
cell_reinit_layers = None
cell_extract_layers = None
cell_function_layers = 'mean'
cell_without_context = True

# 'organ': "everything",



