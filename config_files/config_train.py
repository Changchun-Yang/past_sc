
# config parameter for pa model:

#commented out variables are handled by argparse in main.py
debug = True
# batch_size = 128
# num_workers = 0
lr = 1e-3
weight_decay = 1e-3
patience = 2
factor = 0.5
# epochs = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'resnet50'
image_embedding = 2048
spot_embedding = 3467 #number of shared hvgs (change for each dataset)
cell_embedding = 20340

pretrained = True
trainable = True 
temperature = 1.0

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1


#  'vit_base_patch32_224',
#  'vit_base_patch32_224_clip_laion2b',
#  'vit_base_patch32_224_in21k',
#  'vit_base_patch32_224_sam',


# config parameter for st model:
pretrained_path = "/ibex/user/yangc0h/nicheformer/nicheformer.ckpt"
retake_training = False
dim_feedforward = 1024
nheads = 16
masking_p = 0.
nlayers = 12
dropout = 0.0
dim_model = 512
batch_first = True
n_tokens = 20340
batch_size = 2
context_length = 1500
lr = 7e-4
warmup = 100000
max_epochs = 30661140
autoregressive = False
pool = None
supervised_task = False
learnable_pe = True
# 'organ': "everything",
organ = 'test'
specie = True
assay = True
modality = True
contrastive = False

