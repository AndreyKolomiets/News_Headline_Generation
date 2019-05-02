import os

root_dir = os.path.expanduser("~")

# train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
train_data_path = '/workspace/_Headline_generation/dataset_chunks/'
eval_data_path = '/workspace/_Headline_generation/test/'
decode_data_path = '/workspace/_Headline_generation/val/'
vocab_path = "/workspace/_Headline_generation/vocab"
log_root = "/workspace/_Headline_generation/log_pointer_summarizer/log"

# Hyperparameters
hidden_dim = 256
emb_dim = 128
batch_size = 8
max_enc_steps = 400
max_dec_steps = 100
beam_size = 4
min_dec_steps = 35
vocab_size = 200000

lr = 0.15
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

use_gpu = True

lr_coverage = 0.15
