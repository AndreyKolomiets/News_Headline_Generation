import os
import re

root_dir = os.path.expanduser("~")

# train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
first_sentence_only = True
train_data_path = '/workspace/_Headline_generation/dataset_chunks/'
eval_data_path = '/workspace/_Headline_generation/test/'
decode_data_path = '/workspace/_Headline_generation/val/'
vocab_path = "/workspace/_Headline_generation/vocab"
if first_sentence_only:
    regex = re.compile('/$')
    train_data_path = regex.sub('_1st_sent/', train_data_path)
    eval_data_path = regex.sub('_1st_sent/', eval_data_path)
    decode_data_path = regex.sub('_1st_sent/', decode_data_path)
    vocab_path = regex.sub('_1st_sent/', vocab_path)
bpe_vocab_path = '/workspace/_Headline_generation/bpe_encoder.pkl'
log_root = "/workspace/_Headline_generation/log_pointer_summarizer/log"
use_bpe = True

# Hyperparameters
hidden_dim = 256
emb_dim = 256
batch_size = 8
max_enc_steps = 400
max_dec_steps = 100
beam_size = 4
min_dec_steps = 35
vocab_size = 200000
bpe_vocab_size = 8196  # собственно словарь плюс служебные символы.
# Плюс почему-то для маленького словаря все очень тормозит

lr = 0.3
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0

pointer_gen = True and not use_bpe
is_coverage = False
cov_loss_wt = 0.2

eps = 1e-12
max_iterations = 500000

use_gpu = True

lr_coverage = 0.15
