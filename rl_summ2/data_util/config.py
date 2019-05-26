import re

first_sentence_only = False
train_data_path = '/workspace/_Headline_generation/dataset_chunks/'
test_data_path = '/workspace/_Headline_generation/test/'
valid_data_path = '/workspace/_Headline_generation/val/'
vocab_path = "/workspace/_Headline_generation/vocab"
if first_sentence_only:
    regex = re.compile('/$')
    train_data_path = regex.sub('_1st_sent/', train_data_path)
    test_data_path = regex.sub('_1st_sent/', test_data_path)
    valid_data_path = regex.sub('_1st_sent/', valid_data_path)
    vocab_path = regex.sub('_1st_sent/', vocab_path)
bpe_vocab_path = '/workspace/_Headline_generation/bpe_encoder.pkl'
log_root = "/workspace/_Headline_generation/log_rl_summarizer/"
# TODO: пока это не прикручено к RL summarizer
use_bpe = True

# Hyperparameters
hidden_dim = 256
emb_dim = 256
batch_size = 200
max_enc_steps = 150
max_dec_steps = 15  # 99% of the titles are within length 15
beam_size = 4
min_dec_steps = 3
vocab_size = 50000

lr = 0.0005
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4

eps = 1e-12
max_iterations = 500000

save_model_path = "data/saved_models"

intra_encoder = True
intra_decoder = True
