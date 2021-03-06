import os
# жесткий хардкод, лучше проставлять ручками при запуске
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from typing import List

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from rl_summ2.model import Model
import datetime

from rl_summ2.data_util import config, data
if not config.use_bpe:
    from rl_summ2.data_util.batcher import Batcher
else:
    from pointer_summarizer.data_util.batcher_bpe import Batcher
from rl_summ2.data_util.data import Vocab, make_bpe_vocab
from rl_summ2.train_util import get_cuda, get_enc_data
from rl_summ2.beam_search import beam_search
from rouge import Rouge
import argparse
import pickle
from pointer_summarizer.training_ptr_gen.train_util import init_logger


class Evaluate(object):
    def __init__(self, data_path, opt, batch_size=config.batch_size):
        if config.use_bpe:
            self.vocab = make_bpe_vocab(config.bpe_vocab_path)
            # self.start_id = self.vocab.word_vocab[data.START_DECODING]
            # self.end_id = self.vocab.word_vocab[data.STOP_DECODING]
            # self.pad_id = self.vocab.word_vocab[data.PAD_TOKEN]
            # self.unk_id = self.vocab.word_vocab[data.UNKNOWN_TOKEN]
        else:
            self.vocab = Vocab(config.vocab_path, config.vocab_size)
            # self.start_id = self.vocab.word2id(data.START_DECODING)
            # self.end_id = self.vocab.word2id(data.STOP_DECODING)
            # self.pad_id = self.vocab.word2id(data.PAD_TOKEN)
            # self.unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        self.batcher = Batcher(data_path, self.vocab, mode='eval',
                               batch_size=batch_size, single_pass=True)
        self.opt = opt
        self.logger = init_logger('eval', config.log_root + opt.model_name + '/logfile.log')
        time.sleep(5)

    def setup_valid(self):
        self.model = Model()
        self.model = get_cuda(self.model)
        self.logger.info(f'loading model {self.opt.load_model}')
        print(f'loading model {self.opt.load_model}')
        checkpoint = T.load(self.opt.load_model)
        self.model.load_state_dict(checkpoint["model_dict"])

    def print_original_predicted(self, decoded_sents, ref_sents, article_sents, loadfile):
        filename = "test_" + loadfile.split(".")[0] + ".txt"

        with open(os.path.join("data", filename), "w") as f:
            for i in range(len(decoded_sents)):
                f.write("article: " + article_sents[i] + "\n")
                f.write("ref: " + ref_sents[i] + "\n")
                f.write("dec: " + decoded_sents[i] + "\n\n")

    def evaluate_batch(self, print_sents=False):

        self.setup_valid()
        batch = self.batcher.next_batch()
        if not config.use_bpe:
            start_id = self.vocab.word2id(data.START_DECODING)
            end_id = self.vocab.word2id(data.STOP_DECODING)
            unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        else:
            start_id = self.vocab.word_vocab[data.START_DECODING]
            end_id = self.vocab.word_vocab[data.STOP_DECODING]
            unk_id = self.vocab.word_vocab[data.UNKNOWN_TOKEN]
        decoded_sents: List[str] = []
        ref_sents = []
        article_sents = []
        rouge = Rouge()
        ii = 0
        t_start = datetime.datetime.now()
        while batch is not None:
            # print('new batch ')
            enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e = get_enc_data(batch)

            with T.autograd.no_grad():
                enc_batch = self.model.embeds(enc_batch)
                enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)

            # -----------------------Summarization----------------------------------------------------
            with T.autograd.no_grad():
                pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab,
                                       self.model, start_id, end_id, unk_id)

            for i in range(len(pred_ids)):
                if hasattr(batch, 'art_oovs'):
                    art_oovs = batch.art_oovs[i]
                else:
                    art_oovs = None
                decoded_words = data.outputids2words(pred_ids[i], self.vocab, art_oovs)
                if len(decoded_words) < 2:
                    decoded_words = "xxx"
                else:
                    decoded_words = " ".join(decoded_words)
                decoded_sents.append(decoded_words)
                abstract = batch.original_abstracts[i]
                article = batch.original_articles[i]
                ref_sents.append(abstract)
                article_sents.append(article)
            ii += 1
            if ii % 10 == 0:
                print(f'10 batches processed in {datetime.datetime.now() - t_start}, total {ii} batches, {self.batcher.batch_size * ii} samples')
                t_start = datetime.datetime.now()
            if ii % 40 == 0:
                with open(self.opt.load_model + '_decoded.pkl', 'wb') as f:
                    pickle.dump((decoded_sents, ref_sents), f, protocol=4)

            batch = self.batcher.next_batch()
        load_file = self.opt.load_model

        if print_sents:
            self.print_original_predicted(decoded_sents, ref_sents, article_sents, load_file)
        with open(self.opt.load_model + '_decoded.pkl', 'wb') as f:
            pickle.dump((decoded_sents, ref_sents), f)
        scores = rouge.get_scores(decoded_sents, ref_sents, avg=True)
        if self.opt.task == "test":
            print(load_file, "scores:", scores)
        else:
            rouge_l = scores["rouge-l"]["f"]
            print(load_file, "rouge_l:", "%.4f" % rouge_l)
            mean_rouge = sum(_['f'] for _ in scores.values()) / 3
            print(load_file, "mean_rouge:", "%.4f" % mean_rouge)
            self.logger.info("mean_rouge: %.4f" % mean_rouge)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="validate", choices=["validate", "test"])
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default=200)
    opt = parser.parse_args()

    if opt.task == "validate":
        root = config.log_root + opt.model_name + '/'
        # opt.root = root
        # TODO: здесь нужно как-то обновлять список сериализованных моделей, при этом не нарваться на SetChangedSize
        saved_models = [root + f
                        for f in os.listdir(root)
                        if (not f.startswith('events.')) and (not f.startswith('logfile'))]
        saved_models.sort()
        saved_models = saved_models[opt.start_from:]
        print(saved_models)
        for f in saved_models:
            opt.load_model = f
            eval_processor = Evaluate(config.valid_data_path, opt, batch_size=opt.batch_size)
            eval_processor.evaluate_batch()
    else:  # test
        eval_processor = Evaluate(config.test_data_path, opt)
        eval_processor.evaluate_batch()
