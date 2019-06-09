from flask import Flask, request, jsonify
import numpy as np
from stanfordcorenlp.corenlp import StanfordCoreNLP
from nltk.tokenize import wordpunct_tokenize
import re
import json
from termcolor import colored
import argparse
import torch
from rl_summ2.model import Model
from rl_summ2.data_util.data import Vocab, make_bpe_vocab
from rl_summ2.train_util import get_cuda, get_enc_data
from rl_summ2.beam_search import beam_search
from rl_summ2.data_util import config, data
if not config.use_bpe:
    from rl_summ2.data_util.batcher import Batch, Example
else:
    from pointer_summarizer.data_util.batcher_bpe import Batch, Example

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
args = parser.parse_args()



app = Flask(__name__)
replacement_dict = {chr(8212): '--',
                    '(': '-lrb-',
                    ')': '-rrb-'}

def preprocess_text(text: str):
    tokens = wordpunct_tokenize(text.lower())
    is_closing = False
    for i in range(len(tokens)):
        if tokens[i] == '"':
            if is_closing:
                tokens[i] = "''"

            else:
                tokens[i] = '``'
                is_closing = True
        if tokens[i] == '—':
            tokens[i] = '--'
    return ' '.join(tokens)

@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')


@app.before_first_request
def load_model():
    app.model = Model()
    app.model = get_cuda(app.model)
    print(f'loading model {args.model_path}')
    checkpoint = torch.load(args.model_path)
    app.model.load_state_dict(checkpoint["model_dict"])
    if config.use_bpe:
        app.vocab = make_bpe_vocab(config.bpe_vocab_path)
    else:
        app.vocab = Vocab(config.vocab_path, config.vocab_size)


@app.route('/api/',  methods=['POST'])
def get_headlines():
    query = request.get_json()#.decode(encoding='utf-8')
    # print(query)
    text = preprocess_text(query['text'])

    # print(text)
    if config.use_bpe:
        bpe_enc = next(app.vocab.transform([text]))[:config.max_enc_steps]
        used_text = next(app.vocab.inverse_transform([bpe_enc]))
        print(colored(f'Использованный участок текста: {used_text}', color='green'))
    text = bytes(text, encoding='utf-8')

    if not config.use_bpe:
        start_id = app.vocab.word2id(data.START_DECODING)
        end_id = app.vocab.word2id(data.STOP_DECODING)
        unk_id = app.vocab.word2id(data.UNKNOWN_TOKEN)
    else:
        start_id = app.vocab.word_vocab[data.START_DECODING]
        end_id = app.vocab.word_vocab[data.STOP_DECODING]
        unk_id = app.vocab.word_vocab[data.UNKNOWN_TOKEN]
    ex = Example(text, ['РИА Новости'], app.vocab)
    batch = Batch([ex], app.vocab, batch_size=1)
    enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e = get_enc_data(batch)

    with torch.autograd.no_grad():
        enc_batch = app.model.embeds(enc_batch)
        enc_out, enc_hidden = app.model.encoder(enc_batch, enc_lens)

    with torch.autograd.no_grad():
        pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab,
                               app.model, start_id, end_id, unk_id)

    for i in range(len(pred_ids)):
        if hasattr(batch, 'art_oovs'):
            art_oovs = batch.art_oovs[i]
        else:
            art_oovs = None
        decoded_words = data.outputids2words(pred_ids[i], app.vocab, art_oovs)
        if len(decoded_words) < 2:
            decoded_words = "Не получилось декодировать"
        else:
            decoded_words = " ".join(decoded_words)
    print(colored(f'Декодированное: {decoded_words}', color='red'))
    return jsonify({'res': decoded_words})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)