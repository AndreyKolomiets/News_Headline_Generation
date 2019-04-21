import os
import struct
import collections
import numpy
from tensorflow.core.example import example_pb2
from multiprocessing import Pool
import argparse
from utils import load_data, parse_source


# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
VOCAB_SIZE = 200000
CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_data', type=str, default='/home/jovyan/work/_Headline_generation/ria.json')
parser.add_argument('--n_cores', type=int, default=10)
args = parser.parse_args()

path_to_data = args.path_to_data
path_to_tokenized = '/'.join(path_to_data.split('/')[:-1]) + '/tokenized'

texts, titles = load_data(path_to_data)
pool = Pool(10)
parsed = pool.map(parse_source, texts)


def read_text_file(text_file: str):
    lines = []
    with open(text_file, "r", encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def write_to_bin(article_dir: str, title_dir: str, out_file: str, finished_files_dir: str, makevocab=False):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file
    and writes them to a out_file."""

    story_fnames = [s.replace('.txt', '') + ".story" for s in os.listdir(article_dir)]
    num_stories = len(story_fnames)

    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        for idx, s in enumerate(story_fnames):
            if idx % 1000 == 0:
                print("Writing story %i of %i; %.2f percent done" % (
                idx, num_stories, float(idx) * 100.0 / float(num_stories)))

            path_to_article = f'{article_dir}/{idx}.txt'
            path_to_title = f'{title_dir}/{idx}.txt'
            # Get the strings to write to .bin file
            article = ' '.join(read_text_file(path_to_article)).lower()
            title_lines = read_text_file(path_to_title)
            title = "%s %s %s" % (SENTENCE_START, ' '.join(title_lines), SENTENCE_END)

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([article])
            tf_example.features.feature['abstract'].bytes_list.value.extend([title])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = article.split(' ')
                abs_tokens = title.split(' ')
                abs_tokens = [t for t in abs_tokens if
                              t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


if __name__ == '__main__':
    article_dir = '/home/jovyan/work/_Headline_generation/articles_tokenized/'
    title_dir = '/home/jovyan/work/_Headline_generation/titles_tokenized/'
    out_file = '/home/jovyan/work/_Headline_generation/pointer_summarizer/dataset.bin'
    finished_files_dir = '/home/jovyan/work/_Headline_generation/pointer_summarizer/'
    write_to_bin(article_dir, title_dir, out_file, finished_files_dir, makevocab=True)
