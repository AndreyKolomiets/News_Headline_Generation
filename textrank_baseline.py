from gensim.summarization.summarizer import summarize
import numpy as np
import re
import utils
regex_word = re.compile('\w+')


def apply_textrank(text: str, headline_average_length=9):
    """

    :param text:
    :param headline_average_length:
    :return:
    """
    summarized = summarize(text, word_count=headline_average_length + np.random.randint(-2, 3))
    return summarized



