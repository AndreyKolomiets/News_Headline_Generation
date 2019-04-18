import json
from typing import List, Iterable, Set
import string
from tqdm import tqdm
from rouge import Rouge
from bs4 import BeautifulSoup
from collections import Counter
import re


def load_data(path: str):
    texts = []
    titles = []
    with open(path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            texts.append(data['text'])
            titles.append(data['title'])
    assert len(texts) == len(titles), f'Количество текстов ({len(texts)}) не равно количеству заголовков {len(titles)}'
    return texts, titles


def get_metrics(true_titles: List[str], predicted_titles: List[str]) -> dict:
    scorer = Rouge()
    scores = scorer.get_scores(predicted_titles, true_titles, avg=True)
    return scores


def parse_source(text: str) -> str:
    """
    Метод извлекает текст из html статьи.

    :param text: html source
    :return: Текст без html тегов
    """
    return BeautifulSoup(text, 'lxml').text


class FirstSentenceTokenizer:
    regex_split = re.compile('([а-яёa-z\"»)]{2,}(\.\s|\n))')
    regex_date = re.compile(', \d+\s(янв|фев|мар|апр|мая|июн|июл|авг|сен|окт|ноя|дек)')
    regex_date_and_agency = re.compile(
        ', \d+\s(янв(аря)?|фев(раля)?|мар(та)?|апр(еля)?|мая|июня?|июля?|авг(уста)?|сен(тября)?|окт(ября)?|ноя(бря)?|дек(абря)?)\s[\-—–]\s(риа новости|риа \"новости\"|р\-спорт|риа\.туризм|рапси|прайм)')
    regex_author_and_agency = re.compile('\w+\s\w+,\s(обозреватель )?риа новости')

    def __init__(self):
        self.span = None

    # TODO: выпилить лишние символы в начале и в конце возвращаемого
    def tokenize(self, text: str) -> str:
        """
        Метод, извлекающий из новости первое содержательное предложение
        :param text:
        :return:
        """
        x = self.regex_split.search(text)

        if x is None:
            # Вариант для текста из одного предложения
            self.span = (0, len(text))
            return text
        else:
            start = x.span()[1]
            candidate = text[:start]
            if self.regex_date_and_agency.search(candidate) or self.regex_author_and_agency.search(candidate):
                y = self.regex_split.search(text[start:])
                if y is not None:
                    self.span = (start, start+y.span()[1]-2)
                    return text[start:start+y.span()[1]-2]
                else:
                    # Если после вступления есть только одно предложение
                    self.span = (start, len(text))
                    return text[start:]
            else:
                # Если все же первое предложение содержательное
                self.span = (0, start-2)
                return text[:start-2]

    def is_non_informative(self, sentence: str) -> bool:
        """
        Проверяем, является ли предложение неинформативным.
        Алгоритм тот же, что и в методе tokenize, но предполагаем, что уже сделано правильное разбиение
        :param sentence:
        :return:
        """
        if self.regex_date_and_agency.search(sentence) or self.regex_author_and_agency.search(sentence):
            return True
        return False


regex_word = re.compile('[а-яёa-z\-]+')


def get_vocab(texts: Iterable[str]) -> Counter:

    cnt = Counter()
    for text in tqdm(texts):
        for word in regex_word.findall(text):
            cnt[word] += 1
    return cnt


def filter_vocab(vocab: Counter, n=None, min_count=None) -> Set[str]:
    if (n is not None) and (min_count is not None):
        raise ValueError
    if n is not None:
        return {_[0] for _ in vocab.most_common(n)}
    if min_count is not None:
        words = set()
        for word, cnt in vocab.most_common(len(vocab)):
            if cnt < min_count:
                break
            words.add(word)
        return words


def filter_texts(texts: List[str], vocab: Set[str]) -> List[str]:
    return [' '.join([w for w in regex_word.findall(text) if w in vocab]) for text in tqdm(texts)]


regex_word_and_punctuation = re.compile('[а-яёa-z\-]+|[{}]'.format(string.punctuation))
punctuation = set(string.punctuation)


def filter_texts_with_punctuation(texts: List[str], vocab: Set[str]):
    res = []
    for text in tqdm(texts):
        filtered = [tok for tok in regex_word_and_punctuation.findall(text) if (tok in vocab) or (tok in punctuation)]
        res.append(' '.join(filtered))
    return res


regex_punctuation_replace = re.compile('\s([!%,.:;?])')


def filter_texts_with_punctuation_spaces(texts: List[str], vocab: Set[str]):
    res = []
    for text in tqdm(texts):
        filtered = [tok for tok in regex_word_and_punctuation.findall(text) if (tok in vocab) or (tok in punctuation)]
        res.append(regex_punctuation_replace.sub(r'\1', ' '.join(filtered)))
    return res
