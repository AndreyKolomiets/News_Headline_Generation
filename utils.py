import json
from typing import List
from tqdm import tqdm
from rouge import Rouge
from bs4 import BeautifulSoup
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
