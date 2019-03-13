import json
from typing import List
from tqdm import tqdm
from rouge import Rouge
from bs4 import BeautifulSoup


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
    scorer = Rouge(apply_avg=True)
    scores = scorer.get_scores(predicted_titles, true_titles)
    return scores


def parse_source(text: str) -> str:
    """
    Метод извлекает текст из html статьи.

    :param text: html source
    :return: Текст без html тегов
    """
    return BeautifulSoup(text, 'lxml').text