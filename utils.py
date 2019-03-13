import json
from typing import List
from rouge import Rouge


def load_data(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def get_metrics(true_titles: List[str], predicted_titles: List[str]) -> dict:
    scorer = Rouge(apply_avg=True)
    scores = scorer.get_scores(predicted_titles, true_titles)
    return scores
