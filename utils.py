import json


def load_data(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    return data