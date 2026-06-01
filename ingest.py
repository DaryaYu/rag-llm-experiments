import json
from minsearch import Index


def load_data():
    DATA_PATH = 'data/data_science_qa_dataset.json'

    with open(DATA_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data


def build_index(data):
    index = Index(
        text_fields=['question', 'answer'],
        keyword_fields=['category']
    )
    index.fit(data)
    return index