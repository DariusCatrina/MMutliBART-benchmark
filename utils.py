from datasets import load_metric

import re
import string

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    #TODO: remove remove_articles
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))


def compute_rouge(predictions, references, rouge_types=None, use_stemmer=False):
    rouge_list = ['rouge1', 'rouge2', 'rougeL']
    scores = {}

    predictions = [normalize_text(prediction).split() for prediction in predictions]
    references = [normalize_text(reference).split() for reference in references]

    metric = load_metric("rouge")
    rouge = metric.compute(predictions=predictions, references=references, use_agregator=False)

    for rouge_type in rouge_list:
        score_list = [score.fmeasure for score in rouge[rouge_type]]
        scores[rouge_type] = (sum(score_list) * 100.0) / len(score_list)
    
    return scores
       