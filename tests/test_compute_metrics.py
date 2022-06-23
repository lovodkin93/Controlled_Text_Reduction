from datasets import load_metric
from src.compute_metrics import compute_rouge_metrics


def test_compute_rouge_metrics__with_function_words():
    predictions = ["word"]
    references = ["word"]

    metric = load_metric("rouge")
    result = compute_rouge_metrics(predictions, references, metric, should_filter_function_words=False, prefix="prefix")
    assert result[f'prefix_rouge1'] == 100.0

def test_compute_rouge_metrics__without_function_words():
    predictions = ["word"]
    references = ["the word!"]

    metric = load_metric("rouge")
    result = compute_rouge_metrics(predictions, references, metric, should_filter_function_words=True, prefix="prefix")
    assert result['prefix_rouge1'] == 100.0
