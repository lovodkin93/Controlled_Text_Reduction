import numpy as np

from src.utils import filter_function_words

def compute_rouge_metrics(predictions: list, references: list, metric, prefix: str, should_filter_function_words: bool = False) -> dict:
    assert len(predictions) == len(references)

    filtered_predictions = predictions
    filtered_references = references
    if should_filter_function_words:
        filtered_predictions = []
        for prediction in predictions:
            filtered_predictions.append(filter_function_words(prediction))

        filtered_references = []
        for reference in references:
            filtered_references.append(filter_function_words(reference))
    result = metric.compute(predictions=filtered_predictions,
                            references=filtered_references, use_stemmer=True)
    # Extract a few results from ROUGE
    result_parsed = {f"{prefix}_{key}": value*100 for key, value in result.items()}

    # Add also precision and recall
    # result_parsed.update({f"{prefix}_{key}_precision": value.mid.precision * 100 for key, value in result.items()})
    # result_parsed.update({f"{prefix}_{key}_recall": value.mid.recall * 100 for key, value in result.items()})

    result_parsed = {k: round(v, 4) for k, v in result_parsed.items()}

    return result_parsed

def compute_summac_metrics(inputs, predictions, model) -> dict:
    result = model.score(inputs, predictions)
    return {
        "summac": np.mean(result['scores'])
    }