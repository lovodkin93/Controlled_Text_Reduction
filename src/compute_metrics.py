def compute_rouge_metrics(predictions: list, references: list, metric) -> dict:
    assert len(predictions) == len(references)

    result = metric.compute(predictions=predictions,
                            references=references, use_stemmer=True)
    # Extract a few results from ROUGE
    result_parsed = {key: value.mid.fmeasure *
                100 for key, value in result.items()}

    # Add also precision and recall
    result_parsed.update({f"{key}_precision": value.mid.precision * 100 for key, value in result.items()})
    result_parsed.update({f"{key}_recall": value.mid.precision * 100 for key, value in result.items()})

    result_parsed = {k: round(v, 4) for k, v in result_parsed.items()}

    return result_parsed