def compute_rouge_metrics(predictions: list, references: list, metric) -> dict:
    assert len(predictions) == len(references)

    result = metric.compute(predictions=predictions,
                            references=references, use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure *
                100 for key, value in result.items()}

    result = {k: round(v, 4) for k, v in result.items()}

    return result