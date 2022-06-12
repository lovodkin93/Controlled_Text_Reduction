"""
As a simple baseline, we want to simply concatenate all highlights
"""

import sys
import logging
import json
from datasets import load_metric

import pandas as pd
from src.compute_metrics import compute_rouge_metrics, compute_summac_metrics
from src.predictions_analyzer import PredictionsAnalyzer

from src.preprocessor import merge_overlapping_intervals



def concatenate_highlights(row):
    highlighted_spans = row['highlight_spans']
    if isinstance(highlighted_spans, str):
        highlighted_spans = json.loads(highlighted_spans)

    highlight_spans = merge_overlapping_intervals(highlighted_spans)

    highlights_texts = []
    for highlight_span in highlight_spans:
        highlight_text = row['doc_text'][highlight_span[0]: highlight_span[1]]
        highlights_texts.append(highlight_text)

    return ". ".join(highlights_texts)


def main(config: dict, summaries_to_test_key: str):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    df = pd.read_csv(config['test_file'])

    inputs = list(df['doc_text'])

    summaries = list(df['summary_text'])

    if summaries_to_test_key == "gold_summaries":
        summaries_to_test = summaries
    elif summaries_to_test_key == "simple_concatenation":
        summaries_to_test = list(df.apply(concatenate_highlights, axis=1))
    else:
        raise ValueError(f"Unexpeceted key for summaries_to_text ; {summaries_to_test_key}")

    # Calc rouge
    metric = load_metric("rouge")
    result = compute_rouge_metrics(summaries_to_test, summaries, metric)

    # Calc Summac
    sys.path.append('summac')  # Will fail if you didn't load the submodule (https://git-scm.com/book/en/v2/Git-Tools-Submodules)
    from summac.model_summac import SummaCZS
    model = SummaCZS(granularity="sentence", model_name="vitc")
    result.update(compute_summac_metrics(inputs, summaries_to_test, model))

    # Extract predictions file
    PredictionsAnalyzer(None, config['output_dir']).write_predictions_to_file(summaries_to_test, inputs, is_tokenized=False)

    print(result)

