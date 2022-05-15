"""
As a simple baseline, we want to simply concatenate all highlights
"""

import argparse
import logging
import json
from datasets import load_metric

import pandas as pd
from src.compute_metrics import compute_rouge_metrics

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

    return " ".join(highlights_texts)


def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_data_dir')
    parser.add_argument('--test_file')
    parser.add_argument('--output_dir')
    args, unknown = parser.parse_known_args()

    df = pd.read_csv(args.test_file)

    outputs = df.apply(concatenate_highlights, axis=1)
    metric = load_metric("rouge")
    summaries = list(df['summary_text'])
    result = compute_rouge_metrics(outputs, summaries, metric)
    print(result)

