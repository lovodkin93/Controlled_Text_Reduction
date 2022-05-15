from src.run import main as seq_to_seq_main
from src.simple_concatenation_baseline import main as simple_concatenation_main
import pandas as pd
import argparse
from src.doc_reader import DocReader
from src.preprocessor import convert_highlight_rows_to_document_highlights
import os
import sys
import json

def preprocess_highlight_rows_file():
    """
    Raw files are in a format of highlight rows.
    We need to convert it to a format for training (document in each row)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_data_dir')
    parser.add_argument('--train_file_highlight_rows')
    parser.add_argument('--output_dir')
    args, unknown = parser.parse_known_args()

    highlight_rows = pd.read_csv(args.train_file_highlight_rows)
    doc_reader = DocReader(args.doc_data_dir)

    docs_and_highlighted_spans = convert_highlight_rows_to_document_highlights(doc_reader, highlight_rows)

    train_file_path = f"train.csv"
    highlights_df = pd.DataFrame(docs_and_highlighted_spans, columns=["doc_text", "summary_text", "highlight_spans"])
    # Convert column to proper json before dumping to file
    highlights_df['highlight_spans'] = highlights_df['highlight_spans'].apply(json.dumps)
    highlights_df.to_csv(train_file_path, index=False)
    
    sys.argv.extend(["--train_file", train_file_path])


preprocess_highlight_rows_file()

seq_to_seq_main()
# simple_concatenation_main()
