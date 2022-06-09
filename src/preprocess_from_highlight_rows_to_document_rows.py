import os
import sys
import json
import argparse
import pandas as pd
from src.doc_reader import DocReader
from src.preprocessor import convert_highlight_rows_to_document_highlights
from typing import List


def preprocess_from_highlight_rows_to_document_rows(doc_data_dir: str, train_file_highlight_rows: str, output_file_path: str) -> None:
    """
    Raw files are in a format of highlight rows.
    We need to convert it to a format for training / evaluation: document in each row.
    """


    highlight_rows = pd.read_csv(train_file_highlight_rows)
    doc_reader = DocReader(doc_data_dir)

    docs_and_highlighted_spans = convert_highlight_rows_to_document_highlights(doc_reader, highlight_rows)

    highlights_df = pd.DataFrame(docs_and_highlighted_spans, columns=["doc_text", "summary_text", "highlight_spans"])
    # Convert column to proper json before dumping to file
    highlights_df['highlight_spans'] = highlights_df['highlight_spans'].apply(json.dumps)
    highlights_df.to_csv(output_file_path, index=False)


def concat_files(data_files_dir: str, files_to_concat: List[str], output_file_path: str):
    df = pd.DataFrame()
    for file in files_to_concat:
        df_to_concat = pd.read_csv(f"{data_files_dir}/{file}")
        print(f"File {file} has shape: {df_to_concat.shape}")
        df = pd.concat([df, df_to_concat])
    
    df.to_csv(output_file_path, index=False)
    print(f"Concatenated file {output_file_path} has shape: {df.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file_path')        

    # Preprocess
    parser.add_argument('--do_preprocess')
    parser.add_argument('--doc_data_dir')
    parser.add_argument('--train_file_highlight_rows')
    
    # Concat
    parser.add_argument('--do_concat')
    parser.add_argument('--data_files_dir')
    parser.add_argument('--files_to_concat')

    args, unknown = parser.parse_known_args()

    if args.do_preprocess:
        preprocess_from_highlight_rows_to_document_rows(args.doc_data_dir, args.train_file_highlight_rows, args.output_file_path)
    elif args.do_concat:
        concat_files(args.data_files_dir, args.files_to_concat.split(","), args.output_file_path)
        


if __name__ == "__main__":
    main()
