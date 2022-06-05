import os
import argparse


class DocReader:
    """
    Reads the docs by id
    """

    def __init__(self, doc_data_dir: str):
        self.doc_data_dir = doc_data_dir

    def read_doc(self, topic: str, document_file: str) -> str:
        with open(f"{self.doc_data_dir}/{topic}/{document_file}") as f:
            return f.read()

    def read_summary(self, summary_file: str) -> str:
        with open(f"{self.doc_data_dir}/summaries/{summary_file}") as f:
            return f.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_data_dir')
    args, unknown = parser.parse_known_args()
    doc_reader = DocReader(args.doc_data_dir)
    topic = "WSJ910405-0154_d06aa"
    document_file = "WSJ910405-0154"
    doc = doc_reader.read_doc(topic, document_file)
    print(doc)
    summary = doc_reader.read_summary(topic, document_file)
    print(summary)    
