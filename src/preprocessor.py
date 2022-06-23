from collections import defaultdict
from typing import List, Tuple
import pandas as pd
import json

from src.concatenate_highlights import combine_text_parts_to_str, concatenate_highlights_row, merge_overlapping_intervals


class Preprocessor:
    """
    Preprocess inputs and outputs
    """

    def __init__(self, prefix, special_tokens_constants, should_add_highlights: bool = True, only_sents_with_highlights: bool = False, keep_only_highlights: bool = False):
        self.prefix = prefix
        self.special_tokens_constants = special_tokens_constants
        self.should_add_highlights = should_add_highlights
        self.only_sents_with_highlights = only_sents_with_highlights
        self.keep_only_highlights = keep_only_highlights

    def preprocess_input(self, source_text, highlighted_spans) -> str:
        """
        Converts input to str
        """

        # Collect all indices of tokens that need to be added
        idx_to_tokens = defaultdict(list)

        if self.keep_only_highlights:
            final_text = concatenate_highlights_row({
                "doc_text": source_text,
                "highlight_spans": highlighted_spans
            }, keep_full_sentences=False, return_str=True)
        elif self.only_sents_with_highlights:
            text_parts = concatenate_highlights_row({
                "doc_text": source_text,
                "highlight_spans": highlighted_spans
            }, keep_full_sentences=True, return_str=False)

            for text_part in text_parts:
                if text_part.is_highlight:
                    text_part.prefix = self.special_tokens_constants['highlight_start']
                    text_part.postfix = self.special_tokens_constants['highlight_end']
            final_text = combine_text_parts_to_str(text_parts, keep_full_sentences=True)
        else:
            if not self.should_add_highlights:
                highlighted_spans = []
            else:
                if isinstance(highlighted_spans, str):
                    highlighted_spans = json.loads(highlighted_spans)

                # We don't care about nested highlights / consecutive highlights
                highlighted_spans = merge_overlapping_intervals(highlighted_spans)

                for start, end in highlighted_spans:
                    idx_to_tokens[start].append(self.special_tokens_constants['highlight_start'])
                    idx_to_tokens[end].append(self.special_tokens_constants['highlight_end'])

            # Build concatenated text by running over the text in parts
            source_text_with_highlighted_spans = ""
            last_idx = 0
            for idx in sorted(idx_to_tokens.keys()):
                # Take text up to the current point
                source_text_with_highlighted_spans += source_text[last_idx:idx]

                # Add the necessary tokens
                tokens = idx_to_tokens[idx]
                for token in tokens:
                    source_text_with_highlighted_spans += token
                last_idx = idx

            source_text_with_highlighted_spans += source_text[last_idx:]

            final_text = source_text_with_highlighted_spans
            
        # Return text with prefix
        return f"{self.prefix} {final_text}"


    def preprocess_output(self, summary_text) -> str:
        """
        Converts output to str
        """

        return summary_text

def get_special_tokens_constants(is_t5_model: bool) -> dict:
    """
    Constants used for preprocessing input and output
    """

    special_tokens_constants = {}
    if is_t5_model:
        # T5 model has 100 special tokens by default
        special_tokens_constants['highlight_start'] = "<extra_id_1>"
        special_tokens_constants['highlight_end'] = "<extra_id_2>"
    else:
        special_tokens_constants['highlight_start'] = "<highlight_start>"
        special_tokens_constants['highlight_end'] = "<highlight_end>"
    return special_tokens_constants


def convert_row_spans_str_to_list_of_highlights(spans_str) -> List[Tuple[int, int]]:
    """
    A single row's spans string can have spaces and be non-continuous. Example: "5361, 5374;5380, 5446"
    """

    highlights = []
    start_end_strs = spans_str.split(";")
    for start_end_str in start_end_strs:
        split = start_end_str.split(",")
        start = int(split[0].strip())
        end = int(split[1].strip())
        highlights.append((start, end))

    return highlights

def convert_highlight_rows_to_document_highlights(doc_reader, highlight_rows: pd.DataFrame) -> List[List[Tuple[str, str, list]]]:
    """
    Convert from multiple highlight rows (csv) to document highlights
    """

    def handle_document_rows(doc_rows):
        any_row = doc_rows.iloc[0]
        doc = doc_reader.read_doc(any_row['topic'], any_row['documentFile'])

        # Each topic is a summary
        summary = doc_reader.read_summary(any_row['summaryFile'])
        highlight_spans = doc_rows['docSpanOffsets'].apply(convert_row_spans_str_to_list_of_highlights)
        flattened_highlight_spans = [span for spans in highlight_spans.to_list() for span in spans]

        return [{
            # "doc_id": any_row['documentFile'],
            # "example_id": any_row['summaryFile'],
            "doc_text": doc,
            "summary_text": summary,
            "highlight_spans": flattened_highlight_spans
        }]


    document_highlights_df = highlight_rows.groupby('summaryFile').apply(handle_document_rows)
    # Flatten list of lists to a list
    return [document_highlight for document_highlights in document_highlights_df.to_list() for document_highlight in document_highlights]


