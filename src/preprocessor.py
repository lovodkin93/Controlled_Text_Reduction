from collections import defaultdict
from typing import List, Tuple
import pandas as pd
import json


class Preprocessor:
    """
    Preprocess inputs and outputs
    """

    def __init__(self, prefix, special_tokens_constants, should_add_highlights: bool = True):
        self.prefix = prefix
        self.special_tokens_constants = special_tokens_constants
        self.should_add_highlights = should_add_highlights


    def preprocess_input(self, source_text, highlighted_spans) -> str:
        """
        Converts input to str
        """

        # Collect all indices of tokens that need to be added
        idx_to_tokens = defaultdict(list)

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
        
        # Return text with prefix
        return f"{self.prefix}{source_text_with_highlighted_spans}"


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

        return [(doc, summary, flattened_highlight_spans)]


    document_highlights_df = highlight_rows.groupby('summaryFile').apply(handle_document_rows)
    # Flatten list of lists to a list
    return [document_highlight for document_highlights in document_highlights_df.to_list() for document_highlight in document_highlights]


def merge_overlapping_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merges overlapping / consecutive intervals.
    See more details here https://leetcode.com/problems/merge-intervals
    """

    intervals = sorted(intervals, key=lambda x: x[0])

    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous and it's not its consecutive, simply append it.
        if not merged or merged[-1][1] + 1 < interval[0]:
            interval_copy = list(interval) if isinstance(interval, tuple) else interval.copy()
            merged.append(interval_copy)
        # otherwise, there is overlap, so we merge the current and previous
        # intervals.
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged
