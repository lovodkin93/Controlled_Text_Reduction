import json
from typing import Tuple
from src.preprocessor import merge_overlapping_intervals
import nltk

def concatenate_highlights(df):
    return list(df.apply(concatenate_highlights_row, axis=1))

import spacy
nlp = spacy.load("en_core_web_sm")

def concatenate_highlights_row(row):
    """
    Creates a concatenated string of the highlights. Highlights from different sentences will be separated with a dot between them.
    """

    highlighted_spans = row['highlight_spans']
    if isinstance(highlighted_spans, str):
        highlighted_spans = json.loads(highlighted_spans)

    highlighted_spans = merge_overlapping_intervals(highlighted_spans)

    sents_ranges = _text_to_sentences_ranges(row['doc_text'])
    highlighted_spans = split_spans_across_sentences(highlighted_spans, sents_ranges)

    def combine_multiple_highlights_to_one_string(highlights_sub_texts) -> str:
        """
        We want multiple sub texts (highlights that come from the same sentence) to look as a sentence as much as possible
        """
        
        highlight_text = " ".join(highlights_sub_texts)

        if not highlight_text.endswith("."):
            highlight_text += "."
        
        return highlight_text


    highlights_texts = []
    # Collects multiple texts into one item
    highlight_sub_texts = []
    previous_sentence_id = None
    for highlight_span in highlighted_spans:
        # Decide which sentence the span is in based on where it starts
        current_sentence_id = _find_idx_in_sentence_dict(highlight_span[0], sents_ranges)
        # spans should not cross sentences (taken care of earlier)
        assert current_sentence_id == _find_idx_in_sentence_dict(highlight_span[1], sents_ranges, is_range_end=True)

        did_sentence_change: bool = previous_sentence_id is not None and current_sentence_id != previous_sentence_id

        # If sentence changed, first close the previous sub_texts
        if did_sentence_change:
            highlights_texts.append(combine_multiple_highlights_to_one_string(highlight_sub_texts))
            highlight_sub_texts = []
        
        previous_sentence_id = current_sentence_id
        highligh_sub_text = row['doc_text'][highlight_span[0]: highlight_span[1]]
        highlight_sub_texts.append(highligh_sub_text)


    if any(highlight_sub_texts):
        highlights_texts.append(combine_multiple_highlights_to_one_string(highlight_sub_texts))

    # spacy sentencizer sometimes thinks the sentence starts in \n so we want to remove these
    highlights_texts = [x[1:] if x.startswith("\n") else x for x in highlights_texts]
    # make concatenation look clean
    highlights_texts = [x.strip() for x in highlights_texts]

    return " ".join(highlights_texts)

def split_spans_across_sentences(highlighted_spans, sents_ranges):
    """
    Splits a span into two if it crosses a sentence boundary
    """

    new_highlight_spans = []
    for highlight_span in highlighted_spans:
        start_sentence_id = _find_idx_in_sentence_dict(highlight_span[0], sents_ranges)
        end_sentence_id = _find_idx_in_sentence_dict(highlight_span[1], sents_ranges, is_range_end=True)

        if start_sentence_id != end_sentence_id:
            for sentence_id in range(start_sentence_id, end_sentence_id + 1):
                sentence_range = sents_ranges[sentence_id]

                new_highlight_start = max(highlight_span[0], sentence_range[0])
                new_highlight_end = min(highlight_span[1], sentence_range[1])
                new_highlight_spans.append((new_highlight_start, new_highlight_end))
        else:
            new_highlight_spans.append(highlight_span)

    return new_highlight_spans




def _text_to_sentences_ranges(text: str) -> dict:
    """
    Given a text, returns the ranges where each sentence start and ends
    """

    # Use NLTK tokenizer directly because want the span_tokenize functionality which returns indices
    sents_ranges = []

    for sent in nlp(text).sents:
        sent_range = (sent.start_char, sent.end_char)
        sents_ranges.append(sent_range)

    offset = 0
    # Spacy doesn't count the last space as a sentence
    if text.endswith(" "):
        offset = 1

    # Make sure the ending range is the length of the text
    assert sents_ranges[-1][1] + offset == len(text)

    return sents_ranges

def _find_idx_in_sentence_dict(idx: int, sents_ranges: dict, is_range_end=False) -> int:
    """
    Find to which sentence does the idx belong
    """

    # Use minus 1 if this is the end of a range, because the span range ending includes the idx of the character from the next sentence 
    if is_range_end:
        idx = idx - 1

    # Assumes sents_ranges is sorted
    for sent_id, sentence_range in enumerate(sents_ranges):
        # Minus 1 because the end range is always including the index of the next character
        if idx <= sentence_range[1] - 1:
            return sent_id
    
    raise ValueError("didn't find any sentence for this range")
