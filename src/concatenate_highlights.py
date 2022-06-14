import json
from typing import Tuple
from src.preprocessor import merge_overlapping_intervals
import nltk

def concatenate_highlights(df):
    return list(df.apply(concatenate_highlights_row, axis=1))

def concatenate_highlights_row(row):
    """
    Creates a concatenated string of the highlights. Highlights from different sentences will be separated with a dot between them.
    """

    highlighted_spans = row['highlight_spans']
    if isinstance(highlighted_spans, str):
        highlighted_spans = json.loads(highlighted_spans)

    highlight_spans = merge_overlapping_intervals(highlighted_spans)
    sentences_dict = _text_to_sentences_ranges(row['doc_text'])

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
    previous_sentence_id = 1
    for highlight_span in highlight_spans:
        current_sentence_id = _find_range_in_sentence_dict(highlight_span, sentences_dict)
        did_sentence_change: bool = current_sentence_id != previous_sentence_id

        # If sentence changed, first close the previous sub_texts
        if did_sentence_change:
            highlights_texts.append(combine_multiple_highlights_to_one_string(highlight_sub_texts))
            highlight_sub_texts = []
            previous_sentence_id = current_sentence_id

        highligh_sub_text = row['doc_text'][highlight_span[0]: highlight_span[1]]
        highlight_sub_texts.append(highligh_sub_text)


    if any(highlight_sub_texts):
        highlights_texts.append(combine_multiple_highlights_to_one_string(highlight_sub_texts))

    return " ".join(highlights_texts)

def _text_to_sentences_ranges(text: str) -> dict:
    """
    Given a text, returns the ranges where each sentence start and ends
    """

    sentences = nltk.tokenize.sent_tokenize(text)
    sentences_dict = {}
    range_count = 0
    for sent_idx, sent in enumerate(sentences):
        sentences_dict[sent_idx + 1] = (range_count, range_count + len(sent))
        range_count += len(sent)

    return sentences_dict

def _find_range_in_sentence_dict(range: Tuple[int, int], sentences_dict: dict) -> int:
    for sent_id, sentence_range in sentences_dict.items():
        if range[0] <= sentence_range[1]:
            return sent_id
    
    raise ValueError("didn't find any sentence for this range")
