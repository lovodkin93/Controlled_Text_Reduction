from collections import defaultdict
from dataclasses import dataclass
import json
from typing import List, Tuple


def concatenate_highlights(df):
    return list(df.apply(concatenate_highlights_row, axis=1))

import spacy
nlp = spacy.load("en_core_web_sm")

@dataclass
class TextPart:
    text: str
    range: Tuple[int, int]
    sent_idx: int    
    is_highlight: bool
    prefix: str = None  # To enable adding highlight tokens
    postfix: str = None  # To enable adding highlight tokens

    def text_part_to_text(self) -> str:
        if self.prefix is not None and self.postfix is not None:
            return f"{self.prefix}{self.text}{self.postfix}"
        return self.text

def concatenate_highlights_row(row, keep_full_sentences: bool = False, return_str: bool = True) -> List[TextPart]:
    """
    Creates a concatenated string of the highlights. Highlights from different sentences will be separated with a dot between them.
    param keep_full_sentences controls whether the whole sentence will be kept or only the highlights.
    """

    sents_ranges, sents = _text_to_sentences_ranges(row['doc_text'])
    highlighted_spans = fix_highlights(row['highlight_spans'], sents_ranges)
    highlights_per_sent = collect_highlights_per_sent(highlighted_spans, sents_ranges)
    all_text_parts = stitch_spans_with_sent_text(highlights_per_sent, sents, sents_ranges, keep_full_sentences)

    if return_str:
        all_text_parts = combine_text_parts_to_str(all_text_parts, keep_full_sentences)

    return all_text_parts


def fix_highlights(highlighted_spans: list, sents_ranges) -> list:
    """
    Merges subsequent highlights, splits spans that go beyond sentences
    """

    if isinstance(highlighted_spans, str):
        highlighted_spans = json.loads(highlighted_spans)

    highlighted_spans = merge_overlapping_intervals(highlighted_spans)
    highlighted_spans = split_spans_across_sentences(highlighted_spans, sents_ranges)

    return highlighted_spans

def collect_highlights_per_sent(highlighted_spans, sents_ranges) -> dict:
    highlights_per_sent = defaultdict(list)
    for highlight_span in highlighted_spans:
        # Decide which sentence the span is in based on where it starts
        curr_sentence_idx = _find_idx_in_sentence_dict(highlight_span[0], sents_ranges)

        # Assert spans are not crossing sentences (taken care of earlier)
        assert curr_sentence_idx == _find_idx_in_sentence_dict(highlight_span[1], sents_ranges, is_range_end=True)

        highlights_per_sent[curr_sentence_idx].append(highlight_span)

    return highlights_per_sent

def stitch_spans_with_sent_text(highlights_per_sent, sents, sents_ranges, keep_full_sentences):
    all_text_parts: List[TextPart] = []
    for sent_idx, sent_range in enumerate(sents_ranges):
        curr_highlights_spans = highlights_per_sent[sent_idx]

        # For concatenation we don't care about sentences without highlights
        if not any(curr_highlights_spans):
            continue

        curr_sentence_text = sents[sent_idx]

        # Keep track of last text part from this sentence
        last_text_part_char_idx = 0

        # Break sentence text into text parts
        for curr_highlight_span in curr_highlights_spans:            
            # Find the highlight text
            highlight_sub_text = curr_sentence_text[curr_highlight_span[0] - sent_range[0]: curr_highlight_span[1] - sent_range[0]]
            highlight_text_part = TextPart(highlight_sub_text, curr_highlight_span, sent_idx, is_highlight=True)

            # Skip subtexts that were only a space or \n (caused if the user ends the highlight in \n), otherwise there will be empty highlight tags / floating dots
            if highlight_sub_text == "":
                continue

            if keep_full_sentences:
                text_up_to_highlight = curr_sentence_text[last_text_part_char_idx:curr_highlight_span[0] - sent_range[0]]
                all_text_parts.append(TextPart(text_up_to_highlight, (last_text_part_char_idx, curr_highlight_span[0] - sent_range[0]), sent_idx, is_highlight=False))

            all_text_parts.append(highlight_text_part)

            last_text_part_char_idx = curr_highlight_span[1] - sent_range[0]

        if keep_full_sentences:
            text_remaining_in_sent = curr_sentence_text[last_text_part_char_idx:len(curr_sentence_text)]
            if text_remaining_in_sent != "":
                all_text_parts.append(TextPart(text_remaining_in_sent, (last_text_part_char_idx, len(curr_sentence_text)), sent_idx, is_highlight=False))

    return all_text_parts


def combine_text_parts_to_str(text_parts, keep_full_sentences) -> str:
    """
    We want multiple sub texts (highlights that come from the same sentence) to look as a sentence as much as possible
    """
    
    sent_idx = None
    sent_text_parts = []

    def sent_text_parts_to_str(sent_text_parts: List[TextPart]) -> str:

        # if we keep full sentences, we don't need to add artificial spaces
        if keep_full_sentences:
            joiner = ""
            should_strip = False
        # if we cocnatenate highlights, we need artificial spaces
        else:
            joiner = " "
            should_strip = True

        texts = [text_part.text_part_to_text() for text_part in sent_text_parts]
        if should_strip:
            texts = [text.strip() for text in texts]
        sent_text = joiner.join(texts).strip()
        
        if not sent_text_parts[-1].text.endswith("."):
            sent_text += "."

        return sent_text

    final_texts = []
    for text_part in text_parts:
        did_sent_change = sent_idx is not None and text_part.sent_idx != sent_idx

        if did_sent_change:
            final_texts.append(sent_text_parts_to_str(sent_text_parts))
            sent_text_parts = []

        sent_text_parts.append(text_part)
        sent_idx = text_part.sent_idx

    if any(sent_text_parts):
        final_texts.append(sent_text_parts_to_str(sent_text_parts))

    return " ".join(final_texts)


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

                # Can happen if the user highlighted a \n
                if new_highlight_start != new_highlight_end:
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
    sents = []

    for spacy_sent in nlp(text).sents:
        starting_start_char = spacy_sent.start_char
        starting_end_char = spacy_sent.end_char

        # Help spacy in \n cases
        split_by_newline = spacy_sent.text.split("\n")
        for sent in split_by_newline:
            new_sent_start_char = starting_start_char
            new_sent_end_char = starting_start_char + len(sent)

            sent_range = (new_sent_start_char, new_sent_end_char)
            sents_ranges.append(sent_range)
            sents.append(sent)

            starting_start_char += len(sent) + 1  # Plus 1 to compensate on the deleted \n

        # Validation
        assert sent_range[-1] == starting_end_char

    offset = 0
    # Spacy doesn't count the last space as a sentence
    if text.endswith(" "):
        offset = 1

    # Make sure the ending range is the length of the text
    assert sents_ranges[-1][1] + offset == len(text)

    return sents_ranges, sents

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
