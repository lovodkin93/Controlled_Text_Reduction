from src.concatenate_highlights import concatenate_highlights_row
import pandas as pd

# Note that the indices are non-inclusive, meaning that [0, 4] are the characters 0,1,2,3

def test_concatenate_highlights_row__naive():
    row = {
        'doc_text': 'This is the first sentence. This is the second sentence.',
        'highlight_spans': [[0,4], [8,21], [21,26], [40, 56]]
    }

    result = concatenate_highlights_row(row)
    expected = "This the first sentence. second sentence."  # Spans from the same sentence should stay in the same sentence after concatenation
    assert result == expected

def test_concatenate_highlights_row__highlight_across_sentences():
    row = {
        'doc_text': 'This is the first sentence.\n This is the second sentence.',
        'highlight_spans': [[0,4], [8, 56]]
    }

    result = concatenate_highlights_row(row)
    expected = "This the first sentence. This is the second sentence."  # Spans from the same sentence should stay in the same sentence after concatenation
    assert result == expected

def test_concatenate_highlights_row__highlight_starts_in_new_sent_idx():
    row = {
        'doc_text': 'This is the first sentence.This is the second sentence.',
        'highlight_spans': [[27,31]]
    }

    result = concatenate_highlights_row(row)
    expected = "This."  # Spans from the same sentence should stay in the same sentence after concatenation
    assert result == expected


def test_concatenate_highlights_row__highlight_real_usecase():
    df = pd.read_csv("data/dev__highlights.csv")
    row = df.iloc[0]

    result = concatenate_highlights_row(row)
    expected = "opening of the restaurant will take place March 24. the success of the American restaurant depends on its acceptance by Yugoslavians who are long accustomed to the hamburger-like Pljeskavica. Pljeskavica is made of pork and onions is served on bread and eaten with the hands. McDonald's. John Onoda, a spokesman at McDonald's Oak Brook Ill., headquarters, said it was the first of the chain's outlets in a communist country. The next East European McDonald's is scheduled to be opened in Budapest, Hungary, by the end of this year. Negotiations have been going on for years to the Soviet Union, but no agreement has been announced."
    assert result == expected
