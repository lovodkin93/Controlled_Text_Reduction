from src.concatenate_highlights import concatenate_highlights_row


def test_concatenate_highlights_row__naive():
    row = {
        'doc_text': 'This is the first sentence. This is the second sentence.',
        'highlight_spans': [[0,4], [8,27], [40, 61]]
    }

    result = concatenate_highlights_row(row)
    expected = "This the first sentence. second sentence."  # Spans from the same sentence should stay in the same sentence after concatenation
    assert result == expected
