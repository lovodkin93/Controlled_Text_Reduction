from controlled_reduction.src.preprocessor import Preprocessor
from mock import MagicMock
import pandas as pd

PREFIX = "prefix:"
SPECIAL_TOKENS_CONSTANTS = {"highlight_start": "<h>", "highlight_end": "</h>"}


class TestPreprocessor:
    def test_preprocess_input__sanity(self):
        SPECIAL_TOKENS_CONSTANTS = {
            "highlight_start": "<h>", "highlight_end": "</h>"}
        preprocessor = Preprocessor(PREFIX, SPECIAL_TOKENS_CONSTANTS, None)
        source_text = "abc"
        highlighted_spans = [(1, 2)]
        prep_input = preprocessor.preprocess_input(
            source_text, highlighted_spans)
        assert prep_input == f"{PREFIX}a{SPECIAL_TOKENS_CONSTANTS['highlight_start']}b{SPECIAL_TOKENS_CONSTANTS['highlight_end']}c"

    def test_preprocess_output__sanity(self):
        preprocessor: Preprocessor = Preprocessor("", {}, None)
        summary_text = "abc"
        prep_output = preprocessor.preprocess_output(summary_text)
        assert prep_output == summary_text

    def test_convert_row_spans_str_to_list_of_highlights(self):
        preprocessor: Preprocessor = Preprocessor("", {}, None)
        result = preprocessor.convert_row_spans_str_to_list_of_highlights(
            "5361, 5374;5380, 5446")
        assert result == [(5361, 5374), (5380, 5446)]

    def test_convert_highlight_rows_to_document_highlights(self):
        doc_text = "some_text"
        summary_text = "some_summary_text"
        doc_reader_mock = MagicMock()
        doc_reader_mock.read_doc.return_value = doc_text
        doc_reader_mock.read_summary.return_value = summary_text        
        preprocessor: Preprocessor = Preprocessor("", {}, doc_reader_mock)
        rows = pd.DataFrame(
            [{
                "topic": "abc",
                "documentFile": "abc",
                "docSpanOffsets": "5361, 5374;5380, 5446"
            }, {
                "topic": "abc",
                "documentFile": "abc",
                "docSpanOffsets": "0, 5"
            }])
        result = preprocessor.convert_highlight_rows_to_document_highlights(rows)
        assert result == [(doc_text, summary_text, [(5361, 5374), (5380, 5446), (0, 5)])]
