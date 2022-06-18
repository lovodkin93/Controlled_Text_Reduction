from src.preprocessor import Preprocessor, convert_row_spans_str_to_list_of_highlights, convert_highlight_rows_to_document_highlights
from mock import MagicMock
import pandas as pd

PREFIX = "prefix:"
SPECIAL_TOKENS_CONSTANTS = {"highlight_start": "<h>", "highlight_end": "</h>"}


class TestPreprocessor:
    def test_preprocess_input__sanity(self):
        SPECIAL_TOKENS_CONSTANTS = {
            "highlight_start": "<h>", "highlight_end": "</h>"}
        preprocessor = Preprocessor(PREFIX, SPECIAL_TOKENS_CONSTANTS)
        source_text = "abc"
        highlighted_spans = [(1, 2)]
        prep_input = preprocessor.preprocess_input(
            source_text, highlighted_spans)
        assert prep_input == f"{PREFIX} a{SPECIAL_TOKENS_CONSTANTS['highlight_start']}b{SPECIAL_TOKENS_CONSTANTS['highlight_end']}c"

    def test_preprocess_input__without_highlights(self):
        SPECIAL_TOKENS_CONSTANTS = {
            "highlight_start": "<h>", "highlight_end": "</h>"}
        preprocessor = Preprocessor(PREFIX, SPECIAL_TOKENS_CONSTANTS, should_add_highlights=False)
        source_text = "abc"
        highlighted_spans = [(1, 2)]
        prep_input = preprocessor.preprocess_input(
            source_text, highlighted_spans)
        assert prep_input == f"{PREFIX} abc"


    def test_preprocess_input__keep_only_highlight_sentences(self):
        SPECIAL_TOKENS_CONSTANTS = {
            "highlight_start": "<h>", "highlight_end": "</h>"}
        preprocessor = Preprocessor(PREFIX, SPECIAL_TOKENS_CONSTANTS, should_add_highlights=True, only_sents_with_highlights=True)
        source_text = "wxz.\n abc.\n def."
        highlighted_spans = [(7, 8)]
        prep_input = preprocessor.preprocess_input(
            source_text, highlighted_spans)
        assert prep_input == f"{PREFIX} a{SPECIAL_TOKENS_CONSTANTS['highlight_start']}b{SPECIAL_TOKENS_CONSTANTS['highlight_end']}c."


    def test_preprocess_input__keep_only_highlight_sentences_multiple_highlights(self):
        """
        - Highlight in more than one sentence, and also have a sentence without highlght
        - More than one highlight in same sentence
        - Highlight also the dot and make sure we don't have this: <h>3.</h>.
        """

        SPECIAL_TOKENS_CONSTANTS = {
            "highlight_start": "<h>", "highlight_end": "</h>"}
        preprocessor = Preprocessor(PREFIX, SPECIAL_TOKENS_CONSTANTS, should_add_highlights=True, only_sents_with_highlights=True)
        source_text = "xyz.\n ab, 123.\n def."
        highlighted_spans = [(7, 8), (11, 14), (17, 18)]
        prep_input = preprocessor.preprocess_input(
            source_text, highlighted_spans)
        assert prep_input == f"{PREFIX} a{SPECIAL_TOKENS_CONSTANTS['highlight_start']}b{SPECIAL_TOKENS_CONSTANTS['highlight_end']}, 1{SPECIAL_TOKENS_CONSTANTS['highlight_start']}23.{SPECIAL_TOKENS_CONSTANTS['highlight_end']} d{SPECIAL_TOKENS_CONSTANTS['highlight_start']}e{SPECIAL_TOKENS_CONSTANTS['highlight_end']}f."

    def test_preprocess_input__keep_only_highlight_sentences_real_usecase(self):
        df = pd.read_csv("data/dev__highlights.csv")
        row = df.iloc[0]

        SPECIAL_TOKENS_CONSTANTS = {
            "highlight_start": "<h>", "highlight_end": "</h>"}
        preprocessor = Preprocessor(PREFIX, SPECIAL_TOKENS_CONSTANTS, should_add_highlights=True, only_sents_with_highlights=True)

        source_text = row['doc_text']
        highlighted_spans = row['highlight_spans']
        prep_input = preprocessor.preprocess_input(
            source_text, highlighted_spans)

        expected = f'{PREFIX} The long-awaited <h>opening of the restaurant</h> on one of Belgrade\'s main downtown squares <h>will take place March 24</h>, the Yugoslav news agency Tanjug reported, and it will offer Big Macs, fries and the other specialities familiar to McDonald\'s customers in the West. The Belgrade media have suggested that <h>the success of the American restaurant depends on its acceptance by Yugoslavians who are long accustomed to the hamburger-like Pljeskavica.</h> <h>Pljeskavica is made of</h> ground <h>pork and onions</h>, and it <h>is served on bread and eaten with the hands</h>. "In fact, this is a clash between the Big Mac and Pljeskavica," said an official of Genex, Yugoslavia\'s largest state-run enterprise that will operate the <h>McDonald\'s.</h> <h>John Onoda, a spokesman at McDonald\'s Oak Brook</h>, <h>Ill., headquarters, said it was the first of the chain\'s outlets in a communist country</h>. <h>The next East European McDonald\'s is scheduled to be opened in Budapest, Hungary, by the end of this year</h>, said Vesna Milosevic, another Genex official. <h>Negotiations have been going on for years</h> for expanding the fast-food chain <h>to the Soviet Union, but no agreement has been announced</h>.'
        assert prep_input == expected

    def test_preprocess_output__sanity(self):
        preprocessor: Preprocessor = Preprocessor("", {})
        summary_text = "abc"
        prep_output = preprocessor.preprocess_output(summary_text)
        assert prep_output == summary_text

    def test_convert_row_spans_str_to_list_of_highlights(self):
        result = convert_row_spans_str_to_list_of_highlights(
            "5361, 5374;5380, 5446")
        assert result == [(5361, 5374), (5380, 5446)]

    def test_convert_highlight_rows_to_document_highlights(self):
        doc_text = "some_text"
        summary_text = "some_summary_text"
        doc_reader_mock = MagicMock()
        doc_reader_mock.read_doc.return_value = doc_text
        doc_reader_mock.read_summary.return_value = summary_text        
        rows = pd.DataFrame(
            [{
                "topic": "abc",
                "summaryFile": "abc",
                "documentFile": "abc",
                "docSpanOffsets": "5361, 5374;5380, 5446"
            }, {
                "topic": "abc",
                "summaryFile": "abc",
                "documentFile": "abc",
                "docSpanOffsets": "0, 5"
            }])
        result = convert_highlight_rows_to_document_highlights(doc_reader_mock, rows)
        assert result == [(doc_text, summary_text, [(5361, 5374), (5380, 5446), (0, 5)])]
