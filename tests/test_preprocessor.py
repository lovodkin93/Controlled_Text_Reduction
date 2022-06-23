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

    def test_preprocess_input__keep_only_highlight_sentences__text_start_with_space(self):
        SPECIAL_TOKENS_CONSTANTS = {
            "highlight_start": "<h>", "highlight_end": "</h>"}
        preprocessor = Preprocessor(PREFIX, SPECIAL_TOKENS_CONSTANTS, should_add_highlights=True, only_sents_with_highlights=True)
        source_text = " wxz.\n abc.\n def."
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


    def test_preprocess_input__keep_only_highlight_sentences_real_usecase_1(self):
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

    def test_preprocess_input__keep_only_highlight_sentences__real_usecase_2(self):
        """
        There is a bug with the highlight ranges when the source text starts with space
        """

        SPECIAL_TOKENS_CONSTANTS = {
            "highlight_start": "<h>", "highlight_end": "</h>"}
        preprocessor = Preprocessor(PREFIX, SPECIAL_TOKENS_CONSTANTS, should_add_highlights=True, only_sents_with_highlights=True)
        source_text = ' In local government elections across Britain yesterday, the Conservatives suffered their worst defeat ever, losing control of 17 regional councils and 444 seats.\nAs the ruling party, the Tories took just 27 percent of the vote, two percentage points lower than their rating in the polls.\nThe results of the local elections showed that the Tories were reduced to the third party nationally, behind the Liberal Democrats and the Labour Party.\nThe Labour Party controls 90 local councils, whereas the Conservatives only control 13, with a sharp contrast in strength between the two sides.\nThe Liberal Democrats had a net gain of 378 seats.\nEven some former Conservative members of parliament announced publicly to reporters that they would vote for the Liberal Democrats.\nOf course, this was a measure taken by Tories under circumstances of an inevitable defeat to reduce losses by encouraging constituents, who resent the Tories, to vote for the Liberal Democrats rather than Labour.\nIn the deep sorrow of defeat, some Conservatives are secretly feeling happy.\nThey resent John Major\'s leadership, and would most willingly link Major\'s incompetence and lack of vision with the bitter defeat in the local elections.\nEven before all of the results were known, some Tories openly announced their determination to challenge John Major\'s position and remove him from office as early as possible.\nThe right-winger Michael Portillo added salt to the wound by starting a dispute over the European issue, and highlighted John Major\'s image as a loser who fails to command.\nObviously, all of these actions were long planned and are aimed at making John Major suffer an even greater defeat in the European Parliament elections scheduled for 9 June.\nBefore 9 June, nobody will officially challenge for the position of party leader.\nBut he who is in office must suffer the pain of electoral defeat.\nAfter 9 June, John Major will have to answer for all of the disasters, and it is time to use him as a scapegoat.\nJohn Major is someone who was propped up by Mrs. Thatcher, who supported appointing the mediocre John Major as her successor when she was utterly isolated so that one day she would be able to pull strings from behind the scenes and continue to pursue the Thatcher line.\nNevertheless, people around John Major seriously pushed him to betray Thatcher\'s doctrine, while pursuing entirely new "leftist" policies and affecting John Major\'s moves.\nFor a very long period, right-wingers and left-wingers argued over many domestic and world issues.\nJohn Major has always been weak and powerless, and vacillates to the left and to the right resulting in his own "crisis of honor."\nOn the European issue, he stressed the imperative of maintaining the sovereignty, cultural traditions, and life-style of each country of the future European Union, while reflecting the islanders\' mind set.\nOn the one hand, he expressed the will to join the European family; on the other, he made one big stride closer to the right.\nWhile great skill is needed to have one\'s bread buttered on both sides, John Major has been relatively stupid in such circumstance.\nWhen expounding on such concepts, he set out a campaign to "return to basic values," with an emphasis on citizen self-discipline, taking social responsibility and obligations, and being upright.\nHowever, one scandal surfaced after another in his cabinet; consequently, British citizens saw how bombastic Major could be and how he failed to know himself.\nThe scandals caused Major\'s reputation to collapse.\nFor a long time, John Major has failed to resolve the economic crisis and has seriously jeopardized the interests of British people, who have become poorer with each passing day and suffer from a high rate of unemployment.\nAt this critical juncture, John Major went back on promises made during the election campaign by increasing value-added tax on domestic fuel; as a result, things became even harder for the poor.\nHe promised to reduce public expenditure, but on the contrary, it rose by 45.5 percent.\n[as published] John Major has become an expert in making "empty promises," while constituents continue to scorn this dishonest prime minister.\nMany Conservatives believe that John Major is driven by a gambler\'s mind set.\nAt a time when his reputation was waning, he wanted to start an offensive in the arena of difficult world issues, regardless of British national strength.\nThis being the case, he wanted to try his luck in Bosnia, as well as policy toward China; however, he was helpless in dealing with domestic economic issues which would only lead Britain to an impasse.\nIt will be difficult for John Major to lead the Tories in the next general election; therefore, news spread earlier this year that John Major would be replaced.\nThe setback which has resulted from defeats in the local elections is heavier than expected.\nTo accelerate a leadership reshuffle, Conservatives have spread the opinion that "John Major is not up to his office."\nHowever, if politicians from various factions compete for office by taking a laissez-faire attitude, the Tory unity will be jeopardized.\nThis being the case, a planned transfer of power, especially the reorganization of the cabinet and the replacement of the party chairman, is necessary.\nWhen the new party chairman is elected, the question of who will replace John Major will be settled too.\nWord has spread that John Major wants Chris Patten to become Conservative Party chairman; however, this proposal was rejected.\nThat should have been a party secret; making public such an inside story would only provide evidence of political reality, namely, many Conservatives did not agree to Chris Patten contending for this position.\nThere are no grounds to allow a politician who has lost a previous electoral campaign to canvass votes in the next one under unfavorable electoral circumstances.\nThat would only aggravate the Tory\'s poor image as a party which enjoys little public support and has few talents.\nIn actual fact, should John Major set out such a line in appointing people, that would only give more evidence of his poor ability, insight, and lack of judgment, as well as his self-knowledge.\nNow, it is just a matter of time until he leaves office; but nobody will let him make arrangements for the transfer of power.\nWork to seek a new leader will be conducted by veteran Tories; how can John Major be qualified for such a task?'
        highlighted_spans = '[[0, 54], [56, 106], [0, 29], [56, 73], [108, 161], [4783, 4875], [4783, 4849], [4861, 4875], [0, 29], [56, 73], [108, 161], [335, 388], [982, 1010], [1012, 1058], [2281, 2356], [2281, 2305], [2364, 2438], [2619, 2667], [3487, 3538], [1059, 1063], [1100, 1212]]'
        prep_input = preprocessor.preprocess_input(
            source_text, highlighted_spans)
        assert prep_input == f'{PREFIX} <h>In local government elections across Britain yesterday</h>, <h>the Conservatives suffered their worst defeat ever</h>, <h>losing control of 17 regional councils and 444 seats.</h> The results of the local elections showed that <h>the Tories were reduced to the third party nationally</h>, behind the Liberal Democrats and the Labour Party. <h>In the deep sorrow of defeat</h>, <h>some Conservatives are secretly feeling happy.</h> <h>They</h> resent John Major\'s leadership, and <h>would most willingly link Major\'s incompetence and lack of vision with the bitter defeat in the local elections.</h> Nevertheless, <h>people around John Major seriously pushed him to betray Thatcher\'s doctrine</h>, while <h>pursuing entirely new "leftist" policies and affecting John Major\'s moves.</h> John Major has always been weak and powerless, and vacillates to the left and to <h>the right resulting in his own "crisis of honor.</h> <h>The scandals caused Major\'s reputation to collapse.</h> <h>The setback which has resulted from defeats in the local elections is heavier than expected.</h>'

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
