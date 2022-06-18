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


def test_concatenate_highlights_row__highlight_ends_in_newline():
    row = {
        'doc_text': 'This is the first sentence.\n This is the second sentence.',
        'highlight_spans': [[0,28]]
    }

    result = concatenate_highlights_row(row)
    expected = "This is the first sentence."  # Spans from the same sentence should stay in the same sentence after concatenation
    assert result == expected

def test_concatenate_highlights_row__spacy_sentencizer_bug():
    """
    Spacy didn't split this sentence correctly, so added logic to split based on \n
    """

    row = {
        'doc_text': "Three hours of steady rain Monday afternoon provided a much-needed edge for crews working to douse a blaze that seared 1,650 acres in Shoshone National Forest in Wyoming, and rainfall in neighboring Yellowstone National Park calmed three smaller fires there.\nMeanwhile, illegal fireworks were blamed for causing a blaze that raged across 2,200 acres near Yosemite National Park in California over the weekend, and firefighters brought a four-day fire in Michigan's Hiawatha National Forest under control Monday after the blaze consumed more than 1,100 acres.",
        'highlight_spans': [[0, 33], [44, 106], [131, 169], [259, 408], [410, 510]]
    }

    result = concatenate_highlights_row(row)
    expected = "Three hours of steady rain Monday provided a much-needed edge for crews working to douse a blaze in Shoshone National Forest in Wyoming. Meanwhile, illegal fireworks were blamed for causing a blaze that raged across 2,200 acres near Yosemite National Park in California over the weekend and firefighters brought a four-day fire in Michigan's Hiawatha National Forest under control Monday."
    assert result == expected


def test_concatenate_highlights_row__highlight_real_usecase():
    df = pd.read_csv("data/dev__highlights.csv")
    row = df.iloc[0]

    result = concatenate_highlights_row(row)
    expected = "opening of the restaurant will take place March 24. the success of the American restaurant depends on its acceptance by Yugoslavians who are long accustomed to the hamburger-like Pljeskavica. Pljeskavica is made of pork and onions is served on bread and eaten with the hands. McDonald's. John Onoda, a spokesman at McDonald's Oak Brook Ill., headquarters, said it was the first of the chain's outlets in a communist country. The next East European McDonald's is scheduled to be opened in Budapest, Hungary, by the end of this year. Negotiations have been going on for years to the Soviet Union, but no agreement has been announced."
    assert result == expected

def test_concatenate_highlights_row__highlight_real_usecase_2():
    df = pd.read_csv("data/dev__highlights.csv")
    row = df.iloc[9]

    result = concatenate_highlights_row(row)
    expected = "Three hours of steady rain Monday provided a much-needed edge for crews working to douse a blaze in Shoshone National Forest in Wyoming. Meanwhile, illegal fireworks were blamed for causing a blaze that raged across 2,200 acres near Yosemite National Park in California over the weekend and firefighters brought a four-day fire in Michigan's Hiawatha National Forest under control Monday. Shoshone fire. Four 20-person crews from Colorado, Utah and South Dakota were expected to arrive at the fire by Monday night putting the total number of firefighters at about 430. At least 60 firefighters brought the Hiawatha forest fire under control Monday."
    assert result == expected


def test_concatenate_highlights_row__highlight_real_usecase_3():
    df = pd.read_csv("data/dev__highlights.csv")
    row = df.iloc[35]

    result = concatenate_highlights_row(row)
    expected = "family and friends mourned the deaths of five school children gunned down during recess while the killer was buried nearly unnoticed in a nearby town. Tuesday. Patrick Purdy opened fire outside the Cleveland Elementary School with an AK-47 killing five children and wounding 29 others and one teacher. Counselors and were assisting teachers with children to comfort and reassure them. Cleveland School is in the heart of California's third-largest community of refugees from Southeast Asia. All of the dead children were from refugee families, as were 19 of the wounded. authorities said. that this is not a racist act ..."
    assert result == expected
