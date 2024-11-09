import pytest
from summarization import summarize_text, create_gists

# Mock the pipeline to return a summary without actually calling the model
class MockSummarizer:
    def __call__(self, text, max_length=None, min_length=None, do_sample=None):
        # Mock behavior: return the first half of the input text as "summary"
        if len(text.split()) > 10:
            return [{"summary_text": " ".join(text.split()[:len(text.split())//2])}]
        else:
            return [{"summary_text": text}]

@pytest.mark.parametrize("text,expected", [
    ("This is a test sentence. It is only a test.", "This is a test sentence."),
    ("Short text.", "Short text.")
])
def test_summarize_text(text, expected):
    summarizer = MockSummarizer()
    summary = summarize_text(text, summarizer, chunk_size=5)
    assert summary == expected, "The summarized text does not match the expected output."

def test_summarize_empty_string():
    summarizer = MockSummarizer()
    summary = summarize_text("", summarizer)
    assert summary == "", "The summary of an empty string should be an empty string."

def test_create_gists():
    episodes = [
        "This is the content of episode 1. It contains detailed information about the topic.",
        "This is the content of episode 2. It also includes important information."
    ]
    expected_number_of_gists = len(episodes)
    gists = create_gists(episodes, model_name='sshleifer/distilbart-cnn-12-6', chunk_size=5)
    assert len(gists) == expected_number_of_gists, "The number of gists created does not match the number of episodes."

def test_create_gists_with_empty_string():
    episodes = [""]
    gists = create_gists(episodes, model_name='sshleifer/distilbart-cnn-12-6')
    assert gists == [""], "The gist of an empty episode should be an empty string."

# Remember to replace 'from summarization import ...' with the actual import statement based on your file structure