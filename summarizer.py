import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

nltk.download("punkt")


def generate_summary(text, summary_length=7):

    sentences = sent_tokenize(text)

    if len(sentences) <= summary_length:
        return text

    words = word_tokenize(text.lower())

    word_frequencies = Counter(words)

    sentence_scores = {}

    for sentence in sentences:

        sentence_words = word_tokenize(sentence.lower())

        score = 0

        for word in sentence_words:

            score += word_frequencies[word]

        sentence_scores[sentence] = score

    ranked_sentences = sorted(
        sentence_scores,
        key=sentence_scores.get,
        reverse=True
    )

    selected_sentences = ranked_sentences[:summary_length]

    summary = []

    for sentence in sentences:

        if sentence in selected_sentences:
            summary.append(sentence)

    return " ".join(summary)