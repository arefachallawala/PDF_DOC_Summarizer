import re
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


def generate_summary(text, summary_type="Medium"):

    sentences = sent_tokenize(text)

    if not sentences:
        return "No text available for summarization."

    if summary_type == "Short":
        summary_length = max(1, round(len(sentences) * 0.20))

    elif summary_type == "Medium":
        summary_length = max(3, round(len(sentences) * 0.40))

    elif summary_type == "Detailed":
        summary_length = max(5, round(len(sentences) * 0.70))

    elif summary_type == "Bullet Points":
        summary_length = max(4, round(len(sentences) * 0.40))

    else:
        summary_length = max(3, round(len(sentences) * 0.40))

    summary_length = min(summary_length, len(sentences))

    stop_words = set(stopwords.words("english"))

    words = word_tokenize(text.lower())

    useful_words = []

    for word in words:

        if (
            word.isalpha()
            and word not in stop_words
            and len(word) > 2
        ):
            useful_words.append(word)

    word_frequencies = Counter(useful_words)

    sentence_scores = {}

    for index, sentence in enumerate(sentences):

        sentence_words = word_tokenize(sentence.lower())

        score = 0

        for word in sentence_words:

            if word in word_frequencies:
                score += word_frequencies[word]

        if len(sentence_words) > 0:
            score = score / len(sentence_words)

        sentence_scores[index] = score

    ranked_sentences = sorted(
        sentence_scores,
        key=sentence_scores.get,
        reverse=True
    )

    selected_indexes = ranked_sentences[:summary_length]

    selected_indexes.sort()

    selected_sentences = [
        sentences[index].strip()
        for index in selected_indexes
    ]

    if summary_type == "Bullet Points":

        bullet_points = []

        for sentence in selected_sentences:

            sentence = re.sub(
                r"\s+",
                " ",
                sentence
            ).strip()

            bullet_points.append(
                f"- {sentence}"
            )

        return "\n\n".join(bullet_points)

    return " ".join(selected_sentences)