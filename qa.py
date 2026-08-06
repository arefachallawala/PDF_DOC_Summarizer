import re

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(text):

    stop_words = set(stopwords.words("english"))

    words = word_tokenize(text.lower())

    cleaned_words = []

    for word in words:

        if word.isalpha() and word not in stop_words:

            cleaned_words.append(word)

    return " ".join(cleaned_words)


def answer_question(document_text, question):

    paragraphs = []

    for paragraph in document_text.split("\n"):

        paragraph = paragraph.strip()

        if len(paragraph) > 20:

            paragraphs.append(paragraph)

    if len(paragraphs) == 0:

        paragraphs = sent_tokenize(document_text)

    cleaned_paragraphs = []

    for paragraph in paragraphs:

        cleaned_paragraphs.append(

            preprocess(paragraph)

        )

    cleaned_question = preprocess(question)

    corpus = cleaned_paragraphs + [

        cleaned_question

    ]

    vectorizer = TfidfVectorizer(

        ngram_range=(1,2)

    )

    tfidf = vectorizer.fit_transform(

        corpus

    )

    similarity = cosine_similarity(

        tfidf[-1],

        tfidf[:-1]

    )

    best = similarity.argmax()

    score = similarity[0][best]

    if score < 0.08:

        return "I could not find a relevant answer in the document."

    answer = paragraphs[best]

    answer = re.sub(

        r"\n+",

        " ",

        answer

    )

    return answer