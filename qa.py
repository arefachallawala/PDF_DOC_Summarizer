from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize


def answer_question(document_text, question):

    sentences = sent_tokenize(document_text)

    if len(sentences) == 0:
        return "No content found."

    corpus = sentences + [question]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(corpus)

    question_vector = tfidf_matrix[-1]

    sentence_vectors = tfidf_matrix[:-1]

    similarity_scores = cosine_similarity(
        question_vector,
        sentence_vectors
    )

    best_match = similarity_scores.argmax()

    confidence = similarity_scores[0][best_match]

    if confidence < 0.1:
        return "Sorry, I couldn't find a relevant answer in the document."

    return sentences[best_match]