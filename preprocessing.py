import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    words = word_tokenize(text)

    stop_words = set(stopwords.words("english"))

    filtered_words = []

    for word in words:

        if word not in stop_words:

            filtered_words.append(word)

    lemmatizer = WordNetLemmatizer()

    cleaned_words = []

    for word in filtered_words:

        cleaned_words.append(
            lemmatizer.lemmatize(word)
        )

    cleaned_text = " ".join(cleaned_words)

    return cleaned_text