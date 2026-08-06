import re
import math
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except ImportError:
    nlp = None


def extract_entities(text):
    """
    Extract Named Entities (Persons, Organizations, Geopolitical Entities/Locations, Dates, Money)
    using SpaCy or a reliable regex/NLP fallback.
    """
    entities = {
        "PERSON": set(),
        "ORG": set(),
        "GPE": set(),
        "DATE": set(),
        "MONEY": set()
    }

    if nlp is not None:
        # Truncate text if extremely long to avoid memory issues
        doc = nlp(text[:50000])
        for ent in doc.ents:
            clean_ent = ent.text.strip()
            if len(clean_ent) < 2:
                continue
            if ent.label_ in ["PERSON"]:
                entities["PERSON"].add(clean_ent)
            elif ent.label_ in ["ORG"]:
                entities["ORG"].add(clean_ent)
            elif ent.label_ in ["GPE", "LOC"]:
                entities["GPE"].add(clean_ent)
            elif ent.label_ in ["DATE", "TIME"]:
                entities["DATE"].add(clean_ent)
            elif ent.label_ in ["MONEY", "PERCENT", "CARDINAL"]:
                entities["MONEY"].add(clean_ent)
    else:
        # Regex & Heuristic Fallback
        # Dates (e.g. 2024, Jan 2023, 12th March)
        dates = re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b|\b\d{4}\b', text)
        entities["DATE"].update(dates[:10])

        # Money / Percentages
        money = re.findall(r'\$\d+(?:\.\d+)?|\b\d+%\b|\b\d+ million\b|\b\d+ billion\b', text, re.IGNORECASE)
        entities["MONEY"].update(money[:10])

        # Capitalized multi-word proper nouns (Orgs/Persons heuristic)
        capitalized = re.findall(r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+\s?)+\b', text)
        for name in set(capitalized[:15]):
            if any(kw in name for kw in ["Inc", "Corp", "Ltd", "University", "Company", "Group", "Department", "Association", "Organization"]):
                entities["ORG"].add(name)
            else:
                entities["PERSON"].add(name)

    # Convert sets to sorted lists
    return {k: sorted(list(v)) for k, v in entities.items()}


def analyze_sentiment(text):
    """
    Perform lexicon-based sentiment analysis on text.
    Returns scores for positive, negative, neutral, and overall label.
    """
    positive_words = {
        "good", "great", "excellent", "positive", "growth", "increase", "improve",
        "profit", "success", "effective", "benefit", "strong", "best", "gain",
        "opportunity", "advancement", "innovative", "solution", "valuable", "achieve"
    }
    negative_words = {
        "bad", "poor", "loss", "decrease", "decline", "risk", "failure", "problem",
        "issue", "threat", "damage", "error", "difficult", "concern", "negative",
        "reduction", "weak", "delay", "challenge", "costly", "drop"
    }

    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return {"pos": 33, "neu": 34, "neg": 33, "label": "Neutral", "score": 0.0}

    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    total_matched = pos_count + neg_count

    if total_matched == 0:
        return {"pos": 20, "neu": 60, "neg": 20, "label": "Neutral / Balanced", "score": 0.0}

    pos_pct = round((pos_count / len(words)) * 100 * 5, 1)  # scaled
    neg_pct = round((neg_count / len(words)) * 100 * 5, 1)

    pos_pct = min(pos_pct, 80.0)
    neg_pct = min(neg_pct, 80.0)
    neu_pct = round(100.0 - pos_pct - neg_pct, 1)
    if neu_pct < 10.0:
        neu_pct = 10.0
        pos_pct = round((100.0 - neu_pct) * (pos_count / (pos_count + neg_count)), 1)
        neg_pct = round(100.0 - neu_pct - pos_pct, 1)

    score = round((pos_count - neg_count) / max(total_matched, 1), 2)

    if score > 0.2:
        label = "Positive"
    elif score < -0.2:
        label = "Negative"
    else:
        label = "Neutral / Objective"

    return {
        "pos": pos_pct,
        "neu": neu_pct,
        "neg": neg_pct,
        "label": label,
        "score": score
    }


def calculate_readability(text):
    """
    Calculate Flesch Reading Ease score and Grade Level estimation.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\b[a-zA-Z]+\b', text)

    num_sentences = max(len(sentences), 1)
    num_words = max(len(words), 1)

    # Syllable counter approximation
    def count_syllables(word):
        word = word.lower()
        if len(word) <= 3:
            return 1
        count = len(re.findall(r'[aeiouy]{1,2}', word))
        if word.endswith('e'):
            count -= 1
        return max(count, 1)

    num_syllables = sum(count_syllables(w) for w in words)

    # Flesch Reading Ease Formula
    # 206.835 - 1.015*(total words / total sentences) - 84.6*(total syllables / total words)
    words_per_sentence = num_words / num_sentences
    syllables_per_word = num_syllables / num_words

    flesch_score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
    flesch_score = round(max(0, min(100, flesch_score)), 1)

    # Flesch-Kincaid Grade Level Formula
    grade_level = round(0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59, 1)
    grade_level = max(1.0, grade_level)

    if flesch_score >= 80:
        difficulty = "Very Easy (Elementary Level)"
    elif flesch_score >= 60:
        difficulty = "Standard (Plain English / Conversational)"
    elif flesch_score >= 40:
        difficulty = "Fairly Difficult (College / Technical Level)"
    else:
        difficulty = "Very Complex (Academic / Legal Level)"

    return {
        "flesch_score": flesch_score,
        "grade_level": grade_level,
        "difficulty": difficulty,
        "words_per_sentence": round(words_per_sentence, 1)
    }


def create_sentiment_chart(sentiment_dict):
    """
    Returns a Plotly Donut Chart figure for Sentiment distribution.
    """
    labels = ["Positive", "Neutral", "Negative"]
    values = [sentiment_dict["pos"], sentiment_dict["neu"], sentiment_dict["neg"]]
    colors = ["#2ECC71", "#3498DB", "#E74C3C"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.55,
        marker_colors=colors,
        textinfo='percent+label',
        hoverinfo='label+percent',
        showlegend=False
    )])

    fig.update_layout(
        title={
            'text': f"Overall Sentiment: <b>{sentiment_dict['label']}</b>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': '#2C3E50'}
        },
        margin=dict(t=50, b=20, l=20, r=20),
        height=260,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_keyword_chart(common_words):
    """
    Returns a Plotly horizontal bar chart for top keywords.
    """
    if not common_words:
        return go.Figure()

    words = [w[0] for w in reversed(common_words)]
    counts = [w[1] for w in reversed(common_words)]

    fig = go.Figure(go.Bar(
        x=counts,
        y=words,
        orientation='h',
        marker=dict(
            color=counts,
            colorscale='Peach',
            line=dict(color='#FF6B6B', width=1.5)
        )
    ))

    fig.update_layout(
        title={
            'text': "<b>Top 10 Most Frequent Keywords</b>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': '#2C3E50'}
        },
        xaxis_title="Frequency",
        yaxis_title="Keywords",
        margin=dict(t=50, b=40, l=80, r=20),
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig
