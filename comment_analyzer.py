import joblib
import pandas as pd
import numpy as np

import re
import string
import contractions
import emoji

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


import nltk

resources = {
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
}

for path, package in resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(package)
# ==========================================================
# Load Model
# ==========================================================

pipeline = joblib.load("pipeline.pkl")
thresholds = joblib.load("thresholds.pkl")

# ==========================================================
# Labels
# ==========================================================

LABELS = [
    "Toxic",
    "Severe Toxic",
    "Obscene",
    "Threat",
    "Insult",
    "Identity Hate"
]

# ==========================================================
# NLP
# ==========================================================

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))

for word in ["not", "no", "nor", "never"]:
    stop_words.discard(word)

# ==========================================================
# Preprocessing
# ==========================================================

def preprocess(text):

    text = text.lower()

    text = contractions.fix(text)

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"\S+@\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = text.replace("#", "")

    text = emoji.replace_emoji(text, replace="")

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    words = word_tokenize(text)

    words = [

        lemmatizer.lemmatize(word, pos="v")

        for word in words

        if word not in stop_words

    ]

    return " ".join(words)


# ==========================================================
# Predict One Comment
# ==========================================================

def predict_single_comment(comment):

    cleaned = preprocess(comment)

    df = pd.DataFrame({

        "clean_comment": [cleaned]

    })

    probabilities = pipeline.predict_proba(df)[0]

    predictions = (

        probabilities >= thresholds

    ).astype(int)

    confidence = round(

        np.max(probabilities) * 100,

        2

    )

    status = (

        "🟢 Non Toxic"

        if np.sum(predictions) == 0

        else "🔴 Toxic"

    )

    return {

        "Original Comment": comment,

        "Cleaned Comment": cleaned,

        "Status": status,

        "Confidence": confidence,

        "Probabilities": probabilities,

        "Predictions": predictions

    }


# ==========================================================
# Analyze All Comments
# ==========================================================

def analyze_all_comments(comment_list):

    results = []

    for row in comment_list:

        prediction = predict_single_comment(

            row["Comment"]

        )

        results.append({

            "Username": row["Username"],

            "Comment": row["Comment"],

            "Prediction": prediction["Status"],

            "Confidence (%)": prediction["Confidence"]

        })

    return pd.DataFrame(results)