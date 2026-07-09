# ==========================================================
# Imports
# ==========================================================


from youtube_utils import fetch_comments
from comment_analyzer import analyze_all_comments
from dashboard import generate_dashboard
import gradio as gr

import pandas as pd
import numpy as np



import matplotlib.pyplot as plt

import joblib

import re
import string
import contractions
import emoji

import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ==========================================================
# Gemini AI Import
# ==========================================================

import google.generativeai as genai

import os
from dotenv import load_dotenv
load_dotenv()

# ==========================================================
# Gemini API Configuration
# ==========================================================

# NEVER hardcode API keys
# Add GOOGLE_API_KEY in Koyeb/Render environment variables
CURRENT_COMMENTS = []
genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)


# Gemini model

gemini_model = genai.GenerativeModel(
    "gemini-1.5-flash"
)


# ==========================================================
# NLTK Resources
# ==========================================================

nltk.download("punkt")
nltk.download("punkt_tab")

nltk.download("stopwords")

nltk.download("wordnet")

nltk.download("omw-1.4")


# ==========================================================
# Load Machine Learning Model
# ==========================================================

pipeline = joblib.load(
    "pipeline.pkl"
)


# Load optimized thresholds

thresholds = joblib.load(
    "thresholds.pkl"
)


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
# NLP Objects
# ==========================================================

lemmatizer = WordNetLemmatizer()

stop_words = set(
    stopwords.words("english")
)

# Keep negative words
for word in ["not", "no", "nor", "never"]:
    stop_words.discard(word)


# ==========================================================
# Text Preprocessing Function
# ==========================================================

def preprocess(text):

    # lowercase
    text = text.lower()

    # expand contractions
    text = contractions.fix(text)

    # remove HTML
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # remove emails
    text = re.sub(
        r"\S+@\S+",
        "",
        text
    )

    # remove mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # remove hashtag symbol
    text = text.replace(
        "#",
        ""
    )

    # remove emojis
    text = emoji.replace_emoji(
        text,
        replace=""
    )

    # remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()


    # tokenize
    words = word_tokenize(text)


    # remove stopwords + lemmatization
    words = [

        lemmatizer.lemmatize(
            word,
            pos="v"
        )

        for word in words

        if word not in stop_words

    ]


    return " ".join(words)



# ==========================================================
# Prediction Function
# ==========================================================

def predict_comment(user_comment):


    original_comment = user_comment


    cleaned_comment = preprocess(
        user_comment
    )


    input_df = pd.DataFrame(
        {
            "clean_comment":
            [cleaned_comment]
        }
    )


    probabilities = pipeline.predict_proba(
        input_df
    )[0]


    predictions = (
        probabilities >= thresholds
    ).astype(int)


    confidence = (
        np.max(probabilities)
        * 100
    )


    return (

        original_comment,

        cleaned_comment,

        probabilities,

        predictions,

        confidence

    )



# ==========================================================
# Overall Status
# ==========================================================

def get_overall_status(predictions):

    if np.sum(predictions) == 0:

        return "🟢 NON TOXIC COMMENT"

    else:

        return "🔴 TOXIC COMMENT DETECTED"


# ==========================================================
# Create Probability Table
# ==========================================================

def create_probability_table(
    probabilities,
    predictions
):

    probability_df = pd.DataFrame({

        "Category": LABELS,

        "Probability (%)":
        np.round(
            probabilities * 100,
            2
        ),

        "Prediction":
        [
            "Positive" if pred == 1
            else "Negative"

            for pred in predictions
        ]

    })


    return probability_df



# ==========================================================
# Create Probability Chart
# ==========================================================

def create_probability_chart(
    probabilities
):

    probabilities = probabilities * 100


    fig, ax = plt.subplots(
        figsize=(8,5)
    )


    bars = ax.barh(
        LABELS,
        probabilities
    )


    ax.set_xlim(
        0,
        100
    )


    ax.set_xlabel(
        "Probability (%)"
    )


    ax.set_title(
        "Toxicity Probability Distribution"
    )


    for bar in bars:

        width = bar.get_width()


        ax.text(

            width + 1,

            bar.get_y()
            +
            bar.get_height()/2,

            f"{width:.1f}%",

            va="center"

        )


    plt.tight_layout()


    return fig



# ==========================================================
# Gemini AI Explanation Function
# ==========================================================

def generate_gemini_summary(
    original_comment,
    probabilities,
    predictions
):


    detected_categories = []


    for label, prob, pred in zip(

        LABELS,

        probabilities,

        predictions

    ):


        if pred == 1:

            detected_categories.append(

                f"{label}: {prob*100:.1f}%"

            )



    if len(detected_categories) == 0:

        detected_text = "No toxic categories detected."

    else:

        detected_text = "\n".join(
            detected_categories
        )



    prompt = f"""

You are a content moderation AI assistant.

Analyze this comment:

Comment:
{original_comment}


Detected categories:
{detected_text}


Provide:

1. A short explanation of why this comment was classified this way.
2. The possible harmful impact.
3. A safer alternative way to express the message.

Keep the response concise.

"""


    try:

        response = gemini_model.generate_content(
            prompt
        )


        return response.text


    except Exception as e:

        return (
            "AI explanation unavailable."
            f"\nError: {str(e)}"
        )



# ==========================================================
# Final Analysis Function
# ==========================================================

def analyze_comment(
    user_comment
):


    (
        original_comment,

        cleaned_comment,

        probabilities,

        predictions,

        confidence

    ) = predict_comment(
        user_comment
    )



    status = get_overall_status(
        predictions
    )


    probability_table = create_probability_table(

        probabilities,

        predictions

    )


    chart = create_probability_chart(

        probabilities

    )


    ai_summary = generate_gemini_summary(

        original_comment,

        probabilities,

        predictions

    )


    confidence = f"{confidence:.2f}%"



    return (

        original_comment,

        cleaned_comment,

        status,

        confidence,

        probability_table,

        chart,

        ai_summary

    )

def analyze_youtube_video(video_url, max_comments):

    global CURRENT_COMMENTS

    CURRENT_COMMENTS = fetch_comments(
        video_url,
        max_comments
    )

    df = analyze_all_comments(CURRENT_COMMENTS)

    dashboard = generate_dashboard(df)

    return (
        dashboard["Total Comments"],
        dashboard["Safe Comments"],
        dashboard["Toxic Comments"],
        f"{dashboard['Toxicity Rate']}%",
        df
    )

def show_comment_details(evt: gr.SelectData):

    global CURRENT_COMMENTS

    row = evt.index[0]

    comment = CURRENT_COMMENTS[row]["Comment"]

    return analyze_comment(comment)

def export_csv():

    global CURRENT_COMMENTS

    df = analyze_all_comments(CURRENT_COMMENTS)

    file = "analysis.csv"

    df.to_csv(file,index=False)

    return file
    
# ==========================================================
# CSS Styling
# ==========================================================

css = """

footer {
    visibility: hidden;
}


.gradio-container {

    max-width: 1300px !important;

    margin: auto;

}


h1 {

    text-align:center;

}

"""




# ==========================================================
# Gradio Interface
# ==========================================================

with gr.Blocks(

    css=css,

    theme=gr.themes.Soft(

        primary_hue="blue",

        secondary_hue="slate"

    ),

    title="AI Toxic Comment Detection System"

) as demo:

    gr.Markdown("""

# 🛡️ AI YouTube Toxic Comment Detection System

Paste a YouTube video URL.

The system will:

✅ Fetch comments

✅ Analyze every comment

✅ Show dashboard statistics

✅ Display all comments

✅ Click any comment to view detailed ML analysis

""")

    # ======================================================
    # INPUT
    # ======================================================

    with gr.Row():

        video_url = gr.Textbox(

            label="YouTube Video URL",

            placeholder="https://www.youtube.com/watch?v=..."

        )

    max_comments = gr.Slider(

        minimum=10,

        maximum=500,

        value=100,

        step=10,

        label="Maximum Comments"

    )

    fetch_button = gr.Button(

        "Fetch & Analyze Comments",

        variant="primary"

    )

    # ======================================================
    # DASHBOARD
    # ======================================================

    gr.Markdown("---")
    gr.Markdown("## 📊 Dashboard")

    with gr.Row():

        total_output = gr.Textbox(

            label="Total Comments",

            interactive=False

        )

        safe_output = gr.Textbox(

            label="Safe Comments",

            interactive=False

        )

        toxic_output = gr.Textbox(

            label="Toxic Comments",

            interactive=False

        )

        toxicity_output = gr.Textbox(

            label="Toxicity Rate",

            interactive=False

        )

    # ======================================================
    # COMMENTS TABLE
    # ======================================================

    gr.Markdown("---")
    gr.Markdown("## 💬 YouTube Comments")
    
    # download_file = gr.File(
    #     label="Download Analysis"
    # )

    # download_button = gr.Button(
    #     "Download CSV"
    # )
    
    comments_table = gr.Dataframe(

        headers=[

            "Username",

            "Comment",

            "Prediction",

            "Confidence (%)"

        ],

        interactive=True,

        label="Fetched Comments"

    )

    # ======================================================
    # COMMENT DETAILS
    # ======================================================

    gr.Markdown("---")
    gr.Markdown("## 🔍 Selected Comment Analysis")

    with gr.Row():

        original_output = gr.Textbox(

            label="Original Comment",

            lines=4,

            interactive=False

        )

        cleaned_output = gr.Textbox(

            label="Preprocessed Comment",

            lines=4,

            interactive=False

        )

    with gr.Row():

        status_output = gr.Textbox(

            label="Prediction",

            interactive=False

        )

        confidence_output = gr.Textbox(

            label="Confidence",

            interactive=False

        )

    # ======================================================
    # PROBABILITY TABLE
    # ======================================================

    gr.Markdown("---")

    probability_table = gr.Dataframe(

        headers=[

            "Category",

            "Probability (%)",

            "Prediction"

        ],

        interactive=False,

        label="Prediction Details"

    )

    # ======================================================
    # GRAPH
    # ======================================================

    gr.Markdown("---")

    probability_plot = gr.Plot(

        label="Probability Distribution"

    )

    # ======================================================
    # GEMINI
    # ======================================================

    # gr.Markdown("---")

    # summary_output = gr.Textbox(

    #     label="Gemini AI Explanation",

    #     lines=12,

    #     interactive=False

    # )

    # ======================================================
    # EVENTS
    # ======================================================

    # download_button.click(
    #     export_csv,
    #     outputs=download_file
    # )
    
    fetch_button.click(

        fn=analyze_youtube_video,

        inputs=[

            video_url,

            max_comments

        ],

        outputs=[

            total_output,

            safe_output,

            toxic_output,

            toxicity_output,

            comments_table

        ]

    )
    
    comments_table.select(
        fn=show_comment_details,
        inputs=[],
        outputs=[
            original_output,
            cleaned_output,
            status_output,
            confidence_output,
            probability_table,
            probability_plot
            # summary_output
        ]
    )
# ==========================================================
# Launch Application
# ==========================================================

import os

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )

