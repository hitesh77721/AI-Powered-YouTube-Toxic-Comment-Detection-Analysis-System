import pandas as pd


def generate_dashboard(df):

    total_comments = len(df)

    toxic_comments = len(
        df[df["Prediction"] == "🔴 Toxic"]
    )

    safe_comments = len(
        df[df["Prediction"] == "🟢 Non Toxic"]
    )

    toxicity_rate = round(
        (toxic_comments / total_comments) * 100,
        2
    ) if total_comments > 0 else 0

    return {

        "Total Comments": total_comments,

        "Toxic Comments": toxic_comments,

        "Safe Comments": safe_comments,

        "Toxicity Rate": f"{toxicity_rate}%"

    }