import os
import re

from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = build(
    "youtube",
    "v3",
    developerKey=YOUTUBE_API_KEY
)


def extract_video_id(url):

    patterns = [

        r"v=([a-zA-Z0-9_-]{11})",

        r"youtu\.be/([a-zA-Z0-9_-]{11})",

        r"shorts/([a-zA-Z0-9_-]{11})"

    ]

    for pattern in patterns:

        match = re.search(pattern, url)

        if match:

            return match.group(1)

    return None


def fetch_comments(video_url, max_comments=100):

    video_id = extract_video_id(video_url)

    if video_id is None:

        raise ValueError("Invalid YouTube URL")

    comments = []

    next_page_token = None

    while len(comments) < max_comments:

        request = youtube.commentThreads().list(

            part="snippet",

            videoId=video_id,

            maxResults=min(100, max_comments-len(comments)),

            pageToken=next_page_token,

            textFormat="plainText"

        )

        response = request.execute()

        for item in response["items"]:

            snippet = item["snippet"]["topLevelComment"]["snippet"]

            comments.append(

                {

                    "Username":
                    snippet["authorDisplayName"],

                    "Comment":
                    snippet["textDisplay"],

                    "Likes":
                    snippet["likeCount"],

                    "Published":
                    snippet["publishedAt"]

                }

            )

        next_page_token = response.get("nextPageToken")

        if not next_page_token:

            break

    return comments