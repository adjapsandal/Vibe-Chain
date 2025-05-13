import os
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=KEY)


def get_video_snippet(id_video: str) -> dict | None:
    request = youtube.videos().list(
        part="snippet",
        id=id_video
    )
    response = request.execute()
    items = response.get("items", [])
    if not items:
        return None

    snippet = items[0]["snippet"]
    thumbnails = snippet.get("thumbnails", {})
    # выбираем наиболее подходящую миниатюру (default, medium, high)
    thumb_url = None
    for size in ["high", "medium", "default"]:
        if thumbnails.get(size) and thumbnails[size].get("url"):
            thumb_url = thumbnails[size]["url"]
            break

    return {
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "tags": snippet.get("tags", []),
        "thumbnail_url": thumb_url
    }


def extract_video_id(url: str) -> str | None:
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or ''

    if hostname == 'youtu.be':
        return parsed_url.path[1:]

    if hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        if parsed_url.path == '/watch':
            params = parse_qs(parsed_url.query)
            return params.get('v', [None])[0]
        if parsed_url.path.startswith(('/embed/', '/v/')):
            return parsed_url.path.split('/')[2]

    return None
