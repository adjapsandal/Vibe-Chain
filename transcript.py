from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable


def get_transcript(video_id):
    lang = 'en'
    try:
        yt_api = YouTubeTranscriptApi()
        transcripts = yt_api.list(video_id)
        try:
            transcript = transcripts.find_manually_created_transcript([lang])
        except NoTranscriptFound:
            try:
                transcript = transcripts.find_generated_transcript([lang])
            except NoTranscriptFound:
                try:
                    transcript = transcripts.find_manually_created_transcript(
                        transcripts._manually_created_transcripts.keys()
                    )
                except Exception:
                    transcript = transcripts.find_generated_transcript(
                        transcripts._generated_transcripts.keys()
                    )
                try:
                    transcript = transcript.translate(lang)
                except Exception:
                    pass

        if not transcript:
            print(f"Не удалось получить транскрипт для видео {video_id}")
            return ""

        fetched = transcript.fetch()
        texts = [t.text for t in fetched]
        return " ".join(texts)

    except VideoUnavailable:
        print(f"Видео {video_id} недоступно или удалено")
    except TranscriptsDisabled:
        print(f"У видео {video_id} субтитры отключены")
    except Exception as e:
        print(f"Ошибка при получении транскрипта: {e}")

    return ""