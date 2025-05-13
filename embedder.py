import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from fetch_metadata import get_video_snippet, extract_video_id
from transcript import get_transcript
import requests
from io import BytesIO
from pytube import YouTube
import cv2

sbert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

'''
def download_video_stream(url, max_seconds=30):
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension='mp4', progressive=True).order_by('resolution').desc().first()
    buffer = BytesIO()
    stream.stream_to_buffer(buffer)
    buffer.seek(0)
    return buffer  # BytesIO с mp4

def sample_frames_from_buffer(buffer, num_frames=5):
    # сохраняем временно в файл или напрямую используем opencv
    cap = cv2.VideoCapture(buffer)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def encode_visual_from_url(url, num_frames=5):
    buf = download_video_stream(url)
    frames = sample_frames_from_buffer(buf, num_frames)
    pil_frames = [Image.fromarray(f).convert('RGB') for f in frames]
    embs = [encode_image(img) for img in pil_frames]
    vis_emb = np.mean(embs, axis=0)
    return vis_emb / np.linalg.norm(vis_emb)
'''


def encode_text(text: str):
    return sbert.encode(text, normalize_embeddings=True)


def encode_image(image: Image.Image):
    inputs = processor(images=image, return_tensors='pt')
    features = clip.get_image_features(**inputs)
    emb = features.detach().cpu().numpy().squeeze(0)
    return emb / np.linalg.norm(emb)


def embed_video(url: str):
    vid = extract_video_id(url)
    meta = get_video_snippet(vid)
    meta_str = meta['title'] + ' ' + meta['description'] + ' ' + ' '.join(meta.get('tags', []))
    tr = get_transcript(vid)
    text_emb = encode_text(meta_str + ' ' + tr)
    thumb_url = meta['thumbnail_url']
    if thumb_url:
        resp = requests.get(thumb_url)
        image = Image.open(BytesIO(resp.content)).convert('RGB')
        img_emb = encode_image(image)
        return np.concatenate([text_emb, img_emb])
    else:
        return text_emb
