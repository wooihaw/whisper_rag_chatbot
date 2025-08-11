"""
Chatbot with local LLM (Ollama), local Whisper STT, KittenTTS TTS, and optional RAG.

RAG sources (both optional):
  --urls   Comma-separated list of URLs to index.
  --folder Path to a folder containing .txt or .md files.

Dependencies:
  pip install ollama requests beautifulsoup4 sounddevice soundfile keyboard numpy kittentts openai-whisper

Hints:
  - Pull an embedding model in Ollama, e.g.:
      ollama pull nomic-embed-text
  - Pull your LLM in Ollama, e.g.:
      ollama pull llama3.1
"""

import argparse
import os
import re
import time
import math
import whisper
import sounddevice as sd
import numpy as np
import requests
from bs4 import BeautifulSoup
import keyboard
from typing import List, Dict, Any, Tuple
from kittentts import KittenTTS
from ollama import Client

# -------------------------
# Configurable defaults
# -------------------------
SAMPLE_RATE = 16000
DEFAULT_LLM = "sea_turtle_llama3_2_3b_q4_k_m_v5"
DEFAULT_EMBED = "nomic-embed-text"
TOP_K = 4
MAX_CONTEXT_CHARS = 1800
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
REQUEST_TIMEOUT = 20

# -------------------------
# LLM / STT / TTS init
# -------------------------
client = Client(host="http://127.0.0.1:11434")

print("Loading Whisper model…")
whisper_model = whisper.load_model("small")

ktts = KittenTTS("KittenML/kitten-tts-nano-0.1")


# -------------------------
# Utilities
# -------------------------
def clean_text(txt: str) -> str:
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{2,}", "\n\n", txt)
    return txt.strip()


def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embed_texts(texts: List[str], embed_model: str) -> np.ndarray:
    vecs = []
    for t in texts:
        r = client.embeddings(model=embed_model, prompt=t)
        vec = r["embedding"] if isinstance(r, dict) else r.embedding
        vecs.append(vec)
    return np.array(vecs, dtype=np.float32)


# -------------------------
# Source loaders
# -------------------------
def load_url(url: str) -> str:
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "RAGBot/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove script and style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        doc = f"{title}\n\n{text}"
        return clean_text(doc)
    except Exception as e:
        print(f"[WARN] Failed to fetch {url}: {e}")
        return ""


def load_folder_texts(folder: str) -> List[Tuple[str, str]]:
    docs = []
    if not folder or not os.path.isdir(folder):
        return docs
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith((".txt", ".md")):
                path = os.path.join(root, fn)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        txt = f.read()
                    docs.append((path, clean_text(txt)))
                except Exception as e:
                    print(f"[WARN] Failed to read {path}: {e}")
    return docs


# -------------------------
# RAG index
# -------------------------
class RAGIndex:
    def __init__(self, embed_model: str):
        self.embed_model = embed_model
        self.chunks: List[Dict[str, Any]] = []  # {text, source, id}
        self.matrix: np.ndarray | None = None

    def build(self,
              urls: List[str] | None = None,
              folder: str | None = None) -> None:
        corpus_chunks = []

        # URLs
        if urls:
            for u in urls:
                raw = load_url(u)
                if not raw:
                    continue
                for i, ch in enumerate(chunk_text(raw)):
                    corpus_chunks.append({"text": ch, "source": u, "id": f"url::{u}::chunk{i}"})

        # Local folder
        if folder:
            for path, txt in load_folder_texts(folder):
                for i, ch in enumerate(chunk_text(txt)):
                    corpus_chunks.append({"text": ch, "source": path, "id": f"file::{path}::chunk{i}"})

        if not corpus_chunks:
            self.chunks = []
            self.matrix = None
            print("RAG: no sources provided. Retrieval disabled.")
            return

        texts = [c["text"] for c in corpus_chunks]
        print(f"RAG: embedding {len(texts)} chunks…")
        mat = embed_texts(texts, self.embed_model)
        self.chunks = corpus_chunks
        self.matrix = mat
        print(f"RAG: index ready. {len(self.chunks)} chunks.")

    def retrieve(self, query: str, top_k: int = TOP_K,
                 max_context_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, List[Tuple[int, str]]]:
        if self.matrix is None or not self.chunks:
            return "", []

        qvec = embed_texts([query], self.embed_model)[0]
        sims = self.matrix @ qvec / (np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(qvec) + 1e-12)
        order = np.argsort(-sims)[:max(1, top_k)]

        context_parts = []
        citations = []
        total = 0
        rank = 1
        for idx in order:
            piece = self.chunks[idx]["text"]
            src = self.chunks[idx]["source"]
            snippet = piece.strip()
            if total + len(snippet) + 10 > max_context_chars:
                # truncate last chunk if needed
                remaining = max_context_chars - total - 10
                if remaining > 50:
                    snippet = snippet[:remaining]
                else:
                    break
            context_parts.append(f"[{rank}] {snippet}")
            citations.append((rank, src))
            total += len(snippet) + 10
            rank += 1

        return "\n\n".join(context_parts), citations


# -------------------------
# STT / TTS
# -------------------------
def listen_and_recognize() -> str:
    print("Press SPACE to start recording.")
    keyboard.wait('space')
    print("Recording… Press SPACE again to stop.")
    frames = []

    def callback(indata, frames_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        keyboard.wait('space')

    print("Recording stopped.")
    audio = np.concatenate(frames, axis=0)

    result = whisper_model.transcribe(
        audio.flatten(),
        language="en",
        fp16=False
    )
    text = result.get("text", "").strip()
    return text


def speak(text: str):
    wav = ktts.generate(text + '...', voice='expr-voice-3-f')
    sd.play(wav, samplerate=24000)
    sd.wait()


# -------------------------
# Chat loop with optional RAG
# -------------------------
def converse(model: str,
             rag: RAGIndex | None):
    print("You may start. Say 'exit' to quit.")
    history: List[Dict[str, str]] = []

    while True:
        user_text = listen_and_recognize()
        print(f"You said: {user_text}")
        if user_text.lower() in ("exit", "quit"):
            speak("Goodbye.")
            break

        # Build RAG context if available
        rag_context = ""
        citations: List[Tuple[int, str]] = []
        if rag is not None and rag.matrix is not None:
            rag_context, citations = rag.retrieve(user_text, top_k=TOP_K, max_context_chars=MAX_CONTEXT_CHARS)

        system_style = {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer in one clear sentence, not more than 30 words. "
                "Be factual and concise."
            )
        }

        grounding_msg = None
        if rag_context:
            grounding_msg = {
                "role": "system",
                "content": (
                    "Grounding context follows. Use it if relevant; otherwise answer from general knowledge. "
                    "Do not invent details.\n\n"
                    f"{rag_context}"
                )
            }

        messages = [system_style]
        if grounding_msg:
            messages.append(grounding_msg)
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.2},
        )
        # Support both dict and attribute return styles
        reply = getattr(getattr(response, "message", {}), "content", None)
        if reply is None:
            reply = response.get("message", {}).get("content", "")

        print(f"Assistant: {reply}")
        if citations:
            print("Sources:")
            for idx, src in citations:
                print(f"  [{idx}] {src}")

        speak(reply)

        history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ])


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Local LLM chatbot with Whisper STT, KittenTTS TTS, and optional RAG.")
    p.add_argument("--urls", type=str, default="", help="Comma-separated list of URLs to index.")
    p.add_argument("--folder", type=str, default="", help="Path to folder with .txt/.md files.")
    p.add_argument("--model", type=str, default=DEFAULT_LLM, help="Ollama LLM name.")
    p.add_argument("--embed-model", type=str, default=DEFAULT_EMBED, help="Ollama embedding model name.")
    p.add_argument("--top-k", type=int, default=TOP_K, help="Number of chunks to retrieve.")
    p.add_argument("--max-context-chars", type=int, default=MAX_CONTEXT_CHARS, help="Max characters of injected context.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    urls = [u.strip() for u in args.urls.split(",") if u.strip()] if args.urls else []
    folder = args.folder if args.folder else ""

    # Build RAG index if sources are provided
    rag_index = None
    if urls or folder:
        rag_index = RAGIndex(embed_model=args.embed_model)
        rag_index.build(urls=urls, folder=folder)
    else:
        print("No RAG sources provided. Proceeding without retrieval.")

    try:
        converse(model=args.model, rag=rag_index)
    except KeyboardInterrupt:
        print("\nSession terminated.")
