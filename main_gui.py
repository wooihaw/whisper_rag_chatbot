#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI Chatbot with local LLM (Ollama), Whisper STT, KittenTTS TTS, and optional RAG.
GUI built with FreeSimpleGUI. Options are accessed via the menu.

This version adds a "Demo Mode" that reads a text file of Q&A pairs in the
format:

Question: ...
Answer: ...

Question: ...
Answer: ...

When Demo Mode is enabled, the app:
- Randomly narrates each pair with different voices for Q and A.
- Shows all pairs in the chat window.
- Disables Send/Record/input and demo file picker until Demo Mode is disabled.
"""

from __future__ import annotations

import os
import re
import threading
import time
import random
from typing import Any, Dict, List, Tuple

from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import whisper
from bs4 import BeautifulSoup
from kittentts import KittenTTS
from ollama import Client
import FreeSimpleGUI as sg

# -------------------------
# Global defaults
# -------------------------
CONFIG_PATH = '.rag_chat_gui_config.json'
sg.theme('DarkBlue14')
THEME_BG = '#0f172a'
THEME_FG = '#e5e7eb'
THEME_BG_2 = '#111827'
THEME_FG_2 = '#e5e7eb'
# --- UI polish (added earlier) ---
ASSETS_DIR = "assets"
SEND_IMG  = os.path.join(ASSETS_DIR, "send_02.png")
REC_IMG   = os.path.join(ASSETS_DIR, "record_02.png")
STOP_IMG  = os.path.join(ASSETS_DIR, "stop_02.png")
HEADER_IMG = os.path.join(ASSETS_DIR, "Sea-Turtle-Banner.png")
# -------------------------

# Chat font configuration
CHAT_FONT_FAMILY = 'Segoe UI Emoji'  # Emoji-capable font
CHAT_FONT_SIZE = 20

SAMPLE_RATE = 16000
DEFAULT_LLM = "sea_turtle_llama3_2_3b_q4_k_m_v5"
DEFAULT_EMBED = "nomic-embed-text"
TOP_K = 4
MAX_CONTEXT_CHARS = 1800
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
REQUEST_TIMEOUT = 20
DEFAULT_VOICE = "expr-voice-3-f"  # Adjust to your voice id

# --- Demo Mode defaults ---
DEMO_Q_VOICE_DEFAULT = "expr-voice-2-m"
DEMO_A_VOICE_DEFAULT = "expr-voice-2-f"
DEMO_GAP_QA_DEFAULT = 3.0   # seconds between Q and A
DEMO_GAP_PAIR_DEFAULT = 6.0 # seconds between pairs
PAIR_REGEX = re.compile(
    r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?=\n\s*\n|$)",
    flags=re.IGNORECASE | re.DOTALL,
)

def load_config() -> dict:
    try:
        if os.path.isfile(CONFIG_PATH):
            import json
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_config(cfg: dict) -> None:
    try:
        import json
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

# -------------------------
# LLM / STT / TTS init
# -------------------------
client = Client(host="http://127.0.0.1:11434")

sg.popup_quick_message("Loading MMU Chatbot…", auto_close=True, non_blocking=True, background_color="black", text_color="white")
whisper_model = whisper.load_model("small.en")

ktts = KittenTTS("KittenML/kitten-tts-nano-0.1")

# -------------------------
# Text utils
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

def embed_texts(texts: List[str], embed_model: str) -> np.ndarray:
    vecs = []
    for t in texts:
        r = client.embeddings(model=embed_model, prompt=t)
        vec = r["embedding"] if isinstance(r, dict) else getattr(r, "embedding", None)
        if vec is None:
            raise RuntimeError("Failed to compute embeddings from Ollama client response.")
        vecs.append(vec)
    return np.array(vecs, dtype=np.float32)

# -------------------------
# RAG utilities
# -------------------------
def load_url(url: str) -> str:
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "RAGBot/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="n")
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

class RAGIndex:
    def __init__(self, embed_model: str):
        self.embed_model = embed_model
        self.chunks: List[Dict[str, Any]] = []
        self.matrix: np.ndarray | None = None

    def clear(self):
        self.chunks = []
        self.matrix = None

    def build(self, urls: List[str] | None, folder: str | None, window: sg.Window | None = None):
        corpus_chunks: List[Dict[str, Any]] = []

        if urls:
            for u in urls:
                raw = load_url(u)
                if not raw:
                    continue
                for i, ch in enumerate(chunk_text(raw)):
                    corpus_chunks.append({"text": ch, "source": u, "id": f"url::{u}::chunk{i}"})

        if folder:
            for path, txt in load_folder_texts(folder):
                for i, ch in enumerate(chunk_text(txt)):
                    corpus_chunks.append({"text": ch, "source": path, "id": f"file::{path}::chunk{i}"})

        if not corpus_chunks:
            self.clear()
            return 0

        texts = [c["text"] for c in corpus_chunks]
        if window:
            window["-STATUS-"].update("Embedding chunks…")
            window.refresh()

        mat = embed_texts(texts, self.embed_model)
        self.chunks = corpus_chunks
        self.matrix = mat
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int, max_context_chars: int) -> Tuple[str, List[Tuple[int, str]]]:
        if self.matrix is None or not self.chunks:
            return "", []

        qvec = embed_texts([query], self.embed_model)[0]
        denom = (np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(qvec) + 1e-12)
        sims = (self.matrix @ qvec) / denom
        order = np.argsort(-sims)[:max(1, top_k)]

        context_parts: List[str] = []
        citations: List[Tuple[int, str]] = []
        total = 0
        rank = 1
        for idx in order:
            piece = self.chunks[idx]["text"].strip()
            src = self.chunks[idx]["source"]
            if total + len(piece) + 10 > max_context_chars:
                remaining = max_context_chars - total - 10
                if remaining > 50:
                    piece = piece[:remaining]
                else:
                    break
            context_parts.append(f"[{rank}] {piece}")
            citations.append((rank, src))
            total += len(piece) + 10
            rank += 1

        return "\n\n".join(context_parts), citations

# -------------------------
# Audio recorder (start/stop)
# -------------------------
class AudioRecorder:
    def __init__(self, samplerate: int = SAMPLE_RATE):
        self.samplerate = samplerate
        self.frames: List[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.lock = threading.Lock()
        self.recording = False

    def _callback(self, indata, frames, time_info, status):
        with self.lock:
            self.frames.append(indata.copy())

    def start(self):
        if self.recording:
            return
        self.frames = []
        self.stream = sd.InputStream(samplerate=self.samplerate, channels=1, callback=self._callback)
        self.stream.start()
        self.recording = True

    def stop(self) -> np.ndarray:
        if not self.recording:
            return np.zeros((0, 1), dtype=np.float32)
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None
            self.recording = False

        with self.lock:
            if not self.frames:
                return np.zeros((0, 1), dtype=np.float32)
            audio = np.concatenate(self.frames, axis=0)
            return audio

# -------------------------
# Chat logic
# -------------------------
def build_messages(user_text: str,
                   history: List[Dict[str, str]],
                   rag_context: str | None) -> List[Dict[str, str]]:
    system_style = {
        "role": "system",
        "content": "You are a helpful assistant about sea turtles. Answer must be one short sentence with not more than 30 words. Be factual and concise."
    }
    msgs = [system_style]
    if rag_context:
        msgs.append({
            "role": "system",
            "content": (
                "Grounding context follows. Use it if relevant; otherwise answer from general knowledge. "
                "Do not invent details.\n\n" + rag_context
            )
        })
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_text})
    return msgs

def ollama_chat(model: str, messages: List[Dict[str, str]]) -> str:
    resp = client.chat(model=model, messages=messages, options={"temperature": 0.2})
    msg = getattr(getattr(resp, "message", {}), "content", None)
    if msg is None:
        msg = resp.get("message", {}).get("content", "")
    else:
        msg = shorten_text_by_sentence(msg)
    return msg or ""

def transcribe_audio(audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""
    result = whisper_model.transcribe(audio.flatten(), language="en", fp16=False)
    text = result.get("text", "").strip()
    return text

def shorten_text_by_sentence(text: str) -> str:
    """
    Shortens a string of text to 50 words or less by removing full sentences
    from the end.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    word_count = len(text.split())
    while word_count > 50 and len(sentences) > 1:
        sentences.pop()
        shortened_text = ' '.join(sentences)
        word_count = len(shortened_text.split())
    return ' '.join(sentences).strip()

def speak(text: str, voice: str = DEFAULT_VOICE):
    """Blocking TTS using KittenTTS at 24 kHz."""
    try:
        sd.stop()
        wav = ktts.generate(text, voice=voice)
        sd.play(wav, samplerate=24000)
        sd.wait()
    except Exception as e:
        sg.popup("TTS error", f"{e}", keep_on_top=True)

def speak_async(text: str, voice: str, window: sg.Window, state: 'AppState'):
    state.speaking = True
    try:
        window['-SEND-'].update(disabled=True)
    except Exception:
        pass
    try:
        window['-REC-'].update(disabled=True)
    except Exception:
        pass
    def _worker():
        try:
            speak(text, voice)
        except Exception:
            pass
        finally:
            window.write_event_value('-TTS_DONE-', None)
    th = threading.Thread(target=_worker, daemon=True)
    th.start()

# -------------------------
# Demo Mode helpers
# -------------------------
def load_demo_pairs(path: str) -> List[Tuple[str, str]]:
    """Read and parse Q&A pairs from a text file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Q&A file not found: {p}")
    text = p.read_text(encoding='utf-8')
    matches = PAIR_REGEX.findall(text)
    pairs: List[Tuple[str,str]] = []
    for q, a in matches:
        q = q.strip()
        a = a.strip()
        if q and a:
            pairs.append((q, a))
    if not pairs:
        raise ValueError("No valid pairs found. Use 'Question:' and 'Answer:' labels with blank line between pairs.")
    return pairs

def safe_sleep(seconds: float, stop_event: threading.Event) -> bool:
    """Sleep in small steps so we can exit early when stop_event is set."""
    t = 0.0
    step = 0.1
    while t < seconds:
        if stop_event.is_set():
            return False
        time.sleep(step)
        t += step
    return True

# -------------------------
# App state
# -------------------------
class AppState:
    def __init__(self):
        self.urls: List[str] = []
        self.show_sources: bool = False
        self.speaking: bool = False

        self.folder: str | None = None
        self.model: str = DEFAULT_LLM
        self.embed_model: str = DEFAULT_EMBED
        self.top_k: int = TOP_K
        self.max_context_chars: int = MAX_CONTEXT_CHARS
        self.voice: str = DEFAULT_VOICE
        self.auto_speak: bool = True

        self.rag = RAGIndex(embed_model=self.embed_model)
        self.history: List[Dict[str, str]] = []

        # --- Demo Mode fields ---
        self.demo_enabled: bool = False
        self.demo_file: str | None = None
        self.demo_pairs: List[Tuple[str,str]] = []
        self.demo_q_voice: str = DEMO_Q_VOICE_DEFAULT
        self.demo_a_voice: str = DEMO_A_VOICE_DEFAULT
        self.demo_gap_qa: float = DEMO_GAP_QA_DEFAULT
        self.demo_gap_pair: float = DEMO_GAP_PAIR_DEFAULT
        self.demo_thread: threading.Thread | None = None
        self.demo_stop_event: threading.Event = threading.Event()

    def rebuild_index(self, window: sg.Window) -> int:
        self.rag = RAGIndex(embed_model=self.embed_model)
        return self.rag.build(self.urls, self.folder, window)

    def to_config(self) -> dict:
        return {
            'urls': self.urls,
            'folder': self.folder,
            'model': self.model,
            'embed_model': self.embed_model,
            'top_k': self.top_k,
            'max_context_chars': self.max_context_chars,
            'voice': self.voice,
            'auto_speak': self.auto_speak,
            'show_sources': self.show_sources,
            # Demo
            'demo_file': self.demo_file,
            'demo_q_voice': self.demo_q_voice,
            'demo_a_voice': self.demo_a_voice,
            'demo_gap_qa': self.demo_gap_qa,
            'demo_gap_pair': self.demo_gap_pair,
        }

    def apply_config(self, cfg: dict):
        self.urls = list(cfg.get('urls', []))
        self.folder = cfg.get('folder')
        self.model = cfg.get('model', self.model)
        self.embed_model = cfg.get('embed_model', self.embed_model)
        self.top_k = int(cfg.get('top_k', self.top_k))
        self.max_context_chars = int(cfg.get('max_context_chars', self.max_context_chars))
        self.voice = cfg.get('voice', self.voice)
        self.auto_speak = bool(cfg.get('auto_speak', self.auto_speak))
        self.show_sources = bool(cfg.get('show_sources', self.show_sources))
        # Demo
        self.demo_file = cfg.get('demo_file', self.demo_file)
        self.demo_q_voice = cfg.get('demo_q_voice', self.demo_q_voice)
        self.demo_a_voice = cfg.get('demo_a_voice', self.demo_a_voice)
        self.demo_gap_qa = float(cfg.get('demo_gap_qa', self.demo_gap_qa))
        self.demo_gap_pair = float(cfg.get('demo_gap_pair', self.demo_gap_pair))

# -------------------------
# GUI helpers
# -------------------------
def dialog_edit_urls(urls: List[str]) -> List[str] | None:
    layout = [
        [sg.Text("Enter one URL per line:")],
        [sg.Multiline("\n".join(urls), size=(80, 15), key="-URLS-")],
        [sg.Button("Save"), sg.Button("Cancel")]
    ]
    win = sg.Window("Configure URLs", layout, modal=True, finalize=True)
    event, values = win.read()
    if event == "Save":
        raw = values["-URLS-"]
        new_urls = [u.strip() for u in raw.splitlines() if u.strip()]
        win.close()
        return new_urls
    win.close()
    return None

def dialog_select_folder(current: str | None) -> str | None:
    folder = sg.popup_get_folder("Select folder with .txt/.md files", default_path=current or "", keep_on_top=True)
    return folder

def dialog_models(current_llm: str, current_embed: str) -> Tuple[str, str] | None:
    try:
        listed = client.list()
        models = []
        if isinstance(listed, dict) and "models" in listed:
            models = [m.get("name", "") for m in listed["models"] if m.get("name")]
        elif hasattr(listed, "models"):
            models = [getattr(m, "name", "") for m in listed.models]
        models = sorted(set(models))
    except Exception:
        models = []

    llm_combo = sg.Combo(values=models or [], default_value=current_llm, size=(40, 1), key="-LLM-", readonly=False)
    emb_combo = sg.Combo(values=models or [], default_value=current_embed, size=(40, 1), key="-EMB-", readonly=False)

    layout = [
        [sg.Text("LLM model (Ollama):")],
        [llm_combo],
        [sg.Text("Embedding model (Ollama):")],
        [emb_combo],
        [sg.Button("Save"), sg.Button("Cancel")]
    ]
    win = sg.Window("Model Settings", layout, modal=True, finalize=True)
    event, values = win.read()
    if event == "Save":
        llm = values["-LLM-"].strip() or current_llm
        emb = values["-EMB-"].strip() or current_embed
        win.close()
        return llm, emb
    win.close()
    return None

def dialog_retrieval(top_k: int, max_chars: int) -> Tuple[int, int] | None:
    layout = [
        [sg.Text("Top-K chunks:"), sg.Input(str(top_k), key="-K-", size=(6, 1))],
        [sg.Text("Max context characters:"), sg.Input(str(max_chars), key="-MC-", size=(8, 1))],
        [sg.Button("Save"), sg.Button("Cancel")]
    ]
    win = sg.Window("Retrieval Settings", layout, modal=True, finalize=True)
    event, values = win.read()
    if event == "Save":
        try:
            k = max(1, int(values["-K-"]))
            mc = max(200, int(values["-MC-"]))
            win.close()
            return k, mc
        except Exception:
            sg.popup("Invalid values", keep_on_top=True)
    win.close()
    return None

def dialog_tts(voice: str, auto: bool) -> Tuple[str, bool] | None:
    layout = [
        [sg.Text("Voice id / name:"), sg.Input(voice, key="-V-", size=(30, 1))],
        [sg.Checkbox("Speak responses automatically", default=auto, key="-AUTO-")],
        [sg.Button("Save"), sg.Button("Cancel")]
    ]
    win = sg.Window("TTS Settings", layout, modal=True, finalize=True)
    event, values = win.read()
    if event == "Save":
        v = values["-V-"].strip() or voice
        a = bool(values["-AUTO-"])
        win.close()
        return v, a
    win.close()
    return None

# -------------------------
# GUI layout
# -------------------------
def build_main_window() -> sg.Window:
    button_size = (10, 1)
    menu_def = [
        ["File", ["Save Log", "---", "Exit"]],
        ["RAG", ["Configure URLs", "Select Folder", "Build Index", "Clear Index"]],
        ["Settings", ["Model…", "Retrieval…", "TTS…"]],
        ["View", ["Show Sources"]],
        ["Help", ["About"]],
    ]

    use_icons = all(os.path.isfile(p) for p in (SEND_IMG, REC_IMG, STOP_IMG))
    header_widget = sg.Image(HEADER_IMG, pad=(0, 6), expand_x=True, expand_y=True) if os.path.isfile(HEADER_IMG) else sg.Text("Sea Turtle Tutor", font=("Segoe UI", 18, "bold"))

    if use_icons:
        send_btn = sg.Button(
            "",
            key="-SEND-",
            image_filename=SEND_IMG,
            button_color=(sg.theme_background_color(), sg.theme_background_color()),
            border_width=0,
            tooltip="Send",
            bind_return_key=True,
        )
        rec_btn = sg.Button(
            "",
            key="-REC-",
            image_filename=REC_IMG,
            button_color=(sg.theme_background_color(), sg.theme_background_color()),
            border_width=0,
            tooltip="Record / Stop",
        )
    else:
        send_btn = sg.Button("Send", key="-SEND-", size=button_size, button_color=("white", "blue"), bind_return_key=True)
        rec_btn  = sg.Button("Record", key="-REC-", size=button_size, button_color=("white", "green"))

    # --- Demo controls row ---
    demo_row1 = [
        sg.Button("Choose demo Q&A file", key="-DEMO_CHOOSE-", button_color=("white", "#1f2937")),
        sg.Text("(no file)", key="-DEMO_FILE_LABEL-", size=(30,1)),
    ]

    demo_row2 = [
        sg.Text("AI Chatbot can make mistakes, so double-check it", font=('Times New Roman', 14, 'italic')),
    ]

    demo_row3 = [
        sg.Button("Start Demo", key="-DEMO_TOGGLE-", button_color=("white", "#33a821"), font=('Helvetica', 16, 'bold')),
    ]

    chat_col = [
        [header_widget],
        [sg.Multiline("", size=(80, 12), key="-CHAT-", autoscroll=True, disabled=True, expand_x=True, expand_y=True,
                      background_color=THEME_BG, text_color=THEME_FG, font=(CHAT_FONT_FAMILY, CHAT_FONT_SIZE))],
        [sg.Input("", key="-INPUT-", expand_x=True, focus=True, font=(CHAT_FONT_FAMILY, CHAT_FONT_SIZE)), send_btn, rec_btn],
        [
            sg.Column([demo_row1], justification='left', element_justification='left', expand_x=True),
            sg.Column([demo_row2], justification='left', element_justification='left', expand_x=True),
            sg.Column([demo_row3], justification='right', element_justification='right', expand_x=True),
        ],
        [sg.Multiline("", size=(80, 3), key="-SOURCES-", disabled=True, expand_x=True, expand_y=False,
                      background_color=THEME_BG_2, text_color=THEME_FG_2, visible=False)],
        [sg.StatusBar("Ready.", key="-STATUS-", text_color=THEME_FG, background_color=THEME_BG)]
    ]

    layout = [
        [sg.Menu(menu_def, visible=False)],
        [sg.Column(chat_col, expand_x=True, expand_y=True)],
    ]

    window = sg.Window("MMU Sea Turtle Chatbot", layout, resizable=True, finalize=True)
    try:
        window.metadata = {"use_icons": use_icons}
    except Exception:
        window.metadata = {"use_icons": False}
    window.bind("<Escape>", "ESC")
    window.TKroot.attributes('-fullscreen', True)

    return window

# -------------------------
# Main event loop
# -------------------------
def main():
    state = AppState()
    recorder = AudioRecorder()

    window = build_main_window()
    chat = window["-CHAT-"]
    sources = window["-SOURCES-"]
    status = window["-STATUS-"]

    # Load previous settings
    cfg = load_config()
    if cfg:
        state.apply_config(cfg)
        try:
            sources.update(visible=state.show_sources)
        except Exception:
            pass
        status.update('Settings loaded.')
        if state.demo_file:
            try:
                window["-DEMO_FILE_LABEL-"].update(Path(state.demo_file).name)
            except Exception:
                window["-DEMO_FILE_LABEL-"].update("(no file)")

    # Auto-build index if saved folder valid
    try:
        valid_folder = isinstance(state.folder, str) and os.path.isdir(state.folder)
        if valid_folder:
            status.update('Building index from saved folder…')
            window.refresh()
            count = state.rebuild_index(window)
            if count > 0:
                status.update(f'Index ready. {count} chunks.')
            else:
                status.update('No chunks found in saved folder.')
    except Exception:
        status.update('Ready.')

    def append_chat(prefix: str, text: str, textcolor:str=None):
        chat.update(disabled=False)
        chat.print(f"{prefix}: {text}", text_color=textcolor)
        chat.update(disabled=True)

    def set_demo_controls(running: bool):
        """Enable/disable widgets during demo."""
        state.demo_enabled = running
        try:
            window['-SEND-'].update(disabled=running)
        except Exception:
            pass
        try:
            window['-REC-'].update(disabled=running)
        except Exception:
            pass
        try:
            window['-INPUT-'].update(disabled=running)
        except Exception:
            pass
        try:
            window['-DEMO_CHOOSE-'].update(disabled=running)
        except Exception:
            pass
        try:
            window['-DEMO_TOGGLE-'].update(
                text="Stop Demo" if running else "Start Demo",
                button_color=("white", "#b91c1c") if running else ("white", "#33a821")
            )
        except Exception:
            pass

    def start_demo_thread():
        """Launch demo worker in a background thread."""
        state.demo_stop_event.clear()

        def _worker():
            try:
                # Load pairs if not already cached or file changed
                if not state.demo_pairs:
                    state.demo_pairs = load_demo_pairs(state.demo_file)
                # Show all pairs in chat (requirement)
                # batch_lines = []
                # for i, (q, a) in enumerate(state.demo_pairs, 1):
                #     batch_lines.append(f"[Pair {i}] Q: {q}")
                #     batch_lines.append(f"[Pair {i}] A: {a}")
                # window.write_event_value("-DEMO_PRINT_BATCH-", "\n".join(batch_lines))

                order = state.demo_pairs[:]
                random.shuffle(order)

                for idx, (q, a) in enumerate(order, 1):
                    if state.demo_stop_event.is_set():
                        break
                    window.write_event_value("-DEMO_PRINT-", ("🧑 Q", q, 'yellow'))
                    try:
                        speak('...' + q, voice=state.demo_q_voice)
                    except Exception:
                        pass
                    if not safe_sleep(state.demo_gap_qa, state.demo_stop_event):
                        break

                    if state.demo_stop_event.is_set():
                        break
                    window.write_event_value("-DEMO_PRINT-", ("🤖 A", a + '\n', 'white'))
                    try:
                        speak('...' + a + '...', voice=state.demo_a_voice)
                    except Exception:
                        pass

                    if idx < len(order):
                        if not safe_sleep(state.demo_gap_pair, state.demo_stop_event):
                            break
            finally:
                window.write_event_value("-DEMO_DONE-", None)

        th = threading.Thread(target=_worker, daemon=True)
        state.demo_thread = th
        th.start()

    while True:
        event, values = window.read(timeout=200)

        if event in (sg.WIN_CLOSED, "Exit"):
            try:
                save_config(state.to_config())
            except Exception:
                pass
            # Stop recorder if running
            if recorder.recording:
                recorder.stop()
            # Stop demo if running
            if state.demo_enabled:
                state.demo_stop_event.set()
                try:
                    sd.stop()
                except Exception:
                    pass
            break

        if event == "About":
            sg.popup("MMU Sea Turtle Chatbot\nOllama LLM + RAG\nWhisper STT + KittenTTS TTS\nFreeSimpleGUI interface.\nDemo Mode enabled.", keep_on_top=True)

        elif event == "Save Log":
            path = sg.popup_get_file("Save chat log", save_as=True, no_window=True, default_extension=".txt",
                                     file_types=(("Text", "*.txt"),))
            if path:
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(chat.get())
                    sg.popup("Saved.", keep_on_top=True)
                except Exception as e:
                    sg.popup("Save failed", f"{e}", keep_on_top=True)

        elif event == "Configure URLs":
            updated = dialog_edit_urls(state.urls)
            save_config(state.to_config())
            if updated is not None:
                state.urls = updated
                status.update(f"URLs set: {len(state.urls)}")
                save_config(state.to_config())

        elif event == "Select Folder":
            folder = dialog_select_folder(state.folder)
            save_config(state.to_config())
            if folder:
                state.folder = folder
                status.update(f"Folder set: {state.folder}")
                save_config(state.to_config())

        elif event == "Build Index":
            status.update("Building index…")
            window.refresh()
            try:
                count = state.rebuild_index(window)
                if count == 0:
                    sg.popup("No sources found. Add URLs or a folder.", keep_on_top=True)
                    status.update("No sources. Retrieval disabled.")
                else:
                    status.update(f"Index ready. {count} chunks.")
                save_config(state.to_config())
            except Exception as e:
                sg.popup("Indexing failed", f"{e}", keep_on_top=True)
                status.update("Indexing failed.")

        elif event == "Clear Index":
            state.rag.clear()
            status.update("Index cleared.")

        elif event == "Model…":
            out = dialog_models(state.model, state.embed_model)
            save_config(state.to_config())
            if out:
                state.model, state.embed_model = out
                status.update(f"Models set. LLM={state.model}, EMB={state.embed_model}")
                save_config(state.to_config())

        elif event == "Retrieval…":
            out = dialog_retrieval(state.top_k, state.max_context_chars)
            save_config(state.to_config())
            if out:
                state.top_k, state.max_context_chars = out
                status.update(f"Retrieval set. top_k={state.top_k}, max_chars={state.max_context_chars}")
                save_config(state.to_config())

        elif event == "TTS…":
            out = dialog_tts(state.voice, state.auto_speak)
            save_config(state.to_config())
            if out:
                state.voice, state.auto_speak = out
                status.update("TTS settings updated.")
                save_config(state.to_config())

        elif event == "Show Sources":
            state.show_sources = not state.show_sources
            sources.update(visible=state.show_sources)
            status.update("Sources shown." if state.show_sources else "Sources hidden.")
            save_config(state.to_config())

        # --- Demo Mode events ---
        elif event == "-DEMO_CHOOSE-":
            path = sg.popup_get_file("Choose Q&A text file", file_types=(("Text", "*.txt"),), keep_on_top=True)
            if path:
                try:
                    # Validate file by loading pairs (do not cache yet)
                    _pairs = load_demo_pairs(path)
                    state.demo_file = path
                    state.demo_pairs = _pairs  # cache after successful parse
                    window["-DEMO_FILE_LABEL-"].update(Path(path).name)
                    status.update(f"Loaded Q&A file with {len(_pairs)} pairs.")
                    save_config(state.to_config())
                except Exception as e:
                    sg.popup("Invalid Q&A file", f"{e}", keep_on_top=True)
                    status.update("Ready.")

        elif event == "-DEMO_TOGGLE-":
            if not state.demo_enabled:
                if not state.demo_file:
                    sg.popup("Please choose a Q&A text file first.", keep_on_top=True)
                    continue
                try:
                    # Ensure pairs are available
                    if not state.demo_pairs:
                        state.demo_pairs = load_demo_pairs(state.demo_file)
                except Exception as e:
                    sg.popup("Cannot start demo", f"{e}", keep_on_top=True)
                    continue
                # If recorder is active, stop it
                if recorder.recording:
                    _ = recorder.stop()
                    try:
                        window["-REC-"].update(image_filename=REC_IMG) if window.metadata.get("use_icons") else window["-REC-"].update("Record", button_color=("white", "green"))
                    except Exception:
                        pass
                set_demo_controls(True)
                status.update(f"Demo running… {len(state.demo_pairs)} pairs.")
                start_demo_thread()
            else:
                # Request stop
                state.demo_stop_event.set()
                try:
                    sd.stop()  # interrupt current TTS
                except Exception:
                    pass
                status.update("Stopping demo…")

        elif event == "-DEMO_PRINT-":
            prefix, text, textcolor = values["-DEMO_PRINT-"]
            append_chat(prefix, text, textcolor)

        elif event == "-DEMO_PRINT_BATCH-":
            batch_text = values["-DEMO_PRINT_BATCH-"]
            append_chat("📄 Pairs loaded", "\n" + batch_text + "\n")

        elif event == "-DEMO_DONE-":
            set_demo_controls(False)
            status.update("Demo stopped.")
            # Keep pairs cached for quick restart

        elif event == '-TTS_DONE-':
            state.speaking = False
            try:
                if not recorder.recording:
                    window['-SEND-'].update(disabled=False)
                window['-REC-'].update(disabled=False)
            except Exception:
                pass
            status.update('Ready.')

        elif event == "-REC-":
            if state.demo_enabled:
                continue  # ignore during demo
            if not recorder.recording:
                try:
                    recorder.start()
                    window['-SEND-'].update(disabled=True)
                    window["-REC-"].update(image_filename=STOP_IMG) if window.metadata.get("use_icons") else window["-REC-"].update("Stop", button_color=("white", "firebrick3"))
                    status.update("Recording… Press Stop or Esc.")
                except Exception as e:
                    sg.popup("Cannot start recording", f"{e}", keep_on_top=True)
                    status.update("Ready.")
            else:
                audio = recorder.stop()
                window['-SEND-'].update(disabled=False if not getattr(state, 'speaking', False) else True)
                window["-REC-"].update(image_filename=REC_IMG) if window.metadata.get("use_icons") else window["-REC-"].update("Record", button_color=("white", "green"))
                status.update("Transcribing…")
                window.refresh()
                try:
                    text = transcribe_audio(audio)
                except Exception as e:
                    text = ""
                    sg.popup("Transcription error", f"{e}", keep_on_top=True)
                if text:
                    window["-INPUT-"].update(text)
                    window.write_event_value("-SEND-", None)  # auto-send transcribed text
                    window.write_event_value("-SEND-", None)  # duplicated per previous behavior
                    status.update("Ready.")
                else:
                    status.update("No speech detected.")

        elif event == "ESC":
            if state.demo_enabled:
                continue  # ignore during demo
            if recorder.recording:
                audio = recorder.stop()
                window['-SEND-'].update(disabled=False if not getattr(state, 'speaking', False) else True)
                window["-REC-"].update(image_filename=REC_IMG) if window.metadata.get("use_icons") else window["-REC-"].update("Record", button_color=("white", "green"))
                status.update("Transcribing…")
                window.refresh()
                try:
                    text = transcribe_audio(audio)
                except Exception as e:
                    text = ""
                    sg.popup("Transcription error", f"{e}", keep_on_top=True)
                if text:
                    window["-INPUT-"].update(text)
                    window.write_event_value("-SEND-", None)  # auto-send transcribed text
                status.update("Ready.")
            elif not recorder.recording:
                try:
                    recorder.start()
                    window['-SEND-'].update(disabled=True)
                    window["-REC-"].update(image_filename=STOP_IMG) if window.metadata.get("use_icons") else window["-REC-"].update("Stop", button_color=("white", "firebrick3"))
                    status.update("Recording… Press Stop or Esc.")
                except Exception as e:
                    sg.popup("Cannot start recording", f"{e}", keep_on_top=True)
                    status.update("Ready.")

        elif event == "-SEND-":
            if state.demo_enabled:
                continue  # ignore during demo
            user_text = values.get("-INPUT-", "").strip()
            if not user_text:
                continue

            append_chat("🧑 Q", user_text, 'yellow')
            window["-INPUT-"].update("")
            sources.update("")
            status.update("Thinking…")
            window.refresh()

            try:
                rag_context, cites = state.rag.retrieve(user_text, state.top_k, state.max_context_chars)
            except Exception:
                rag_context, cites = ("", [])

            messages = build_messages(user_text, state.history, rag_context or None)
            try:
                reply = ollama_chat(state.model, messages)
            except Exception as e:
                reply = f"(LLM error: {e})"

            append_chat("🤖 A", reply + '\n')
            window.refresh()

            if cites and state.show_sources:
                src_text = "Sources:\n" + "\n".join([f"  [{i}] {s}" for i, s in cites])
                sources.update(src_text)
            else:
                sources.update("")

            if state.auto_speak and reply.strip():
                speak_async(reply + '...', voice=state.voice, window=window, state=state)

            state.history.extend([{"role": "user", "content": user_text},
                                  {"role": "assistant", "content": reply}])

            status.update("Ready.")

    window.close()

if __name__ == "__main__":
    main()
