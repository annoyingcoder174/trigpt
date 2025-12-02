# src/api.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any
import io
import time
import json
import os


import cv2
import numpy as np

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from PIL import Image

# Optional HEIC/HEIF support (requires pillow-heif to be installed)
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    print("HEIC/HEIF support enabled in api.py.")
except ImportError:
    print("pillow-heif not installed; HEIC/HEIF images may not work in image endpoints.")

from .llm_client import LLMClient
from .doc_ingestion import load_pdf_text, load_pdf_text_from_bytes, chunk_text
from .vector_store import add_document, query_document, list_documents
from .vision_model import FaceClassifier
from .s3_storage import upload_pdf_bytes
from .question_classifier import QuestionTypeClassifier
from .reranker import Reranker
from .emotion_model import EmotionClassifier
from .object_detector import ObjectDetector
from .identity_db import IDENTITY_DB, get_identity_summary, render_profile_details
from .irene_style_examples import STYLE_EXAMPLES
from .irene_dynamic_examples import load_dynamic_style_examples

app = FastAPI(title="TriGPT API", version="0.3.0")

llm_client = LLMClient()

# ----- CORS (allow frontend to call this API) -----
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # later you can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Identity summaries ----------

IDENTITY_INFO: Dict[str, Optional[str]] = {
    label: get_identity_summary(label) for label in IDENTITY_DB.keys()
}

# Style examples: static (hand-written) + dynamic (from feedback)
DYNAMIC_STYLE_EXAMPLES = load_dynamic_style_examples()

# Feedback logs
FEEDBACK_FILE = Path("data/chat_feedback.jsonl")

# ---------- Model loading ----------

try:
    face_classifier = FaceClassifier(checkpoint_path="models/face_classifier.pt")
    print("✅ Face classifier loaded for API.")
except FileNotFoundError:
    face_classifier = None
    print("⚠️ Face classifier checkpoint not found. /predict_face, /secure_ask and identity features limited.")

try:
    question_classifier = QuestionTypeClassifier(model_dir="models/question_classifier")
    print("✅ Question-type classifier loaded for API.")
except FileNotFoundError:
    question_classifier = None
    print("⚠️ Question-type classifier not found. Intent detection disabled.")

try:
    reranker = Reranker()
    print("✅ Reranker loaded for API.")
except Exception as e:
    reranker = None
    print(f"⚠️ Reranker not available: {e}")

try:
    emotion_classifier = EmotionClassifier(checkpoint_path="models/emotion_classifier.pt")
    print("✅ Emotion classifier loaded for API.")
except FileNotFoundError:
    emotion_classifier = None
    print("⚠️ Emotion classifier checkpoint not found. /live_emotion and emotion in /vision_qa limited.")

try:
    object_detector = ObjectDetector(score_thresh=0.7)
    print("✅ Object detector loaded for API (score_thresh=0.7).")
except Exception as e:
    object_detector = None
    print(f"⚠️ Object detector not available: {e}")

# ---------- Vision / identity thresholds ----------

# Below this, we **do not trust** the identity prediction
IDENTITY_CONF_THRESHOLD: float = 0.70


def postprocess_identity_label(
    raw_label: str | None, raw_conf: float | None
) -> tuple[str | None, float | None]:
    """
    Apply per-class confidence thresholds and fallback.

    - If confidence is None → ('strangers', None)
    - If label is None → ('strangers', conf)
    - Some classes (like PTri's Muse) require higher confidence.
    """
    if raw_conf is None:
        return "strangers", None

    if raw_label is None:
        return "strangers", raw_conf

    # Per-class stricter thresholds (tune as needed)
    special_thresholds: dict[str, float] = {
        "PTri's Muse": 0.9,  # make Muse harder to trigger to avoid hallucinations
    }

    thr = special_thresholds.get(raw_label, IDENTITY_CONF_THRESHOLD)

    if raw_conf < thr:
        return "strangers", raw_conf

    return raw_label, raw_conf



class IngestRequest(BaseModel):
    pdf_path: str


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5
    doc_id: str | None = None  # optional: ask within one document


class AnswerResponse(BaseModel):
    answer: str
    used_sources: List[str]
    intent: str | None = None
    intent_confidence: float | None = None


class FacePredictionResponse(BaseModel):
    predicted_class: str
    confidence: float  # 0–1


class SecureAskResponse(BaseModel):
    allowed: bool
    identity: str | None
    confidence: float | None
    answer: str | None
    used_sources: List[str] = []
    intent: str | None = None
    intent_confidence: float | None = None
    emotion: str | None = None
    emotion_confidence: float | None = None


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    num_chunks: int


class EmotionPredictionResponse(BaseModel):
    emotion: str
    confidence: float


class VisionQAResponse(BaseModel):
    answer: str
    identity: str | None = None
    identity_confidence: float | None = None
    emotion: str | None = None
    emotion_confidence: float | None = None


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


class DetectionBox(BaseModel):
    # High-level category for the object
    # ex: "human", "car", "dog", "cat", "tree", "phone"
    label: str

    # Raw detector label from the underlying model (e.g. "person", "cell phone")
    raw_label: str | None = None

    score: float
    x: float  # 0-1, left
    y: float  # 0-1, top
    w: float  # 0-1, width
    h: float  # 0-1, height

    # Only filled when label == "human" AND face_classifier is available
    identity: str | None = None  # PTri, Lanh, MTuan, BHa, PTri's Muse, strangers
    identity_confidence: float | None = None

    # Short info string from IDENTITY_INFO
    identity_info: str | None = None


class DetectionResponse(BaseModel):
    detections: List[DetectionBox]


class ChatFeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int  # 1–5
    comment: str | None = None
    mode: str | None = None  # "doc", "global", "general", "vision"


# ---------- Helpers ----------


def is_image_upload(file: UploadFile) -> bool:
    """
    Accept standard image MIME types and common extensions,
    including HEIC/HEIF.
    """
    ct = (file.content_type or "").lower()
    name = (file.filename or "").lower()

    if ct.startswith("image/"):
        return True

    # Fallback by extension (for browsers that send octet-stream)
    if name.endswith((".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")):
        return True

    return False


def build_context_from_query(
    question: str,
    top_k: int = 5,
    doc_id: str | None = None,
):
    """
    Run a vector search for the question.
    If doc_id is provided, search only within that document.
    If reranker is available, rerank the retrieved chunks.
    """
    result = query_document(question, n_results=top_k * 2, doc_id=doc_id)

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]

    if not docs:
        return "NO_RELEVANT_CONTEXT_FOUND", []

    if reranker is not None and len(docs) > 1:
        indices, scores = reranker.rerank(question, docs, top_k=top_k)
        docs = [docs[i] for i in indices]
        metas = [metas[i] for i in indices]
    else:
        docs = docs[:top_k]
        metas = metas[:top_k]

    context_parts: list[str] = []
    sources: list[str] = []

    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        source = meta.get("source", "unknown.pdf")
        chunk_index = meta.get("chunk_index", "?")
        tag = f"[Source {i} – {source}, chunk {chunk_index}]"
        context_parts.append(f"{tag}\n{doc}")
        sources.append(tag)

    return "\n\n".join(context_parts), sources


def ingest_pdf_from_path(pdf_path: Path):
    """
    Shared helper to ingest a PDF at a given path into the vector store.
    Returns doc_id and num_chunks.
    """
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found")

    full_text = load_pdf_text(pdf_path)
    chunks = chunk_text(full_text, chunk_size=800, overlap=200)

    if not chunks:
        raise HTTPException(
            status_code=400, detail="No text could be extracted from the PDF"
        )

    doc_id = pdf_path.stem
    add_document(
        doc_id=doc_id,
        chunks=chunks,
        metadata={"source": pdf_path.name},
    )

    return doc_id, len(chunks)


def looks_vietnamese(text: str) -> bool:
    """
    Rough check if text is likely Vietnamese.
    We use common particles + presence of Vietnamese diacritics.
    """
    if not text:
        return False
    t = text.lower()

    vi_markers = [
        " không",
        " ko ",
        "k0 ",
        "k dc",
        " là ",
        "của",
        "nhé",
        "nhỉ",
        "nha",
        "đi mà",
        " với",
        "vậy",
        "thế",
        "được",
        "mình",
        "cậu",
        "tớ",
        "anh ",
        "chị ",
        "em ",
        "ơi",
        "ảnh",       # "trong ảnh này"
        "trong ảnh", # very common in your use case
    ]

    if any(m in t for m in vi_markers):
        return True

    # Vietnamese accented characters
    vi_chars = (
        "ăâêôơưđ"
        "áàảãạấầẩẫậắằẳẵặ"
        "éèẻẽẹếềểễệ"
        "óòỏõọốồổỗộớờởỡợ"
        "úùủũụứừửữự"
        "íìỉĩị"
        "ýỳỷỹỵ"
    )
    return any(ch in t for ch in vi_chars)


def select_style_examples(
    identity_label: str | None,
    question: str,
    max_examples: int = 4,
):
    """
    Choose a few style examples based on:
    - user language (English vs Vietnamese)
    - identity label if available (PTri, Muse, etc.)
    - static style examples + dynamic ones from feedback
    """
    lang = "vi" if looks_vietnamese(question) else "en"

    static_examples = STYLE_EXAMPLES.get(lang, [])
    dynamic_examples = DYNAMIC_STYLE_EXAMPLES.get(lang, [])

    # dynamic examples all have label=None, behave as general tone examples
    examples_all = static_examples + dynamic_examples
    if not examples_all:
        return lang, []

    label_examples: list[Dict[str, Any]] = []
    other_examples: list[Dict[str, Any]] = []

    for ex in examples_all:
        ex_label = ex.get("label")
        if identity_label is not None and ex_label == identity_label:
            label_examples.append(ex)
        else:
            other_examples.append(ex)

    selected: list[Dict[str, Any]] = []

    # Prioritize examples with matching identity label
    for ex in label_examples:
        if len(selected) >= max_examples:
            break
        selected.append(ex)

    # Then fill with general / other examples
    for ex in other_examples:
        if len(selected) >= max_examples:
            break
        selected.append(ex)

    return lang, selected


def build_system_prompt(
    context: str,
    intent: str | None,
    emotion: str | None = None,
) -> str:
    """
    Build a system prompt for IreneAdler based on detected intent and (optionally) emotion.
    Used for /ask and /secure_ask (document Q&A).
    """
    base = (
        "You are an AI assistant named IreneAdler, part of the local system TriGPT built and owned by PTri.\n"
        "Use the provided document context to answer the question as clearly and concretely as possible.\n"
        "If the answer is clearly not in the context, say that you don't know or that the document doesn't contain the information.\n\n"
        "Language rules:\n"
        "- Detect the user's language from their message.\n"
        "- If the user writes in Vietnamese, reply in natural, fluent Vietnamese "
        "(nhẹ nhàng, gần gũi nhưng rõ ràng, không vòng vo).\n"
        "- Otherwise, reply in the same language the user used.\n\n"
        "Relationship rules:\n"
        "- The people in the database are all related to PTri, not to you personally.\n"
        "- Do NOT say 'my friend', 'my crush', 'my muse', 'my owner', etc.\n"
        "- Instead, say things like 'PTri's friend', 'PTri's muse', 'someone important to PTri'.\n"
        "- When talking about BHa, default to 'a friend of PTri'. Only mention that she used to be a crush "
        "if the user explicitly asks about romance or crushes.\n\n"
        "Answering style:\n"
        "- Be specific and to the point. Avoid vague motivational talk and filler.\n"
        "- If the user asks for facts, give short, factual sentences or structured bullet points.\n"
        "- If you need to say you don't know, just say it directly without over-explaining.\n"
    )

    if intent == "summary":
        style = (
            "Focus on giving a concise summary using short paragraphs or bullet points with key ideas."
        )
    elif intent == "explain":
        style = (
            "Explain concepts in simple language using examples. Assume the user is smart but not an expert."
        )
    elif intent == "compare":
        style = (
            "Compare the key items directly. Highlight similarities and differences, ideally in a structured list."
        )
    elif intent == "definition":
        style = (
            "Start with a short, clear definition in one or two sentences, then add minimal extra context if helpful."
        )
    else:
        style = "Answer in a clear, structured, and helpful way without unnecessary filler."

    emotion_note = ""
    if emotion in {"sad", "tired"}:
        emotion_note = (
            "The user may feel down or tired. Be kind, but still keep the answer short and concrete."
        )
    elif emotion == "happy":
        emotion_note = "The user seems happy. A slightly positive tone is fine."
    elif emotion == "angry":
        emotion_note = (
            "The user may be frustrated. Be calm, empathetic, and focus on giving a direct solution."
        )

    intent_str = intent or "unknown"
    emotion_str = emotion or "unknown"

    return (
        f"{base}\n\n"
        f"Detected intent: {intent_str}.\n"
        f"Detected emotion: {emotion_str}.\n"
        f"{style}\n"
        f"{emotion_note}\n\n"
        f"Document context:\n{context}"
    )


# ---------- Routes ----------


@app.get("/")
def root():
    return {"message": "TriGPT API is running. Go to /docs to try it."}


@app.get("/documents", response_model=List[DocumentInfo])
def get_documents():
    """
    List all ingested documents in the vector store.
    """
    docs = list_documents()
    return [DocumentInfo(**d) for d in docs]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "face_classifier": face_classifier is not None,
        "question_classifier": question_classifier is not None,
        "emotion_classifier": emotion_classifier is not None
        if "emotion_classifier" in globals()
        else False,
        "reranker": reranker is not None if "reranker" in globals() else False,
        "object_detector": object_detector is not None
        if "object_detector" in globals()
        else False,
    }


@app.post("/ingest_pdf")
def ingest_pdf(body: IngestRequest):
    """
    Ingest a PDF from a local path on disk (dev only).
    """
    pdf_path = Path(body.pdf_path)
    doc_id, num_chunks = ingest_pdf_from_path(pdf_path)

    return {
        "message": "PDF ingested successfully",
        "doc_id": doc_id,
        "num_chunks": num_chunks,
    }


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF from the client, store it in S3, and ingest into the vector store.
    """
    if file.content_type not in ("application/pdf",):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    filename = file.filename or "uploaded.pdf"

    contents = await file.read()

    doc_id, s3_key = upload_pdf_bytes(contents, filename)

    full_text = load_pdf_text_from_bytes(contents)
    chunks = chunk_text(full_text, chunk_size=800, overlap=200)

    if not chunks:
        raise HTTPException(
            status_code=400, detail="No text could be extracted from the uploaded PDF"
        )

    add_document(
        doc_id=doc_id,
        chunks=chunks,
        metadata={
            "source": filename,
            "s3_key": s3_key,
        },
    )

    return {
        "message": "Uploaded to S3 and ingested PDF successfully",
        "filename": filename,
        "doc_id": doc_id,
        "s3_key": s3_key,
        "num_chunks": len(chunks),
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_question(body: QuestionRequest):
    """
    Main document Q&A endpoint with intent detection and reranking.
    """
    intent_label: str | None = None
    intent_conf: float | None = None
    if question_classifier is not None:
        intent_label, intent_conf = question_classifier.classify(body.question)

    context, sources = build_context_from_query(
        body.question,
        top_k=body.top_k,
        doc_id=body.doc_id,
    )

    if context == "NO_RELEVANT_CONTEXT_FOUND":
        return AnswerResponse(
            answer="I couldn't find any relevant parts of the indexed documents for that question.",
            used_sources=[],
            intent=intent_label,
            intent_confidence=intent_conf,
        )

    system_prompt = build_system_prompt(context, intent_label, emotion=None)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": body.question},
    ]

    try:
        answer = llm_client.chat(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return AnswerResponse(
        answer=answer,
        used_sources=sources,
        intent=intent_label,
        intent_confidence=intent_conf,
    )


@app.post("/predict_face", response_model=FacePredictionResponse)
async def predict_face(file: UploadFile = File(...)):
    """
    Predict identity class (PTri / Lanh / MTuan / BHa / PTri's Muse / strangers / BinhLe / HThuong / XViet / KNguyen)
    from an uploaded face image.
    """
    if face_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Face classifier not available. Train the model first.",
        )

    if not is_image_upload(file):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    try:
        image_bytes = await file.read()
        class_name, confidence = face_classifier.predict_from_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

    return FacePredictionResponse(
        predicted_class=class_name,
        confidence=confidence,
    )


@app.post("/secure_ask", response_model=SecureAskResponse)
async def secure_ask(
    question: str = Form(...),
    doc_id: str | None = Form(None),
    file: UploadFile = File(...),
):
    """
    Vision-gated question answering.

    Treats 'PTri' as the owner. Only answers if the face classifier
    sees PTri with enough confidence.
    """
    if face_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Face classifier not available. Train the model first.",
        )

    if not is_image_upload(file):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    try:
        image_bytes = await file.read()
        identity, conf = face_classifier.predict_from_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

    emotion_label: str | None = None
    emotion_conf: float | None = None
    if "emotion_classifier" in globals() and emotion_classifier is not None:
        try:
            emotion_label, emotion_conf = emotion_classifier.predict_from_bytes(
                image_bytes
            )
        except Exception:
            emotion_label, emotion_conf = None, None

    threshold = 0.7
    if identity != "PTri" or conf < threshold:
        msg = (
            f"Access denied. I see '{identity}' with confidence {conf:.2f}, "
            f"but I only answer when I'm confident it's PTri."
        )
        return SecureAskResponse(
            allowed=False,
            identity=identity,
            confidence=conf,
            answer=msg,
            used_sources=[],
            intent=None,
            intent_confidence=None,
            emotion=emotion_label,
            emotion_confidence=emotion_conf,
        )

    intent_label: str | None = None
    intent_conf: float | None = None
    if question_classifier is not None:
        intent_label, intent_conf = question_classifier.classify(question)

    context, sources = build_context_from_query(
        question,
        top_k=5,
        doc_id=doc_id,
    )

    if context == "NO_RELEVANT_CONTEXT_FOUND":
        return SecureAskResponse(
            allowed=True,
            identity=identity,
            confidence=conf,
            answer="I couldn't find any relevant parts of the indexed documents for that question.",
            used_sources=[],
            intent=intent_label,
            intent_confidence=intent_conf,
            emotion=emotion_label,
            emotion_confidence=emotion_conf,
        )

    system_prompt = build_system_prompt(
        context=context,
        intent=intent_label,
        emotion=emotion_label,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    try:
        answer_text = llm_client.chat(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return SecureAskResponse(
        allowed=True,
        identity=identity,
        confidence=conf,
        answer=answer_text,
        used_sources=sources,
        intent=intent_label,
        intent_confidence=intent_conf,
        emotion=emotion_label,
        emotion_confidence=emotion_conf,
    )


@app.post("/vision_qa", response_model=VisionQAResponse)
async def vision_qa(
    question: str = Form("Describe the people in this image."),
    file: UploadFile = File(...),
):
    """
    Vision Q&A.

    Goals:
    - Support MULTIPLE people in one photo (like PTri + MTuan).
    - Still be fast / not crash your laptop.
    - Avoid hallucinating extra people (like Muse appearing when she is not there).
    - If confidence is low, call them 'strangers' instead of guessing.

    Logic:
    1) Decode image once.
    2) Use object_detector to find 'person' boxes.
       - Filter tiny boxes.
       - Apply simple NMS so we don't get 7 boxes around one face.
       - Keep only the largest few humans (MAX_HUMANS).
    3) For each kept human:
       - Crop.
       - Run face_classifier + emotion_classifier on the crop.
       - If identity_conf < IDENTITY_CONF_THRESHOLD -> label = "strangers".
    4) If no human found at all:
       - Fallback = run classifiers on the whole image once.
    5) Build a factual summary: number of people, identity_label per #, emotions.
       Pass that + short style examples to LLM with strict rules.
    """

    if not is_image_upload(file):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {e}")

    # ---------- Decode image once ----------
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_rgb = np.array(img)
        img_bgr = img_rgb[:, :, ::-1].copy()
        img_h, img_w = img_bgr.shape[:2]
        img_area = float(img_h * img_w)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding image: {e}")

    label_map = {
        "person": "human",
        "car": "car",
        "truck": "car",
        "bus": "car",
        "motorcycle": "car",
        "bicycle": "car",
        "dog": "dog",
        "cat": "cat",
        "cell phone": "phone",
        "mobile phone": "phone",
        "potted plant": "tree",
        "tree": "tree",
    }

    human_infos: list[dict[str, Any]] = []

    # ---------- 1) Human detection via object_detector ----------
    if object_detector is not None:
        try:
            dets = object_detector.detect(image_bytes, max_dets=15)
        except Exception:
            dets = []
    else:
        dets = []

    # small helper: IoU to merge near-duplicate boxes
    def iou(a, b) -> float:
        ax1 = a["x"]
        ay1 = a["y"]
        ax2 = a["x"] + a["w"]
        ay2 = a["y"] + a["h"]

        bx1 = b["x"]
        by1 = b["y"]
        bx2 = b["x"] + b["w"]
        by2 = b["y"] + b["h"]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / max(area_a + area_b - inter, 1e-6)

    # Filter to persons, remove tiny boxes and duplicates
    MIN_AREA_RATIO = 0.02   # ignore boxes < 2% of image
    MAX_IOU = 0.55          # if two boxes overlap more than this, keep only one

    human_boxes: list[dict[str, float]] = []

    for d in dets:
        raw_label = d.get("label", "")
        if label_map.get(raw_label) != "human":
            continue

        x = float(d.get("x", 0.0))
        y = float(d.get("y", 0.0))
        w = float(d.get("w", 0.0))
        h = float(d.get("h", 0.0))
        score = float(d.get("score", 0.0))

        if w <= 0.0 or h <= 0.0:
            continue

        box_area = w * h * img_area
        if box_area < MIN_AREA_RATIO * img_area:
            continue

        candidate = {"x": x, "y": y, "w": w, "h": h, "score": score}

        duplicate = False
        for kept in human_boxes:
            if iou(candidate, kept) > MAX_IOU:
                duplicate = True
                break
        if not duplicate:
            human_boxes.append(candidate)

    # Sort by area (largest first) and keep only a few humans
    MAX_HUMANS = 3
    human_boxes.sort(key=lambda b: b["w"] * b["h"], reverse=True)
    human_boxes = human_boxes[:MAX_HUMANS]

    # ---------- 2) Per-human classification ----------
    for box in human_boxes:
        x_norm = box["x"]
        y_norm = box["y"]
        w_norm = box["w"]
        h_norm = box["h"]

        x1 = max(int(x_norm * img_w), 0)
        y1 = max(int(y_norm * img_h), 0)
        x2 = min(int((x_norm + w_norm) * img_w), img_w - 1)
        y2 = min(int((y_norm + h_norm) * img_h), img_h - 1)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Encode crop once
        try:
            success, buffer = cv2.imencode(".jpg", crop)
            crop_bytes = buffer.tobytes() if success else None
        except Exception:
            crop_bytes = None

        identity_label: str | None = None
        identity_conf: float | None = None
        emotion_label: str | None = None
        emotion_conf: float | None = None

        if crop_bytes is not None and face_classifier is not None:
            try:
                identity_label, identity_conf = face_classifier.predict_from_bytes(
                    crop_bytes
                )
            except Exception:
                identity_label, identity_conf = None, None

            if (
                identity_conf is None
                or identity_conf < IDENTITY_CONF_THRESHOLD
                or identity_label is None
            ):
                identity_label = "strangers"

        if crop_bytes is not None and emotion_classifier is not None:
            try:
                emotion_label, emotion_conf = emotion_classifier.predict_from_bytes(
                    crop_bytes
                )
            except Exception:
                emotion_label, emotion_conf = None, None

        human_infos.append(
            {
                "identity_label": identity_label,
                "identity_conf": identity_conf,
                "emotion_label": emotion_label,
                "emotion_conf": emotion_conf,
            }
        )

    # ---------- 3) Fallback: whole-image classification if no humans ----------
    if not human_infos:
        fallback_identity: str | None = None
        fallback_identity_conf: float | None = None
        fallback_emotion: str | None = None
        fallback_emotion_conf: float | None = None

        if face_classifier is not None:
            try:
                fallback_identity, fallback_identity_conf = \
                    face_classifier.predict_from_bytes(image_bytes)
            except Exception:
                fallback_identity, fallback_identity_conf = None, None

            if (
                fallback_identity_conf is None
                or fallback_identity_conf < IDENTITY_CONF_THRESHOLD
                or fallback_identity is None
            ):
                fallback_identity = "strangers"

        if emotion_classifier is not None:
            try:
                fallback_emotion, fallback_emotion_conf = \
                    emotion_classifier.predict_from_bytes(image_bytes)
            except Exception:
                fallback_emotion, fallback_emotion_conf = None, None

        if fallback_identity is not None or fallback_emotion is not None:
            human_infos.append(
                {
                    "identity_label": fallback_identity,
                    "identity_conf": fallback_identity_conf,
                    "emotion_label": fallback_emotion,
                    "emotion_conf": fallback_emotion_conf,
                }
            )

    human_count = len(human_infos)

    # ---------- 4) Build factual context (NO guessing) ----------
    detection_lines: list[str] = []
    if human_count == 0:
        detection_lines.append(
            "No clear human face was detected in this image."
        )
    else:
        detection_lines.append(
            f"Number of distinct human faces detected in the image: {human_count}."
        )
        for idx, h in enumerate(human_infos, start=1):
            ident = h.get("identity_label")
            iconf = h.get("identity_conf")
            emo = h.get("emotion_label")
            econf = h.get("emotion_conf")

            if ident is not None:
                detection_lines.append(
                    f"Person #{idx}: identity_label = {ident} "
                    f"(confidence {iconf:.2f} if available)."
                )
                details = render_profile_details(ident)
                if details:
                    detection_lines.append(
                        f"Structured profile for person #{idx} (facts only):\n{details}"
                    )
            else:
                detection_lines.append(
                    f"Person #{idx}: no stable identity label from the classifier."
                )

            if emo is not None:
                detection_lines.append(
                    f"Person #{idx}: facial emotion = {emo} "
                    f"(confidence {econf:.2f} if available)."
                )

    factual_context = "\n".join(detection_lines) if detection_lines else "No detections."

    # ---------- 5) Style examples (small, not heavy) ----------
    primary_identity = human_infos[0].get("identity_label") if human_infos else None
    lang_code, style_examples = select_style_examples(
        primary_identity, question, max_examples=3
    )

    examples_text_parts: list[str] = []
    if style_examples:
        examples_text_parts.append(
            "These Q&A pairs show the tone you should follow. "
            "Use them only as a STYLE reference; do NOT copy sentences exactly."
        )
        for ex in style_examples:
            u = (ex.get("user") or "").strip()
            a = (ex.get("assistant") or "").strip()
            if not u or not a:
                continue
            examples_text_parts.append(f"User: {u}\nAssistant: {a}")

    examples_text = "\n\n".join(examples_text_parts)

    # ---------- 6) System prompt with strict rules ----------
    is_vi = looks_vietnamese(question)
    language_instruction = (
        "The user is writing in Vietnamese; you MUST answer in natural Vietnamese (không được trả lời bằng tiếng Anh)."
        if is_vi
        else "Answer in the same language the user used."
    )

    system_prompt = (
        "You are IreneAdler, an AI assistant describing the people in ONE photo.\n\n"
        "You are given:\n"
        "- A factual summary from local vision models (how many humans, identity_label for each, emotions).\n"
        "- Structured profiles for those identity labels in PTri's private database.\n"
        "- A few example Q&A pairs that show the desired tone.\n\n"
        "STRICT rules:\n"
        f"- The number of humans in this photo is EXACTLY {human_count}. "
        "Do NOT invent extra people and do NOT ignore existing ones.\n"
        "- You are ONLY allowed to mention identity names that appear in the factual context section below. "
        "Do NOT guess or introduce new names like 'PTri's Muse' unless that label appears explicitly.\n"
        "- If identity_label == 'strangers', it means you know nothing special about them; "
        "do NOT fabricate any personality or backstory, just say they're strangers.\n"
        "- Use the structured profiles as factual context (relationship to PTri, hobbies, etc.) but do NOT add facts that are not implied.\n"
        "- Keep the answer reasonably short and friendly, like a close friend explaining who is in the picture.\n"
        f"- {language_instruction}\n\n"
        f"Factual context from detectors and profiles:\n{factual_context}\n\n"
        f"Style examples (language: {lang_code}):\n{examples_text}\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    try:
        answer_text = llm_client.chat(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # For compatibility with frontend: return only first human's labels
    out_identity = human_infos[0].get("identity_label") if human_infos else None
    out_identity_conf = human_infos[0].get("identity_conf") if human_infos else None
    out_emotion = human_infos[0].get("emotion_label") if human_infos else None
    out_emotion_conf = human_infos[0].get("emotion_conf") if human_infos else None

    return VisionQAResponse(
        answer=answer_text,
        identity=out_identity,
        identity_confidence=out_identity_conf,
        emotion=out_emotion,
        emotion_confidence=out_emotion_conf,
    )


@app.post("/chat", response_model=ChatResponse)
def general_chat(body: ChatRequest):
    """
    General knowledge chat with IreneAdler.
    Uses style examples (static + dynamic, label=None) to keep a consistent tone.
    """
    question = body.question

    lang_code = "vi" if looks_vietnamese(question) else "en"

    static_ex = STYLE_EXAMPLES.get(lang_code, [])
    dynamic_ex = DYNAMIC_STYLE_EXAMPLES.get(lang_code, [])

    all_ex = static_ex + dynamic_ex
    # Only general (label=None) examples; keep up to 4
    general_ex = [e for e in all_ex if e.get("label") is None][:4]

    system_prompt = (
        "You are an AI assistant named IreneAdler inside TriGPT.\n"
        "You answer general questions using your world knowledge.\n\n"
        "Language rules:\n"
        "- If the user writes in Vietnamese, reply in natural, fluent Vietnamese.\n"
        "- Otherwise, reply in the same language the user used.\n\n"
        "Relationship rules (if the user asks about people in PTri's database):\n"
        "- Refer to them through PTri, e.g. 'PTri's friend', 'PTri's muse', not 'my friend'.\n"
        "- For BHa, default to 'a friend of PTri'; mention past crush only when the user explicitly asks.\n\n"
        "Answering style:\n"
        "- Be concrete and concise. Avoid long motivational speeches and vague filler.\n"
        "- If something is outside your knowledge, say you don't know instead of guessing.\n\n"
        "Below are a few example answers that show the tone you should follow. "
        "Use them as a style reference only; do NOT copy any sentence word-for-word.\n\n"
    )

    examples_block = ""
    for ex in general_ex:
        u = ex.get("user", "")
        a = ex.get("assistant", "")
        if not u or not a:
            continue
        examples_block += f"User: {u}\nAssistant: {a}\n\n"

    messages = [
        {"role": "system", "content": system_prompt + examples_block},
        {"role": "user", "content": question},
    ]

    try:
        answer = llm_client.chat(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return ChatResponse(answer=answer)


@app.post("/live_emotion", response_model=EmotionPredictionResponse)
async def live_emotion(file: UploadFile = File(...)):
    """
    Fast endpoint for webcam snapshots.
    Returns only emotion + confidence.
    """
    if emotion_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Emotion classifier not available. Train or load it first.",
        )

    if not is_image_upload(file):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    try:
        image_bytes = await file.read()
        label, conf = emotion_classifier.predict_from_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

    return EmotionPredictionResponse(
        emotion=label,
        confidence=conf,
    )


@app.post("/live_detect", response_model=DetectionResponse)
async def live_detect(file: UploadFile = File(...)):
    """
    Object detection for webcam / uploaded snapshots.

    - Uses object_detector to find objects.
    - We only keep a small set of categories (human, car, dog, cat, tree, phone).
    - For humans, we crop and run the face classifier to get identity.
    - If identity confidence is low, we force the label to 'strangers'.
    - We also filter tiny boxes and merge overlapping ones to avoid
      '7 humans' around 2 real people.
    """
    if object_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Object detector not available.",
        )

    if not is_image_upload(file):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {e}")

    # Decode image via Pillow (HEIC supported if pillow-heif is installed)
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_rgb = np.array(img)
        img_bgr = img_rgb[:, :, ::-1].copy()
        img_h, img_w = img_bgr.shape[:2]
        img_area = float(img_h * img_w)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding image: {e}")

    try:
        dets = object_detector.detect(image_bytes, max_dets=15)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error running object detector: {e}")

    # Map model labels -> high-level categories
    label_map = {
        "person": "human",
        "car": "car",
        "truck": "car",
        "bus": "car",
        "motorcycle": "car",
        "bicycle": "car",
        "dog": "dog",
        "cat": "cat",
        "cell phone": "phone",
        "mobile phone": "phone",
        "potted plant": "tree",
        "tree": "tree",
    }

    # --- small helper: IoU for duplicate-removal ---
    def iou(a, b) -> float:
        ax1 = a["x"]
        ay1 = a["y"]
        ax2 = a["x"] + a["w"]
        ay2 = a["y"] + a["h"]

        bx1 = b["x"]
        by1 = b["y"]
        bx2 = b["x"] + b["w"]
        by2 = b["y"] + b["h"]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / max(area_a + area_b - inter, 1e-6)

    # --- filter tiny boxes + NMS to prevent duplicates ---
    filtered: list[dict] = []
    MIN_AREA_RATIO = 0.02  # ignore boxes < 2% of image area
    MAX_IOU = 0.55         # if IoU > this with a kept box, treat as duplicate

    for d in dets:
        raw_label = d.get("label", "")
        category = label_map.get(raw_label)
        if category is None:
            continue

        x = float(d.get("x", 0.0))
        y = float(d.get("y", 0.0))
        w = float(d.get("w", 0.0))
        h = float(d.get("h", 0.0))
        score = float(d.get("score", 0.0))

        # Filter clearly invalid or tiny boxes
        if w <= 0.0 or h <= 0.0:
            continue
        box_area = w * h * img_area
        if box_area < MIN_AREA_RATIO * img_area:
            continue

        candidate = {"label": raw_label, "score": score, "x": x, "y": y, "w": w, "h": h}

        # NMS: drop if highly overlapping with a better one
        duplicate = False
        for kept in filtered:
            if kept["label"] != raw_label:
                continue
            if iou(candidate, kept) > MAX_IOU:
                duplicate = True
                break
        if not duplicate:
            filtered.append(candidate)

    detections_out: list[DetectionBox] = []

    for d in filtered:
        raw_label = d["label"]
        score = float(d["score"])
        x_norm = float(d["x"])
        y_norm = float(d["y"])
        w_norm = float(d["w"])
        h_norm = float(d["h"])

        category = label_map.get(raw_label)
        if category is None:
            continue

        x = x_norm
        y = y_norm
        w = w_norm
        h = h_norm

        identity_label: str | None = None
        identity_conf: float | None = None
        identity_info: str | None = None

        if category == "human" and face_classifier is not None:
            x1 = max(int(x_norm * img_w), 0)
            y1 = max(int(y_norm * img_h), 0)
            x2 = min(int((x_norm + w_norm) * img_w), img_w - 1)
            y2 = min(int((y_norm + h_norm) * img_h), img_h - 1)

            if x2 > x1 and y2 > y1:
                crop = img_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    success, buffer = cv2.imencode(".jpg", crop)
                    if success:
                        crop_bytes = buffer.tobytes()
                        try:
                            identity_label, identity_conf = \
                                face_classifier.predict_from_bytes(crop_bytes)
                        except Exception:
                            identity_label, identity_conf = None, None

                        # Low confidence → treat as strangers
                        if (
                            identity_conf is None
                            or identity_conf < IDENTITY_CONF_THRESHOLD
                            or identity_label is None
                        ):
                            identity_label = "strangers"

                        if identity_label is not None:
                            identity_info = IDENTITY_INFO.get(identity_label)

        detections_out.append(
            DetectionBox(
                label=category,
                raw_label=raw_label,
                score=score,
                x=x,
                y=y,
                w=w,
                h=h,
                identity=identity_label,
                identity_confidence=identity_conf,
                identity_info=identity_info,
            )
        )

    return DetectionResponse(detections=detections_out)

@app.post("/face_feedback")
async def face_feedback(
    label: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Save a corrected face image into the training folder so the model
    can learn from mistakes later.

    Usage (multiple people in one photo):
    - On the frontend, crop each face separately (e.g. using the bbox from /live_detect).
    - Call this endpoint once per person with:
        - label: one of your class names
          (PTri, Lanh, MTuan, BHa, PTri's Muse, BinhLe, HThuong, XViet, KNguyen, strangers, ...)
        - file: the FACE CROP image (jpg/png/etc.) for that person.
    Each call saves one image:
        data/faces/train/<label>/feedback_*.jpg

    Then re-run: python -m src.train_face_model
    to retrain with this extra feedback data.
    """
    if not is_image_upload(file):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    label = label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="Label must not be empty.")

    if face_classifier is not None:
        known_classes = set(face_classifier.classes)
        if label not in known_classes:
            print(
                f"⚠️ face_feedback received unknown label '{label}'. "
                f"Known classes: {known_classes}"
            )

    try:
        img_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {e}")

    try:
        _ = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding image: {e}")

    save_dir = Path("data/faces/train") / label
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"feedback_{int(time.time())}.jpg"
    save_path = save_dir / filename

    with open(save_path, "wb") as f:
        f.write(img_bytes)

    return {
        "message": "Feedback image saved successfully.",
        "label": label,
        "path": str(save_path),
    }
