# src/api.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any
import io
import time
import json

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


# ---------- Pydantic Models ----------


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
    Very rough check if text is likely Vietnamese.
    Not perfect, just to pick style examples.
    """
    if not text:
        return False
    t = text.lower()
    vi_markers = [
        " không",
        " ko ",
        " là ",
        "của ",
        "nhé",
        "nh nha",
        " nha",
        " ạ",
        "và",
        "được",
        "mình",
        "cậu",
        "tớ",
        "anh ",
        "chị ",
        "em ",
        "ơi",
    ]
    return any(m in t for m in vi_markers) or "đ" in t


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
    Predict identity class (PTri / Lanh / MTuan / BHa / PTri's Muse / strangers)
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
    question: str = Form("Describe the person and their emotion."),
    file: UploadFile = File(...),
):
    """
    Vision Q&A without access control.

    Irene will:
    - Use identity_db profile as factual context.
    - Use style examples (EN or VI) as a guide for human-like tone.
    - Not just copy raw fields; instead talk like a close friend of PTri describing that person.
    """
    if not is_image_upload(file):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {e}")

    identity_label: str | None = None
    identity_conf: float | None = None
    if "face_classifier" in globals() and face_classifier is not None:
        try:
            identity_label, identity_conf = face_classifier.predict_from_bytes(
                image_bytes
            )
        except Exception:
            identity_label, identity_conf = None, None

    emotion_label: str | None = None
    emotion_conf: float | None = None
    if "emotion_classifier" in globals() and emotion_classifier is not None:
        try:
            emotion_label, emotion_conf = emotion_classifier.predict_from_bytes(
                image_bytes
            )
        except Exception:
            emotion_label, emotion_conf = None, None

    # Build factual context from identity_db + detectors
    detection_lines: list[str] = []

    if identity_label is not None:
        detection_lines.append(
            f"Detected identity label (from a local classifier): {identity_label} "
            f"(confidence {identity_conf:.2f} if available). "
            "This is NOT a real-world identity; just a local tag."
        )
        details = render_profile_details(identity_label)
        if details:
            detection_lines.append(
                "Structured profile (facts only, not how you must phrase it):\n"
                + details
            )

    if emotion_label is not None:
        detection_lines.append(
            f"Detected facial emotion: {emotion_label} "
            f"(confidence {emotion_conf:.2f} if available)."
        )

    factual_context = (
        "\n".join(detection_lines) if detection_lines else "No detections available."
    )

    # Select style examples based on language + identity (static + dynamic)
    lang_code, style_examples = select_style_examples(
        identity_label, question, max_examples=4
    )

    examples_text_parts: list[str] = []
    if style_examples:
        examples_text_parts.append(
            "Here are a few example conversations showing the TONE and STYLE you should follow.\n"
            "Important:\n"
            "- Use them only as a style reference.\n"
            "- Do NOT copy sentences or paragraphs from them.\n"
            "- Rewrite the ideas in your own words.\n"
        )
        for ex in style_examples:
            u = (ex.get("user") or "").strip()
            a = (ex.get("assistant") or "").strip()
            if not u or not a:
                continue
            examples_text_parts.append(f"User: {u}\nAssistant: {a}\n")

    examples_text = "\n\n".join(examples_text_parts) if examples_text_parts else ""

    system_prompt = (
        "You are an AI assistant named IreneAdler answering questions about a single image.\n\n"
        "You will be given:\n"
        "1) Detector outputs (identity label, emotion).\n"
        "2) A structured profile for that identity from the user's private database.\n"
        "3) A few example Q&A pairs that show the desired tone and style.\n\n"
        "Your job:\n"
        "- Use the structured profile as factual context (birthday, hobbies, university, achievements, memories, etc.).\n"
        "- Answer in a natural, human-like way, similar in tone to the examples, but using your own wording.\n"
        "- Do NOT simply list the raw fields or copy the profile lines.\n"
        "- Do NOT copy the example answers word-for-word.\n"
        "- Write your own paragraph(s) that feel like a close friend of PTri describing this person.\n"
        "- Keep answers concrete and avoid vague motivational talk.\n\n"
        "Language rules:\n"
        "- Detect the user's language from their question.\n"
        "- If the user writes in Vietnamese, answer in natural Vietnamese.\n"
        "- Otherwise, answer in the user's language.\n\n"
        "Relationship rules (very important):\n"
        "- All people in the database are connected to PTri, not to you.\n"
        "- Never say 'my friend', 'my crush', 'my muse', 'my owner', etc.\n"
        "- Instead, use phrases like 'PTri's friend', 'PTri's muse', 'someone important to PTri'.\n"
        "- When describing BHa, default to 'a friend of PTri'. Only mention that she used to be a crush "
        "if the user explicitly asks about love, crush, or romantic feelings.\n\n"
        f"Factual context:\n{factual_context}\n\n"
        f"Style examples (language code: {lang_code}):\n{examples_text}\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    try:
        answer_text = llm_client.chat(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return VisionQAResponse(
        answer=answer_text,
        identity=identity_label,
        identity_confidence=identity_conf,
        emotion=emotion_label,
        emotion_confidence=emotion_conf,
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
    Object detection for webcam snapshots.

    - Maps raw detector labels to high-level categories:
      human, car, dog, cat, tree, phone, ...
    - If the category is 'human' AND the face_classifier is available,
      it crops the region and runs the face classifier to get an identity label.
    - Adds identity_info from IDENTITY_INFO for humans.
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
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding image: {e}")

    try:
        dets = object_detector.detect(image_bytes, max_dets=10)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error running object detector: {e}")

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

    detections_out: list[DetectionBox] = []

    for d in dets:
        raw_label = d.get("label", "")
        score = float(d.get("score", 0.0))
        x_norm = float(d.get("x", 0.0))
        y_norm = float(d.get("y", 0.0))
        w_norm = float(d.get("w", 0.0))
        h_norm = float(d.get("h", 0.0))

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
                            identity_label, identity_conf = (
                                face_classifier.predict_from_bytes(crop_bytes)
                            )
                            if identity_label is not None:
                                identity_info = IDENTITY_INFO.get(identity_label)
                        except Exception:
                            identity_label, identity_conf, identity_info = (
                                None,
                                None,
                                None,
                            )

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

    Usage:
    - label: one of your class names (PTri, Lanh, MTuan, BHa, PTri's Muse, strangers)
    - file: a FACE CROP image (jpg/png/etc.)
    Saved to: data/faces/train/<label>/feedback_*.jpg

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


@app.post("/chat_feedback")
def chat_feedback(body: ChatFeedbackRequest):
    """
    Store user feedback about a chat answer (1–5 stars).

    - rating <= 2  → considered bad (needs improvement; often with a comment)
    - rating 3     → neutral
    - rating >= 4  → considered good (candidate for future style examples)

    This does NOT instantly change the model, but creates a growing dataset
    you can later use to improve prompts or fine-tune.
    """
    if body.rating < 1 or body.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5.")

    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": int(time.time()),
        "question": body.question,
        "answer": body.answer,
        "rating": body.rating,
        "comment": body.comment,
        "mode": body.mode,
    }

    try:
        with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing feedback: {e}")

    return {"message": "Feedback saved. Cảm ơn nha 🫶"}
