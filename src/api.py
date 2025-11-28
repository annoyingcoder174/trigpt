# src/api.py

from pathlib import Path
from typing import List

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from .llm_client import LLMClient
from .doc_ingestion import load_pdf_text, load_pdf_text_from_bytes, chunk_text
from .vector_store import add_document, query_document, list_documents
from .vision_model import FaceClassifier
from .s3_storage import upload_pdf_bytes
from .question_classifier import QuestionTypeClassifier
from .reranker import Reranker
from .emotion_model import EmotionClassifier
from typing import List

from .object_detector import ObjectDetector


app = FastAPI(title="TriGPT API", version="0.1.0")

llm_client = LLMClient()
# ----- CORS (allow frontend to call this API) -----
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:5173",   # Vite dev server (for future React UI)
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # later you can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to load face classifier at startup
try:
    face_classifier = FaceClassifier(checkpoint_path="models/face_classifier.pt")
    print("✅ Face classifier loaded for API.")
except FileNotFoundError:
    face_classifier = None
    print("⚠️ Face classifier checkpoint not found. /predict_face and /secure_ask will be limited.")

# Try to load question-type classifier at startup
try:
    question_classifier = QuestionTypeClassifier(model_dir="models/question_classifier")
    print("✅ Question-type classifier loaded for API.")
except FileNotFoundError:
    question_classifier = None
    print("⚠️ Question-type classifier not found. Intent detection disabled.")
# Try to load RAG reranker at startup
try:
    reranker = Reranker()
    print("✅ Reranker loaded for API.")
except Exception as e:
    reranker = None
    print(f"⚠️ Reranker not available: {e}")
# Try to load emotion classifier at startup
try:
    emotion_classifier = EmotionClassifier(checkpoint_path="models/emotion_classifier.pt")
    print("✅ Emotion classifier loaded for API.")
except FileNotFoundError:
    emotion_classifier = None
    print("⚠️ Emotion classifier checkpoint not found. /predict_emotion will be disabled.")
try:
    object_detector = ObjectDetector(score_thresh=0.7)
    print("✅ Object detector loaded for API (score_thresh=0.7).")
except Exception as e:
    object_detector = None
    print(f"⚠️ Object detector not available: {e}")




# ---------- Models ----------

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
    label: str
    score: float
    x: float  # 0-1, left
    y: float  # 0-1, top
    w: float  # 0-1, width
    h: float  # 0-1, height


class DetectionResponse(BaseModel):
    detections: List[DetectionBox]





# ---------- Helpers ----------

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
    # Ask Chroma for a bit more than top_k so the reranker has options
    result = query_document(question, n_results=top_k * 2, doc_id=doc_id)

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]

    if not docs:
        # no relevant chunks found
        return "NO_RELEVANT_CONTEXT_FOUND", []

    # --- Rerank if model is available ---
    if reranker is not None and len(docs) > 1:
        indices, scores = reranker.rerank(question, docs, top_k=top_k)
        docs = [docs[i] for i in indices]
        metas = [metas[i] for i in indices]
    else:
        # If no reranker, just use the first top_k from vector search
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
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")

    doc_id = pdf_path.stem
    add_document(
        doc_id=doc_id,
        chunks=chunks,
        metadata={"source": pdf_path.name},
    )

    return doc_id, len(chunks)


def build_system_prompt(
    context: str,
    intent: str | None,
    emotion: str | None = None,
) -> str:
    """
    Build a system prompt for IreneAdler based on detected intent and (optionally) emotion.
    """
    base = (
        "You are an AI assistant named IreneAdler. "
        "Use the provided context to answer the question as clearly as possible. "
        "If the answer is clearly not in the context, say you don't know."
    )

    if intent == "summary":
        style = (
            "Focus on giving a concise summary. "
            "Use short paragraphs or bullet points highlighting the key ideas."
        )
    elif intent == "explain":
        style = (
            "Focus on explaining concepts in simple language, using examples and analogies. "
            "Assume the user is smart but not an expert."
        )
    elif intent == "compare":
        style = (
            "Focus on comparing the key items. "
            "Highlight similarities and differences. "
            "Lists or structured formats are welcome."
        )
    elif intent == "definition":
        style = (
            "Start with a short, clear definition in one or two sentences. "
            "Then, if helpful, add a bit more detail or context."
        )
    else:
        style = "Answer in a clear, structured, and helpful way."

    # Emotion-based tweak
    emotion_note = ""
    if emotion in {"sad", "tired"}:
        emotion_note = (
            "The user seems a bit down or tired. "
            "Be extra kind, encouraging, and keep the answer relatively short."
        )
    elif emotion == "happy":
        emotion_note = (
            "The user seems happy. It's okay to be slightly more upbeat and positive."
        )
    elif emotion == "angry":
        emotion_note = (
            "The user may be frustrated. Be calm, empathetic, and solution-focused."
        )

    intent_str = intent or "unknown"
    emotion_str = emotion or "unknown"

    return (
        f"{base}\n\n"
        f"Detected intent: {intent_str}.\n"
        f"Detected emotion: {emotion_str}.\n"
        f"{style}\n"
        f"{emotion_note}\n\n"
        f"Context:\n{context}"
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
        "emotion_classifier": emotion_classifier is not None if 'emotion_classifier' in globals() else False,
        "reranker": reranker is not None if 'reranker' in globals() else False,
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

    # Read file bytes once
    contents = await file.read()

    # 1) Upload to S3
    doc_id, s3_key = upload_pdf_bytes(contents, filename)

    # 2) Extract text from bytes and index into Chroma
    full_text = load_pdf_text_from_bytes(contents)
    chunks = chunk_text(full_text, chunk_size=800, overlap=200)

    if not chunks:
        raise HTTPException(status_code=400, detail="No text could be extracted from the uploaded PDF")

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
    # 1 detect intent (if classifier is available)
    intent_label: str | None = None
    intent_conf: float | None = None
    if question_classifier is not None:
        intent_label, intent_conf = question_classifier.classify(body.question)

    # 2 retrieve context
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

    # 3 build prompt based on intent
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
    Predict 'me' vs 'other' from an uploaded image.
    """
    if face_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Face classifier not available. Train the model first."
        )

    if not file.content_type.startswith("image/"):
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
    Vision-gated question answering:
    - Uses the face classifier to check identity.
    - Uses the emotion classifier to detect basic emotion.
    - Only runs RAG if the face is recognized as 'me' with enough confidence.
    - Adapts Irene's tone based on detected emotion + intent.
    """
    if face_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Face classifier not available. Train the model first."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    try:
        image_bytes = await file.read()
        # 1) face identity
        identity, conf = face_classifier.predict_from_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

    # 2) optional emotion
    emotion_label: str | None = None
    emotion_conf: float | None = None
    if "emotion_classifier" in globals() and emotion_classifier is not None:
        try:
            emotion_label, emotion_conf = emotion_classifier.predict_from_bytes(image_bytes)
        except Exception:
            # don't break secure_ask if emotion model fails
            emotion_label, emotion_conf = None, None

    # 3) Threshold & identity check
    threshold = 0.7
    if identity != "me" or conf < threshold:
        msg = (
            f"Access denied. I see '{identity}' with confidence {conf:.2f}, "
            f"but I only answer when I'm confident it's 'me'."
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

    # 4) Authorized → detect intent
    intent_label: str | None = None
    intent_conf: float | None = None
    if question_classifier is not None:
        intent_label, intent_conf = question_classifier.classify(question)

    # 5) Do normal RAG (optionally scoped to a doc_id)
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

    # 6) Build system prompt with intent + emotion
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
    Vision Q&A without access control:
    - Uses face + emotion classifiers on the uploaded image.
    - Passes those detections to IreneAdler.
    - Irene must NOT identify real-world people by name.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {e}")

    # Optional: face identity label (e.g., 'me' / 'other')
    identity_label: str | None = None
    identity_conf: float | None = None
    if "face_classifier" in globals() and face_classifier is not None:
        try:
            identity_label, identity_conf = face_classifier.predict_from_bytes(image_bytes)
        except Exception:
            identity_label, identity_conf = None, None

    # Optional: emotion
    emotion_label: str | None = None
    emotion_conf: float | None = None
    if "emotion_classifier" in globals() and emotion_classifier is not None:
        try:
            emotion_label, emotion_conf = emotion_classifier.predict_from_bytes(image_bytes)
        except Exception:
            emotion_label, emotion_conf = None, None

    # Build a context string for Irene
    detection_lines = []
    if identity_label is not None:
        detection_lines.append(
            f"Detected identity label (from a local classifier): {identity_label} "
            f"(confidence {identity_conf:.2f} if available). "
            "This is NOT a real-world identity; just a label like 'me' or 'other'."
        )
    if emotion_label is not None:
        detection_lines.append(
            f"Detected facial emotion: {emotion_label} "
            f"(confidence {emotion_conf:.2f} if available)."
        )

    context = "\n".join(detection_lines) if detection_lines else "No detections available."

    system_prompt = (
        "You are an AI assistant named IreneAdler answering questions about a single image.\n\n"
        "You will be given some detector outputs (a simple identity label and an emotion label). "
        "These labels are only local tags like 'me' or 'other' and generic emotion names.\n\n"
        "IMPORTANT:\n"
        "- You must NOT try to guess or state the real-world identity, real name, or personal details "
        "of anyone in the image.\n"
        "- If the user asks 'who is this?' or for a name, say you cannot identify the person.\n"
        "- You MAY talk about general appearance, expressions, and emotions.\n\n"
        f"Detections:\n{context}\n"
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
    Does NOT use documents or RAG, just the base LLM.
    """
    system_prompt = (
        "You are an AI assistant named IreneAdler. "
        "Answer general questions using your world knowledge. "
        "You are not limited to any uploaded documents. "
        "If you genuinely don't know something or it may be outdated, say that honestly."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": body.question},
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

    if not file.content_type.startswith("image/"):
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
    Returns generic categories like person, dog, car, etc.
    """
    if object_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Object detector not available.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    try:
        image_bytes = await file.read()
        dets = object_detector.detect(image_bytes, max_dets=10)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

    # Filter only categories you care about if you want (optional)
    # e.g., humans/dogs/cats/vehicles
    keep_prefixes = {"person", "dog", "cat", "car", "bus", "truck", "motorcycle", "bicycle"}
    filtered = [
        d for d in dets
        if any(k in d["label"] for k in keep_prefixes)
    ]

    return DetectionResponse(
        detections=[DetectionBox(**d) for d in filtered]
    )
