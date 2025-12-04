import { useEffect, useState, useRef } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const MODES = {
  DOC: "doc",
  GLOBAL: "global",
  GENERAL: "general",
  VISION: "vision",
};

const IDENTITY_OPTIONS = [
  "PTri",
  "Lanh",
  "MTuan",
  "BHa",
  "PTri's Muse",
  "strangers",
  "HThuong",
  "BinhLe",
  "XViet",
  "KNguyen",
  "PTrinh",
];

function App() {
  const [health, setHealth] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState(null);

  const [mode, setMode] = useState(MODES.DOC);

  const [conversations, setConversations] = useState({
    [MODES.DOC]: [],
    [MODES.GLOBAL]: [],
    [MODES.GENERAL]: [],
    [MODES.VISION]: [],
  });

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState(null);

  // Vision-specific state
  const [visionFile, setVisionFile] = useState(null);
  const [visionPreviewUrl, setVisionPreviewUrl] = useState(null);
  const [visionDetections, setVisionDetections] = useState([]);

  // Feedback for face classifier
  const [feedbackLabels, setFeedbackLabels] = useState({});
  const [feedbackStatus, setFeedbackStatus] = useState(null);
  const [feedbackLoading, setFeedbackLoading] = useState(false);

  // Chat answer feedback (1–5 rating per assistant message)
  const [messageFeedback, setMessageFeedback] = useState({});
  const [sendingChatFeedback, setSendingChatFeedback] = useState(false);

  // Camera for Vision snapshot
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [cameraActive, setCameraActive] = useState(false);

  // PDF upload state (Doc mode)
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);

  const currentMessages = conversations[mode] || [];

  // ---------- Effects ----------

  useEffect(() => {
    async function fetchData() {
      try {
        const [healthRes, docsRes] = await Promise.all([
          fetch(`${API_URL}/health`),
          fetch(`${API_URL}/documents`),
        ]);

        if (!healthRes.ok) throw new Error("Health check failed");
        if (!docsRes.ok) throw new Error("Documents fetch failed");

        const healthJson = await healthRes.json();
        const docsJson = await docsRes.json();

        setHealth(healthJson);
        setDocuments(docsJson);
        if (docsJson.length > 0) {
          setSelectedDocId(docsJson[0].doc_id);
        }
      } catch (err) {
        setLoadError(err.message || "Request failed");
      }
    }

    fetchData();
  }, []);

  // Cleanup snapshot camera + preview URL on unmount
  useEffect(() => {
    return () => {
      stopCamera();
      if (visionPreviewUrl) {
        URL.revokeObjectURL(visionPreviewUrl);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------- Helpers ----------

  function setMessagesForMode(modeKey, msgs) {
    setConversations((prev) => ({
      ...prev,
      [modeKey]: msgs,
    }));
  }

  function switchMode(newMode) {
    if (cameraActive) {
      stopCamera();
    }
    setMode(newMode);
    setLoadError(null);
  }

  function getLastUserQuestionForMessage(modeKey, msgIndex) {
    const msgs = conversations[modeKey] || [];
    for (let i = msgIndex - 1; i >= 0; i--) {
      if (msgs[i].role === "user") {
        return msgs[i].content;
      }
    }
    return "";
  }

  // ---------- Chat send ----------

  async function handleSend() {
    const question = input.trim();
    if (!question || loading) return;

    setInput("");
    setLoadError(null);

    const userMsg = { role: "user", content: question };
    const newMessages = [...currentMessages, userMsg];
    setMessagesForMode(mode, newMessages);
    setLoading(true);

    try {
      if (mode === MODES.VISION) {
        if (!visionFile) {
          throw new Error(
            "Please upload an image or capture one from the camera before asking."
          );
        }

        const formData = new FormData();
        formData.append("question", question);
        formData.append("file", visionFile);

        const res = await fetch(`${API_URL}/vision_qa`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          throw new Error(`vision_qa failed with status ${res.status}`);
        }

        const data = await res.json();

        const metaLines = [];

        if (typeof data.identity === "string") {
          const conf =
            typeof data.identity_confidence === "number"
              ? data.identity_confidence.toFixed(2)
              : "unknown";
          metaLines.push(`identity label: ${data.identity} (conf: ${conf})`);
        }

        if (typeof data.emotion === "string") {
          const econf =
            typeof data.emotion_confidence === "number"
              ? data.emotion_confidence.toFixed(2)
              : "unknown";
          metaLines.push(`emotion: ${data.emotion} (conf: ${econf})`);
        }

        const assistantText =
          (data.answer || "I don't know about this image.") +
          (metaLines.length ? `\n\n${metaLines.join("\n")}` : "");

        const finalMessages = [
          ...newMessages,
          { role: "assistant", content: assistantText },
        ];
        setMessagesForMode(mode, finalMessages);
      } else if (mode === MODES.GENERAL) {
        const body = { question };

        const res = await fetch(`${API_URL}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });

        if (!res.ok) {
          throw new Error(`chat failed with status ${res.status}`);
        }

        const data = await res.json();
        const assistantText = data.answer || "I don't know.";

        const finalMessages = [
          ...newMessages,
          { role: "assistant", content: assistantText },
        ];
        setMessagesForMode(mode, finalMessages);
      } else {
        const docIdToUse = mode === MODES.DOC ? selectedDocId : null;

        const body = {
          question,
          top_k: 5,
          doc_id: docIdToUse,
        };

        const res = await fetch(`${API_URL}/ask`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });

        if (!res.ok) {
          throw new Error(`ask failed with status ${res.status}`);
        }

        const data = await res.json();

        const metaLines = [];

        if (data.intent) {
          const conf =
            typeof data.intent_confidence === "number"
              ? data.intent_confidence.toFixed(2)
              : "unknown";
          metaLines.push(`intent: ${data.intent} (conf: ${conf})`);
        }

        if (Array.isArray(data.used_sources) && data.used_sources.length > 0) {
          metaLines.push(
            `sources: ${data.used_sources
              .map((s) => s.replace("[", "").replace("]", ""))
              .join(" | ")}`
          );
        }

        const assistantText =
          (data.answer || "I couldn't find anything about that.") +
          (metaLines.length ? `\n\n${metaLines.join("\n")}` : "");

        const finalMessages = [
          ...newMessages,
          { role: "assistant", content: assistantText },
        ];
        setMessagesForMode(mode, finalMessages);
      }
    } catch (err) {
      console.error(err);
      setLoadError(err.message || "Something went wrong.");
      setMessagesForMode(mode, [
        ...newMessages,
        {
          role: "assistant",
          content:
            "Sorry, something went wrong while answering that question.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  // ---------- Vision: upload / camera (snapshot) ----------

  function handleVisionFileChange(e) {
    const file = e.target.files?.[0];
    if (file) {
      if (visionPreviewUrl) {
        URL.revokeObjectURL(visionPreviewUrl);
      }
      setVisionFile(file);
      setVisionPreviewUrl(URL.createObjectURL(file));
      setVisionDetections([]);
      setFeedbackLabels({});
      setFeedbackStatus(null);
    } else {
      if (visionPreviewUrl) {
        URL.revokeObjectURL(visionPreviewUrl);
      }
      setVisionFile(null);
      setVisionPreviewUrl(null);
      setVisionDetections([]);
      setFeedbackLabels({});
      setFeedbackStatus(null);
    }
  }

  async function startCamera() {
    if (cameraActive) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      setCameraActive(true);
      setLoadError(null);
    } catch (err) {
      console.error("Error starting camera", err);
      setLoadError("Could not access camera.");
    }
  }

  function stopCamera() {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  }

  function captureFromCamera() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;

    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, w, h);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;

        if (visionPreviewUrl) {
          URL.revokeObjectURL(visionPreviewUrl);
        }

        const file = new File([blob], "camera_capture.jpg", {
          type: "image/jpeg",
        });

        setVisionFile(file);
        setVisionPreviewUrl(URL.createObjectURL(blob));
        setVisionDetections([]);
        setFeedbackLabels({});
        setFeedbackStatus(null);

        // release camera after capture
        stopCamera();
      },
      "image/jpeg",
      0.9
    );
  }

  // ---------- Vision: Detect objects & faces (live_detect) ----------

  async function handleVisionDetect() {
    if (!visionFile) {
      setLoadError("Please upload or capture an image first.");
      return;
    }

    setLoading(true);
    setLoadError(null);

    try {
      const formData = new FormData();
      formData.append("file", visionFile);

      const res = await fetch(`${API_URL}/live_detect`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`live_detect failed with status ${res.status}`);
      }

      const data = await res.json();
      const detections = Array.isArray(data.detections)
        ? data.detections
        : [];

      setVisionDetections(detections);
      setFeedbackLabels({});
      setFeedbackStatus(null);

      let text = "";

      if (detections.length === 0) {
        text = "I didn't detect any of the configured objects in this image.";
      } else {
        const lines = ["Objects detected:"];
        for (const d of detections) {
          const label = d.label || d.raw_label || "unknown";
          const score =
            typeof d.score === "number" ? d.score.toFixed(2) : "unknown";

          let line = `- ${label} (score: ${score})`;

          if (label === "human" && d.identity) {
            const iconf =
              typeof d.identity_confidence === "number"
                ? d.identity_confidence.toFixed(2)
                : "unknown";
            line += ` • identity: ${d.identity} (conf: ${iconf})`;

            if (d.identity_info) {
              line += ` • info: ${d.identity_info}`;
            }
          }

          if (
            typeof d.x === "number" &&
            typeof d.y === "number" &&
            typeof d.w === "number" &&
            typeof d.h === "number"
          ) {
            line += ` • bbox: x=${d.x.toFixed(2)}, y=${d.y.toFixed(
              2
            )}, w=${d.w.toFixed(2)}, h=${d.h.toFixed(2)}`;
          }

          lines.push(line);
        }
        text = lines.join("\n");
      }

      const visionMessages = conversations[MODES.VISION] || [];
      setMessagesForMode(MODES.VISION, [
        ...visionMessages,
        { role: "assistant", content: text },
      ]);
    } catch (err) {
      console.error(err);
      setLoadError(err.message || "Something went wrong during detection.");
      const visionMessages = conversations[MODES.VISION] || [];
      setMessagesForMode(MODES.VISION, [
        ...visionMessages,
        {
          role: "assistant",
          content:
            "Sorry, something went wrong while running object & face detection.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  // ---------- Vision feedback: crop & send to /face_feedback ----------

  function onFeedbackLabelChange(idx, value) {
    setFeedbackLabels((prev) => ({
      ...prev,
      [idx]: value,
    }));
  }

  function cropDetectionToBlob(detection) {
    return new Promise((resolve, reject) => {
      if (!visionFile) {
        reject(new Error("No vision file available for feedback."));
        return;
      }

      const img = new Image();
      const url = URL.createObjectURL(visionFile);

      img.onload = () => {
        try {
          const { x, y, w, h } = detection;
          const x1 = Math.max(Math.floor(x * img.width), 0);
          const y1 = Math.max(Math.floor(y * img.height), 0);
          const x2 = Math.min(
            Math.floor((x + w) * img.width),
            img.width
          );
          const y2 = Math.min(
            Math.floor((y + h) * img.height),
            img.height
          );

          const boxW = Math.max(x2 - x1, 1);
          const boxH = Math.max(y2 - y1, 1);

          const canvas = document.createElement("canvas");
          canvas.width = boxW;
          canvas.height = boxH;
          const ctx = canvas.getContext("2d");
          ctx.drawImage(
            img,
            x1,
            y1,
            boxW,
            boxH,
            0,
            0,
            boxW,
            boxH
          );

          canvas.toBlob(
            (blob) => {
              URL.revokeObjectURL(url);
              if (!blob) {
                reject(new Error("Failed to create feedback blob."));
              } else {
                resolve(blob);
              }
            },
            "image/jpeg",
            0.95
          );
        } catch (err) {
          URL.revokeObjectURL(url);
          reject(err);
        }
      };

      img.onerror = (err) => {
        URL.revokeObjectURL(url);
        reject(err);
      };

      img.src = url;
    });
  }

  async function handleSendFeedback(idx) {
    const det = visionDetections[idx];
    if (!det) {
      setFeedbackStatus("Cannot find that detection.");
      return;
    }
    if (det.label !== "human") {
      setFeedbackStatus("Feedback is only for human detections.");
      return;
    }

    const correctedLabel = feedbackLabels[idx];
    if (!correctedLabel) {
      setFeedbackStatus("Choose the correct identity before sending feedback.");
      return;
    }

    try {
      setFeedbackLoading(true);
      setFeedbackStatus("Creating face crop and sending feedback...");

      const blob = await cropDetectionToBlob(det);
      const file = new File([blob], "feedback_crop.jpg", {
        type: "image/jpeg",
      });

      const formData = new FormData();
      formData.append("label", correctedLabel);
      formData.append("file", file);

      const res = await fetch(`${API_URL}/face_feedback`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`face_feedback failed with status ${res.status}`);
      }

      const data = await res.json();
      console.log("Feedback saved:", data);

      setFeedbackStatus(
        `Saved feedback for ${correctedLabel}. It will take effect after you retrain the face model.`
      );
    } catch (err) {
      console.error(err);
      setFeedbackStatus(err.message || "Feedback failed.");
    } finally {
      setFeedbackLoading(false);
    }
  }

  // ---------- Chat feedback (rating 1–5) ----------

  function keyForMessage(modeKey, index) {
    return `${modeKey}-${index}`;
  }

  function handleRateMessage(modeKey, index, rating) {
    const k = keyForMessage(modeKey, index);
    setMessageFeedback((prev) => {
      const existing = prev[k] || {};
      const waitingForComment = rating <= 2; // only ask for comment when rating is 1–2
      return {
        ...prev,
        [k]: {
          ...existing,
          rating,
          waitingForComment,
          comment: existing.comment || "",
          submitted: rating >= 4 ? true : existing.submitted || false, // will mark submitted after API
          error: null,
        },
      };
    });

    if (rating >= 3) {
      // rating 3–5 → send feedback immediately
      submitChatFeedback(modeKey, index, rating, null);
    }
  }

  function handleFeedbackCommentChange(modeKey, index, value) {
    const k = keyForMessage(modeKey, index);
    setMessageFeedback((prev) => ({
      ...prev,
      [k]: {
        ...(prev[k] || {}),
        comment: value,
      },
    }));
  }

  function handleSubmitFeedbackWithComment(modeKey, index, skip) {
    const k = keyForMessage(modeKey, index);
    const state = messageFeedback[k] || {};
    const rating = state.rating;
    if (!rating) return;

    const comment = skip ? null : (state.comment || "").trim() || null;
    submitChatFeedback(modeKey, index, rating, comment);
  }

  async function submitChatFeedback(modeKey, index, rating, comment) {
    try {
      setSendingChatFeedback(true);

      const msgs = conversations[modeKey] || [];
      const answerMsg = msgs[index];
      if (!answerMsg || answerMsg.role !== "assistant") {
        return;
      }
      const answer = answerMsg.content;
      const question = getLastUserQuestionForMessage(modeKey, index);

      const body = {
        question,
        answer,
        rating,
        comment,
        mode: modeKey,
      };

      const res = await fetch(`${API_URL}/chat_feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        throw new Error(`chat_feedback failed with status ${res.status}`);
      }

      const k = keyForMessage(modeKey, index);
      setMessageFeedback((prev) => ({
        ...prev,
        [k]: {
          ...(prev[k] || {}),
          submitted: true,
          waitingForComment: false,
          error: null,
        },
      }));
    } catch (err) {
      console.error(err);
      const k = keyForMessage(modeKey, index);
      setMessageFeedback((prev) => ({
        ...prev,
        [k]: {
          ...(prev[k] || {}),
          error: err.message || "Feedback failed",
        },
      }));
    } finally {
      setSendingChatFeedback(false);
    }
  }

  // ---------- Doc upload ----------

  async function handleUploadPdf() {
    if (!uploadFile) {
      setUploadStatus("Choose a PDF first.");
      return;
    }
    setUploadLoading(true);
    setUploadStatus(null);

    const formData = new FormData();
    formData.append("file", uploadFile);

    try {
      const res = await fetch(`${API_URL}/upload_pdf`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`upload_pdf failed with status ${res.status}`);
      }

      const data = await res.json();

      const newDoc = {
        doc_id: data.doc_id,
        source: data.filename || data.doc_id,
        num_chunks: data.num_chunks,
      };

      setDocuments((prev) => [...prev, newDoc]);
      setSelectedDocId(data.doc_id);
      setUploadStatus(`Uploaded & indexed: ${data.doc_id}`);
      setUploadFile(null);
    } catch (err) {
      console.error(err);
      setUploadStatus(err.message || "Upload failed.");
    } finally {
      setUploadLoading(false);
    }
  }

  function handleUploadFileChange(e) {
    const file = e.target.files?.[0];
    if (file) {
      setUploadFile(file);
      setUploadStatus(null);
    } else {
      setUploadFile(null);
    }
  }

  // ---------- Render ----------

  return (
    <div className="app-root">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo-dot" />
          <div>
            <div className="app-name">TriGPT</div>
            <div className="app-subtitle">IreneAdler • local AI lab</div>
          </div>
        </div>

        {/* Modes */}
        <div className="sidebar-section">
          <div className="section-title">Modes</div>
          <ul className="mode-list">
            <li
              className={
                "mode-item " + (mode === MODES.DOC ? "mode-item-active" : "")
              }
              onClick={() => switchMode(MODES.DOC)}
            >
              <span>Doc Q&amp;A</span>
              <small>Selected PDF only</small>
            </li>
            <li
              className={
                "mode-item " +
                (mode === MODES.GLOBAL ? "mode-item-active" : "")
              }
              onClick={() => switchMode(MODES.GLOBAL)}
            >
              <span>All Docs Q&amp;A</span>
              <small>Across all PDFs</small>
            </li>
            <li
              className={
                "mode-item " +
                (mode === MODES.GENERAL ? "mode-item-active" : "")
              }
              onClick={() => switchMode(MODES.GENERAL)}
            >
              <span>General Chat</span>
              <small>World knowledge</small>
            </li>
            <li
              className={
                "mode-item " +
                (mode === MODES.VISION ? "mode-item-active" : "")
              }
              onClick={() => switchMode(MODES.VISION)}
            >
              <span>Vision (face &amp; emotion)</span>
              <small>Upload or camera</small>
            </li>
          </ul>
        </div>

        {/* Status */}
        <div className="sidebar-section">
          <div className="section-title">Status</div>
          {loadError && (
            <div className="status-badge status-badge-error">
              {loadError}
            </div>
          )}
          {health && !loadError && (
            <div className="status-grid">
              <StatusPill label="Face" ok={health.face_classifier} />
              <StatusPill label="Intent" ok={health.question_classifier} />
              <StatusPill label="Emotion" ok={health.emotion_classifier} />
              <StatusPill label="Reranker" ok={health.reranker} />
              <StatusPill label="Object" ok={health.object_detector} />
            </div>
          )}
        </div>

        {/* Documents & upload: only in Doc mode */}
        {mode === MODES.DOC && (
          <div className="sidebar-section sidebar-docs">
            <div className="section-title">Documents</div>

            <div className="upload-box">
              <label className="file-label">
                <span>Select PDF</span>
                <input
                  type="file"
                  accept="application/pdf"
                  onChange={handleUploadFileChange}
                />
              </label>
              <button
                className="upload-btn"
                onClick={handleUploadPdf}
                disabled={uploadLoading || !uploadFile}
              >
                {uploadLoading ? "Uploading..." : "Upload & index"}
              </button>
              <div className="file-name">
                {uploadFile ? uploadFile.name : "No file selected"}
              </div>
              {uploadStatus && (
                <div className="hint-text small">{uploadStatus}</div>
              )}
            </div>

            {documents.length === 0 ? (
              <div className="hint-text">
                No docs yet. Upload a PDF to get started.
              </div>
            ) : (
              <ul className="doc-list">
                {documents.map((doc) => {
                  const active = doc.doc_id === selectedDocId;
                  return (
                    <li
                      key={doc.doc_id}
                      className={
                        "doc-item" + (active ? " doc-item-active" : "")
                      }
                      onClick={() => setSelectedDocId(doc.doc_id)}
                    >
                      <div className="doc-title">{doc.doc_id}</div>
                      <div className="doc-subtitle">{doc.source}</div>
                      <div className="doc-meta">
                        {doc.num_chunks}{" "}
                        {doc.num_chunks === 1 ? "chunk" : "chunks"}
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        )}

        <div className="sidebar-footer">
          <div className="tiny-text">
            Backend: <code>{API_URL}</code>
          </div>
        </div>
      </aside>

      {/* Chat area */}
      <section className="chat-pane">
        <header className="chat-header">
          <div>
            <h1>Chat with IreneAdler</h1>
            <p>
              Mode:{" "}
              <span className="doc-pill">
                {mode === MODES.DOC
                  ? "Doc Q&A"
                  : mode === MODES.GLOBAL
                    ? "All Docs Q&A"
                    : mode === MODES.GENERAL
                      ? "General Chat"
                      : "Vision (face & emotion)"}
              </span>
            </p>
            {mode === MODES.DOC && (
              <p style={{ marginTop: "0.3rem", fontSize: "0.8rem" }}>
                Document:{" "}
                <span className="doc-pill">
                  {selectedDocId || "None selected"}
                </span>
              </p>
            )}
          </div>
        </header>

        <div className="chat-messages">
          {/* Vision: upload + camera + feedback + live cam */}
          {mode === MODES.VISION && (
            <div className="vision-panel">
              <div className="vision-row">
                <div className="vision-card">
                  <div className="vision-card-title">Upload image</div>
                  <label className="file-label">
                    <span>Choose image</span>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleVisionFileChange}
                    />
                  </label>
                  <div className="file-name">
                    {visionFile ? visionFile.name : "No file selected"}
                  </div>
                </div>

                <div className="vision-card">
                  <div className="vision-card-title">Camera snapshot</div>

                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    className={
                      "camera-video " +
                      (cameraActive ? "" : "camera-video-hidden")
                    }
                  />

                  <div className="camera-controls">
                    {!cameraActive ? (
                      <button
                        className="upload-btn"
                        type="button"
                        onClick={startCamera}
                      >
                        Open camera
                      </button>
                    ) : (
                      <>
                        <button
                          className="upload-btn"
                          type="button"
                          onClick={captureFromCamera}
                        >
                          Capture
                        </button>
                        <button
                          className="secondary-btn"
                          type="button"
                          onClick={stopCamera}
                        >
                          Cancel
                        </button>
                      </>
                    )}
                  </div>

                  <canvas ref={canvasRef} style={{ display: "none" }} />
                </div>
              </div>

              {/* Live Cam (continuous ID using /predict_face) */}
              <div className="vision-row" style={{ marginTop: "1rem" }}>
                <div className="vision-card">
                  <div className="vision-card-title">
                    Live camera (who is in front of me?)
                  </div>
                  <LiveCam />
                </div>
              </div>

              {visionPreviewUrl && (
                <div className="vision-preview">
                  <div className="vision-card-title">
                    Current image &amp; analysis
                  </div>
                  <img
                    src={visionPreviewUrl}
                    alt="Selected"
                    className="vision-preview-img"
                  />
                  <div style={{ marginTop: "0.75rem" }}>
                    <button
                      className="upload-btn"
                      type="button"
                      onClick={handleVisionDetect}
                      disabled={loading || !visionFile}
                    >
                      {loading ? "Analyzing..." : "Analyze objects & faces"}
                    </button>
                  </div>
                </div>
              )}

              {visionDetections.length > 0 && (
                <div className="vision-detections">
                  <div className="vision-card-title">
                    Detections &amp; feedback
                  </div>
                  <ul className="det-list">
                    {visionDetections.map((d, idx) => {
                      const label = d.label || d.raw_label || "unknown";
                      const score =
                        typeof d.score === "number"
                          ? d.score.toFixed(2)
                          : "unknown";

                      const hasBBox =
                        typeof d.x === "number" &&
                        typeof d.y === "number" &&
                        typeof d.w === "number" &&
                        typeof d.h === "number";

                      return (
                        <li key={idx} className="det-item">
                          <div className="det-main">
                            <div>
                              <strong>{label}</strong> (score {score})
                            </div>
                            {label === "human" && (
                              <div className="det-sub">
                                <div>
                                  Model guess:{" "}
                                  <code>
                                    {d.identity || "unknown"} (
                                    {typeof d.identity_confidence === "number"
                                      ? d.identity_confidence.toFixed(2)
                                      : "?"}
                                    )
                                  </code>
                                </div>
                                {d.identity_info && (
                                  <div className="hint-text small">
                                    {d.identity_info}
                                  </div>
                                )}
                              </div>
                            )}
                            {hasBBox && (
                              <div className="tiny-text">
                                bbox: x={d.x.toFixed(2)}, y={d.y.toFixed(
                                  2
                                )}, w={d.w.toFixed(2)}, h={d.h.toFixed(2)}
                              </div>
                            )}
                          </div>

                          {label === "human" && (
                            <div className="det-feedback">
                              <select
                                className="feedback-select"
                                value={feedbackLabels[idx] || ""}
                                onChange={(e) =>
                                  onFeedbackLabelChange(idx, e.target.value)
                                }
                              >
                                <option value="">
                                  Correct identity (optional)
                                </option>
                                {IDENTITY_OPTIONS.map((opt) => (
                                  <option key={opt} value={opt}>
                                    {opt}
                                  </option>
                                ))}
                              </select>
                              <button
                                type="button"
                                className="secondary-btn"
                                onClick={() => handleSendFeedback(idx)}
                                disabled={feedbackLoading}
                              >
                                {feedbackLoading
                                  ? "Sending..."
                                  : "Send feedback"}
                              </button>
                            </div>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                  {feedbackStatus && (
                    <div
                      className="hint-text small"
                      style={{ marginTop: "0.5rem" }}
                    >
                      {feedbackStatus}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {currentMessages.map((m, idx) => (
            <MessageBubble
              key={idx}
              mode={mode}
              index={idx}
              message={m}
              feedback={messageFeedback[keyForMessage(mode, idx)]}
              onRate={handleRateMessage}
              onCommentChange={handleFeedbackCommentChange}
              onSubmitComment={handleSubmitFeedbackWithComment}
              sendingChatFeedback={sendingChatFeedback}
            />
          ))}

          {loading && (
            <div className="typing-indicator">
              Irene is thinking<span className="dots">...</span>
            </div>
          )}
        </div>

        <footer className="chat-input-bar">
          <form
            className="chat-form"
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
          >
            <textarea
              rows={1}
              className="chat-input"
              placeholder={
                mode === MODES.VISION
                  ? "Ask about the uploaded / captured image..."
                  : mode === MODES.DOC
                    ? "Ask anything about this document..."
                    : mode === MODES.GENERAL
                      ? "Ask anything (general knowledge)..."
                      : "Ask anything across your documents..."
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button
              type="submit"
              className="send-btn"
              disabled={loading || !input.trim()}
            >
              {loading ? "Sending..." : "Send"}
            </button>
          </form>

          <div className="hint-text small">
            Enter = send, Shift+Enter = new line.
          </div>
        </footer>
      </section>
    </div>
  );
}

function StatusPill({ label, ok }) {
  return (
    <div className={"status-pill " + (ok ? "ok" : "bad")}>
      <span className="status-dot" />
      <span>{label}</span>
    </div>
  );
}

function MessageBubble({
  mode,
  index,
  message,
  feedback,
  onRate,
  onCommentChange,
  onSubmitComment,
  sendingChatFeedback,
  // optional for face-detection feedback in Vision mode
  onVisionLike,
  onVisionDislike,
}) {
  const isUser = message.role === "user";
  const showFeedback = !isUser; // only on assistant messages

  const rating = feedback?.rating;
  const waitingForComment = feedback?.waitingForComment;
  const submitted = feedback?.submitted;
  const error = feedback?.error;
  const comment = feedback?.comment || "";

  // Collapse user newlines so user bubbles don't become super tall,
  // but keep assistant newlines (for formatting).
  const rawText = message.content || "";
  const displayText = isUser ? rawText.replace(/\s*\n\s*/g, " ") : rawText;

  // Heuristic: this message probably has a face-identity result
  const isVisionIdentityMessage =
    !isUser &&
    mode === "vision" && // or use MODES.VISION if available in this file
    typeof rawText === "string" &&
    rawText.toLowerCase().includes("identity label:");

  return (
    <div
      className={
        "message-row " +
        (isUser ? "message-row-user" : "message-row-assistant")
      }
    >
      <div className="avatar-circle">{isUser ? "😼" : "🤖"}</div>

      <div
        className={
          "message-bubble " +
          (isUser ? "message-bubble-user" : "message-bubble-assistant")
        }
      >
        {/* Brand name: Irene (TriGPT) above assistant bubbles */}
        {!isUser && (
          <div className="assistant-name-inline">
            <span className="assistant-name-main">Irene</span>{" "}
            <span className="assistant-name-brand">(TriGPT)</span>
          </div>
        )}

        {/* Main message text */}
        <p>{displayText}</p>

        {showFeedback && (
          <div className="message-feedback">
            {/* 1–5 star rating: answer quality */}
            {!submitted && !waitingForComment && !rating && (
              <div className="feedback-row">
                <span className="tiny-text">Rate this answer:</span>
                <div className="feedback-stars">
                  {[1, 2, 3, 4, 5].map((r) => (
                    <button
                      key={r}
                      type="button"
                      className={
                        "star-btn " + (rating && rating >= r ? "star-on" : "")
                      }
                      onClick={() => onRate(mode, index, r)}
                    >
                      {r}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* If user gave 1–2★, ask for comment */}
            {waitingForComment && (
              <div className="feedback-comment-block">
                <div className="tiny-text">
                  You gave {rating}★. Nếu câu trả lời chưa ổn, bạn có thể gợi ý
                  Irene nên trả lời thế nào tốt hơn (optional):
                </div>
                <textarea
                  className="feedback-textarea"
                  rows={2}
                  value={comment}
                  onChange={(e) =>
                    onCommentChange(mode, index, e.target.value)
                  }
                  placeholder="Ví dụ: trả lời ngắn gọn hơn / dùng tiếng Việt tự nhiên hơn / tập trung vào X..."
                />
                <div className="feedback-actions">
                  <button
                    type="button"
                    className="secondary-btn"
                    disabled={sendingChatFeedback}
                    onClick={() => onSubmitComment(mode, index, false)}
                  >
                    {sendingChatFeedback ? "Sending..." : "Send feedback"}
                  </button>
                  <button
                    type="button"
                    className="tiny-link-btn"
                    disabled={sendingChatFeedback}
                    onClick={() => onSubmitComment(mode, index, true)}
                  >
                    Skip comment
                  </button>
                </div>
              </div>
            )}

            {/* Thanks text after feedback sent */}
            {submitted && (
              <div className="tiny-text feedback-thanks">
                {rating >= 4
                  ? `Thanks for the ${rating}★!`
                  : rating
                    ? `Thanks for the feedback (${rating}★).`
                    : "Thanks for the feedback."}
              </div>
            )}

            {error && (
              <div className="tiny-text" style={{ color: "#f87171" }}>
                {error}
              </div>
            )}

            {/* 👍 / 👎 for face detection, only in Vision identity answers */}
            {isVisionIdentityMessage && (
              <div className="vision-like-row">
                <span className="tiny-text" style={{ marginRight: "0.3rem" }}>
                  Face detection:
                </span>
                <button
                  type="button"
                  className="thumb-btn"
                  onClick={() =>
                    onVisionLike && onVisionLike(mode, index, message)
                  }
                >
                  👍
                </button>
                <button
                  type="button"
                  className="thumb-btn"
                  onClick={() =>
                    onVisionDislike && onVisionDislike(mode, index, message)
                  }
                >
                  👎
                </button>
                <span className="tiny-text" style={{ marginLeft: "0.3rem" }}>
                  correct / wrong
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}



/**
 * LiveCam component:
 * - Shows webcam preview
 * - Sends a SMALL frame every ~1.2s to /predict_face
 * - Displays who Irene sees right now
 */
function LiveCam() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Start live camera
  const startLiveCamera = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsCameraOn(true);
    } catch (err) {
      console.error("Error starting live camera", err);
      setError("Không mở được camera (check quyền truy cập).");
    }
  };

  const stopLiveCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraOn(false);
    setIsScanning(false);
  };

  // Cleanup when unmount
  useEffect(() => {
    return () => {
      stopLiveCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const captureAndSend = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    if (isBusy) return;

    setIsBusy(true);
    setError(null);

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      // Downscale frame to reduce lag
      const targetWidth = 320;
      const targetHeight = Math.round(
        (video.videoHeight / video.videoWidth) * targetWidth || 240
      );

      canvas.width = targetWidth;
      canvas.height = targetHeight;

      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, targetWidth, targetHeight);

      const blob = await new Promise((resolve) =>
        canvas.toBlob((b) => resolve(b), "image/jpeg", 0.8)
      );
      if (!blob) {
        throw new Error("Failed to capture frame");
      }

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const res = await fetch(`${API_URL}/predict_face`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`API error: ${res.status} ${text}`);
      }

      const data = await res.json();
      setResult({
        label: data.predicted_class,
        confidence: data.confidence,
        ts: new Date().toLocaleTimeString(),
      });
    } catch (err) {
      console.error(err);
      setError(err.message || "Lỗi khi gửi frame.");
    } finally {
      setIsBusy(false);
    }
  };

  // Auto-scan loop (about every 1.2s)
  useEffect(() => {
    if (!isCameraOn || !isScanning) return;

    let cancelled = false;

    const tick = async () => {
      if (cancelled) return;
      await captureAndSend();
      if (cancelled) return;
      setTimeout(tick, 1200);
    };

    tick();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isCameraOn, isScanning]);

  const handleScanOnce = () => {
    if (!isCameraOn || isBusy) return;
    captureAndSend();
  };

  const toggleAutoScan = () => {
    if (!isCameraOn) return;
    setIsScanning((prev) => !prev);
  };

  return (
    <div>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={
          "camera-video " + (isCameraOn ? "" : "camera-video-hidden")
        }
      />
      <canvas ref={canvasRef} style={{ display: "none" }} />

      <div className="camera-controls" style={{ marginTop: "0.5rem" }}>
        {!isCameraOn ? (
          <button
            className="upload-btn"
            type="button"
            onClick={startLiveCamera}
          >
            🎥 Start live camera
          </button>
        ) : (
          <>
            <button
              className="upload-btn"
              type="button"
              onClick={handleScanOnce}
              disabled={isBusy}
            >
              📸 Scan once
            </button>
            <button
              className={
                "secondary-btn " + (isScanning ? "secondary-btn-active" : "")
              }
              type="button"
              onClick={toggleAutoScan}
            >
              {isScanning ? "⏸ Pause auto-scan" : "▶ Auto-scan (slow)"}
            </button>
            <button
              className="secondary-btn"
              type="button"
              onClick={stopLiveCamera}
            >
              🛑 Stop
            </button>
          </>
        )}
      </div>

      <div className="hint-text small" style={{ marginTop: "0.5rem" }}>
        {result ? (
          <>
            <div>
              <strong>Who I see:</strong> {result.label}
            </div>
            <div>
              <strong>Confidence:</strong>{" "}
              {(result.confidence * 100).toFixed(1)}%
            </div>
            <div className="tiny-text">Last updated: {result.ts}</div>
          </>
        ) : (
          <span>
            Start camera, then use “Scan once” or “Auto-scan” to identify
            who is in front of you.
          </span>
        )}
      </div>

      {isBusy && (
        <div className="tiny-text" style={{ marginTop: "0.25rem" }}>
          Đang xử lý frame...
        </div>
      )}
      {error && (
        <div
          className="tiny-text"
          style={{ marginTop: "0.25rem", color: "#f87171" }}
        >
          Lỗi: {error}
        </div>
      )}
    </div>
  );
}

export default App;
