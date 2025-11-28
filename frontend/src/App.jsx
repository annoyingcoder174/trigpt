import { useEffect, useState, useRef } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const MODES = {
  DOC: "doc",
  GLOBAL: "global",
  GENERAL: "general",
  VISION: "vision",
};

function App() {
  const [health, setHealth] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState(null);

  const [mode, setMode] = useState(MODES.DOC);

  const [conversations, setConversations] = useState({
    [MODES.DOC]: [
      {
        role: "assistant",
        content:
          "Hi, I'm IreneAdler. Upload a PDF on the left, select it, and ask me anything about that document.",
      },
    ],
    [MODES.GLOBAL]: [
      {
        role: "assistant",
        content:
          "Hi, I'm IreneAdler. Ask anything across all your indexed documents.",
      },
    ],
    [MODES.GENERAL]: [
      {
        role: "assistant",
        content:
          "Hi, I'm IreneAdler. This is general chat. Ask me anything, not just about your PDFs.",
      },
    ],
    [MODES.VISION]: [
      {
        role: "assistant",
        content:
          "Hi, I'm IreneAdler. Upload a face image or take a snapshot with your camera, then ask a question about it.",
      },
    ],
  });

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState(null);

  // Vision-specific state
  const [visionFile, setVisionFile] = useState(null);
  const [visionPreviewUrl, setVisionPreviewUrl] = useState(null);

  // Camera for Vision
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

  // Cleanup camera + preview URL on unmount
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
        // Vision mode: /vision_qa
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

        setMessagesForMode(mode, [
          ...newMessages,
          { role: "assistant", content: assistantText },
        ]);
      } else if (mode === MODES.GENERAL) {
        // General chat: /chat
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

        setMessagesForMode(mode, [
          ...newMessages,
          { role: "assistant", content: assistantText },
        ]);
      } else {
        // Doc / Global modes: /ask
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

        setMessagesForMode(mode, [
          ...newMessages,
          { role: "assistant", content: assistantText },
        ]);
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

  // ---------- Vision: upload / camera ----------

  function handleVisionFileChange(e) {
    const file = e.target.files?.[0];
    if (file) {
      if (visionPreviewUrl) {
        URL.revokeObjectURL(visionPreviewUrl);
      }
      setVisionFile(file);
      setVisionPreviewUrl(URL.createObjectURL(file));
    } else {
      if (visionPreviewUrl) {
        URL.revokeObjectURL(visionPreviewUrl);
      }
      setVisionFile(null);
      setVisionPreviewUrl(null);
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

        // release camera after capture
        stopCamera();
      },
      "image/jpeg",
      0.9
    );
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
          {/* Vision: upload + camera */}
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

              {visionPreviewUrl && (
                <div className="vision-preview">
                  <div className="vision-card-title">Current image</div>
                  <img
                    src={visionPreviewUrl}
                    alt="Selected"
                    className="vision-preview-img"
                  />
                </div>
              )}
            </div>
          )}

          {currentMessages.map((m, idx) => (
            <MessageBubble key={idx} message={m} />
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

function MessageBubble({ message }) {
  const isUser = message.role === "user";
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
        {message.content.split("\n").map((line, i) => (
          <p key={i}>{line}</p>
        ))}
      </div>
    </div>
  );
}

export default App;
