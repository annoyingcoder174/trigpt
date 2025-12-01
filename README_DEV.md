TriGPT – Local AI Lab (IreneAdler)
==================================

GitHub repo: https://github.com/annoyingcoder174/trigpt


==================================
0. QUICKSTART – HOW TO RUN
==================================

If you just cloned the repo and want to run everything, do this:

1) Create & activate virtualenv (backend)
-----------------------------------------

macOS / Linux:

    cd /path/to/trigpt
    python3 -m venv .venv
    source .venv/bin/activate

Windows (PowerShell):

    cd C:\path\to\trigpt
    python -m venv .venv
    .\.venv\Scripts\activate

You should see `(.venv)` at the start of your terminal prompt.

2) Set up env files
-------------------

Backend `.env` (in project root: `trigpt/.env`):

    OPENAI_API_KEY=sk-...          # if you use OpenAI
    AWS_REGION=us-east-2
    AWS_S3_BUCKET=trigpt-docs-pt123

Frontend `.env` (in `trigpt/frontend/.env`):

    VITE_API_URL=http://127.0.0.1:8000

3) Install backend Python packages
----------------------------------

With venv active and at project root:

    pip install fastapi uvicorn[standard] pydantic
    pip install pillow
    pip install torch torchvision
    pip install chromadb
    pip install python-dotenv
    pip install boto3              # needed because you use S3
    pip install opencv-python
    pip install pillow-heif        # for HEIC images (optional but useful)

(Or later you can switch to `pip install -r requirements.txt` once you have it.)

4) Install frontend dependencies
--------------------------------

    cd frontend
    npm install
    cd ..

==================================
5. RUN BACKEND
==================================
From project root (with venv active):

    cd C:\path\to\trigpt
   python3 -m venv .venv
   source .venv/bin/activate

    uvicorn src.api:app --reload

Backend will be at:

    http://127.0.0.1:8000

Swagger docs:

    http://127.0.0.1:8000/docs

6) Run frontend
---------------

Open a second terminal:

    cd /path/to/trigpt/frontend
    npm run dev

Frontend will be at (by default):

    http://127.0.0.1:5173

Open that in your browser and chat with IreneAdler.


==================================
1. Project layout
==================================

After cloning, the important folders/files are:

- src/
  - api.py                 -> main FastAPI backend (all endpoints live here)
  - vision_model.py        -> face classifier wrapper
  - train_face_model.py    -> script to train the face model
  - identity_db.py         -> structured info about PTri, Muse, friends, etc.
  - irene_style_examples.py-> style examples (EN + VI) for Irene’s tone
  - other helpers:
      doc_ingestion.py     -> PDF text loading + chunking
      vector_store.py      -> Chroma / vector DB logic
      s3_storage.py        -> S3 upload helpers (if used)
      emotion_model.py     -> emotion classifier wrapper
      object_detector.py   -> object detection wrapper
      question_classifier.py, reranker.py, llm_client.py, ...

- frontend/
  - Vite + React app (chat UI / dashboard)

- models/
  - model checkpoints (face classifier, emotion model, etc.)
  - THIS FOLDER IS GIT-IGNORED (not pushed to GitHub)

- data/
  - local runtime data (faces dataset, feedback logs, etc.)
  - THIS FOLDER IS GIT-IGNORED

- .env (backend, in project root)
  - environment variables (API keys, S3 config, etc.)
  - THIS FILE IS GIT-IGNORED

- frontend/.env (frontend)
  - points the React app to the backend API
  - THIS FILE IS ALSO GIT-IGNORED


==================================
2. Cloning the repo
==================================

From any folder you like (for example Desktop):

    cd ~/Desktop
    git clone https://github.com/annoyingcoder174/trigpt.git
    cd trigpt


==================================
3. Python setup (backend) – details
==================================

3.1 Create and activate venv
----------------------------

macOS / Linux:

    python3 -m venv .venv
    source .venv/bin/activate

Windows (PowerShell):

    python -m venv .venv
    .\.venv\Scripts\activate

You’ll know it worked when your terminal prompt starts with:

    (.venv) ...

To leave the virtualenv later, just type:

    deactivate


3.2 Install Python dependencies
-------------------------------

If you have a requirements.txt file in the project later:

    pip install -r requirements.txt

If not, install the key libraries manually (adjust if you add more):

    pip install fastapi uvicorn[standard] pydantic
    pip install pillow
    pip install torch torchvision
    pip install chromadb
    pip install python-dotenv
    pip install boto3              # needed because you use S3
    pip install opencv-python
    pip install pillow-heif        # for HEIC images (optional but useful)

If the backend crashes complaining about a missing package, just:

    pip install <package-name>

and run again.


==================================
4. Frontend setup (React + Vite) – details
==================================

From the project root:

    cd frontend
    npm install
    cd ..

The frontend has its own `.env` file at:

    frontend/.env

In your current setup it contains:

    VITE_API_URL=http://127.0.0.1:8000

This tells the React app to call the backend running on your local machine.


==================================
5. Environment variables (.env)
==================================

Backend `.env` (in project root):

Current example:

    OPENAI_API_KEY=sk-...          # your OpenAI key (if used)
    AWS_REGION=us-east-2
    AWS_S3_BUCKET=trigpt-docs-pt123

These values are read in code like:
- `llm_client.py` (for OPENAI key)
- `s3_storage.py` (for AWS region and S3 bucket)

Frontend `.env` (in frontend/):

    VITE_API_URL=http://127.0.0.1:8000


==================================
6. Running the backend (FastAPI)
==================================

Always make sure your venv is active first:

    source .venv/bin/activate      # macOS / Linux
    .\.venv\Scripts\activate       # Windows

Then, from the project root:

    uvicorn src.api:app --reload

Backend URL:

    http://127.0.0.1:8000

Swagger / docs:

    http://127.0.0.1:8000/docs


==================================
7. Running the frontend (chat UI)
==================================

Open a second terminal.

From the project root:

    cd frontend
    npm run dev

Frontend URL (by default):

    http://127.0.0.1:5173


==================================
8. Face dataset & training the face model
==================================

The face classifier recognizes:

- PTri
- Lanh
- MTuan
- BHa
- PTri's Muse
- strangers

Create the dataset structure:

    data/
      faces/
        train/
          PTri/
          Lanh/
          MTuan/
          BHa/
          PTri's Muse/
          strangers/
        val/
          PTri/
          Lanh/
          MTuan/
          BHa/
          PTri's Muse/
          strangers/

Inside each folder, put images of that person/class.

Supported file types:

- .jpg
- .jpeg
- .png
- .webp
- (and .heic if you installed pillow-heif)

Then, with venv active and from the project root, run:

    python -m src.train_face_model

If training is successful, you should get:

    models/face_classifier.pt

The API will load this automatically on startup.


==================================
9. Emotion & object detection
==================================

- Emotion model:
    models/emotion_classifier.pt

- Object detector:
    created in src.api: `ObjectDetector(score_thresh=0.7)`

If those models are missing, the API still runs, but:

- /live_emotion and emotion in /vision_qa may be disabled
- /live_detect may be disabled

Check component status via:

    GET http://127.0.0.1:8000/health

You get flags like:
- face_classifier: true/false
- question_classifier: true/false
- emotion_classifier: true/false
- reranker: true/false
- object_detector: true/false


==================================
10. Feedback endpoints (for improving Irene later)
==================================

1) Chat feedback
----------------

Endpoint:

    POST /chat_feedback

Body (JSON):

    {
      "question": "what you asked",
      "answer": "what Irene replied",
      "rating": 1-5,
      "comment": "optional comment",
      "mode": "doc" | "global" | "general" | "vision"
    }

Data is saved to:

    data/chat_feedback.jsonl

This does NOT auto-train the model yet, but it gives you a log of:
- which answers were good (4–5 stars)
- which answers were bad (1–2 stars)

Later, you can:
- use good examples to expand irene_style_examples.py
- or build a fine-tuning dataset.


2) Face feedback
----------------

Endpoint:

    POST /face_feedback

Form fields:

    label: one of your classes (PTri, Lanh, MTuan, BHa, "PTri's Muse", strangers)
    file:  image (face crop)

It saves the corrected images to:

    data/faces/train/<label>/feedback_*.jpg

Later, you can retrain:

    python -m src.train_face_model

to improve the face model.


==================================
11. Identity & style (how Irene talks about people)
==================================

- identity_db.py
  - Stores structured info:
    - nickname
    - Instagram
    - birthday
    - gender
    - university
    - careers
    - hobbies
    - favorite music, games, movies
    - how you met
    - special memories
    - priority_level
    - base_text summary

- irene_style_examples.py
  - Stores example Q&A pairs that define how Irene should talk
    (English and Vietnamese).
  - /vision_qa and /chat use these examples to shape tone,
    so Irene sounds more like a close friend, not a database.

If you want Irene to sound more natural over time:
- Add more high-quality examples to STYLE_EXAMPLES["en"] and ["vi"],
- Especially based on real answers you like (4–5 star feedback).


==================================
12. Typical dev workflow (summary)
==================================

Every day when you come back to code:

    cd /path/to/trigpt

1) Activate venv:

    source .venv/bin/activate
    # or on Windows:
    .\.venv\Scripts\activate

2) Start backend:

    uvicorn src.api:app --reload

3) Open another terminal, start frontend:

    cd /path/to/trigpt/frontend
    npm run dev

4) Use in browser:

    http://127.0.0.1:5173  (frontend)
    http://127.0.0.1:8000/docs  (backend docs)

5) When done:

    Ctrl+C in both terminals
    deactivate    # if you want to exit venv


==================================
13. Git usage (for this repo)
==================================

Basic commands:

    # See what changed
    git status

    # Stage everything
    git add .

    # Commit with a message
    git commit -m "Update Irene style and feedback features"

    # Push to GitHub
    git push origin main

If you ever see `(.venv)` in the terminal and want to get out:

    deactivate

Then you can still run git commands normally.


----------------------------------
End of readme.txt
----------------------------------
