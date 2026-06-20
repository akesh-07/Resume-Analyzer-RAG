# Resume RAG Analyzer

AI-powered resume evaluation app using **Retrieval-Augmented Generation (RAG)**. Upload PDF resumes, paste a job description, and get ATS-style match reports powered by Gemini and FAISS.

## Features

- Upload multiple PDF resumes at once
- Custom job description input
- RAG pipeline: chunk → embed → retrieve → evaluate
- Match percentage, skills gap analysis, strengths, weaknesses, and recommendations
- Clean Streamlit UI with per-resume result cards

## Setup

1. **Clone / open the project**

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set your Google API key**

   Copy `.env.example` to `.env` and add your key:

   ```bash
   copy .env.example .env       # Windows
   # cp .env.example .env       # macOS/Linux
   ```

   Get a key from [Google AI Studio](https://aistudio.google.com/apikey).

   You can also enter the key directly in the app sidebar.

## Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Usage

1. Upload one or more PDF resumes
2. Paste or edit the job description
3. Click **Analyze Resumes**
4. Review match scores and detailed evaluations for each candidate

## Project Structure

```
RESUME_RAG/
├── app.py           # Streamlit frontend
├── rag_engine.py    # RAG pipeline (PDF extraction, FAISS, Gemini)
├── requirements.txt
├── .env.example
└── Resume_using_RAG.ipynb   # Original notebook
```

## Notes

- First run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) — may take a minute.
- PDFs must contain selectable text (not scanned images without OCR).
- Never commit `.env` or expose API keys in notebooks or version control.
