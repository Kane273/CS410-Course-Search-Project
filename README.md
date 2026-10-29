# Course Search Engine

Search engine for lecture transcripts and PDF course materials.

## Planned Stack

- React + Vite
- Python + Flask
- PyMuPDF
- BM25
- TF-IDF + cosine similarity
- SQLite

## Pipeline

Raw TXT/PDF
→ extraction
→ cleaning
→ chunking
→ indexing
→ BM25 / TF-IDF ranking
→ Flask API
→ React UI

## Data

Place local course files in:

- `data/raw/transcripts/`
- `data/raw/books/`

Raw course data should not be committed to Git.
