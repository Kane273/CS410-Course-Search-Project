# CS410 Course Search Project — Team Handoff Plan

## Goal

Build an MVP search engine for course transcripts and PDF material.

The system will:

- extract and clean text
- split documents into searchable passages
- rank passages using BM25
- compare BM25 against TF-IDF + cosine similarity
- expose search through a Flask API
- display results in a React frontend
- evaluate ranking quality using Precision@10 and MAP

---

## End-to-End Pipeline

```text
Raw TXT / PDF files
        |
        v
[1] Extract text
        |
        v
[2] Clean text
        |
        v
[3] Chunk + metadata
        |
        v
     chunks.json
        |
   +----+----+
   |         |
   v         v
[4] BM25   [5] TF-IDF
   |         |
   +----+----+
        |
        v
   Ranked results
        |
        v
[6] Flask API
        |
        v
   React frontend
        |
        v
 Evaluation + Demo
```

---

## Team Split

| Person | Area | Input | Output | Main Libraries |
|---|---|---|---|---|
| 1 | Text extraction | TXT + PDF | `documents.json` | Python, PyMuPDF |
| 2 | Cleaning + chunking | `documents.json` | `chunks.json` | Python, `re` |
| 3 | BM25 ranking | `chunks.json` | `search_bm25()` | `rank-bm25` |
| 4 | TF-IDF baseline | `chunks.json` | `search_tfidf()` | scikit-learn |
| 5 | Backend/API | search functions | `/search` API | Flask |
| 6 | Frontend + evaluation | API + ranked results | UI + metrics | React, Vite, Python |

---

## Shared Data Contract

### Extracted Document

```json
{
  "doc_id": "lecture_03",
  "course": "CS410",
  "doc_type": "transcript",
  "title": "Lecture 3",
  "source_file": "lecture_03.txt",
  "text": "Full extracted text..."
}
```

### Searchable Chunk

```json
{
  "chunk_id": "lecture_03_0005",
  "doc_id": "lecture_03",
  "course": "CS410",
  "doc_type": "transcript",
  "title": "Lecture 3",
  "source_file": "lecture_03.txt",
  "section": "Week 3",
  "page": null,
  "text": "BM25 is a ranking function...",
  "tokens": ["bm25", "is", "a", "ranking", "function"]
}
```

### Search Result

```json
{
  "chunk_id": "lecture_03_0005",
  "score": 8.42,
  "title": "Lecture 3",
  "course": "CS410",
  "doc_type": "transcript",
  "source_file": "lecture_03.txt",
  "text": "BM25 is a ranking function..."
}
```

---

## Module Responsibilities

### 1. Text Extraction

**Input**

```text
data/raw/transcripts/*.txt
data/raw/books/*.pdf
```

**Process**

```python
# TXT
text = path.read_text()

# PDF
doc = fitz.open(path)
text = "\n".join(page.get_text() for page in doc)
```

**Output**

```text
data/processed/documents.json
```

---

### 2. Cleaning + Chunking

**Input**

```text
documents.json
```

**Process**

```python
text = re.sub(r"\s+", " ", text).strip()
```

Split into approximately 300–500 word chunks with small overlap.

```python
chunk_size = 400
overlap = 50
```

**Output**

```text
data/processed/chunks.json
```

---

### 3. BM25 Search

**Input**

```text
chunks.json
```

**Process**

```python
bm25 = BM25Okapi([chunk["tokens"] for chunk in chunks])
scores = bm25.get_scores(query.lower().split())
```

**Interface**

```python
search_bm25(query, top_k=10)
```

**Output**

Ranked search-result objects.

---

### 4. TF-IDF + Cosine Search

**Input**

```text
chunks.json
```

**Process**

```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

q = vectorizer.transform([query])
scores = cosine_similarity(q, X)
```

**Interface**

```python
search_tfidf(query, top_k=10)
```

**Output**

Ranked search-result objects.

---

### 5. Flask Backend

**Input**

```python
search_bm25()
search_tfidf()
```

**API**

```text
GET /search?q=document+ranking&method=bm25
```

Example:

```python
@app.route("/search")
def search():
    query = request.args.get("q")
    method = request.args.get("method", "bm25")

    if method == "tfidf":
        results = search_tfidf(query)
    else:
        results = search_bm25(query)

    return jsonify(results)
```

---

### 6. React Frontend + Evaluation

Frontend sends:

```javascript
fetch(`/search?q=${query}&method=${method}`)
```

UI displays:

```text
Lecture 3
Transcript | CS410

"BM25 is a ranking function used by search engines..."

Score: 8.42
Source: lecture_03.txt
```

Evaluation uses manually judged queries:

```json
{
  "query": "bm25 ranking",
  "relevant_chunks": [
    "lecture_03_0005",
    "book_0008"
  ]
}
```

Metrics:

```text
Precision@10
Mean Average Precision (MAP)
```

---

## Repository Ownership

```text
pipeline/
    extract.py          Person 1
    clean.py            Person 2
    chunk.py            Person 2

backend/search/
    bm25.py             Person 3
    tfidf.py            Person 4

backend/
    main.py             Person 5
    api/routes.py       Person 5

frontend/               Person 6

evaluation/             Person 6 + Persons 3/4
```

---

## Suggested Branches

```text
feature/extraction
feature/clean-chunk
feature/bm25
feature/tfidf
feature/backend
feature/frontend-eval
```

---

## MVP Milestones

### Milestone 1 — Data

```text
TXT/PDF
  -> documents.json
  -> chunks.json
```

### Milestone 2 — Ranking

Both work locally:

```python
search_bm25("information retrieval")

search_tfidf("information retrieval")
```

### Milestone 3 — Integration

```text
React
  -> Flask
      -> BM25 / TF-IDF
          -> ranked JSON
```

### Milestone 4 — Evaluation

Run the same test queries through both methods and compare:

```text
BM25 Precision@10 / MAP
vs.
TF-IDF Precision@10 / MAP
```

---

## Integration Rule

Every person should know:

1. **What is my input?**
2. **What transformation do I perform?**
3. **What exact output do I produce?**
4. **Who consumes my output?**

The most important shared contract is:

```text
documents.json
      |
      v
chunks.json
      |
  +---+---+
  |       |
 BM25   TF-IDF
  |       |
  +---+---+
      |
 search-result JSON
      |
    Flask
      |
    React
```

For the first MVP, keep the scope limited to:

```text
TXT transcripts
+ one PDF book
+ BM25
+ TF-IDF baseline
+ Flask API
+ React search page
+ Precision@10 / MAP
```

Add filters, highlighting, SQLite, and UI polish only after the full pipeline works end to end.