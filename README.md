<div align="center">

# AinSeba (আইনসেবা)

**A bilingual RAG assistant that answers Bangladesh legal questions with section-level citations.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6B6B)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Live Demo](#) · [API Docs](#api-reference) · [Engineering Notes](#engineering-notes)

</div>

---

*AinSeba* means "law service" in Bangla. Legal information in Bangladesh is locked inside scanned PDFs on government portals, written in dense statutory English that most of the 170 million people it governs cannot readily parse. AinSeba lets someone ask **"my employer hasn't paid me for three months, what can I do?"** — in English, Bangla, or romanised Banglish — and get an answer grounded in the actual text of the Bangladesh Labour Act, with the section number attached so they can verify it themselves.

Every answer cites its sources. When the corpus does not cover a question, the system says so rather than guessing.

<p align="center">
  <img src="docs/images/chat-english.png" alt="AinSeba answering a question about theft penalties with Penal Code citations" width="850">
</p>

---

## Contents

- [What it does](#what-it-does)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Legal corpus](#legal-corpus)
- [Engineering notes](#engineering-notes)
- [Quickstart](#quickstart)
- [API reference](#api-reference)
- [Project structure](#project-structure)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

---

## What it does

**Citation-grounded answers.** Retrieval runs over 1,155 passages chunked along statutory boundaries, so every claim traces back to a specific section. Answers are structured as *Relevant Law → What it means for you → Recommended next steps*, and always name the sections they rely on.

**Trilingual input.** `langdetect` plus a heuristic classifier routes English, Bangla, and Banglish (Bangla written in Latin script — how most Bangladeshis actually type). Non-English questions are translated to English for retrieval, then the answer is translated back, so a Bangla question gets a Bangla answer citing English section numbers.

**Two-stage retrieval.** Dense vector search over `text-embedding-3-small` pulls 20 candidates; a `ms-marco-MiniLM-L-6-v2` cross-encoder reranks them to the top 5. The reranker earns its place — on *"maximum daily working hours"* it scored the correct Section 100 at **3.381** while pushing a superficially similar adolescent-hours clause down to **−1.219**.

**History-aware retrieval.** Follow-ups like *"what section covers that?"* carry no searchable subject on their own. Short or anaphoric questions are condensed into standalone queries against the conversation history before retrieval, which is what makes multi-turn conversation work at all.

**Honest refusals.** Five acts in the registry have no source PDF. Asked about consumer rights or the Cyber Security Act, the system refuses instead of stretching an unrelated statute to fit.

---

## Screenshots

<table>
<tr>
<td width="50%"><img src="docs/images/chat-bangla.png" alt="Bangla question answered in Bangla with English section citations"><br><em>Bangla in, Bangla out — citations stay in English</em></td>
<td width="50%"><img src="docs/images/sources-panel.png" alt="Expanded sources panel showing citations with similarity and rerank scores"><br><em>Every answer is auditable: act, chapter, section, and both scores</em></td>
</tr>
<tr>
<td><img src="docs/images/followup.png" alt="Follow-up question resolved against conversation history"><br><em>Follow-ups resolve against history</em></td>
<td><img src="docs/images/api-docs.png" alt="FastAPI interactive OpenAPI documentation"><br><em>Auto-generated OpenAPI docs</em></td>
</tr>
</table>

---

## Architecture

```
                        ┌──────────────────────────┐
   Question             │  Streamlit  (port 8501)  │
   EN / BN / Banglish   │  chat · filters · sources│
                        └────────────┬─────────────┘
                                     │  SSE  /api/query/stream
                        ┌────────────▼─────────────┐
                        │  FastAPI    (port 8000)  │
                        │  rate limit · sessions   │
                        └────────────┬─────────────┘
                                     │
                   ┌─────────────────▼──────────────────┐
                   │        BilingualRAGChain           │
                   │  detect → translate → answer → BN  │
                   └─────────────────┬──────────────────┘
                                     │
                   ┌─────────────────▼──────────────────┐
                   │          LegalRAGChain             │
                   │  condense follow-up → retrieve →   │
                   │  prompt → GPT-4o-mini → cite       │
                   └─────────────────┬──────────────────┘
                                     │
              ┌──────────────────────▼───────────────────────┐
              │              LegalRetriever                  │
              │  ChromaDB top-20  →  cross-encoder top-5     │
              └──────────────────────┬───────────────────────┘
                                     │
              ┌──────────────────────▼───────────────────────┐
              │   ChromaDB · 1,155 passages · 1536-dim       │
              │   metadata: act · chapter · section · year   │
              └─────────────────────────────────────────────┘
```

**Ingestion pipeline** (offline, run once per corpus change):

```
PDF → PyMuPDF extract → clean → strip contents pages
    → detect Part/Chapter/Section → chunk (600 tok, 100 overlap)
    → quality report → embed → ChromaDB
```

### Stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangChain 0.3 (LCEL) | composable chain, streaming built in |
| LLM | `gpt-4o-mini` | citation-following at ~$0.0007/query |
| Embeddings | `text-embedding-3-small` | 1536-dim, $0.02/1M tokens |
| Reranker | `ms-marco-MiniLM-L-6-v2` | runs on CPU, no API cost |
| Vector store | ChromaDB | metadata filtering, embeds in-process |
| API | FastAPI + Pydantic | typed schemas, free OpenAPI docs |
| UI | Streamlit | fastest path to a demoable chat surface |
| PDF | PyMuPDF | fastest reliable text extraction |

---

## Legal corpus

Sourced from [bdlaws.minlaw.gov.bd](http://bdlaws.minlaw.gov.bd/), the official legislative portal.

| Act | Year | Category | Passages | Sections |
|---|---|---|---|---|
| The Penal Code | 1860 | Criminal Law | 520 | 498 |
| Bangladesh Labour Act | 2006 | Employment | 381 | 352 |
| State Acquisition and Tenancy Act | 1950 | Property Law | 224 | 175 |
| Bangladesh Environment Conservation Act | 1995 | Environmental Law | 20 | 16 |
| Muslim Family Laws Ordinance | 1961 | Family Law | 10 | 10 |
| **Total** | | | **1,155** | **1,051** |

Mean chunk size ~215 tokens; no chunk exceeds the 8,191-token embedding ceiling. Full indexing costs about **$0.005**.

Five further acts sit in the registry awaiting source PDFs — Consumer Rights Protection 2009, Cyber Security 2023, Rent Control 1991, Companies Act 1994, and the Constitution. The system refuses questions in those areas rather than answering from adjacent law.

---

## Engineering notes

Most of the work on this project was not building the chain. It was discovering that the chain was answering confidently from a corpus that was quietly broken. These are the failures worth reading about.

### Half a statute was invisible to the parser

Old bdlaws PDFs print section titles in a **left margin column**. PyMuPDF interleaves that column into the body text, so Section 379 extracts as:

```
Punishment for 379. Whoever commits theft shall be punished with theft. imprisonment...
```

The heading is not at the start of the line — `Punishment for` is. A line-anchored `^(\d+)\.` regex never matches it. In the Penal Code body, **86 headings are displaced this way against 83 at line start**, so roughly half the statute was unreachable, including s.379, the punishment for theft. Section 382 survived only because its marginal note happened to land after the number.

The fix matches headings behind an optional short prefix, then filters candidates to the **longest strictly ascending run** of section numbers. Statutes number sequentially, so anything going backwards is a list item or a cross-reference. A naive greedy filter was tried first and regressed the Environment Act from 15 sections to 9 — one stray `12A` appearing early set the floor and rejected everything after it. Solving for the longest ascending subsequence discards the outlier instead.

Result: Penal Code sections went from **227 to 498**.

### Contents pages were being indexed as law

The first several hundred lines of these PDFs are a table of contents whose entries are indistinguishable from section headings. They were being embedded as content and retrieved as citations. Detection now finds where section numbering climbs high and resets near 1, and cuts there — and the harvested contents entries are reused as clean section titles, since the body copies are fragmented by the marginal-note splice.

### The token floor deleted exactly the useful sections

A 50-token minimum discarded any chunk shorter than that. Penal code punishment sections are deliberately terse — s.379 is 39 tokens — so the filter was removing precisely the sections users ask about. Numbered sections now have a much lower floor than loose prose.

### Oversized chunks failed silently

Paragraph-level splitting never subdivided a paragraph that alone exceeded the budget, producing chunks of 18,000 and 22,000 tokens. Those exceed `text-embedding-3-small`'s 8,191-token limit and were rejected at index time without surfacing an error. Splitting is now paragraph → sentence → hard token slice.

### Follow-up questions retrieved nothing

*"What section covers that?"* embeds to a vector with no legal subject in it, so it retrieved five arbitrary passages and the model correctly reported it had no information. Conversation memory was working fine — retrieval was the problem. Short or anaphoric questions are now condensed into standalone queries before embedding, with a free fallback to prepending the previous turn if the rewrite call fails.

### Smaller fixes

- `stream()` built a source list and discarded it, so streaming answers had no citations
- Streaming ignored the target language: a Bangla question got a Bangla answer from `/api/query` and an English one from `/api/query/stream`
- The chain built lazily, so the first request paid ~40s of cross-encoder loading; it now warms at startup
- `allow_origins=["*"]` with `allow_credentials=True` is rejected outright by browsers
- "Clear chat" wiped only the browser copy while the backend kept replaying old turns into every prompt
- Streamlit widget keys derived from `hash(answer[:50])` collided whenever two answers opened alike

---

## Quickstart

**Prerequisites:** Python 3.11+, an OpenAI API key, ~$0.01 of credit.

```bash
git clone https://github.com/mh-hamim/Ain-Seba.git
cd ainseba

python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # then add your OPENAI_API_KEY
```

### Build the index

Place the source PDFs in `data/raw/`, then:

```bash
python scripts/run_pipeline.py --all          # extract → clean → chunk
python -m src.vectorstore.populate --all      # embed → ChromaDB
python -m src.vectorstore.populate --stats    # expect ~1,155 documents
```

### Run

```bash
uvicorn src.api.app:app --port 8000           # wait for "Warm-up complete"
streamlit run frontend/app.py                 # second terminal → :8501
```

The API warms the chain at startup, so give it ~40 seconds on first boot while the cross-encoder loads. Point the UI at a remote backend with `AINSEBA_API_URL`.

### Configuration

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | required |
| `LLM_MODEL` | `gpt-4o-mini` | |
| `LLM_MAX_TOKENS` | `1500` | lower to `800` to cut cost by ~⅓ |
| `USE_RERANKER` | `true` | set `false` on memory-constrained hosts |
| `RETRIEVAL_TOP_K` | `20` | candidates before reranking |
| `RERANK_TOP_N` | `5` | passages sent to the LLM |
| `API_RATE_LIMIT` | `30` | requests per window, per IP |
| `CORS_ORIGINS` | `*` | comma-separated in production |
| `AINSEBA_API_URL` | `http://localhost:8000` | read by the frontend |

---

## API reference

Interactive docs at `/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/query` | ask a question, get answer + sources |
| `POST` | `/api/query/stream` | same, as SSE: `metadata → token* → sources → done` |
| `GET` | `/api/health` | status, document count, indexed acts |
| `GET` | `/api/sources` | available acts and categories |
| `POST` | `/api/feedback` | 1–5 rating on a response |
| `GET` | `/api/session/{id}` | conversation history |
| `DELETE` | `/api/session/{id}` | clear server-side memory |

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the penalty for theft?", "session_id": "demo"}'
```

```json
{
  "answer": "According to Section 379 of the Penal Code 1860, whoever commits theft...",
  "sources": [
    {
      "citation": "The Penal Code 1860, Chapter XVII: OF OFFENCES AGAINST PROPERTY, Section 379",
      "act_id": "penal_code_1860",
      "section_number": "379",
      "section_title": "Punishment for theft",
      "similarity_score": 0.611,
      "rerank_score": 4.911
    }
  ],
  "detected_language": "en",
  "response_language": "en",
  "session_id": "demo"
}
```

---

## Project structure

```
ainseba/
├── src/
│   ├── config.py                 # central config, law registry
│   ├── vectorstore/              # embeddings, ChromaDB, populate CLI
│   ├── retrieval/                # retriever + cross-encoder reranker
│   ├── prompts/templates.py      # system prompt, context formatting
│   ├── chain/                    # RAG chain, memory, builder
│   ├── language/                 # detector, translator, bilingual wrapper
│   ├── models/schemas.py         # Pydantic request/response models
│   └── api/                      # FastAPI app + rate limiter
├── scripts/
│   ├── run_pipeline.py           # ingestion CLI
│   └── ingestion/                # extractor, cleaner, chunker, quality
├── frontend/app.py               # Streamlit chat UI
├── evaluation/                   # retrieval + answer metrics
├── tests/                        # pytest suite
├── data/{raw,processed}/
└── docs/images/
```

---

## Evaluation

```bash
pytest tests/ -v
python -m evaluation.run_evaluation
```

The ingestion pipeline emits a per-act quality report scoring extraction completeness, section coverage, and chunk size distribution. Current corpus average: **91.4 / 100**.

Retrieval is spot-checked against a fixed question set covering all five acts, including two out-of-corpus questions that must be refused. Refusal behaviour matters as much as recall here — a legal tool that confidently cites the wrong statute is more dangerous than one that admits ignorance.

---

## Limitations

**Read this before relying on any answer.**

- **Not legal advice.** Educational information only. Consult a qualified lawyer for any actual matter.
- **Static snapshot.** The corpus reflects the law as printed in the source PDFs. It does not track subsequent amendments, repeals, or provisions read down or struck by the courts. Verify against bdlaws before acting.
- **Five acts unindexed.** Consumer rights, cyber security, rent control, company law, and constitutional questions are out of scope and are refused.
- **Marginal-note artefacts.** Penal Code and Tenancy Act passages carry title fragments spliced mid-sentence by PDF extraction. Retrieval and generation handle this, but raw source snippets read slightly mangled. Fixing it properly needs column-aware extraction.
- **English-only case law.** No judicial interpretation, only statutory text.
- **In-memory sessions.** Conversation state does not survive a backend restart.

---

## Roadmap

- [ ] Column-aware PDF extraction to remove marginal-note splicing
- [ ] Index the five remaining acts
- [ ] Minimum relevance threshold so refusals show no sources
- [ ] Persist sessions and feedback to Postgres
- [ ] Hybrid BM25 + dense retrieval for exact section lookups
- [ ] Bangla-native embeddings to skip the translation round-trip

---

## License

MIT — see [LICENSE](LICENSE).

Statutory texts are Government of Bangladesh publications, reproduced from the official portal for educational use.

---

<div align="center">

Built by **[Mahmudul Hasan Hamim](https://www.linkedin.com/in/mahmudul-hasan-hamim-9088733ab/)**

*If this was useful, a star helps.*

</div>
