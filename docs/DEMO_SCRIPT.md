# Demo script (~4 minutes)

Target: show a realistic multimodal dataset being ingested and a query whose answer
requires more than one modality, with traceability to the source (brief §6–§8).
Modalities on screen: audio (transcript), video (frames), image (diagram/OCR), document (PDF).

Record with screen + mic. Terminal on the left, browser on the right. Font size 16+.

---

## 0. Before recording (not on camera)

```sh
cd Multimodal_RAG
source .venv/bin/activate
export PYTHONPATH=src
kill $(pgrep -f "mmrag.cli serve") 2>/dev/null   # stop any old server
rm -f data/processed/evidence.db                  # so §2 ingestion runs live, from cache (~60 s, no API calls)
```

Do NOT start the server yet: it loads the vector index once at first search, so it must start
**after** ingestion (§2). One terminal is enough; the server goes in the background with `&`.

Have the Excalidraw-style architecture diagram (docs/superpowers spec §3) open in a tab.

---

## 1. The problem — 30 s

> "This is a 40-minute recording of an ex-Atlassian engineer explaining their provisioning
> architecture while drawing it on a whiteboard. Normal RAG transcribes the audio, chunks it,
> and searches. Ask it *who explained the async task flow and where the diagram was shown* and
> it can't answer — the speaker, the diagram, the text inside it, and the moment they coincided
> are four different things, and chunking throws the links away. We built a pipeline that keeps
> them."

Show the video for 5 seconds at 08:15 — the FastAPI/SQS/worker drawing in progress.

---

## 2. Ingestion — 60 s

```sh
python -m mmrag.cli ingest data/raw/Test_video.mp4 data/raw/docs/*.pdf \
  --presenter "Ex-Atlassian Senior Engineer"
```

While it runs, narrate the log lines as they appear:

> "One command. ffmpeg splits audio and samples frames — every five seconds plus scene changes,
> de-duplicated by perceptual hash, so 40 minutes becomes 93 meaningful screenshots. Whisper
> gives timestamped sentences. Gemini reads each frame: what the screen shows, the text on it,
> the systems drawn. PyMuPDF chunks the PDFs without crossing pages. A claim pass turns speech
> and documents into propositions — *'the worker writes the result to DynamoDB'* — tied to
> entities. Every API call is cached by content hash, which is why this finishes in a minute."

When `[done]` prints, start the server in the same terminal (`python -m mmrag.cli serve &`) and read the numbers:

> "1,200 nodes, 3,400 edges. 707 of those edges say *this sentence was spoken while this frame
> was on screen*. 48 say *this diagram illustrates this claim* — which only happens when the
> frame shows the same entity **and** was visible while the claim was being said."

---

## 3. The representation — 30 s

Switch to the architecture diagram tab.

> "Everything is a node in one SQLite table — sentence, frame, OCR block, PDF chunk, claim,
> entity, person — with its modality, source file, timestamp or page, confidence, and
> provenance down to the model and prompt version. Edges carry the relationships: co-occurs-at
> with the overlap in seconds, expresses, illustrates, supports, spoken-by, same-topic across
> files. No Neo4j, no vector database — a graph this size fits in a file; the design is what
> matters, and the same schema ports to Neo4j later."

---

## 4. The query — 90 s

Browser → http://127.0.0.1:8000. Type:

> **How does the async task flow work between the web server, the worker and DynamoDB, who explained it, and where was the diagram shown?**

Run it three times, changing the mode dropdown:

**text_only** (baseline)
> "This is ordinary text RAG over the transcript and PDFs. It finds the right sentences at
> 08:04 to 09:09 — and stops. No speaker, no diagram."

**flat_multimodal** (ablation)
> "Same store, frames included, still plain vector search. Now it finds diagram frames by
> similarity — but nothing connects them to the sentences, and still no speaker. Having frames
> is not the win."

**graph**
> "Now the graph walks. Seeds are taken per modality, then it follows typed paths: sentence →
> claim → the frame that illustrates it; sentence → person who spoke it; sentence → the page
> that supports it."

Point at the cards as you read the answer:

> "The answer: client hits FastAPI, FastAPI drops the task into SQS [E2 at 08:30], the worker
> provisions Route53 and CloudFront [E5 at 08:49], writes to DynamoDB [E9 at 09:04], the client
> polls [E14 at 08:56]. Explained by the presenter. Diagram at 08:00–09:30 — here's the frame.
> And the supporting document page — here. Every card shows the path that reached it, so the
> evidence is traceable, not just plausible."

Click one frame thumbnail, then scrub the real video to that timestamp. Same diagram.

---

## 5. Evaluation — 30 s

```sh
python -m mmrag.cli eval eval/questions.json
```

> "Five labelled questions, thirteen required pieces of evidence across transcript, frames and
> PDF pages. Recall: text-only 0.54, flat multimodal 0.62, graph 0.92." 

> "The gap between the last two is the graph's contribution — not just access to more data."

---

## 6. Close — 20 s

> "Built in 13 hours by two agents working in separate branches against one frozen contract.
> Next: Neo4j plus a dedicated vector store for scale, speaker diarization, region-level
> provenance on diagrams, and temporal 'how did this change' queries. The repo, the design
> spec, and every decision are on GitHub."

---

## Checklist before hitting record

- [x] PDFs ingested (5 in `data/raw/docs/`); graph mode returns a `pdf_chunk` for the demo question
- [x] `eval/questions.json` labelled; numbers in §5
- [ ] Presenter name set (re-run ingest if changed)
- [ ] Browser zoom 125%, dark theme matching the Excalidraw board
- [ ] `data/processed/evidence.db` deleted so §2 runs live
