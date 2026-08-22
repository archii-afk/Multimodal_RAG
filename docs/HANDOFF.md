# Current handoff

Keep this page current while work is in progress. Replace stale task details after a handoff; durable facts belong in `PROJECT_CONTEXT.md` or `DECISIONS.md`.

## Active tasks

### Claude — demo hardening
- Primary video ingested end-to-end on `main` (`data/processed/evidence.db`): 1204 nodes / 3437 edges, 93 frames (49 diagrams), 172 claims, 48 `illustrates` links. Demo query returns transcript + diagram frames + speaker with timestamps.
- Next: source 2–3 overlapping public docs as PDFs (see system names below), ingest them, label `eval/questions.json` (5 questions), run `mmrag eval`, README architecture + future-work sections, rehearse `mmrag serve`.

### Codex — available
- Candidate task: source and export PDFs for the systems below (public Atlassian engineering posts / specs), save under `data/raw/docs/`, and report which entity names each one contains.

## System / architecture names used in the primary video (from transcript + OCR, by frequency)
Envoy (sidecar proxy, dynamic config/xDS), Open Service Broker API, FastAPI, DynamoDB, SQS, CloudFront, Route53, AMI / HashiCorp Packer, CloudFormation, NLB, IAM Role, EC2, "Sovereign" (Atlassian data sovereignty / sovereign cloud), KeyPair, IGW, Subnet, AutoscalingGroup, SecurityGroup, ACM, S3, VPC, Excalidraw (tool), Atlassian.
Key narratives: (1) client → FastAPI → SQS → worker → DynamoDB async provisioning with client polling (08:00–09:30); (2) Open Service Broker provisioning flow (≈20:00–21:00); (3) Envoy sidecar / management-server configuration baked into AMIs (≈13:00–15:00).

## Completed

- Requirements reviewed; design spec accepted; Codex design review incorporated (`docs/DECISIONS.md` 2026-08-23).
- `src/mmrag/model.py` contract + 8 passing tests; `uv` venv; `.env.example`.
- Five ingestion workers: audio, hybrid video frames, Gemini vision/OCR, page-aware PDF, and claim/entity extraction.

## Checks run

- Before workers: `.venv/bin/python -m pytest -q` → 8 passed.
- After integration + demo fixes: `.venv/bin/python -m pytest -q` → 46 passed; real ingestion run with 0 warnings.

## Open questions

- Primary video is still pending in `data/raw/`; system/architecture names cannot be extracted yet.
- PDF sources: decide after the first real transcript pass.

## Next action

- Claude reviews and integrates `codex/ingestion`, applying the `extract_claims` pipeline assignment noted above.
- Once the primary video appears, run the real audio pass, record its system names here, and choose overlapping Atlassian PDFs.

## Integration rule

Claude merges. Rebase on `main` before opening for integration; never merge the other agent's unreviewed work.
