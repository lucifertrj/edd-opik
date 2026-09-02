# RAG Loop

This project answers questions about aircraft systems from a PDF.

It uses LangGraph for the RAG steps, Gemini for generation, FastEmbed (local, keyless) for
embeddings, Qdrant Cloud for PDF search, and Opik for traces, datasets, and scores.

## Initial Setup

Run once, before the loop starts.

1. Create a virtualenv and install the package:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. Copy `.env.example` to `.env` and fill in `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, and
   your Opik credentials. Model choice, chunking, and retrieval settings live in
   `src/rag_loop/config.py`, not in `.env`.

```sh
cp .env.example .env
```

3. Build the search index in Qdrant Cloud:

```sh
rag-loop index
```

Re-run `rag-loop index` any time `chunk_size`/`chunk_overlap` or the PDF changes — those changes
invalidate the existing collection silently otherwise.

4. Sanity check with a question:

```sh
rag-loop ask "What is the function of the auxiliary power unit?"
```

## The Loop

Run repeatedly, one change at a time, once setup is done. This is what an agent should follow for
eval-driven development on this repo — the full version, with rules and preconditions, is in
[`.claude/skills/edd/SKILL.md`](.claude/skills/edd/SKILL.md).

1. **Grow `evals/seed.jsonl`** — add goldens with `expected_output` lifted from the actual PDF
   text (never invented), then push a new versioned Opik dataset:

```sh
rag-loop seed
```

2. **Write one change** as a new entry in `evals/experiments.json` — one variable only
   (`chat_model`, `retrieval_k`, a prompt tweak in `src/rag_loop/rag.py`), e.g.:

```json
{
  "name": "topk-6",
  "change": "retrieval_k 4 -> 6",
  "retrieval_k": 6,
  "acceptance": { "compare_against": "baseline", "required_metrics": [...] }
}
```

`compare_against` is the name of the entry to score against — `"baseline"` here means "keep this
only if it's flat or better than baseline on every required metric." It must name an entry that's
already been run (`rag-loop run` needs that entry's `evals/results/<name>.json` to diff against).

3. **Run it by that same `name`** — `rag-loop run` looks up an existing entry, it doesn't create
   one, so the entry from step 2 has to exist first. This persists its scores, decides keep/reject
   against `compare_against`, records the decision back into the entry, and handles git for you:
   commits the change on keep, reverts `src/` on reject.

```sh
rag-loop run topk-6
```

4. **If rejected, analyze before trying again** — read the worst-scoring cases to see what's
   actually failing, then go back to step 2 with a different single-variable change:

```sh
rag-loop report <name>
```

**"Plateaued"** means: kept (no regression), but 2 kept runs in a row moved no metric at all. It's
not an error — it's a signal that prompt/config tweaking has stopped paying off, and the next
change should be structural (retrieval strategy, chunking, reranking) rather than another wording
edit.

For a one-off multi-config sweep instead of the iterative accept/reject flow (e.g. comparing
several `retrieval_k` values at once, with no keep/reject bookkeeping), use `evals/configs.json`:

```sh
rag-loop loop --config evals/configs.json
```

See the [flowchart](flowchart.md).

## Example: running your first experiment

The dataset has to exist in Opik before anything can be scored against it — `rag-loop seed` is
step 1 for a reason. A full first pass looks like this:

```sh
# 1. Push evals/seed.jsonl to Opik as a new dataset version (needed before any run can score)
rag-loop seed

# 2. evals/experiments.json already has a "baseline" entry — run it.
#    compare_against is null, so this always keeps and becomes the reference point.
rag-loop run baseline

# 3. Add a new entry, e.g. "topk-6" (retrieval_k: 6, compare_against: "baseline"), then run it by
#    that same name:
rag-loop run topk-6

# 4. If topk-6 was rejected, see why before writing the next entry:
rag-loop report topk-6
```

## Inspecting the loop's history

`rag-loop run` never overwrites history — it's all in two places:

- **`git log --oneline`** — one commit per run. Kept experiments commit as `<name>: <change>` with
  the actual code diff; rejected ones commit as `<name> (rejected): <change>` with the record only
  (the code edit itself was reverted). `git show <sha>` on a kept commit shows what changed.
- **`evals/experiments.json`** — every entry keeps a `result` block (`decision`, `reason`,
  per-metric `comparisons`) once it's been run, so the file itself is a running scoreboard of every
  experiment tried, in order.
