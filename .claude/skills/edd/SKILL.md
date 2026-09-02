---
name: edd
description: Eval-driven development loop for this repo (rag-loop, an aircraft-systems RAG app on LangGraph, Gemini, Qdrant Cloud, and Opik). Use when the user wants to run an experiment, grow the eval dataset, or decide keep-or-reject on a config/prompt change.
---

# EDD loop for rag-loop (aircraft systems)

Orchestration for this repo's eval loop: which command to run, in what order, and what's automated
vs. what needs judgment. For Opik/Qdrant SDK mechanics (dataset version API, collection/search
calls, flushing, span types, etc.), use the `opik` and `qdrant-clients-sdk` skills instead of
re-deriving them here — this skill only covers what's specific to this repo's wiring.

Stack: LangGraph orchestrates retrieve → generate; retrieval embeddings are FastEmbed
(`BAAI/bge-small-en-v1.5`, local, keyless); generation uses Gemini (`gemini-3.1-flash-lite`);
the three Opik LLM-judge metrics (`answer_relevance`, `hallucination`, `answer_quality`) use
OpenAI (`gpt-5.4-nano`) — different provider from generation, on purpose, so the judge isn't the
same model family grading its own output; vectors live in Qdrant Cloud (`QDRANT_URL`/
`QDRANT_API_KEY`, not a local path). `OPIK_PROJECT_NAME` in `.env` is read by the Opik SDK
directly — nothing in this repo's code passes `project_name` explicitly, don't add it back.

Two rules:

1. **Never invent an `expected_output`.** Every golden's expected answer must be lifted from the
   actual PDF text, not written from general knowledge of aircraft systems.
2. **Grade outcomes, not paths.** A judge scores whether the final answer is correct and grounded,
   not whether retrieval happened in some specific sequence.

## Preconditions

Everything except reading/editing files needs `.env` populated: `GEMINI_API_KEY`, `OPENAI_API_KEY`,
`QDRANT_URL`, `QDRANT_API_KEY`, `OPIK_API_KEY`, `OPIK_WORKSPACE`. `rag-loop run` also needs a git
repo, since it commits accepted changes and reverts rejected ones. If any are missing, say which
step is blocked — don't simulate results.

## What's automated vs. what's your job

`rag-loop run <name>` (`src/rag_loop/loop.py: run_iteration`) does the mechanical part of one
iteration: runs the experiment, persists its scores to `evals/results/<name>.json`, compares
against `acceptance.compare_against` on `acceptance.required_metrics`, writes the verdict back
onto that entry in `experiments.json`, then either commits `src/` (keep) or reverts it via git
(reject). See `evals/README.md` for the persisted-result format.

Your job is what a formula can't do: **read the failures, form a hypothesis, write the next
one-variable change.** That loop:

1. `rag-loop report <name>` — read the worst-scoring cases (input / expected / actual / judge
   reason) for the experiment that just ran.
2. Form a hypothesis about *why* those cases scored low.
3. Write one new `evals/experiments.json` entry with exactly one changed variable (`chat_model`,
   `retrieval_k`, or a prompt edit in `src/rag_loop/rag.py`) and `compare_against` set to the last
   *kept* experiment's name.
4. `rag-loop run <that-name>` and read the printed verdict.
5. Repeat from step 1 on reject. On `Plateaued` (a couple of kept runs in a row with no
   improvement), stop tweaking the prompt — a structural change (retrieval strategy, chunking,
   reranking) needs a human decision, not another wording edit.

## Dataset versioning is in Opik, not the local file

`evals/seed.jsonl` is only the source file for `rag-loop seed`. Every run creates a **new** Opik
dataset version (`src/rag_loop/evals.py`), it doesn't mutate the last one. `run_iteration` and
`run_experiment_loop` both freeze `load_dataset().get_current_version_name()` once per run — never
compare experiments scored against different dataset versions.

## The loop, in order

### 1. Index the source PDF into Qdrant Cloud

```sh
rag-loop index
```

Rebuild if `chunk_size`/`chunk_overlap` in `src/rag_loop/config.py` change, or the PDF changes —
those changes invalidate the existing collection silently otherwise.

### 2. Grow `evals/seed.jsonl`, then push it

See `evals/README.md` for the golden format (`input`/`expected_output`/`metadata`, plus
`label`/`critique`) and the two sources (PDF passages, real traces via `create_dataset_from_traces`).

```sh
rag-loop seed
```

### 3. Run one experiment at a time

```sh
rag-loop run <name>
```

`<name>` must already exist as an entry's `"name"` field in `evals/experiments.json` — `run`
looks one up, it doesn't create one. Write the entry first (step in "What's automated vs. what's
your job" above), then run it by that same name. `compare_against` on a new entry must name an
entry that's *already been run* — `decide()` needs its `evals/results/<name>.json` to diff against.

Example, from a fresh `experiments.json` with only `baseline` in it:

```sh
rag-loop seed                              # dataset must exist before anything can be scored
rag-loop run baseline                      # compare_against: null → always keeps, becomes the reference
# add a "topk-6" entry (retrieval_k: 6, compare_against: "baseline") to experiments.json, then:
rag-loop run topk-6
```

For a one-off multi-config sweep with no keep/reject bookkeeping (e.g. eyeballing several
`retrieval_k` values before committing to one as the next experiment entry), use
`rag-loop loop --config evals/configs.json` or `rag-loop experiment-loop` instead — neither of
those touches git or records a verdict.

### 4. On reject, analyze before writing the next change

```sh
rag-loop report <name>
```

Open the actual failing cases, not just the aggregate delta, before proposing what to change next.
