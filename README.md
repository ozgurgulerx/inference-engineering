# Inference Engineering

Daily notes, Runpod experiments, and the source for the book
**Building an AI Data Center from Scratch**.

Live site: https://ozgurgulerx.github.io/inference-engineering/
Live book: https://ozgurgulerx.github.io/inference-engineering/book/

## Purpose

This repo is the canonical workspace. The goal is to write each day in my own
words, run experiments on Runpod, and promote the strongest observations into a
coherent book about inference engineering, AI hardware, and data center design.

The public site is the front door. The durable work lives in:

- `days/` - daily notes written in first person
- `experiments/runpod/` - Runpod experiment logs and configs
- `results/` - benchmark outputs, plots, tables, and artifacts
- `notes/` - cleaned concept notes promoted from daily work
- `book/` - Quarto book source
- `maps/day-to-book.yml` - the promotion map from journal entries to chapters
- `prompts/` - the salvaged daily-coach prompt system
- `source-material/` - prior journal material used as source evidence
- `templates/` - repeatable note and experiment templates

## How the Two Layers Work Together

The journal is raw evidence. The book is the curated explanation.

```text
days/ -> experiments/ -> results/ -> notes/ -> book/
```

Use `days/` for first-person learning and messy observations. Use `book/` only
after an idea has enough evidence, a clearer mental model, or a reusable
decision framework.

## Prior Work

The earlier `inference-journal` project is preserved under
`source-material/prior-inference-journal/`. It contains the original 19-day lab
sequence, the `days02/` full-stack curriculum, topic notes, benchmark scripts,
and the older mdBook draft.

The prompt system that generated those daily exercises is promoted to
`prompts/`. Use `prompts/daily-inference-coach.md` with the measurement-first
template to generate new 3-tier daily plans, then map the useful parts through
`maps/day-to-book.yml`.

## Daily Workflow

1. Copy `templates/day.md` into `days/YYYY-MM-DD-day-NNN-topic.md`.
2. Write the day in my own words before polishing anything.
3. If I run a benchmark, copy `templates/runpod-experiment.md` into
   `experiments/runpod/`.
4. Put raw outputs in `results/`.
5. Update `maps/day-to-book.yml` with the chapter connections.
6. Promote the durable part into `notes/` or `book/`.
7. Commit the work with a short message.
8. Push to GitHub.

## Runpod Lab Ladder

The first Runpod experiments should reuse the strongest prior work:

1. vLLM smoke test from the old GPU node bring-up day.
2. vLLM capacity and OOM grid.
3. TTFT and prefix caching probes.

The detailed adaptation plan is in
`experiments/runpod/prior-journal-lab-plan.md`.

## Local Preview

The journal front door is plain HTML/CSS/JS:

```bash
open index.html
```

Render the combined local site into `_site/`:

```bash
./scripts/build-site.sh
open _site/index.html
```

Preview only the Quarto book:

```bash
cd book
./scripts/quarto preview
```

## Suggested Commit Style

```bash
git add .
git commit -m "day: add runpod vllm smoke test"
git push
```
