# Building an AI Data Center from Scratch

Live book: https://ozgurgulerx.github.io/inference-engineering/book/

This folder contains the Quarto book source inside the canonical
`inference-engineering` repo. The daily journal, Runpod experiments, raw
results, and polished book now live together.

## Book Structure

The active outline is organized as:

1. AI data center design.
2. Silicon, nodes, and hardware architecture.
3. Model mechanics that matter for inference.
4. Kernels, compilers, and runtime primitives.
5. Framework-first implementation with vLLM and Runpod.
6. Advanced inference architectures.
7. Production inference engineering.

Prior source material is preserved under
`../source-material/prior-inference-journal/` and mapped in
`appendices/source-map.qmd`.

## Quick Start

Install Quarto:

```bash
brew install --cask quarto
```

This machine also has a project-local Quarto CLI extracted under
`.local-tools/`. Use the wrapper when `quarto` is not installed system-wide:

```bash
./scripts/quarto --version
```

Preview the book locally:

```bash
cd inference-engineering/book
./scripts/quarto preview
```

Render the static site:

```bash
./scripts/quarto render
```

The rendered book is written to `_book/`.

## Publishing

The root workflow publishes the journal opening page and this book together.
The public routes are:

- Journal front door: `https://ozgurgulerx.github.io/inference-engineering/`
- Book: `https://ozgurgulerx.github.io/inference-engineering/book/`

## Writing Workflow

- Keep chapters in `chapters/`.
- Keep runnable exercises in `labs/`.
- Keep reusable diagrams in `diagrams/`.
- Promote daily notes into chapters through `../maps/day-to-book.yml`.
- Use PRs for chapter review and GitHub Issues for topic backlog.
- Prefer short, reviewable changes: one chapter section, one diagram, or one lab at a time.

## Local Checks

```bash
./scripts/quarto check
./scripts/quarto render
```

When Python-backed examples are added:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
