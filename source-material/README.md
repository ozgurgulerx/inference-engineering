# Source Material

This folder preserves prior work that should feed the journal and book.

## Prior Inference Journal

`prior-inference-journal/` is the earlier inference engineering journal. It is
not dead material. Treat it as a source library:

- `JOURNAL_INDEX.md` gives the original 19-day sequence.
- `days/` contains detailed daily exercises, tiered lab plans, scripts, and
  notes.
- `days02/` contains a stronger full-stack curriculum map from workloads and
  evals through GPU architecture, kernels, serving engines, networking,
  deployment, observability, and SRE economics.
- `topics/` contains cleaned concept notes that can be promoted into the new
  Quarto book.
- `books/inference-engineering/` contains the older mdBook attempt.

The old journal is intentionally kept separate from the new `days/` folder so
new writing can stay personal and chronological while prior material remains
available for mining.

## Promotion Rule

Use the prior journal in this order:

1. Pull the experiment idea or prompt into `maps/day-to-book.yml`.
2. Run or adapt the experiment in the current repo under `experiments/runpod/`.
3. Put raw outputs in `results/`.
4. Write a cleaned note in `notes/`.
5. Promote only the durable explanation into `book/`.

Do not paste old material directly into chapters without checking it against a
new measurement, current docs, or a clearly stated assumption.
