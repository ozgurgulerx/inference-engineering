# Prompts

This folder preserves the prompt system used to generate the earlier daily
inference engineering exercises.

## Files

- `daily-inference-coach.md` - the main reusable meta-prompt for generating
  history-aware 3-tier daily lab plans.
- `measurement-first-daily-log.md` - the daily log template that forces each
  experiment to capture setup, metrics, cost, quality, reliability, and next
  steps.
- `technical-focus.md` - the technical theme map behind the daily exercises.
- `100-day-learning-goals.md` - the earlier 100-day curriculum plan.
- `success-outcomes.md` - the target outcomes the curriculum was designed to
  produce.

## How To Use

For a new day, use `daily-inference-coach.md` as the assistant setup prompt and
fill in the daily input with:

- the day number,
- available time,
- the last 1-3 days of concrete work,
- the Runpod/hardware target,
- the book chapter or theme you want to advance.

The output should become a draft day note, not a final chapter. After the
experiment runs, update `maps/day-to-book.yml` and promote the durable part into
`notes/` or `book/`.

## What To Keep

Keep the 3-tier structure because it works well:

- Tier 1: minimum useful experiment.
- Tier 2: deeper measurement.
- Tier 3: stretch investigation.

Keep the measurement-first constraint. Every useful day should produce code,
numbers, config, or a concrete written artifact.
