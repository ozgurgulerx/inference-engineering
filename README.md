# Inference Engineering

Daily notes and experiments for learning inference engineering in public.

Live site: https://ozgurgulerx.github.io/inference-engineering/

## Purpose

This repo is a working lab notebook. The goal is to write each day in my own
words, run experiments on Runpod, and turn raw observations into reusable
inference engineering knowledge.

The public site is the front door. The durable work lives in:

- `days/` - daily notes written in first person
- `experiments/runpod/` - Runpod experiment logs and configs
- `templates/` - repeatable note and experiment templates
- `results/` - benchmark outputs, plots, tables, and artifacts
- `notes/` - cleaned concept notes promoted from daily work

## Daily Workflow

1. Copy `templates/day.md` into `days/YYYY-MM-DD-day-NNN-topic.md`.
2. Write the day in my own words before polishing anything.
3. If I run a benchmark, copy `templates/runpod-experiment.md` into
   `experiments/runpod/`.
4. Put raw outputs in `results/`.
5. Commit the work with a short message.
6. Push to GitHub.

## Local Preview

The site is plain HTML/CSS/JS. Open `index.html` in a browser.

```bash
open index.html
```

## Suggested Commit Style

```bash
git add .
git commit -m "day: add runpod vllm smoke test"
git push
```
