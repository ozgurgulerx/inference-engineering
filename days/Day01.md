# Day 01 - Starting the Inference Engineering Lab

Date: 2026-05-11

## What I am trying to do

I am starting a public inference engineering lab. The goal is not to write
perfect notes. The goal is to build a habit: run real experiments, write what I
observe in my own words, and slowly turn those observations into a systems-level
understanding of inference.

The path I care about is practical:

- How tokens move through hardware.
- How vLLM and other runtimes schedule work.
- How GPU memory, KV cache, batching, and context length affect latency.
- How to run useful experiments on Runpod without fooling myself.
- How to connect measurements to cost, reliability, and production design.

## Today's setup

Today is about creating the container for the work.

I now have:

- A GitHub repo for the journal.
- A public opening page.
- A `days/` folder for daily writing.
- A Runpod experiment folder.
- Templates for daily notes, Runpod runs, and benchmark results.
- A merged book layer for turning the strongest days into chapters.

This is enough structure to start. I do not need a perfect curriculum before
running the first experiment.

## Book placement

- Primary book part: Preface and Part V, framework-first implementation.
- Candidate chapters: Executive map, vLLM from first principles, Runpod practical deployment.
- Promotion status: raw day note with book map entry.

## First experiment direction

The first real experiment should be a small vLLM smoke test on Runpod.

The target is simple:

1. Start one GPU pod.
2. Serve one model with vLLM.
3. Send one request through the OpenAI-compatible endpoint.
4. Measure the basic request path.
5. Write down what happened.

The first useful metrics are:

- TTFT: time to first token.
- Total latency: time until the full response is done.
- Output tokens per second.
- GPU memory used.
- Approximate cost for the run.

## Questions I want to answer

- Which GPU is the simplest baseline for a first vLLM run?
- Which small model should I start with?
- What does vLLM log during startup?
- What breaks first: model download, memory, endpoint exposure, or benchmark code?
- What does a single request feel like before any tuning?

## Runpod checklist

- Pick a GPU with enough VRAM for a small model.
- Record the exact GPU name and memory.
- Record the container image.
- Record the vLLM version.
- Record the model name.
- Save the launch command.
- Save one raw request and response.
- Save timing numbers.
- Stop the pod when done.

## Commands placeholder

```bash
# Fill this in after the first Runpod session.
```

## My current mental model

Inference engineering is not just "serve a model." It is the stack around model
execution:

- Hardware gives the physical limits.
- The model architecture defines the work.
- The runtime schedules that work.
- The benchmark reveals the bottleneck.
- Production constraints decide whether the result matters.

Day 01 is the start of making that stack concrete.

## Next

- Run the first Runpod vLLM smoke test.
- Fill in `experiments/runpod/2026-05-11-vllm-smoke-test.md`.
- Put any raw results in `results/`.
- Write Day 02 from the actual experiment, not from theory.
