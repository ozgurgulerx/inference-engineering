# Runpod Lab Plan From Prior Journal

The prior journal already contains a useful sequence of experiments. This page
adapts that work to the current repo and to Runpod as the default on-demand GPU
lab.

## Why Runpod Fits This Project

Runpod is useful here because the book needs repeatable GPU experiments without
assuming a permanent cluster. The right use is not "click around until it works";
the right use is a disciplined lab substrate:

- choose a GPU and record its exact name and VRAM,
- start from a known container image,
- pin model, runtime, and benchmark versions,
- expose a vLLM-compatible endpoint,
- run one experiment at a time,
- save logs, commands, results, and cost notes,
- stop the pod when the run is done.

## Adaptation Pattern

For each prior experiment:

1. Pick the old day folder from `source-material/prior-inference-journal/days/`.
2. Create a current experiment note from `templates/runpod-experiment.md`.
3. Translate any local-node assumptions into Runpod fields:
   - provider and pod type,
   - GPU model and VRAM,
   - container image,
   - mounted volume or model cache,
   - exposed HTTP port,
   - vLLM flags.
4. Run the minimum Tier 1 version first.
5. Store raw outputs under `results/`.
6. Update `maps/day-to-book.yml`.
7. Promote the durable takeaway into `notes/` or `book/`.

## Experiment Ladder

| Ladder | Prior source | Runpod adaptation | Book target |
| --- | --- | --- | --- |
| vLLM smoke test | `day-002-GPU-node-bring-up` | Serve one small model through the OpenAI-compatible API | `21-runpodai-practical-deployment` |
| Capacity and OOM | `day-003-vLLM-capacity-and-OOM` | Sweep `max_model_len`, `max_num_seqs`, concurrency, and output length | `20-vllm-from-first-principles`, `30-cost-and-capacity-planning` |
| Quantization | `day-004-quantization-vs-bf16` | Compare BF16 and quantized serving on the same pod class | `18-memory-bandwidth-quantization-numerics` |
| SLM probe | `day-006-slm-memory` | Use a small model to test cold/warm load, allocator, and TTFT behavior | `08-cpu-gpu-node-architecture`, `20-vllm-from-first-principles` |
| Prefix caching | `day-007-vllm-runtime-probes` | Reuse `prefix_prompts.jsonl` against vLLM with prefix caching on/off | `14-attention-kv-cache-long-context` |
| SLO lab | `day-009-latency-metrics-and-slo-lab` | Run streaming TTFT and p95/p99 load tests | `06-inference-metrics-first-principles` |
| Runtime shootout | `day-011-runtime-shootout` | Keep request shapes identical across vLLM, TGI, TensorRT-LLM, or hosted endpoints | `19-serving-engines-and-runtime-map` |
| Long context vs RAG | `day-012-long-context-vs-rag` | Compare long prompt buckets against retrieval-shaped prompts | `14-attention-kv-cache-long-context` |
| Speculative decoding | `day-013-speculative-decoding` | Measure draft/target speedup, rejection, and quality drift | `23-speculative-decoding` |
| Multi-GPU scaling | `day-014-multi-gpu-scaling` | Use a multi-GPU pod for tensor/data parallel comparisons | `26-multi-gpu-and-multi-node-scaling` |
| Multi-tenant QoS | `day-015-multi-tenant-qos` | Mix chat and batch traffic, then add simple priority/rate policies | `27-multitenancy-and-qos` |
| Cost accounting | `day-016-cost-efficiency-accounting` | Convert throughput and pod price into cost per 1M tokens | `30-cost-and-capacity-planning` |
| Reliability and guardrails | `day-017-reliability-guardrails` | Trigger OOM, timeout, restart, and guardrail-overhead scenarios | `29-observability-reliability-security` |
| Alignment serving | `day-018-alignment-e2e-demo` | Serve baseline and tuned variants with the same benchmark harness | `16-rl-and-policy-inference`, `28-evals-and-regression-testing` |
| MoE or state-space | `day-019-state-space-or-moe` | Compare architecture-specific latency, memory, and runtime support | `15-moe-state-space-and-specialized-models` |

## Minimum Runpod Record

Every Runpod experiment should capture:

- pod provider, region, GPU, VRAM, hourly rate, and runtime duration,
- container image and Python/runtime versions,
- model, precision, context length, and vLLM flags,
- benchmark command and exact prompt workload,
- p50/p95/p99 latency, TTFT, output tokens/sec, request throughput,
- GPU memory, GPU utilization, CPU utilization, and any vLLM startup logs,
- raw output path in `results/`,
- one short "what changed in my mental model" note.

## First Three Runs

Start with these because they unlock the rest of the book:

1. vLLM smoke test from `day-002-GPU-node-bring-up`.
2. Capacity and OOM grid from `day-003-vLLM-capacity-and-OOM`.
3. TTFT and prefix caching from `day-007-vllm-runtime-probes`.

After those three, the book has enough evidence to explain the Runpod workflow,
vLLM capacity, KV cache behavior, and benchmark discipline without staying
purely theoretical.
