# Runpod Experiment - vLLM Smoke Test

Date: 2026-05-11

## Goal

Start one GPU pod, serve one model with vLLM, and confirm I can measure a basic
request path.

## Runpod Setup

- Pod type:
- GPU:
- GPU memory:
- Region:
- Container image:
- Persistent volume:
- Estimated hourly cost:

## Server Command

```bash
python -m vllm.entrypoints.openai.api_server \
  --model MODEL_NAME \
  --host 0.0.0.0 \
  --port 8000
```

## Client Command

```bash
curl http://POD_HOST:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MODEL_NAME",
    "messages": [{"role": "user", "content": "Explain TTFT in one paragraph."}],
    "max_tokens": 128
  }'
```

## Results

| Metric | Value |
| --- | ---: |
| TTFT |  |
| Total latency |  |
| Tokens/sec |  |
| Error rate |  |

## Notes

Write raw observations here.

