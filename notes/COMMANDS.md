### 1. `llama-server`

```bash
./build/bin/llama-server \
  -m ./models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  --mmproj ./models/mmproj-BF16.gguf \
  --alias gemma-4-E4B-it \
  --host 127.0.0.1 \
  --port 9379 \
  --parallel 1 \
  --n-gpu-layers all \
  --mmproj-offload \
  --flash-attn on \
  --ctx-size 4096 \
  --threads 4 \
  --threads-batch 8 \
  --reasoning on \
  --reasoning-budget 256 \
  --metrics \
  --perf
```

This limits reasoning to 256 tokens rather than allowing it to consume the entire completion budget. `--reasoning-budget` accepts a positive reasoning-token limit, while `-1` is unrestricted. ([GitHub][1])

### 2. RobotMDAR producer

```bash
uv run python -m mjlab_textop.robotmdar.produce \
  --ckpt /tmp/textop-data/TextOpRobotMDAR/logs/pretrained/checkpoint/ckpt_200000.pth \
  --datadir /tmp/textop-data/TextOpRobotMDAR/dataset/PRIVATE-DATA/ \
  --skeleton-asset-root /tmp/textop-data/TextOpRobotMDAR/description/robots/g1 \
  --planner vlm \
  --prompt "stand" \
  --observation-listen-port 8766 \
  --query-every-blocks 4 \
  --vlm-base-url http://127.0.0.1:9379 \
  --vlm-model gemma-4-E4B-it \
  --vlm-system-prompt ./sys.md \
  --vlm-user-prompt ./user.md \
  --vlm-max-tokens 320
```

The important changes are:

```text
--query-every-blocks 20  →  4
--vlm-max-tokens 1024    →  320
```

The server reserves up to 256 of those tokens for reasoning, leaving room for the final command.

### 3. `play-live`

```bash
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
uv run --extra cu128 mjlab-textop play-live \
  --onnx-file "$ONNX_PATH" \
  --task straight \
  --reference-debug-vis \
  observation:observation-params
```

### More conservative VLM cadence

Use this only if querying every four blocks noticeably slows the overall system:

```bash
uv run python -m mjlab_textop.robotmdar.produce \
  --ckpt /tmp/textop-data/TextOpRobotMDAR/logs/pretrained/checkpoint/ckpt_200000.pth \
  --datadir /tmp/textop-data/TextOpRobotMDAR/dataset/PRIVATE-DATA/ \
  --skeleton-asset-root /tmp/textop-data/TextOpRobotMDAR/description/robots/g1 \
  --planner vlm \
  --prompt "stand" \
  --observation-listen-port 8766 \
  --query-every-blocks 8 \
  --vlm-base-url http://127.0.0.1:9379 \
  --vlm-model gemma-4-E4B-it \
  --vlm-system-prompt ./sys.md \
  --vlm-user-prompt ./user.md \
  --vlm-max-tokens 320
```

Start with **four blocks**. Switch to **eight** only if requests overlap or the VLM cannot keep up.

[1]: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md?utm_source=chatgpt.com "llama.cpp/tools/server/README.md at master · ggml-org ..."
