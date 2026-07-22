# Optimized `play-live` stack

Start these commands in order. They use the existing TCP/HTTP transports.

## 1. `llama-server`

Use the FP16 multimodal projector on the RTX 2080 Ti:

```bash
./build/bin/llama-server \
  -m ./models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  --mmproj ./models/mmproj-F16.gguf \
  --alias gemma-4-E4B-it \
  --host 127.0.0.1 \
  --port 9379 \
  --parallel 1 \
  --n-gpu-layers all \
  --mmproj-offload \
  --flash-attn on \
  --ctx-size 8192 \
  --cache-prompt \
  --cache-reuse 256 \
  --threads 4 \
  --threads-batch 8 \
  --reasoning on \
  --reasoning-budget 256 \
  --metrics \
  --perf
```

This limits reasoning to 256 tokens rather than allowing it to consume the
entire completion budget. `--reasoning-budget` accepts a positive
reasoning-token limit, while `-1` is unrestricted. ([GitHub][1])

For more difficult scenes, try a 384-token reasoning budget and increase the
producer completion budget to 448 tokens. Increase both values together; the
completion budget must still leave room for the final motion command:

```text
llama-server: --reasoning-budget 384
producer:     --vlm-max-tokens 448
```

More reasoning increases VLM latency, so keep the 256/320 pair as the baseline.

## 2. RobotMDAR producer

Run this in the TextOp/RobotMDAR environment:

```bash
uv run python -m mjlab_textop.robotmdar.produce \
  --ckpt /tmp/textop-data/TextOpRobotMDAR/logs/pretrained/checkpoint/ckpt_200000.pth \
  --datadir /tmp/textop-data/TextOpRobotMDAR/dataset/PRIVATE-DATA/ \
  --skeleton-asset-root /tmp/textop-data/TextOpRobotMDAR/description/robots/g1 \
  --device cuda \
  --planner vlm \
  --prompt "stand" \
  --observation-listen-port 8766 \
  --vlm-base-url http://127.0.0.1:9379 \
  --vlm-model gemma-4-E4B-it \
  --vlm-system-prompt ./sys.md \
  --vlm-user-prompt ./user.md \
  --vlm-history-length 5 \
  --vlm-max-tokens 320 \
  --vlm-reasoning
```

The first VLM request starts after the initial motion block is generated. Later
requests only use new images; images received during inference are coalesced to
the newest one. `--observation.every-frames` on `play-live` controls the maximum
query rate. Repeated RobotMDAR prompts reuse a bounded text-embedding cache
automatically. Each VLM request includes at most five user-image turns: four
completed user/assistant pairs plus the current image. The 8192-token context
provides headroom for that window, while prompt caching reuses compatible KV
cache regions. The server reserves up to 256 of the 320 completion tokens for
reasoning, leaving room for the final command. Set `--vlm-history-length 1` to
restore stateless requests.

## 3. `play-live`

Run the MJLab simulation and ONNX actor on CUDA. Observation settings are shown
explicitly so the VLM receives 320x240 images at most every 20 TextOp frames
(2.5 images per second at 50 Hz):

```bash
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
uv run --extra cu128 mjlab-textop play-live \
  --onnx-file "$ONNX_PATH" \
  --onnx-provider cuda \
  --device cuda:0 \
  --task portrait-corridors \
  observation:observation-params \
  --observation.url http://127.0.0.1:8766/observation \
  --observation.every-frames 20 \
  --observation.image-width 320 \
  --observation.image-height 240
```

Motion arrays are transferred to the MJLab device once per field and block,
rather than once per frame. Add `--reference-debug-vis` before
`observation:observation-params` only when the ghost reference is useful.

### CPU ONNX fallback

If CUDA ONNX Runtime still crashes, keep MJLab on `cuda:0` and change only the
provider:

```bash
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
uv run --extra cu128 mjlab-textop play-live \
  --onnx-file "$ONNX_PATH" \
  --onnx-provider cpu \
  --device cuda:0 \
  --task straight \
  observation:observation-params \
  --observation.url http://127.0.0.1:8766/observation \
  --observation.every-frames 20 \
  --observation.image-width 320 \
  --observation.image-height 240
```

### More conservative VLM cadence

If VLM inference noticeably slows the shared GPU, publish images less often:

```bash
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
uv run --extra cu128 mjlab-textop play-live \
  --onnx-file "$ONNX_PATH" \
  --onnx-provider cuda \
  --device cuda:0 \
  --task straight \
  observation:observation-params \
  --observation.url http://127.0.0.1:8766/observation \
  --observation.every-frames 40 \
  --observation.image-width 320 \
  --observation.image-height 240
```

[1]: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md "llama.cpp server README"
