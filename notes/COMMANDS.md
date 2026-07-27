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
  --reasoning-budget -1 \
  --n-predict - \
  --metrics \
  --perf
```

```text
llama-server: --reasoning-budget 384 --n-predict 448
```

## 2. RobotMDAR producer

Run this in the TextOp/RobotMDAR environment. Point `PYTHONPATH` at this
repository's `src` directory first:

```bash
export PYTHONPATH=/absolute/path/to/mjlab_textop/src
```

```bash
uv run python -m robotmdar_textop.produce \
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
  --vlm-invariant \
  --vlm-user-prompt ./user.md \
  --vlm-history-length -1 \
  --vlm-reasoning
```

The first VLM request starts after the initial motion block is generated. Later
requests only use new images; images received during inference are coalesced to
the newest one. `--obs.every-frames` after `obs` controls the maximum
query rate. Repeated RobotMDAR prompts reuse a bounded text-embedding cache
automatically. Each VLM request includes the full conversation history by
default. Returned reasoning is preserved with each assistant turn so Gemma 4
can continue thinking across the conversation, while prompt caching reuses
compatible KV cache regions. The server reserves up to 256 of the 320
completion tokens for reasoning, leaving room for the final command. Set
`--vlm-history-length` to a positive number to bound the number of user-image
turns, or to `1` for stateless requests.

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
  --device cuda:0 \
  --task portrait-corridors \
  obs \
  --obs.url http://127.0.0.1:8766/observation \
  --obs.every-frames 20 \
  --obs.image-size 320 240
```

Motion arrays are transferred to the MJLab device once per field and block,
rather than once per frame. Add `--ref-vis` before `obs` only when the
ghost reference is useful.

### More conservative VLM cadence

If VLM inference noticeably slows the shared GPU, publish images less often:

```bash
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
uv run --extra cu128 mjlab-textop play-live \
  --onnx-file "$ONNX_PATH" \
  --device cuda:0 \
  --task straight \
  obs \
  --obs.url http://127.0.0.1:8766/observation \
  --obs.every-frames 40 \
  --obs.image-size 320 240
```

[1]: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md "llama.cpp server README"
