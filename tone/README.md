# Nous Tone

Adjust the tone of a self-hosted open LLM from inside the model, without fine-tuning and without changing the prompt. While the reply streams, Nous Tone also reads which tone the model is in, token by token, for avatars, robots and safety. The two connect: give it a target intensity, and it raises or lowers the steering until the readout gets there.

> Status: 0.1, internal. It comes out of the E01–E06 research in this repository. Head selection uses attribution patching (the GCM recipe, arXiv 2602.16080), which beat the repository's own two-stage selection in E06.

## What it does

| | How |
|---|---|
| **Steer** | Each attribute is a *tone pack*: about 12 attention heads, each with a direction vector. During generation the vectors are added to those heads' outputs, from the last user turn onward and at every new token. `tone: {"anger": -1}` pushes away from anger; `{"warmth": 1.5}` pushes toward warmth. |
| **Read** | A multinomial linear probe on one middle layer's residual (reference / anger / joy / sadness / warmth) reads the running mean over the reply. Before the first token it gives the pre-reply state; after that it updates every token. Readouts map to avatar expression weights (`happy` / `sad` / `angry` / `relaxed` / `neutral`). |
| **Target mode** | `tone_target: {"warmth": 0.7}`. The strength is picked up front from the pack's held-out calibration (strength → mean readout), because a reply's tone is set in its first tokens. Within the reply, the readout only lowers lambda if it overshoots. Token-by-token closed-loop control was tried first and did not work (see `EVAL.md`). |
| **Guard** | Each pack's default lambda is the largest one whose fluency cost stays under 0.3 nats/token, measured as the rise in NLL of the model's own neutral text. `max_lambda` is twice the default. The research found that strong steering costs fluency and factual accuracy (E04–E06). |

## Quick start

```bash
pip install -r requirements.txt
# 1. build tone packs (one per attribute) and the readout probe for your model
python -m nous_tone.build   MODEL_DIR anger   packs/anger      # ~4-5 min for Qwen3-1.7B on an M4
python -m nous_tone.build   MODEL_DIR warmth  packs/warmth
python -m nous_tone.readout MODEL_DIR packs/readout            # 5-way probe, layer chosen by CV
python -m nous_tone.evaluate MODEL_DIR packs/* eval.jsonl --readout packs/readout   # held-out replies
python -m nous_tone.calibrate eval.jsonl packs/anger packs/warmth                    # enables target mode
# 2. try it
python -m nous_tone.demo    MODEL_DIR packs/anger packs/warmth --say "老板让我周末加班。" --tone warmth=1.5
# 3. serve (OpenAI-compatible, 127.0.0.1 only)
python -m nous_tone.server  MODEL_DIR packs/anger packs/warmth --readout packs/readout --port 7880
```

Request:

```json
POST /v1/chat/completions
{"messages": [{"role": "user", "content": "快递又丢了"}], "stream": true,
 "tone": {"anger": -1.0}, "tone_target": {"warmth": 0.7}}
```

Each streamed chunk carries `x_tone`: `readout` (per tone, 0–1), `expression` (avatar weights) and `lambda` (the current steering).

## Modules

| File | Role |
|---|---|
| `nous_tone/build.py` | Builds a tone pack in five steps: contrast pairs → self-generated replies → attribution patching over every head at the aligned user turn → vectors → dose curve and default lambda |
| `nous_tone/readout.py` | Trains the readout probe |
| `nous_tone/engine.py` | Runs the KV-cached decoding loop with steering, readouts, closed loop and expression weights |
| `nous_tone/server.py` | The OpenAI-compatible API |
| `nous_tone/demo.py` | Terminal demo |
| `nous_tone/evaluate.py` | Held-out generations for blind rating |
| `nous_tone/hooks.py` | Hooks on every layer's o_proj input (the per-head outputs) |
| `nous_tone/pack.py` | The pack format: `pack.json` + `vectors.npz` |
| `tests/test_tone.py` | Tests on a tiny random Qwen3 (seconds, CPU) |

The tests cover:
- the attribution gradient against finite differences;
- the decoding loop against full-recompute greedy decoding;
- the closed loop;
- the build end to end;
- the API.

## Known limits (0.1)

- **Model coverage:** the hooks assume a HuggingFace decoder (`model.model.layers[i].self_attn.o_proj`). Tested on Qwen3 only.
- **Concurrency:** one request at a time, under a global lock. Batch steering is not done.
- **Readout prompt mismatch:** the probe reads text under the reference system prompt. At runtime the conversation's own prompt is used, so expect some shift; calibrate per product.
- **Pack quality on Qwen3-1.7B** (blind-rated held-out set, `EVAL.md`):
  - **warmth** and **joy** work in both directions: +1.2 / +1.1 out of 4 at strength 2, −0.7 / −0.4 at −1;
  - **sadness** is weak (+0.3);
  - **anger** does not steer on this model (+0.15);
  - the four packs share most heads, and joy and warmth spill into each other.
- **Readout quality:** the readout tracks raters at ρ 0.66 for warmth and 0.48 for joy, but tracks sadness and anger poorly.
- **Human calibration:** none yet. Ratings come from two LLM raters.
