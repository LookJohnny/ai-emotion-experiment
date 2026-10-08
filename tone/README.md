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

Each streamed chunk carries `x_tone`. It has three fields:
- `readout`: per tone, 0–1;
- `expression`: avatar weights;
- `lambda`: the current steering.

### Serving an application with one local model

These additions were made while serving SoulForge's chat brain:

| Feature | Request / API | Why |
|---|---|---|
| **Per-sentence readout of finished text** | `POST /v1/tone/read {"text", "segments": [[s, e], ...], "user"}` | The text is forced in as a reply to `user` under the reference prompt (as the probe was trained), so the reading reflects the words, not the JSON or prompt that carried them. A segment's mean is shrunk toward the whole text's mean (8 pseudo-tokens), so a short sentence keeps the reply's tone. |
| **Readout-only mode** | start without packs: `server MODEL --readout DIR` | Expressions need only the probe; steering packs need gradients to build. |
| **Priority** | `"priority": 1` (−10..10) | One generation at a time. When the model frees, the highest-priority waiter goes next, so a user's turn never queues behind background work. Reads default to high. |
| **Prefill** | `"prefill": "{\"selected_intent\": \""` | Forces how the reply opens; the prefill is part of the returned text. On real prompts a 1.7B model otherwise echoed its input JSON or answered in its history's plain-text style. |
| **JSON mode** | `"response_format": {"type": "json_object"}` | Generation stops when the top-level object closes, and the reply is that object only. |
| **Request log** | always on; `NOUS_TONE_LOG_TEXT=1` adds outputs | Each request logs priority, prompt and output tokens, queue wait, generation time and tok/s. The output log is for local debugging only. |

### Readout probe for spoken lines

- **Sentence samples.** `python -m nous_tone.readout MODEL OUT [--short]` trains on whole replies and
  also on each sentence of them, built exactly as `read()` builds it.
- **Calibration.** The logits get a temperature fitted on out-of-fold predictions.
- **Short replies.** `--short` also generates every tone as a one- or two-sentence reply, which is
  the shape of companion speech.
- **Evaluation.** `python -m nous_tone.eval_lines MODEL DIR...` scores probes on
  `eval/companion_lines.json`. That file holds 40 short lines, 8 per tone, from one annotator, so it
  is a dev set and not a human study.

## Modules

| File | Role |
|---|---|
| `nous_tone/build.py` | Builds a tone pack in five steps: contrast pairs → self-generated replies → attribution patching over every head at the aligned user turn → vectors → dose curve and default lambda |
| `nous_tone/readout.py` | Trains the readout probe: whole replies plus sentences, optional short replies, temperature calibration |
| `nous_tone/eval_lines.py` | Scores probes on labelled short lines |
| `nous_tone/engine.py` | Runs the KV-cached decoding loop with steering, readouts, closed loop and expression weights. Also provides `read()` for finished text, the priority lock, prefill and the JSON object end |
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
- **Concurrency:** one request at a time, under a priority lock. Batch steering is not done.
- **Readout prompt mismatch:** the probe reads text under the reference system prompt. At runtime the conversation's own prompt is used, so expect some shift; calibrate per product.
- **Pack quality on Qwen3-1.7B** (blind-rated held-out set, `EVAL.md`):
  - **warmth** and **joy** work in both directions: +1.2 / +1.1 out of 4 at strength 2, −0.7 / −0.4 at −1;
  - **sadness** is weak (+0.3);
  - **anger** does not steer on this model (+0.15);
  - the four packs share most heads, and joy and warmth spill into each other.
- **Readout quality:** the readout tracks raters at ρ 0.66 for warmth and 0.48 for joy, but tracks sadness and anger poorly.
- **Sentence readout** on the 40-line companion dev set (`eval/companion_lines.json`, one annotator):

  | Probe | Accuracy | Valence flips | Notes |
  |---|---|---|---|
  | `readout` (0.1) | 0.70 | — | uncalibrated, saturates at 0/1 |
  | `readout_v2` (sentences + temperature 1.41) | 0.70 | 5 | neutral recall 1.0 |
  | `readout_v3` (+ `--short`) | 0.725 | 9 | |

  Cross-validated within the training distribution, v2 scores 0.865 on replies and 0.781 on
  sentences. Warmth is the weakest class on short lines: recall 0.12–0.25, usually misread as
  neutral or sadness. Prefer the probe that fails toward neutral.
- **Human calibration:** none yet. Ratings come from two LLM raters.
