"""Nous Tone: steer and read the tone of an open LLM from inside, with the same attention heads.
build -> a tone pack per attribute (heads selected by attribution patching, per-head direction vectors, readout calibration)
engine -> generate with tone strengths and/or target intensities, per-token readouts and avatar expression weights
server -> OpenAI-compatible API with `tone` / `tone_target` fields and readouts in the stream"""
__version__ = "0.1.0"
