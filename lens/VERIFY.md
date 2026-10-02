# Nous Lens 0.1 verification record (2026-10-02, MacBook M4 16 GB, MPS)

`scripts/verify.py` re-reads every reply from the pre-registered reader experiment (`2026-09-28/monitor/reader`) through the product engine. The engine differs from the research pipeline in three ways:
- it does one forward pass instead of token-by-token teacher forcing;
- it loads only the first 23 layers;
- it pads inputs to fixed lengths.

The engine then reproduces the research scores:

| Readout | n | Lens AUROC | Research AUROC | Correlation | Max abs. difference |
|---|---|---|---|---|---|
| Hostility (C1) | 143 (18 hostile) | 0.973 | 0.973 | 1.0000 | 0.0035 |
| Fabrication (C3-H, experimental) | 338 (53 wrong) | 0.738 | 0.737 | 0.9999 | — |

Speed: 0.9–2 s per reply. Memory: about 5 GB (layers ≥ 23 and the LM head are not loaded).

Fresh-install test: unzip the package, then run `start-mac.command` with an empty home directory.
- The environment installs automatically. The domestic pip mirror was unavailable during the test, and the script fell back to the official index.
- The model loads and the service comes up.

Extension test: headless Chromium with the extension loaded, on the simulated chat page `/chat`.
- A hostile reply gets the "在贬低你" marker, and only the insulting sentence is shown.
- Switching to a fabricated answer gets "可能在编造（实验）".
- The popup shows the counts.

## Not tested
- Real ChatGPT / Kimi / Doubao pages. Their selectors are best-effort and need checking on a logged-in page; right-click checking works on any site.
- The Windows script (`start-windows.bat`). The 5080 host was offline.
- Hostility on knowledge-work chats. The probe was validated on companion conversations, and false alarms there are not measured.
