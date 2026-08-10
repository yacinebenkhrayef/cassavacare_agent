# §6.4 — Qualitative UX Scorecard (5 representative scenarios)

One row per disease class (FR2's 5 classes), scored against the live dashboard at
http://localhost:8501 — not the API directly.

**Stopwatch convention:** start the instant you click **Diagnose**; stop the instant the result
tabs are fully rendered and interactive (spinner gone — Grad-CAM is already loaded by then,
since Part 2 fetches it automatically, so you don't need to click into that tab first).

| Class | Image | Upload→Display (s) | Clarity of explanation (1–5) | Heatmap intuitiveness (1–5) | Notes |
|---|---|---|---|---|---|
| CBB | | | | | |
| CBSD | | | | | |
| CGM | | | | | |
| CMD | | | | | |
| Healthy | | | | N/A | |

**Targets (§6.4):** every "Upload→Display" cell should read under 10s. The cahier des charges
sets no numeric bar for the qualitative columns — but flag anything scoring ≤2 as a concrete UX
issue to note (and ideally fix) before Phase 6.