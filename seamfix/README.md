# seamfix

Repairs the joins of a repeat tile for textile and wallpaper print, and changes nothing else.

## Run the app

```bash
cd seamfix
uv sync                                   # Python 3.12, CPU torch
uv run uvicorn seamfix.server:app --port 8000
```

Open http://localhost:8000, drop in a tile and press **Repair tile**.

- **Full** mode: Nano Banana 2.1 paints, Gemini 3.1 Pro judges. It needs `GEMINI_API_KEY` in `../.env` or the shell.
- **Free** mode: local LaMa only. It makes no API calls.
- **Results** tab: every run, plus the edge-case suite, shown as before/after 2×2 with a slider.

Command line: `uv run python -m seamfix ../pattern.jpg [--free] [--no-judge] [--attempts 3]`

## What happens to a tile

| The tile… | What you get |
|---|---|
| passes the seam check | the original file, byte for byte (`*_seamless.*`) |
| contains a smaller repeat (cropped too big, half-drop) | the true repeat, cut pixel for pixel (`*_repeat.png`). The size changes. |
| is a photo, mockup or has transparency | refused, with the reason |
| has joins that fail | only a band along the failing stretches is repainted (`*_fixed.png`) |
| fails, and no repair passes every check | the original is kept (`*_original.*`), with `*_partial_for_review.png` |

Each repair is checked twice. The deterministic gates come first: pixels outside the band unchanged, join ratio, no new seam at the band edges, detail versus neighbours, no new colours. The judge then looks for invented motifs, ghosting and a still-visible join. If any check fails, the next attempt gets the failure notes. A repair is never shipped unless it passes.

Outputs go to `runs/<id>/`, which holds `report.json`, `events.jsonl` and every step's images.

## Budget

Every paid call goes through `runs/ledger.json`, against a hard cap (default $1.50) set by `SEAMFIX_BUDGET_USD`. When the cap is reached, the run keeps the original instead of overspending. Costs measured so far: about $0.077 per paint, $0.023 per judge call and $0.002 per analyst call.

## Tests

```bash
uv run pytest                         # 24 offline tests, no API calls
uv run python -m seamfix.cases        # edge cases -> test_cases/index.html (also in the Results tab)
uv run python -m seamfix.cases --samples
```

## Layout

```
seamfix/
  checker.py     the user-confirmed seam check, ported unchanged (a test compares it with v1)
  period.py      finds a smaller or half-drop repeat inside the image and cuts it
  plan.py        chooses the fewest stretches to repaint, cut into native-resolution windows that wrap around the tile edges; the corner is painted once
  composite.py   aligns and colour-matches the model output, blends only inside the band
  gates.py       deterministic checks
  repair.py      one attempt: paint, fit, sharpen to match neighbours, gate
  pipeline.py    the LangGraph state machine (workflow.png, steps 1-8)
  agents/        prompts.py (every prompt), analyst, judge, schemas, llm (the only Gemini client)
  painters/      lama.py (local), nano_banana.py
  server.py      FastAPI + server-sent events; web/ is the page
```

`experiments/` holds the paid probe used to tune the prompts (`nb_probe.py`), an aligned before/after viewer (`compare.py`) and the raw Nano Banana outputs from tuning.

## Known limits, measured

- LaMa never passed the gates on detailed ornament or on the synthetic tiles: it smears. Nano Banana is the painter that works.
- On `pattern.jpg`, Nano Banana fixes the join, but it redraws a motif cut by the join instead of completing it, and its edges are softer. Matching sharpening fixes the softness.
- The judge, Gemini 3.1 Pro, under-scores finish by about 1 point, even on untouched artwork. Its acceptance rule is calibrated for this: finish ≥ 3.
- The confirmed seam check can fail a perfect tile when an outline runs along the join, mostly on small or busy tiles. A cut that is proven exact (the pixels past it repeat it) is accepted on that axis regardless.
