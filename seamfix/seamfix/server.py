"""The web app: upload a tile, watch each step, download the result.

    uvicorn seamfix.server:app --port 8000

Runs go to runs/<id>/ and their events to events.jsonl, which the page reads
as server-sent events. Test cases from `python -m seamfix.cases` are served
from test_cases/ for the Results tab.
"""

import asyncio
import contextlib
import json
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from seamfix import config, pipeline
from seamfix.budget import LEDGER

WEB = config.ROOT / "web"
CASES = config.ROOT / "test_cases"
UPLOADS = config.RUNS / "uploads"
for folder in (config.RUNS, UPLOADS, CASES):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="seamfix")
app.mount("/web", StaticFiles(directory=WEB), name="web")
app.mount("/runs", StaticFiles(directory=config.RUNS), name="runs")
app.mount("/cases", StaticFiles(directory=CASES), name="cases")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/budget")
def budget():
    return {"spent": LEDGER.spent, "cap": LEDGER.cap, "left": LEDGER.remaining}


@app.post("/api/runs")
async def start(file: UploadFile = File(...), mode: str = Form("full"), attempts: int = Form(3)):
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename or "tile.png").name)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{Path(name).stem}-{uuid.uuid4().hex[:4]}"
    src = UPLOADS / run_id / name  # keeps the designer's file name for outputs and results
    src.parent.mkdir(parents=True)
    src.write_bytes(await file.read())
    free = mode == "free"
    options = {"painters": ["lama"] if free else ["lama", "nano"], "judge": not free,
               "analyst": not free, "attempts": max(1, min(attempts, 4))}

    def work():
        with contextlib.suppress(Exception):  # the error is in events.jsonl for the page
            pipeline.run(src, options, run_id=run_id)

    threading.Thread(target=work, daemon=True).start()
    return {"run_id": run_id}


@app.get("/api/runs/{run_id}/events")
async def events(run_id: str):
    if not re.fullmatch(r"[\w.-]+", run_id):
        raise HTTPException(400)
    log = config.RUNS / run_id / "events.jsonl"

    async def stream():
        sent, waited = 0, 0.0
        while not log.exists() and waited < 30:
            await asyncio.sleep(0.25)
            waited += 0.25
        while True:
            lines = log.read_text().splitlines() if log.exists() else []
            for line in lines[sent:]:
                yield f"data: {line}\n\n"
                if json.loads(line).get("type") == "end":
                    return
            sent = len(lines)
            await asyncio.sleep(0.4)

    return StreamingResponse(stream(), media_type="text/event-stream")


def _summary(report: Path, base: str, extra: dict | None = None) -> dict:
    r = json.loads(report.read_text())
    folder = report.parent
    after = {"fixed": "fixed_2x2.png", "cut": "cut_2x2.png"}.get(r["status"], "input_2x2.png")
    return {
        "status": r["status"], "reason": r.get("reason", ""), "input": r["input"]["name"],
        "size_in": r["input"]["size"], "size_out": r.get("output_size"),
        "before": f"{base}/input_2x2.png" if (folder / "input_2x2.png").exists() else None,
        "after": f"{base}/{after}" if (folder / after).exists() else None,
        "marked": f"{base}/input_marked.png" if (folder / "input_marked.png").exists() else None,
        "design": f"{base}/{r['files']['design']}" if r.get("files", {}).get("design") else None,
        "check": r.get("check"), "windows": r.get("windows", []),
        "modified": folder.stat().st_mtime,
    } | (extra or {})


@app.get("/api/results")
def results():
    out = []
    for report in sorted(config.RUNS.glob("*/report.json")):
        out.append(_summary(report, f"/runs/{report.parent.name}", {"kind": "run", "name": report.parent.name}))
    cases = json.loads((CASES / "results.json").read_text()) if (CASES / "results.json").exists() else []
    for case in cases:
        folder = CASES / (case["name"] if not case["name"].startswith("sample") else f"sample_{Path(case['input']).stem}")
        report = folder / "run" / "report.json"
        if report.exists():
            out.append(_summary(report, f"/cases/{folder.name}/run", {
                "kind": "case", "name": case["name"], "tests": case["tests"], "ok": case["ok"],
                "expect": case["expect"], "checks": case["checks"]}))
    return sorted(out, key=lambda r: (r["kind"] != "run", -r["modified"]))
