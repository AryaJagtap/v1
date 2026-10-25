"""The workflow, as a LangGraph state machine.

    measure ──ok──────────────────────────────────────────────► finish (original, byte for byte)
       │ fails
       ▼
    repeat ──holds a smaller repeat──► (cut it out) ──► measure
       │ none
       ▼
    diagnose ──photo or mockup──────────────────────────────────► finish (refused)
       │ repeat tile, or artwork never made to repeat (wide band)
       ▼
    plan ──► paint ──► review ──retry with the notes────► paint
                         │ window settled, more left ─► paint
                         ▼ all settled
                       assemble ──still failing, a round left──► plan
                         │
                         ▼
                       finish (fixed, or the original kept for review)

Code decides every edge. Models only propose pixels, describe the design and
criticise a candidate. Every step is streamed as an event; images are written
to the run directory and events carry their paths.
"""

import json
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from PIL import Image, ImageDraw

from seamfix import checker, config, gates, imaging, period, repair
from seamfix import plan as planner
from seamfix.agents import analyst, judge
from seamfix.budget import LEDGER, BudgetExceeded
from seamfix.painters import Request
from seamfix.painters.lama import Lama
from seamfix.painters.nano_banana import NanoBanana

# what a failed gate tells the painter for its next attempt
GATE_NOTES = {
    "join": "The join is still visible along the centre line; make every line, dot row and fill meet across it.",
    "edges": "The strip's edges show against the untouched artwork; carry the artwork right up to them.",
    "detail": "The strip is softer or emptier than its surroundings; keep the same density of small marks and the same crisp edges.",
    "detail_local": "Part of the strip lost the ground texture; keep the fine texture all the way along it.",
    "new_colour": "The strip holds colours the design does not use; use only the design's palette.",
    "drift": "The picture moved or changed scale; keep the framing exactly as given.",
}


class Run(TypedDict, total=False):
    run_id: str
    src: str
    dir: str
    options: dict
    tile: imaging.Tile
    original: np.ndarray  # RGB the repair starts from: the input, or its cut
    work: np.ndarray  # RGB being repaired
    check: checker.SeamCheck
    cut: dict
    tried_repeat: bool
    brief: dict
    difficulty: str
    whole: bool
    plan: planner.Plan
    claimed: np.ndarray  # every pixel any window of any round was allowed to change
    window: int
    attempt: int
    feedback: str
    best: Any  # the best candidate of the current window
    last: Any  # the newest candidate, until review settles it
    windows: list[dict]  # what happened in each window, every round
    round: int
    status: str
    reason: str
    result: dict


def emit(event: str, /, **data) -> None:
    get_stream_writer()({"type": event, "at": round(time.time(), 2)} | data)


def _save(state: Run, name: str, image: Image.Image | np.ndarray) -> str:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    path = Path(state["dir"]) / f"{name}.png"
    image.save(path)
    return path.name


def _preview(state: Run, prefix: str, image: Image.Image, check: checker.SeamCheck) -> dict:
    grid = imaging.tile_2x2(image)
    return {
        "2x2": _save(state, f"{prefix}_2x2", imaging.thumbnail(grid, 1600)),
        "diamond": _save(state, f"{prefix}_diamond", imaging.thumbnail(imaging.diamond(grid), 1600)),
        "marked": _save(state, f"{prefix}_marked", imaging.thumbnail(imaging.mark_joins(grid, check), 1600)),
    }


# ---------- nodes ----------


def measure(state: Run) -> dict:
    tile = state.get("tile") or imaging.load(Path(state["src"]))
    emit("stage", stage="measure", message=f"{tile.size[0]}x{tile.size[1]} {tile.image.mode}", notes=tile.notes)
    check = checker.check(tile.rgb)
    first = "cut" if state.get("cut") else "input"
    images = {"tile": _save(state, f"{first}_tile", imaging.thumbnail(tile.image, 1200))}
    images |= _preview(state, first, tile.image, check)
    emit("check", which=first, check=check.summary(), images=images)
    update: dict = {"tile": tile, "check": check}
    if "original" not in state:
        update |= {"original": tile.rgb.copy(), "work": tile.rgb.copy(), "windows": [], "round": 1}
    if not tile.opaque:
        update |= {"status": "refused", "reason": "the tile has transparent pixels; print repair needs an opaque tile"}
    return update


def after_measure(state: Run) -> str:
    if state.get("status") == "refused":
        return "finish"
    if not _failing(state):
        return "finish"
    return "diagnose" if state.get("tried_repeat") else "repeat"


def _failing(state: Run, check: checker.SeamCheck | None = None) -> tuple[str, ...]:
    """Joins still to repair: failing the seam check and not proven pixel-exact
    by a cut."""
    proven = state.get("cut", {}).get("proven", []) if state.get("cut") else []
    return tuple(a for a in (check or state["check"]).failing if a not in proven)


def repeat(state: Run) -> dict:
    emit("stage", stage="repeat", message="looking for a smaller repeat inside the image")
    tile = state["tile"]
    cut = period.best_cut(tile.image, tile.rgb)
    if cut is None:
        emit("repeat", found=False)
        return {"tried_repeat": True}
    piece = imaging.Tile(tile.path, cut.image, np.asarray(cut.image.convert("RGB")).copy(), tile.notes)
    oy, ox = cut.offset
    marked = tile.image.convert("RGB").copy()
    bw, bh = cut.repeat.block
    ImageDraw.Draw(marked).rectangle([ox, oy, ox + bw - 1, oy + bh - 1], outline="red", width=max(3, tile.size[0] // 200))
    info = {"repeat": asdict(cut.repeat), "offset": cut.offset, "describe": cut.repeat.describe(),
            "size": [bw, bh], "passes": cut.check.ok, "proven": list(cut.proven)}
    emit("repeat", found=True, **info, image=_save(state, "repeat_found", imaging.thumbnail(marked, 1200)))
    return {"tried_repeat": True, "cut": info, "tile": piece, "original": piece.rgb.copy(), "work": piece.rgb.copy()}


def after_repeat(state: Run) -> str:
    return "measure" if state.get("cut") else "diagnose"


def diagnose(state: Run) -> dict:
    check, tile = state["check"], state["tile"]
    emit("stage", stage="diagnose", message="describing the design and how hard its joins are")
    # edges never made to meet: the join is bad along most of its length, not
    # merely very bad in one stretch, which a narrow repair handles
    def bad_share(j):
        return sum(r > 2 * check.threshold for _, _, r in j.sections) / len(j.sections)

    unrelated = any(
        j.overall > config.UNRELATED_OVERALL and bad_share(j) >= 0.5 for j in (check.lr, check.tb)
    )
    brief: dict = {}
    try:
        if not state["options"].get("analyst", True):
            raise RuntimeError("analyst off")
        marked = imaging.mark_joins(imaging.tile_2x2(tile.image), check)
        notes = tile.notes + ([f"cut to {state['cut']['describe']}"] if state.get("cut") else [])
        brief = analyst.diagnose(tile.image, marked, check, notes).model_dump()
    except BudgetExceeded as e:
        emit("warning", message=f"analyst skipped: {e}")
    except Exception as e:  # no key, network: the measurements still decide
        emit("warning", message=f"analyst unavailable ({type(e).__name__}: {str(e)[:160]})")
    kind = brief.get("kind") or ("artwork_not_repeat" if unrelated else "repeat_tile")
    difficulty = brief.get("difficulty") or "medium"
    whole = kind == "artwork_not_repeat" or unrelated
    if whole:
        difficulty = "hard"
    emit("diagnosis", kind=kind, difficulty=difficulty, whole=whole, brief=brief, unrelated=unrelated)
    if kind == "photo_or_mockup":
        return {"brief": brief, "status": "refused",
                "reason": brief.get("reason") or "this is a photo or mockup, not a flat repeat tile"}
    return {"brief": brief, "difficulty": difficulty, "whole": whole}


def after_diagnose(state: Run) -> str:
    return "finish" if state.get("status") == "refused" else "plan"


def make_plan(state: Run) -> dict:
    work = state["work"]
    check = checker.check(work)
    emit("stage", stage="plan", message=f"round {state['round']}: choosing what to repaint")
    p = planner.make(work, _failing(state, check), state["difficulty"], whole=state["whole"])
    h, w = work.shape[:2]
    grid = imaging.tile_2x2(Image.fromarray(work)).convert("RGBA")
    shade = np.zeros((2 * h, 2 * w), bool)
    for win in p.windows:
        rows = (np.arange(win.y0, win.y0 + win.size) + h) % (2 * h)
        cols = (np.arange(win.x0, win.x0 + win.size) + w) % (2 * w)
        shade[np.ix_(rows, cols)] |= win.mask
    overlay = Image.new("RGBA", grid.size, (0, 0, 0, 0))
    o = np.asarray(overlay).copy()
    o[shade] = (255, 0, 160, 110)
    grid = Image.alpha_composite(grid, Image.fromarray(o))
    draw = ImageDraw.Draw(grid)
    for win in p.windows:
        x, y = win.x0 + w, win.y0 + h
        draw.rectangle([x, y, x + win.size - 1, y + win.size - 1], outline=(0, 120, 255, 255), width=max(2, w // 300))
    claimed = state.get("claimed")
    claimed = p.claimed.copy() if claimed is None else (claimed | p.claimed)
    emit("plan", round=state["round"], band=p.band, spans={k: v for k, v in p.spans.items()},
         windows=[win.box | {"index": win.index} for win in p.windows],
         image=_save(state, f"plan_r{state['round']}", imaging.thumbnail(grid.convert("RGB"), 1600)))
    return {"plan": p, "claimed": claimed, "window": 0, "attempt": 1, "feedback": "", "best": None}


def _painter(state: Run, attempt: int):
    """Attempt 1 is LaMa, free. Later attempts are Nano Banana with the notes."""
    order = state["options"].get("painters", ["lama", "nano"])
    if attempt == 1 and order[0] == "lama":
        return Lama()
    if "nano" not in order:
        return Lama() if attempt == 1 else None
    return NanoBanana(state["options"].get("model", config.PAINT_MODEL))


def paint(state: Run) -> dict:
    p, index, number = state["plan"], state["window"], state["attempt"]
    win = p.windows[index]
    painter = _painter(state, number)
    if painter is None:  # no painter left for this attempt
        return {"last": None, "reason": state.get("reason") or "no painter available"}
    before = repair.crop(state["work"], win)
    emit("stage", stage="paint", window=index, attempt=number, painter=painter.name,
         message=f"window {index + 1}/{len(p.windows)}, attempt {number} with {painter.name}")
    request = Request(before, win.mask, win.axes, state.get("brief") or {}, state.get("feedback", ""),
                      reference=state["work"] if painter.paid else None)
    tag = f"r{state['round']}_w{index}_a{number}"
    try:
        cand = repair.attempt(state["work"], p, win, painter, number, request)
    except BudgetExceeded as e:
        emit("warning", window=index, message=str(e))
        return {"last": None, "reason": "budget"}
    except Exception as e:  # a refusal or a network failure costs this attempt, not the run
        emit("warning", window=index, message=f"{painter.name} failed: {type(e).__name__}: {str(e)[:200]}")
        return {"last": None}
    files = {"panel": _save(state, f"panel_{tag}", judge.panel(before, cand.after, win.mask))}
    emit("candidate", window=index, attempt=number, painter=cand.painter, seconds=cand.seconds,
         gates=cand.gates.as_list(), gates_ok=cand.gates.ok, score=cand.gates.score,
         fit=cand.fit.as_dict() if cand.fit else None, sharpen=cand.sharpen, images=files)
    best = state.get("best")
    if best is None or (cand.gates.ok, -cand.gates.score) > (best.gates.ok, -best.gates.score):
        best = cand
    return {"best": best, "last": cand} | ({} if cand.gates.ok else {"feedback": _gate_notes(cand.gates)})


def _gate_notes(checks: gates.Gates) -> str:
    notes = []
    for name in checks.failed:
        key = name.rsplit("_", 1)[0] if name not in GATE_NOTES else name
        note = GATE_NOTES.get(key)
        if note and note not in notes:
            notes.append(note)
    return " ".join(notes)


def review(state: Run) -> dict:
    """Judge the window's newest candidate if the gates let it through, then
    settle the window or ask for another attempt."""
    p, index, number = state["plan"], state["window"], state["attempt"]
    win = p.windows[index]
    cand = state.get("last")
    settled, update = False, {}
    if cand is not None and cand.attempt == number and cand.gates.ok:
        if state["options"].get("judge", True):
            try:
                before = repair.crop(state["work"], win)
                verdict = judge.inspect(before, cand.after, win.mask, win.axes, state.get("brief") or {})
                cand.verdict = verdict.model_dump() | {"accept": verdict.accept}
                emit("verdict", window=index, attempt=number, verdict=cand.verdict, accepted=verdict.accept)
                if not verdict.accept:
                    update["feedback"] = verdict.fix or "Make the strip indistinguishable from the artwork around it."
            except BudgetExceeded as e:
                cand.verdict = {"accept": True, "skipped": str(e)}
                emit("warning", window=index, message=f"judge skipped, gates decide: {e}")
            except Exception as e:
                cand.verdict = {"accept": True, "skipped": f"{type(e).__name__}"}
                emit("warning", window=index, message=f"judge unavailable, gates decide: {e}")
        else:
            cand.verdict = {"accept": True, "skipped": "judge off"}
        settled = cand.accepted
    stopped = state.get("reason") in ("budget", "no painter available")
    out_of_tries = number >= state["options"].get("attempts", config.ATTEMPTS) or stopped
    if not settled and not out_of_tries:
        return update | {"attempt": number + 1, "last": None}
    # the window is settled: keep an accepted candidate, otherwise leave the original
    work = state["work"]
    record = {"round": state["round"], "window": index, "box": win.box, "attempts": number}
    if settled:
        repair.apply(work, win, cand.after)
        record |= {"accepted": True, "painter": cand.painter, "score": cand.gates.score}
    else:
        best = state.get("best")
        record |= {"accepted": False, "best_painter": best.painter if best else None}
    emit("window", **record)
    windows = state["windows"] + [record | ({"best": state.get("best")} if not settled else {})]
    return {"work": work, "windows": windows, "window": index + 1, "attempt": 1, "feedback": "",
            "best": None, "last": None}


def after_review(state: Run) -> str:
    more = state["window"] < len(state["plan"].windows)
    return "paint" if more and state.get("reason") != "budget" else "assemble"


def after_plan(state: Run) -> str:
    return "paint" if state["plan"].windows else "assemble"


def assemble(state: Run) -> dict:
    emit("stage", stage="assemble", message="checking the whole tile again (step 8)")
    original, work = state["original"], state["work"]
    before = checker.check(original)
    checks, after = gates.tile(original, work, state["claimed"], before)
    assert checks.value("untouched") == 0, "pixels outside the bands changed"
    accepted = [w for w in state["windows"] if w.get("accepted") and w["round"] == state["round"]]
    images = _preview(state, f"round{state['round']}", Image.fromarray(work), after)
    emit("assemble", round=state["round"], gates=checks.as_list(), check=after.summary(), images=images,
         accepted=len(accepted))
    if not _failing(state, after):
        return {"check": after, "status": "fixed"}
    if accepted and state["round"] < config.ROUNDS and state.get("reason") != "budget":
        return {"check": after, "round": state["round"] + 1}
    return {"check": after, "status": "review",
            "reason": state.get("reason") or "no repair passed every check; the original is kept"}


def after_assemble(state: Run) -> str:
    return "finish" if state.get("status") else "plan"


def finish(state: Run) -> dict:
    src, out = Path(state["src"]), Path(state["dir"])
    tile = state["tile"]
    status = state.get("status") or ("cut" if state.get("cut") else "returned")
    files: dict[str, str] = {}
    if status == "returned":
        dest = out / f"{src.stem}_seamless{src.suffix}"
        shutil.copyfile(src, dest)  # byte for byte
        files["design"] = dest.name
    elif status == "cut":
        files["design"] = imaging.save(tile.image, out / f"{src.stem}_repeat").name
        if not state["check"].ok:
            proven = ", ".join(state["cut"]["proven"])
            state["reason"] = (f"exact repeat: every pixel continues across the {proven} join; the seam "
                               "check reads a busy join as a seam there")
    elif status == "fixed":
        changed = (state["original"] != state["work"]).any(axis=2)
        image = imaging.write_back(tile, state["work"], changed)
        files["design"] = imaging.save(image, out / f"{src.stem}_fixed").name
        files |= _preview(state, "fixed", image, checker.check(state["work"]))
    elif status == "review":
        if state.get("cut"):
            files["design"] = imaging.save(tile.image, out / f"{src.stem}_repeat").name
        else:
            dest = out / f"{src.stem}_original{src.suffix}"
            shutil.copyfile(src, dest)
            files["design"] = dest.name
        partial = (state["original"] != state["work"]).any(axis=2)
        if partial.any():
            image = imaging.write_back(tile, state["work"], partial)
            files["partial"] = imaging.save(image, out / f"{src.stem}_partial_for_review").name
    result = {
        "status": status, "reason": state.get("reason", ""), "files": files,
        "input": {"name": src.name, "size": list(imaging.load(src).size)},
        "output_size": list(tile.size), "cut": state.get("cut"),
        "check": state["check"].summary() if state.get("check") else None,
        "windows": [{k: v for k, v in w.items() if k != "best"} for w in state.get("windows", [])],
        "spent_usd": LEDGER.spent, "budget_left_usd": LEDGER.remaining,
    }
    (out / "report.json").write_text(json.dumps(result, indent=1, default=str))
    emit("result", **result)
    return {"status": status, "result": result}


def build():
    g = StateGraph(Run)
    g.add_node("measure", measure)
    g.add_node("repeat", repeat)
    g.add_node("diagnose", diagnose)
    g.add_node("plan", make_plan)
    g.add_node("paint", paint)
    g.add_node("review", review)
    g.add_node("assemble", assemble)
    g.add_node("finish", finish)
    g.add_edge(START, "measure")
    g.add_conditional_edges("measure", after_measure, ["finish", "repeat", "diagnose"])
    g.add_conditional_edges("repeat", after_repeat, ["measure", "diagnose"])
    g.add_conditional_edges("diagnose", after_diagnose, ["finish", "plan"])
    g.add_conditional_edges("plan", after_plan, ["paint", "assemble"])
    g.add_edge("paint", "review")
    g.add_conditional_edges("review", after_review, ["paint", "assemble"])
    g.add_conditional_edges("assemble", after_assemble, ["finish", "plan"])
    g.add_edge("finish", END)
    return g.compile()


GRAPH = build()


def run(
    src: Path, options: dict | None = None, run_id: str | None = None, on_event=None,
    root: Path | None = None,
) -> dict:
    """Run the whole workflow on one image. Every event goes to `on_event` and to
    events.jsonl in the run directory, under `root` or config.RUNS."""
    run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{Path(src).stem}-{uuid.uuid4().hex[:4]}"
    folder = Path(root or config.RUNS) / run_id
    folder.mkdir(parents=True, exist_ok=True)
    log = (folder / "events.jsonl").open("a")
    state: Run = {"run_id": run_id, "src": str(src), "dir": str(folder), "options": options or {}}
    start = {"type": "run", "run_id": run_id, "src": Path(src).name, "options": options or {},
             "budget": {"spent": LEDGER.spent, "cap": LEDGER.cap}}
    final: dict = {}
    try:
        for event in [start]:
            log.write(json.dumps(event) + "\n")
            if on_event:
                on_event(event)
        for part in GRAPH.stream(state, stream_mode=["custom", "values"], version="v2",
                                 config={"recursion_limit": 400}):
            if part["type"] == "custom":
                event = part["data"]
                log.write(json.dumps(event, default=str) + "\n")
                log.flush()
                if on_event:
                    on_event(event)
            else:
                final = part["data"]
    except Exception as e:
        event = {"type": "error", "message": f"{type(e).__name__}: {e}"}
        log.write(json.dumps(event) + "\n")
        if on_event:
            on_event(event)
        raise
    finally:
        log.write(json.dumps({"type": "end"}) + "\n")
        log.close()
    return final.get("result", {}) | {"run_id": run_id, "dir": str(folder)}
