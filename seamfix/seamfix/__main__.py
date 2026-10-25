"""python -m seamfix IMAGE [--free] [--no-judge] [--attempts N] [--model ID]"""

import argparse
import sys
from pathlib import Path

from seamfix import config, pipeline


def line(event: dict) -> str | None:
    kind = event["type"]
    if kind == "stage":
        return f"· {event['stage']:<9} {event.get('message', '')}"
    if kind == "check":
        c = event["check"]
        return (f"  {event['which']} joins: left-right {c['lr']['overall']}/{c['lr']['worst']} "
                f"{'ok' if c['lr']['ok'] else 'FAIL'}, top-bottom {c['tb']['overall']}/{c['tb']['worst']} "
                f"{'ok' if c['tb']['ok'] else 'FAIL'}")
    if kind == "repeat":
        return f"  {event['describe']} at {tuple(event['offset'])}" if event["found"] else "  no smaller repeat"
    if kind == "diagnosis":
        return f"  {event['kind']}, {event['difficulty']}{', whole join' if event['whole'] else ''}: {event['brief'].get('style', '')}"
    if kind == "plan":
        return f"  {len(event['windows'])} window(s), band {event['band']}, stretches {event['spans']}"
    if kind == "candidate":
        failed = [g["name"] for g in event["gates"] if not g["ok"]]
        return (f"  {event['painter']}: {event['seconds']}s, gates {'pass' if event['gates_ok'] else 'fail ' + ','.join(failed)}"
                f", score {event['score']}")
    if kind == "verdict":
        v = event["verdict"]
        return (f"  judge: {'accept' if event['accepted'] else 'reject'} "
                f"{v.get('continuity')}/{v.get('fidelity')}/{v.get('finish')} {v.get('fix', '')[:120]}")
    if kind == "window":
        return f"  window {event['window'] + 1}: {'accepted' if event['accepted'] else 'kept original'}"
    if kind == "assemble":
        c = event["check"]
        return f"  tile now: left-right {c['lr']['overall']}, top-bottom {c['tb']['overall']}, ok={c['ok']}"
    if kind in ("warning", "error"):
        return f"  ! {event['message']}"
    if kind == "result":
        return (f"= {event['status']}{': ' + event['reason'] if event['reason'] else ''} -> "
                f"{event['files']}  spent ${event['spent_usd']:.3f}, ${event['budget_left_usd']:.3f} left")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(prog="seamfix", description=__doc__)
    ap.add_argument("image", type=Path)
    ap.add_argument("--free", action="store_true", help="LaMa only, no paid calls at all")
    ap.add_argument("--no-judge", action="store_true", help="gates decide alone")
    ap.add_argument("--attempts", type=int, default=config.ATTEMPTS)
    ap.add_argument("--model", default=config.PAINT_MODEL)
    args = ap.parse_args()
    options = {
        "painters": ["lama"] if args.free else ["lama", "nano"],
        "judge": not (args.no_judge or args.free),
        "attempts": args.attempts,
        "model": args.model,
        "analyst": not args.free,  # nothing paid in free mode, the measurements decide
    }
    result = pipeline.run(args.image, options, on_event=lambda e: (s := line(e)) and print(s, flush=True))
    print(f"  run folder: {result['dir']}")
    return 0 if result.get("status") in ("returned", "cut", "fixed") else 1


if __name__ == "__main__":
    sys.exit(main())
