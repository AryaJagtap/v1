"""A hard cap on API spend across every run, kept on disk so that tuning runs
add up instead of starting from zero each time.

A paid call first reserves what it might cost and is refused if that could
cross the cap. Afterwards it settles at the price of the tokens it actually
used, read from the response.
"""

import json
import threading
import time
from pathlib import Path

from seamfix import config


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: Path = config.RUNS / "ledger.json", cap: float = config.BUDGET_USD):
        self.path = Path(path)
        self.cap = cap
        self._lock = threading.Lock()
        self._held = 0.0

    def _read(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"spent": 0.0, "calls": []}

    @property
    def spent(self) -> float:
        return round(self._read()["spent"], 4)

    @property
    def remaining(self) -> float:
        return round(self.cap - self.spent - self._held, 4)

    def reserve(self, estimate: float, what: str) -> float:
        with self._lock:
            if self._read()["spent"] + self._held + estimate > self.cap:
                raise BudgetExceeded(
                    f"{what} needs up to ${estimate:.3f}; ${self.cap - self.spent - self._held:.3f} "
                    f"of the ${self.cap:.2f} cap is left (SEAMFIX_BUDGET_USD)"
                )
            self._held += estimate
            return estimate

    def settle(self, held: float, usd: float, entry: dict) -> None:
        with self._lock:
            data = self._read()
            data["spent"] = round(data["spent"] + usd, 6)
            data["calls"].append({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "usd": round(usd, 5)} | entry)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=1))
            self._held = max(0.0, self._held - held)


def cost(model: str, usage) -> tuple[float, dict]:
    """USD for one response from its usage metadata, at config.PRICES."""
    price_in, price_text, price_image = config.PRICES.get(model, (2.0, 12.0, 120.0))
    prompt = usage.prompt_token_count or 0
    out = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    image = sum(
        d.token_count or 0
        for d in (usage.candidates_tokens_details or [])
        if str(d.modality).upper().endswith("IMAGE")
    )
    usd = (prompt * price_in + (out - image) * price_text + image * price_image) / 1e6
    return usd, {"model": model, "in": prompt, "out": out, "image_out": image}


LEDGER = Ledger()
