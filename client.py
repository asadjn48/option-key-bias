"""Rate-limited, cached, retrying wrapper around client.decisions.create."""
import json
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from .budget import Budget, BudgetExceeded
from .cache import Cache, request_key
from .env import load_secrets

MODEL = "gemini-decision-1"
SAFETY_CODES = {"safety", "prohibited_content", "blocklist", "image_safety", "content_blocked"}
TIERS = {"free": (200, 50_000), "tier1": (1_500, 1_000_000)}  # (RPM, TPM)


class RateLimiter:
    """Sliding 60 s window over requests and tokens. Token cost is estimated before the call
    and corrected with the measured value after it."""

    def __init__(self, rpm: int, tpm: int, safety: float = 0.85):
        self.rpm, self.tpm = int(rpm * safety), int(tpm * safety)
        self.events = deque()  # [timestamp, tokens]
        self.lock = threading.Lock()
        self.ratio = 1.0  # measured / estimated tokens, running

    def _trim(self, now):
        while self.events and now - self.events[0][0] > 60:
            self.events.popleft()

    def acquire(self, est_tokens: int):
        est = int(est_tokens * self.ratio)
        while True:
            with self.lock:
                now = time.time()
                self._trim(now)
                used = sum(e[1] for e in self.events)
                if len(self.events) < self.rpm and used + est <= self.tpm:
                    ev = [now, est]
                    self.events.append(ev)
                    return ev
                wait = 60 - (now - self.events[0][0]) + 0.05 if self.events else 0.5
            time.sleep(min(max(wait, 0.05), 5))

    def settle(self, ev, actual_tokens: int, est_tokens: int):
        with self.lock:
            ev[1] = actual_tokens
            if est_tokens > 0:
                self.ratio = 0.9 * self.ratio + 0.1 * (actual_tokens / est_tokens)


def estimate_tokens(input_, questions) -> int:
    chars = len(json.dumps(input_, ensure_ascii=False)) + len(json.dumps(questions, ensure_ascii=False))
    return int(chars / 3.5) + 120


def _error_code(e):
    body = getattr(e, "body", None)
    if isinstance(body, str):
        try:
            body = json.loads(body.replace("'", '"'))
        except Exception:
            body = None
    if isinstance(body, dict):
        return (body.get("error") or {}).get("code"), (body.get("error") or {}).get("message")
    msg = str(e)
    for c in SAFETY_CODES:
        if f"'{c}'" in msg or f'"{c}"' in msg:
            return c, msg
    return None, msg


class DecisionClient:
    def __init__(self, experiment: str, tier: str = "free", budget: Budget | None = None,
                 safety: float = 0.85):
        load_secrets()
        from google import genai
        self.sdk = genai.Client(http_options={"api_version": "v1beta"})
        self.cache = Cache(experiment)
        self.experiment = experiment
        self.limiter = RateLimiter(*TIERS[tier], safety=safety)
        self.budget = budget or Budget()
        self.tier = tier

    def decide(self, item_id, input_, questions: dict, tag: str = "", meta: dict | None = None) -> dict:
        key = request_key(MODEL, input_, questions, tag)
        hit = self.cache.get(key)
        if hit is not None:
            return {**hit, "cached": True}
        est = estimate_tokens(input_, questions)
        base = {"key": key, "experiment": self.experiment, "item_id": item_id, "model": MODEL,
                "tag": tag, "meta": meta or {}}
        for attempt in range(7):  # 1 try + 6 retries
            self.budget.check()
            ev = self.limiter.acquire(est)
            t0 = time.perf_counter()
            try:
                r = self.sdk.decisions.create(model=MODEL, input=input_, questions=questions)
                latency = time.perf_counter() - t0
                d = r.model_dump(mode="json")
                tokens = d["usage"]["total_input_tokens"]
                self.limiter.settle(ev, tokens, est)
                self.budget.record(tokens)
                rec = {**base, "status": "ok", "answers": d["answers"], "usage": d["usage"],
                       "latency_s": latency, "attempts": attempt + 1,
                       "time_utc": datetime.now(timezone.utc).isoformat()}
                self.cache.put(rec)
                return rec
            except BudgetExceeded:
                raise
            except Exception as e:
                latency = time.perf_counter() - t0
                status = getattr(e, "status_code", None)
                try:
                    status = int(status)
                except (TypeError, ValueError):
                    status = None
                code, msg = _error_code(e)
                if status == 400 and code in SAFETY_CODES:
                    self.limiter.settle(ev, 0, est)
                    rec = {**base, "status": "blocked", "error_code": code, "error": msg,
                           "latency_s": latency, "time_utc": datetime.now(timezone.utc).isoformat()}
                    self.cache.put(rec)
                    return rec
                retryable = status in (429, 500, 502, 503, 504) or status is None
                if not retryable or attempt == 6:
                    rec = {**base, "status": "error", "http_status": status, "error_code": code,
                           "error": str(msg)[:1000], "time_utc": datetime.now(timezone.utc).isoformat()}
                    self.cache.put(rec)
                    return rec
                time.sleep(min(2 ** attempt, 60) * (0.5 + random.random()))

    def run(self, jobs, workers: int = 8, progress_every: int = 100):
        """jobs: iterable of dicts with item_id, input, questions, optional tag/meta."""
        jobs = list(jobs)
        out, n_err, n_blk, t0 = [], 0, 0, time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(self.decide, j["item_id"], j["input"], j["questions"],
                              j.get("tag", ""), j.get("meta")) for j in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    r = f.result()
                except BudgetExceeded as e:
                    print(f"[ABORT] {e}")
                    for g in futs:
                        g.cancel()
                    break
                out.append(r)
                n_err += r["status"] == "error"
                n_blk += r["status"] == "blocked"
                if i % progress_every == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} done, errors={n_err}, blocked={n_blk}, "
                          f"{time.time() - t0:.0f}s elapsed", flush=True)
        return out
