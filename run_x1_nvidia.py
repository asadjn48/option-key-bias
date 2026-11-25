"""X1 cross-model validation on NVIDIA API-catalog models (free trial credits; no paid spend).

Each condition variant from tasks.build() becomes one chat prompt. Options are listed in the SAME order the
Decisions API uses (option keys sorted in UTF-8 byte order) and labelled A, B, C, ...; the line shows the key and,
when present, its description. The model answers with one letter; we read the first output token's top
logprobs and renormalise over the option letters, giving a probability per option (so conditions A-F, calibration
and positional analyses carry over). Reasoning is switched off where the endpoint allows it.

python baselines/run_x1_nvidia.py --smoke --model nemotron       # 20 items per dataset, condition A only
python baselines/run_x1_nvidia.py --model nemotron --conditions A,B,C,D,E,F --items 250 100   (after go-ahead)
"""
import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.cache import Cache, request_key  # noqa: E402
from common.client import RateLimiter  # noqa: E402
from common.env import load_secrets  # noqa: E402
from tasks import INSTRUCTIONS, build, load_items, run_order, sort_keys  # noqa: E402

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODELS = {  # short name -> (model id, extra request fields that switch reasoning off)
    "nemotron": ("nvidia/nemotron-3-ultra-550b-a55b", {"reasoning_effort": "none"}),
    "gemma": ("google/gemma-4-31b-it", {}),
}
LETTERS = "ABCDEFGHIJ"
INPUT_NAME = {"ag_news": "News article", "banking77": "Customer message"}


def prompt_for(ds, text, question):
    keys = sort_keys(question["criteria"])
    lines = []
    for i, k in enumerate(keys):
        d = question["criteria"][k]
        lines.append(f"{LETTERS[i]}. {k}" + (f": {d}" if d else ""))
    return (f"{INSTRUCTIONS[ds]}\n\n{INPUT_NAME[ds]}:\n\"\"\"\n{text}\n\"\"\"\n\nOptions:\n" + "\n".join(lines) +
            f"\n\nAnswer with the letter of the single best option ({LETTERS[0]}-{LETTERS[len(keys) - 1]}) and nothing else."), keys


def letter_probs(top, n):
    """top: list of {token, logprob}. Returns (probs over the n letters, coverage = raw mass on letters)."""
    mass = {}
    for t in top:
        tok = t["token"].strip().rstrip(".)").upper()
        if len(tok) == 1 and tok in LETTERS[:n]:
            mass[tok] = mass.get(tok, 0.0) + math.exp(t["logprob"])
    cov = sum(mass.values())
    probs = [mass.get(LETTERS[i], 0.0) / cov if cov else 0.0 for i in range(n)]
    return probs, cov


class NvClient:
    def __init__(self, short, experiment, rpm):
        load_secrets()
        self.model, self.extra = MODELS[short]
        self.cache = Cache(experiment)
        self.limiter = RateLimiter(rpm, 10_000_000, safety=1.0)

    def call(self, job):
        key = request_key(self.model, job["prompt"], {"top": job["top"]}, "x1")
        hit = self.cache.get(key)
        if hit:
            return {**hit, "cached": True}
        body = {"model": self.model, "messages": [{"role": "user", "content": job["prompt"]}], "max_tokens": 2,
                "temperature": 0, "logprobs": True, "top_logprobs": job["top"], **self.extra}
        base = {"key": key, "model": self.model, **job["meta"]}
        for attempt in range(6):
            ev = self.limiter.acquire(1)
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
                    "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}", "Content-Type": "application/json",
                    "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    d = json.load(r)
                lat = time.perf_counter() - t0
                ch = d["choices"][0]
                lp = (ch.get("logprobs") or {}).get("content") or []
                top = lp[0].get("top_logprobs") or [] if lp else []
                probs, cov = letter_probs(top, job["meta"]["n_options"])
                rec = {**base, "status": "ok" if cov > 0 else "no_letter", "text": (ch["message"].get("content") or "")[:10],
                       "probs": probs, "coverage": cov, "top": [(t["token"], t["logprob"]) for t in top],
                       "usage": d.get("usage"), "latency_s": lat, "attempts": attempt + 1,
                       "time_utc": datetime.now(timezone.utc).isoformat()}
                self.cache.put(rec)
                return rec
            except urllib.error.HTTPError as e:
                code, msg = e.code, e.read().decode()[:300]
            except Exception as e:  # noqa: BLE001
                code, msg = None, str(e)[:300]
            if code not in (429, 500, 502, 503, 504, None) or attempt == 5:
                rec = {**base, "status": "error", "http_status": code, "error": msg,
                       "time_utc": datetime.now(timezone.utc).isoformat()}
                self.cache.put(rec)
                return rec
            time.sleep(min(2 ** attempt * 3, 60) * (0.5 + random.random()))


def jobs(ds, n_items, conditions, b_cyclic_only=False):
    qs, cond = build(ds)
    K = len(next(iter(qs.values()))["criteria"])
    cyclic = {tuple((i + r) % K for i in range(K)) for r in range(K)}
    out = []
    for it in run_order(load_items(ds))[:n_items]:
        for q, c in cond.items():
            if c["cond"] not in conditions:
                continue
            if b_cyclic_only and c["cond"] == "B" and tuple(c["perm"]) not in cyclic:
                continue  # keep only the K cyclic rotations (as in the generative baseline)
            p, keys = prompt_for(ds, it["text"], qs[q])
            out.append({"prompt": p, "top": 20,
                        "meta": {"dataset": ds, "item_id": it["item_id"], "label": it["label"], "question": q,
                                 "condition": c["cond"], "keys_sorted": keys, "n_options": len(keys)}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--conditions", default="A")
    ap.add_argument("--items", nargs=2, type=int, default=[20, 20])
    ap.add_argument("--rpm", type=int, default=30)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--b-cyclic-only", action="store_true", help="B: only the K cyclic rotations (AG News: 4 of 24)")
    a = ap.parse_args()
    conds = set(a.conditions.split(","))
    for ds, n in zip(("ag_news", "banking77"), a.items):
        js = jobs(ds, 20 if a.smoke else n, conds, a.b_cyclic_only)
        cl = NvClient(a.model, f"p02_x1_{a.model}_{ds}", a.rpm)
        print(f"[{a.model} / {ds}] {len(js)} calls", flush=True)
        t0 = time.time(); recs = []
        with ThreadPoolExecutor(a.workers) as ex:
            for f in as_completed([ex.submit(cl.call, j) for j in js]):
                recs.append(f.result())
        ok = [r for r in recs if r["status"] == "ok"]
        acc = sum(max(range(len(r["probs"])), key=lambda i: r["probs"][i]) ==
                  next(i for i, k in enumerate(r["keys_sorted"]) if build(ds)[1][r["question"]]["map"][k] == r["label"])
                  for r in ok if r["condition"] == "A") / max(1, sum(r["condition"] == "A" for r in ok))
        lat = sorted(r["latency_s"] for r in ok if not r.get("cached"))
        toks = [r["usage"]["total_tokens"] for r in ok if r.get("usage")]
        print(json.dumps({"dataset": ds, "calls": len(recs), "ok": len(ok),
                          "no_letter": sum(r["status"] == "no_letter" for r in recs),
                          "errors": sum(r["status"] == "error" for r in recs), "acc_condition_A": round(acc, 3),
                          "mean_letter_coverage": round(sum(r["coverage"] for r in ok) / max(1, len(ok)), 4),
                          "min_letter_coverage": round(min((r["coverage"] for r in ok), default=0), 4),
                          "tokens_mean": round(sum(toks) / max(1, len(toks)), 1),
                          "latency_p50": lat[len(lat) // 2] if lat else None, "wall_s": round(time.time() - t0)}),
              flush=True)
        errs = [r for r in recs if r["status"] == "error"]
        if errs:
            print("  first error:", errs[0].get("http_status"), errs[0].get("error", "")[:200])


if __name__ == "__main__":
    main()
