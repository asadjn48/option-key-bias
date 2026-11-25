"""B1: generative baseline. A Gemini text model sees the same options as a numbered list, in the
same permutations as condition B (descriptions shown, class names hidden), and answers with a number.

Model: gemini-3.5-flash-lite (listed by models.list on 13 Nov 2026; logprobs are not enabled for it,
so only the predicted option is recorded and calibration is not measured).
Design: the cyclic-rotation subset of condition B (AG News: 4 of the 24 permutations; Banking77:
all 10 rotations), so positions stay balanced and the Decision-model numbers can be recomputed on
exactly the same item x permutation cells.

  python baselines/run_generative.py --smoke      # 20 items per dataset + projection
  python baselines/run_generative.py              # full run (after the user confirms)
"""
import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.budget import Budget, BudgetExceeded  # noqa: E402
from common.cache import Cache, request_key  # noqa: E402
from common.client import RateLimiter  # noqa: E402
from common.env import load_secrets  # noqa: E402
from tasks import descriptions, label_names, load_items, run_order  # noqa: E402

GEN_MODEL = "gemini-3.5-flash-lite"
DAILY_STOP = __import__("threading").Event()
N_ITEMS = {"ag_news": 250, "banking77": 100}
TASK = {"ag_news": "Which topic category does this news article belong to?",
        "banking77": "Which intent best describes this customer's message to their bank?"}
INPUT_NAME = {"ag_news": "News article", "banking77": "Customer message"}


def rotations(name):
    K = len(label_names(name))
    return [tuple((i + r) % K for i in range(K)) for r in range(K)]  # asg[class] = position index


def prompt(name, text, asg):
    desc = descriptions(name)
    order = sorted(range(len(asg)), key=lambda c: asg[c])  # classes by position
    opts = "\n".join(f"{p + 1}. {desc[c]}" for p, c in enumerate(order))
    return (f"{TASK[name]}\n\n{INPUT_NAME[name]}:\n\"\"\"\n{text}\n\"\"\"\n\nOptions:\n{opts}\n\n"
            f"Answer with the number of the single best option (1-{len(asg)}) and nothing else.")


def parse(text, K):
    m = re.search(r"\d+", text or "")
    if not m:
        return None
    v = int(m.group())
    return v if 1 <= v <= K else None


class GenClient:
    def __init__(self, experiment, rpm, tpm, budget):
        load_secrets()
        from google import genai
        from google.genai import types
        self.types = types
        self.sdk = genai.Client()
        self.cache = Cache(experiment)
        self.limiter = RateLimiter(rpm, tpm, safety=1.0)
        self.budget = budget
        self.n429 = 0

    def call(self, job):
        key = request_key(GEN_MODEL, job["prompt"], {}, "gen")
        hit = self.cache.get(key)
        if hit:
            return {**hit, "cached": True}
        base = {"key": key, "model": GEN_MODEL, **job["meta"]}
        if DAILY_STOP.is_set():  # daily quota hit: do not call, do not cache
            return {**base, "status": "skipped"}
        cfg = self.types.GenerateContentConfig(temperature=0, max_output_tokens=8,
                                               thinking_config=self.types.ThinkingConfig(thinking_level="minimal"))
        for attempt in range(7):
            self.budget.check()
            ev = self.limiter.acquire(300)
            t0 = time.perf_counter()
            try:
                r = self.sdk.models.generate_content(model=GEN_MODEL, contents=job["prompt"], config=cfg)
                lat = time.perf_counter() - t0
                u = r.usage_metadata
                self.limiter.settle(ev, u.total_token_count or 0, 300)
                self.budget.record(0)  # count the request; generative token cost is computed separately
                cand = r.candidates[0] if r.candidates else None
                fr = str(getattr(cand, "finish_reason", None)) if cand else str(r.prompt_feedback)
                text = r.text if cand and cand.content and cand.content.parts else ""
                status = "blocked" if ("SAFETY" in fr or "PROHIBITED" in fr or "BLOCK" in fr) else "ok"
                rec = {**base, "status": status, "text": text, "finish_reason": fr,
                       "pred_pos": parse(text, job["meta"]["K"]), "latency_s": lat, "attempts": attempt + 1,
                       "usage": {"prompt": u.prompt_token_count, "output": u.candidates_token_count,
                                 "thoughts": u.thoughts_token_count, "total": u.total_token_count},
                       "time_utc": datetime.now(timezone.utc).isoformat()}
                self.cache.put(rec)
                return rec
            except BudgetExceeded:
                raise
            except Exception as e:
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                self.limiter.settle(ev, 0, 300)
                if code == 429:
                    self.n429 += 1
                    if "per day" in str(e):  # free tier: 500 requests/day for this model (13-15 Nov 2026)
                        if not DAILY_STOP.is_set():
                            print("  [STOP] daily request quota reached; re-run after midnight Pacific", flush=True)
                        DAILY_STOP.set()
                        return {**base, "status": "skipped"}
                if code not in (429, 500, 502, 503, 504) or attempt == 6:
                    rec = {**base, "status": "error", "http_status": code, "error": str(e)[:500],
                           "time_utc": datetime.now(timezone.utc).isoformat()}
                    self.cache.put(rec)
                    return rec
                time.sleep(min(2 ** attempt * 2, 90) * (0.5 + random.random()))


def jobs(name, n):
    K = len(label_names(name))
    out = []
    for it in run_order(load_items(name))[:n]:
        for r, asg in enumerate(rotations(name)):
            out.append({"prompt": prompt(name, it["text"], asg),
                        "meta": {"dataset": name, "item_id": it["item_id"], "label": it["label"], "rot": r,
                                 "perm": list(asg), "K": K, "gold_pos": asg[it["label"]] + 1}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rpm", type=int, default=14)  # free tier: 15 RPM (from the 429 message, 13 Nov 2026)
    ap.add_argument("--tpm", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    budget = Budget()
    for name in ("ag_news", "banking77"):
        n = 20 if a.smoke else N_ITEMS[name]
        js = jobs(name, n)
        cl = GenClient(f"p02_gen_{name}", a.rpm, a.tpm, budget)
        print(f"[{name}] {len(js)} requests ({n} items x {len(rotations(name))} rotations)", flush=True)
        t0 = time.time()
        recs = []
        with ThreadPoolExecutor(a.workers) as ex:
            for i, f in enumerate(as_completed([ex.submit(cl.call, j) for j in js]), 1):
                recs.append(f.result())
                if i % 100 == 0 or i == len(js):
                    print(f"  {i}/{len(js)} {time.time() - t0:.0f}s 429s={cl.n429}", flush=True)
        print(f"  skipped (daily quota): {sum(r['status'] == 'skipped' for r in recs)}", flush=True)
        ok = [r for r in recs if r["status"] == "ok"]
        fresh = sorted(r["latency_s"] for r in ok if not r.get("cached"))
        print(json.dumps({"dataset": name, "n": len(recs), "ok": len(ok),
                          "blocked": sum(r["status"] == "blocked" for r in recs),
                          "errors": sum(r["status"] == "error" for r in recs),
                          "unparsed": sum(r["pred_pos"] is None for r in ok),
                          "acc": sum(r["pred_pos"] == r["gold_pos"] for r in ok) / max(len(ok), 1),
                          "prompt_tok_mean": sum(r["usage"]["prompt"] for r in ok) / max(len(ok), 1),
                          "out_tok_mean": sum((r["usage"]["output"] or 0) + (r["usage"]["thoughts"] or 0)
                                              for r in ok) / max(len(ok), 1),
                          "lat_p50": fresh[len(fresh) // 2] if fresh else None,
                          "n429": cl.n429, "wall_s": round(time.time() - t0)}), flush=True)
        for r in recs:
            if r["status"] == "error":
                print("  ERROR", r.get("http_status"), r["error"][:200])
                break
        if a.smoke:
            full = N_ITEMS[name] * len(rotations(name))
            print(f"  full run would be {full} requests", flush=True)
    print("Budget:", budget.summary())


if __name__ == "__main__":
    main()
