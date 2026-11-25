"""Capability probe for NVIDIA API-catalog models: does the endpoint return token logprobs for a one-letter answer?

python baselines/nvidia_probe.py      (run from paper_2/; one tiny request per model)
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.env import load_secrets  # noqa: E402

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODELS = ["moonshotai/kimi-k3.1-pro", "mistralai/mistral-large-4-1t-a52b", "stepfun-ai/step-5-preview",
          "google/gemma-4-31b-it", "google/gemma-3-12b-it", "nvidia/nemotron-3-ultra-550b-a55b", "z-ai/glm-5.3-flash"]
PROMPT = ("Which topic category does this news article belong to?\n\nNews article:\n\"\"\"\nLakers beat Celtics 102-99 "
          "in overtime.\n\"\"\"\n\nOptions:\nA. Business: Business news.\nB. Sci/Tech: Science and technology news.\n"
          "C. Sports: Sports news.\nD. World: International news.\n\nAnswer with the letter of the single best option and "
          "nothing else.")


def call(model, extra):
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 8, "temperature": 0,
            "logprobs": True, "top_logprobs": 5, **extra}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}", "Content-Type": "application/json",
        "Accept": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}
    ch = d["choices"][0]
    lp = (ch.get("logprobs") or {}).get("content") or []
    top = [(t["token"], round(t["logprob"], 3)) for t in (lp[0].get("top_logprobs") or [])] if lp else None
    return {"text": (ch["message"].get("content") or "")[:40],
            "reasoning_chars": len(ch["message"].get("reasoning_content") or ch["message"].get("reasoning") or ""),
            "first_token_top_logprobs": top, "usage": d.get("usage"), "latency_s": round(time.perf_counter() - t0, 2)}


if __name__ == "__main__":
    load_secrets()
    sys.stdout.reconfigure(encoding="utf-8")
    for m in MODELS:
        print(m, json.dumps(call(m, {}), ensure_ascii=False))
        time.sleep(2)
