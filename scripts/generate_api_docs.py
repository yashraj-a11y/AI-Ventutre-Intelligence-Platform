"""Day 20 · Phase D — generate docs/API_DOCUMENTATION.md from the LIVE app.

Boots the real FastAPI app (TestClient), reads its OpenAPI schema, calls every endpoint with a sample
campaign, and captures the error cases it can trigger. Examples are real responses, abbreviated
(long lists / strings / deep nesting are cut — never edited). Needs the trained artifacts (it loads the
full warm stack), so it is a local tool, not a CI step.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/generate_api_docs.py

Writes: docs/API_DOCUMENTATION.md and docs/assets/openapi.json
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402

CAMPAIGN = {
    "name": "EcoTrack smart garden", "main_category": "Technology", "category": "Gadgets",
    "currency": "USD", "country": "US", "usd_goal_real": 25000,
    "launched": "2024-03-04 10:00:00", "deadline": "2024-04-03 10:00:00",
}

# ----------------------------------------------------------------------------- endpoint metadata
# Hand-written: purpose, UI consumer, notes. Everything else (schemas, examples) is captured live.
ENDPOINTS: list[dict[str, Any]] = [
    {"method": "POST", "path": "/predict", "tag": "Prediction",
     "purpose": "Calibrated funding probability for one campaign, the predicted label, and the decision "
                "threshold the label was derived with.",
     "notes": ["`predicted_label` is `successful` when `funding_probability >= decision_threshold`.",
               "The probability is calibrated (sigmoid), so it can be read as a probability, not just a rank."],
     "ui": "Prediction panel (Founder portal)", "extra_errors": []},
    {"method": "POST", "path": "/explain", "tag": "Explainability",
     "purpose": "SHAP local explanation: the real features pushing the probability up and down.",
     "notes": ["Contributions are in **log-odds margin** space, not probability points.",
               "`base_value + sum(all contributions) = margin`. Only the top drivers are listed; the "
               "remainder is not returned, so `margin - base_value - sum(listed)` is the unlisted rest.",
               "Factors are associations the model learned, not causal levers."],
     "ui": "Explanation panel: contribution bar chart and waterfall", "extra_errors": []},
    {"method": "POST", "path": "/retrieve", "tag": "Retrieval",
     "purpose": "Semantically similar historical campaigns (SBERT + FAISS), split by their real "
                "funded / failed outcome, each with a reason for the match.",
     "notes": ["Query parameter `k` (1–25, default 5) sets how many neighbours are returned.",
               "Searches the **train split only**; a campaign's own outcome is never in the index.",
               "The first call after startup is slow (model warm-up)."],
     "ui": "Retrieval dashboard (similar successful / failed tables)",
     "extra_errors": [("503", "`artifact_unavailable`", "retrieval disabled in config (`load_retriever: false`)")]},
    {"method": "POST", "path": "/counterfactual", "tag": "Counterfactuals",
     "purpose": "Single-feature changes ranked by how much they would move the probability.",
     "notes": ["Describes the *model*, not the world: a counterfactual shows what the model would predict, "
               "not what would happen to a real campaign."],
     "ui": "Scenario simulator (counterfactual list)", "extra_errors": []},
    {"method": "POST", "path": "/scenario", "tag": "Scenarios",
     "purpose": "Ranked what-if scenarios over the real levers (goal, duration, timing, title) with the "
                "marginal value of each change.",
     "notes": ["Levers are the features that exist in the data. Revenue, retention, team and burn do not "
               "exist in the dataset and are not scenario inputs."],
     "ui": "Scenario simulator (suggested scenarios)", "extra_errors": []},
    {"method": "POST", "path": "/recommend", "tag": "Recommendation",
     "purpose": "Investment-readiness index, prioritized improvements, founder and investor strategy and a "
                "confidence score, aggregated from the layers below it.",
     "notes": ["Response includes a `deferred` block naming what could not be computed (and why)."],
     "ui": "Recommendation panel (Founder portal)", "extra_errors": []},
    {"method": "POST", "path": "/analyze", "tag": "Analysis",
     "purpose": "The full analysis: committee verdict, overall confidence, all four reports "
                "(founder, investor, memo, executive) and the copilot brief, from one shared bundle.",
     "notes": ["The slowest endpoint — it runs the whole stack once.",
               "Investor objections are inside the reports / copilot output; there is no separate "
               "objections endpoint.",
               "The committee verdict is one of: Strong Invest, Proceed to Due Diligence, Monitor Progress, "
               "Do Not Invest."],
     "ui": "Investor portal (committee verdict)", "extra_errors": []},
    {"method": "POST", "path": "/report/founder", "tag": "Reports",
     "purpose": "Founder Intelligence Report as ordered sections.",
     "notes": ["Each section has an `id`, `title`, `kind` (text / list / metrics / table), a `body` whose "
               "shape depends on `kind`, and a `source` naming the layer that produced it.",
               "`deferred_appendix` lists what could not be computed. There is no PDF endpoint — the "
               "frontend uses the browser's print-to-PDF."],
     "ui": "Report viewer (founder variant)", "extra_errors": []},
    {"method": "POST", "path": "/report/investor", "tag": "Reports",
     "purpose": "Investor Due Diligence Report as ordered sections.",
     "notes": ["Same envelope as `/report/founder`."],
     "ui": "Report viewer (investor variant)", "extra_errors": []},
    {"method": "GET", "path": "/health", "tag": "System",
     "purpose": "Diagnostics: load status, which components are loaded, and host / process resource usage.",
     "notes": ["Use `/live` and `/ready` for orchestrator probes."], "ui": "—", "extra_errors": []},
    {"method": "GET", "path": "/live", "tag": "System",
     "purpose": "Liveness probe: the process is up. Always 200 and independent of model loading.",
     "notes": [], "ui": "—", "extra_errors": []},
    {"method": "GET", "path": "/ready", "tag": "System",
     "purpose": "Readiness probe: 200 once the warm stack is loaded, otherwise 503 with `loading`.",
     "notes": ["Route traffic only once this returns 200."], "ui": "—", "extra_errors": []},
]
BODY_ENDPOINTS = [e for e in ENDPOINTS if e["method"] == "POST"]


# ----------------------------------------------------------------------------- helpers
def ref_name(schema: dict) -> str | None:
    ref = schema.get("$ref") or ""
    return ref.rsplit("/", 1)[-1] if ref else None


def type_of(prop: dict, comps: dict) -> str:
    if "$ref" in prop:
        return ref_name(prop) or "object"
    if "anyOf" in prop:
        parts = [type_of(p, comps) for p in prop["anyOf"]]
        return " \\| ".join(p for p in parts if p != "null") + (" (optional)" if "null" in parts else "")
    t = prop.get("type", "any")
    if t == "array":
        return f"array of {type_of(prop.get('items', {}), comps)}"
    if t == "object" and "additionalProperties" in prop:
        return "object"
    return t


def trim(obj: Any, depth: int, max_list: int = 2, max_str: int = 140) -> Any:
    """Abbreviate a real response for display: cut long lists / strings and deep nesting."""
    if isinstance(obj, dict):
        if depth <= 0:
            return f"{{… {len(obj)} keys}}"
        return {k: trim(v, depth - 1, max_list, max_str) for k, v in obj.items()}
    if isinstance(obj, list):
        if depth <= 0:
            return f"[… {len(obj)} items]"
        out = [trim(v, depth - 1, max_list, max_str) for v in obj[:max_list]]
        if len(obj) > max_list:
            out.append(f"… +{len(obj) - max_list} more")
        return out
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str].rstrip() + "…"
    return obj


def abbreviated(obj: Any, max_lines: int = 70) -> tuple[str, bool]:
    """Largest depth whose pretty-printed form fits in max_lines. Returns (text, was_abbreviated)."""
    full = json.dumps(obj, indent=2, ensure_ascii=False)
    if full.count("\n") + 1 <= max_lines and trim(obj, 99, 99, 10**6) == obj:
        return full, False
    for depth in (6, 5, 4, 3, 2, 1):
        text = json.dumps(trim(obj, depth), indent=2, ensure_ascii=False)
        if text.count("\n") + 1 <= max_lines:
            return text, True
    return json.dumps(trim(obj, 1), indent=2, ensure_ascii=False), True


def key_tree(obj: Any, depth: int = 3, indent: int = 0) -> list[str]:
    """Observed key structure with value types, from a real response."""
    lines: list[str] = []
    pad = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            kind = ("object" if isinstance(v, dict) else
                    f"array[{len(v)}]" if isinstance(v, list) else type(v).__name__)
            lines.append(f"{pad}{k}: {kind}")
            if depth > 1 and isinstance(v, dict):
                lines += key_tree(v, depth - 1, indent + 1)
            elif depth > 1 and isinstance(v, list) and v and isinstance(v[0], dict):
                lines += key_tree(v[0], depth - 1, indent + 1)
    return lines


def curl_for(ep: dict[str, Any]) -> str:
    url = "http://localhost:8000" + ep["path"] + ("?k=3" if ep["path"] == "/retrieve" else "")
    if ep["method"] == "GET":
        return f"curl -s {url}"
    return f"curl -s -X POST '{url}' \\\n  -H 'Content-Type: application/json' \\\n  -d @campaign.json"


# ----------------------------------------------------------------------------- capture
def capture() -> dict[str, Any]:
    cap: dict[str, Any] = {"endpoints": {}, "errors": {}}

    # 503 first, while no registry is loaded (TestClient without `with` does not run the lifespan).
    cold = TestClient(app, raise_server_exceptions=False)
    r = cold.post("/predict", json=CAMPAIGN)
    cap["errors"]["not_ready_predict"] = (r.status_code, r.json())
    r = cold.get("/ready")
    cap["errors"]["not_ready_ready"] = (r.status_code, r.json())

    with TestClient(app, raise_server_exceptions=False) as c:
        cap["openapi"] = app.openapi()
        for ep in ENDPOINTS:
            url = ep["path"] + ("?k=3" if ep["path"] == "/retrieve" else "")
            t0 = time.time()
            r = c.get(url) if ep["method"] == "GET" else c.post(url, json=CAMPAIGN)
            cap["endpoints"][ep["path"]] = {"status": r.status_code, "json": r.json(),
                                             "seconds": round(time.time() - t0, 2),
                                             "request_id": r.headers.get("X-Request-ID")}
            print(f"  {ep['method']:4} {ep['path']:18} {r.status_code}  {time.time() - t0:5.1f}s", flush=True)

        def post(path: str, body: dict) -> tuple[int, Any]:
            r = c.post(path, json=body)
            return r.status_code, r.json()

        cap["errors"]["unknown_field"] = post("/predict", {**CAMPAIGN, "founder_experience_years": 7})
        cap["errors"]["bad_dates"] = post("/predict", {**CAMPAIGN, "deadline": "2024-03-01 10:00:00"})
        cap["errors"]["bad_goal"] = post("/predict", {**CAMPAIGN, "usd_goal_real": 0})
        cap["errors"]["missing_field"] = post("/predict", {k: v for k, v in CAMPAIGN.items() if k != "name"})
        r = c.post("/retrieve?k=0", json=CAMPAIGN)
        cap["errors"]["bad_k"] = (r.status_code, r.json())
        r = c.get("/nope")
        cap["errors"]["unknown_route"] = (r.status_code, r.json())
        r = c.get("/predict")
        cap["errors"]["wrong_method"] = (r.status_code, r.json())
    return cap


# ----------------------------------------------------------------------------- render
def render(cap: dict[str, Any]) -> str:
    oa = cap["openapi"]
    comps = oa["components"]["schemas"]
    out: list[str] = []
    w = out.append

    w("# API Documentation\n")
    w("*Generated by `scripts/generate_api_docs.py` from the **live** FastAPI app: schemas come from its OpenAPI "
      "document, examples are real responses (abbreviated where long — nothing is edited). Regenerate after "
      "any API change. Machine-readable spec: [`assets/openapi.json`](assets/openapi.json); interactive docs "
      "at `/docs` on a running server.*\n")

    w("## 1. Overview\n")
    w("| | |\n|---|---|")
    w(f"| Title / version | {oa['info']['title']} · {oa['info']['version']} |")
    w("| Base URL | `http://localhost:8000` (the `PORT` env var changes it) |")
    w("| Format | JSON in, JSON out (`Content-Type: application/json`) |")
    w("| Authentication | **None.** The service is deterministic and holds no secrets; put it behind your own gateway if exposed |")
    w("| CORS | Deny by default. Set `CORS_ORIGINS` to an explicit comma-separated allow-list (e.g. `http://localhost:5173`) |")
    w("| Tracing | Every response carries an `X-Request-ID` header; the same id appears in the server log |")
    w("| Rate limiting | None implemented |")
    w("| Training data | Kickstarter campaigns launched 2009–2017; read predictions in that light (see [Performance Report §12](PERFORMANCE_REPORT.md#12-weaknesses-and-limits)) |\n")
    w("**The honesty contract.** Requests accept only real Kickstarter campaign fields; anything else is "
      "rejected with `422`. Responses carry an `honesty_note` and/or a `deferred` block saying what could not "
      "be computed, and why. The data has no founder, team, revenue or market signals, so scores that depend on "
      "them (Business-Model, Founder-Strength, Scalability, Market-Fit, Growth-Potential) are returned "
      "as deferred, never invented; market size (TAM/SAM/CAGR) is deferred too.\n")

    # ------------------------------------------------------------------ request schema
    w("## 2. Request schema — `CampaignRequest`\n")
    w("Used as the body of every `POST` endpoint.\n")
    cr = comps["CampaignRequest"]
    req = set(cr.get("required", []))
    w("| Field | Type | Required | Constraints / default | Description |\n|---|---|---|---|---|")
    for name, p in cr["properties"].items():
        cons = []
        if "minLength" in p: cons.append(f"min length {p['minLength']}")
        if "exclusiveMinimum" in p: cons.append(f"> {p['exclusiveMinimum']}")
        if "default" in p: cons.append(f"default `{p['default']}`")
        for alt in p.get("anyOf", []):
            if "exclusiveMinimum" in alt: cons.append(f"> {alt['exclusiveMinimum']}")
        w(f"| `{name}` | {type_of(p, comps)} | {'yes' if name in req else 'no'} | {'; '.join(cons) or '—'} | {p.get('description', '')} |")
    w("")
    w("**Rules** (enforced server-side)")
    w("- `extra=\"forbid\"`: any field not listed above → `422`. Examples that are rejected: founder profile, "
      "revenue / MRR, team size, burn rate, pitch-deck metadata.")
    w("- `launched` and `deadline` must be parseable datetimes, and `deadline` must be after `launched`.")
    w("- `category`, `goal` and `name_clean` are filled in server-side when omitted.")
    w("- Outcome labels (`state`, `target`) are not request fields — that is what is being predicted.\n")
    w("**Example body** — save as `campaign.json`; every `curl` example below sends it with `-d @campaign.json`\n")
    w("```json\n" + json.dumps(CAMPAIGN, indent=2) + "\n```\n")

    # ------------------------------------------------------------------ errors
    w("## 3. Errors and status codes\n")
    w("| Status | When | Body |\n|---|---|---|")
    w("| `200` | success | endpoint-specific, below |")
    w("| `422` | request fails validation (unknown field, missing field, bad value, bad dates, `k` out of range) | FastAPI validation error: `{\"detail\": [ … ]}` |")
    w("| `503` | warm stack not loaded yet (`/ready` is not 200), or a needed artifact is unavailable | `{\"detail\": …}` while loading; `{\"error\": \"artifact_unavailable\", \"detail\": …, \"request_id\": …}` for a missing artifact |")
    w("| `500` | a pipeline failed on a valid request, or an unexpected error | `{\"error\": \"inference_error\", \"detail\": …, \"request_id\": …}`; the traceback is logged server-side only |")
    w("| `404` / `405` | unknown route / wrong method | FastAPI default |\n")
    w("Real captured errors:\n")
    errs = cap["errors"]
    labels = [
        ("unknown_field", "Unknown field (`founder_experience_years`)"),
        ("missing_field", "Missing required field (`name`)"),
        ("bad_goal", "`usd_goal_real` = 0"),
        ("bad_dates", "`deadline` before `launched`"),
        ("bad_k", "`/retrieve?k=0`"),
        ("not_ready_predict", "`/predict` before the stack is loaded"),
        ("not_ready_ready", "`/ready` before the stack is loaded"),
        ("unknown_route", "Unknown route"),
        ("wrong_method", "`GET` on a `POST` route"),
    ]
    w("*For `422` responses only `type`, `loc` and `msg` of each error are shown; the real error objects also "
      "echo `input` and `ctx`.*\n")
    for key, label in labels:
        code, body = errs[key]
        if code == 422 and isinstance(body.get("detail"), list):
            lines = [json.dumps({k: e[k] for k in ("type", "loc", "msg") if k in e}, ensure_ascii=False)
                     for e in body["detail"]]
            txt = "\n".join(lines)
            w(f"**{label} → `{code}`**\n\n```json\n{txt}\n```\n")
        else:
            txt, _ = abbreviated(body, 24)
            w(f"**{label} → `{code}`**\n\n```json\n{txt}\n```\n")

    # ------------------------------------------------------------------ endpoints
    w("## 4. Endpoints\n")
    w("| Method | Path | Purpose |\n|---|---|---|")
    for ep in ENDPOINTS:
        w(f"| {ep['method']} | [`{ep['path']}`](#{ep['method'].lower()}-{ep['path'].strip('/').replace('/', '')}) | {ep['purpose'].split('.')[0]}. |")
    w("")

    for ep in ENDPOINTS:
        path = ep["path"]
        data = cap["endpoints"][path]
        op = oa["paths"][path][ep["method"].lower()]
        w(f"### {ep['method']} {path}\n")
        w(ep["purpose"] + "\n")
        if ep["method"] == "POST":
            w("**Request body:** [`CampaignRequest`](#2-request-schema--campaignrequest)" +
              (" · **Query:** `k` — integer 1–25, default 5" if path == "/retrieve" else "") + "\n")
        sch = op["responses"]["200"]["content"]["application/json"]["schema"]
        rname = ref_name(sch)
        if rname and rname in comps:
            req_f = set(comps[rname].get("required", []))
            w(f"**Response `200`:** `{rname}`\n")
            w("| Field | Type | Always present |\n|---|---|---|")
            for fname, p in comps[rname]["properties"].items():
                w(f"| `{fname}` | {type_of(p, comps)} | {'yes' if fname in req_f else 'no'} |")
            w("")
        else:
            w("**Response `200`:** free-form JSON object.\n")
        if ep["notes"]:
            for n in ep["notes"]:
                w(f"- {n}")
            w("")
        w("**Example request**\n")
        w("```bash\n" + curl_for(ep) + "\n```\n")
        body = data["json"]
        w(f"**Example response** (`{data['status']}`)\n")
        text, cut = abbreviated(body)
        w("```json\n" + text + "\n```")
        if cut:
            w("\n*Abbreviated: long lists, long strings and deep nesting are shortened. Observed structure:*\n")
            w("```text\n" + "\n".join(key_tree(body)) + "\n```")
        w("")
        codes = ["`200` success"]
        if ep["method"] == "POST":
            codes.append("`422` request fails validation")
            codes.append("`503` warm stack not ready")
        for status, name, why in ep["extra_errors"]:
            codes.append(f"`{status}` {name} — {why}")
        if ep["path"] not in ("/live", "/health"):
            if ep["path"] == "/ready":
                codes = ["`200` ready", "`503` still loading"]
            else:
                codes.append("`500` pipeline failure (`inference_error`)")
        w("**Status codes:** " + " · ".join(codes) + "\n")
        w(f"**Used by the UI in:** {ep['ui']}\n")

    w("## 5. Regenerating this document\n")
    w("```bash\nOMP_NUM_THREADS=1 .venv/bin/python scripts/generate_api_docs.py\n```\n")
    w("It loads the full model stack, so it needs the trained artifacts and takes about a minute. It rewrites "
      "this file and `docs/assets/openapi.json`.\n")
    return "\n".join(out)


def main() -> None:
    print("Booting the app and calling every endpoint…", flush=True)
    cap = capture()
    (ROOT / "docs" / "assets").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "assets" / "openapi.json").write_text(json.dumps(cap["openapi"], indent=2) + "\n")
    (ROOT / "docs" / "API_DOCUMENTATION.md").write_text(render(cap))
    print("Wrote docs/API_DOCUMENTATION.md and docs/assets/openapi.json")


if __name__ == "__main__":
    main()
