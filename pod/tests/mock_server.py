import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.bench import LENGTHS, SENTENCES, texts_for


def make_items(count, tag):
    languages = list(SENTENCES)
    table = {language: texts_for(language) for language in languages}
    items = []
    for index in range(count):
        language = languages[index % len(languages)]
        length = LENGTHS[(index // len(languages)) % len(LENGTHS)]
        items.append(
            {
                "id": f"mock-{tag}-{index}",
                "text": f"{table[language][length]} {tag} {index}",
                "language": language,
                "format": "ogg",
            }
        )
    return items


def make_state(count, tag, report_path):
    items = make_items(count, tag)
    return {
        "items": items,
        "languages": {item["id"]: item["language"] for item in items},
        "records": {},
        "progress_log": [],
        "beats": 0,
        "phase": None,
        "complete": None,
        "report_path": report_path,
        "lock": threading.Lock(),
    }


def incremental_stats(state):
    """How results arrived. A server that only hears about clips at the very end has a first
    batch holding nearly everything; a healthy run reports small batches as it goes."""
    log = [entry for entry in state["progress_log"] if entry["done"] > 0]
    total_done = sum(entry["done"] for entry in log)
    if not log:
        return {"progress_calls": 0, "first_done_batch": 0, "done_reported": 0, "spread_seconds": 0.0}
    return {
        "progress_calls": len(log),
        "first_done_batch": log[0]["done"],
        "done_reported": total_done,
        "spread_seconds": round(log[-1]["ts"] - log[0]["ts"], 2),
    }


def summarize(state):
    records = list(state["records"].values())
    done = [r for r in records if r["status"] == "done"]
    failed = [r for r in records if r["status"] != "done"]
    reasons = {}
    for record in failed:
        key = record.get("error") or "unknown"
        reasons[key] = reasons.get(key, 0) + 1
    samples = {}
    by_language = {}
    for record in done:
        language = state["languages"].get(record["id"], "unknown")
        by_language[language] = by_language.get(language, 0) + 1
        samples.setdefault(language, [])
        if len(samples[language]) < 2 and record.get("url"):
            samples[language].append(record["url"])
    return {
        "items_in_manifest": len(state["items"]),
        "done": len(done),
        "failed": len(failed),
        "failure_reasons": reasons,
        "done_by_language": by_language,
        "heartbeats": state["beats"],
        "complete": state["complete"],
        "sample_urls": samples,
        "incremental": incremental_stats(state),
    }


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def send_payload(self, payload, code=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_body(self):
            length = int(self.headers.get("Content-Length", "0"))
            if not length:
                return {}
            return json.loads(self.rfile.read(length))

        def do_GET(self):
            if self.path.startswith("/health"):
                self.send_payload({"ok": True})
                return
            if self.path.startswith("/internal/pod/manifest"):
                self.send_payload({"items": state["items"]})
                return
            self.send_payload({}, 404)

        def do_POST(self):
            data = self.read_body()
            if self.path.startswith("/internal/pod/progress"):
                records = data.get("records", [])
                with state["lock"]:
                    for record in records:
                        state["records"][record["id"]] = record
                    state["progress_log"].append(
                        {
                            "ts": time.time(),
                            "count": len(records),
                            "done": sum(1 for r in records if r.get("status") == "done"),
                        }
                    )
                self.send_payload({})
                return
            if self.path.startswith("/internal/pod/beat"):
                with state["lock"]:
                    state["beats"] += 1
                    state["phase"] = data.get("phase")
                self.send_payload({"stop": False})
                return
            if self.path.startswith("/internal/pod/complete"):
                with state["lock"]:
                    state["complete"] = data
                    report = summarize(state)
                with open(state["report_path"], "w") as handle:
                    json.dump(report, handle, indent=2, ensure_ascii=False)
                sys.stdout.write(json.dumps(report, ensure_ascii=False) + "\n")
                sys.stdout.flush()
                self.send_payload({})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            self.send_payload({}, 404)

    return Handler


def main():
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
    else:
        count = int(os.environ.get("MOCK_COUNT", "200") or 200)
    port = int(os.environ.get("MOCK_PORT", "8000") or 8000)
    tag = os.environ.get("MOCK_TAG", "") or str(int(time.time()))
    report_path = os.environ.get("MOCK_REPORT", "") or "/tmp/mock_report.json"
    state = make_state(count, tag, report_path)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    sys.stdout.write(f"mock server port={port} items={count} tag={tag}\n")
    sys.stdout.flush()
    server.serve_forever()


if __name__ == "__main__":
    main()