"""Step 3: compile every ONNX part to an fp16 QNN context binary on Qualcomm AI Hub.

For each part: upload -> compile (qnn_dlc, --quantize_full_type float16) -> link (context binary)
-> optional profile on the target device. Parts run in parallel threads. Job ids are kept in a
state file, so re-running resumes instead of resubmitting.

Requires an AI Hub account: `pip install qai-hub && qai-hub configure --api_token <token>`.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import qai_hub as hub

_lock = threading.Lock()


def _call_with_deadline(fn, timeout):
    """Run fn() in a daemon thread and give up after `timeout` seconds.

    qai_hub's S3 transfers (model upload, inference inputs/outputs, profiles) have no overall timeout;
    a proxied connection that trickles data can hang them forever. A stuck daemon thread is simply
    abandoned (it cannot be cancelled) and the call is retried.
    """
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"no result within {timeout}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def retry(fn, what, timeout=None, attempts=200):
    """Retry transient AI Hub errors; `timeout` bounds each attempt (use it for calls that move data)."""
    for attempt in range(1, attempts + 1):
        try:
            return _call_with_deadline(fn, timeout) if timeout else fn()
        except Exception as e:  # noqa: BLE001
            print(f"  api error ({what}): {type(e).__name__}: {str(e)[:120]}; retry {attempt}", flush=True)
            time.sleep(min(10 * attempt, 60))
    raise RuntimeError(f"giving up: {what}")


class State:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_text("{}")

    def get(self, key):
        with _lock:
            return json.loads(self.path.read_text()).get(key)

    def set(self, key, value):
        with _lock:
            s = json.loads(self.path.read_text())
            s[key] = value
            self.path.write_text(json.dumps(s, indent=2))


def wait(job_id, label):
    while True:
        st = retry(lambda: hub.get_job(job_id).get_status(), f"status {job_id}")
        if st.finished:
            print(f"[{label}] {job_id} {st.code} {(st.message or '')[:300]}", flush=True)
            return st.success
        time.sleep(60)


def run_part(part_name, onnx_dir, device, state, profile):
    key = lambda k: f"{part_name}.{k}"  # noqa: E731
    if not state.get(key("model")):
        print(f"[{part_name}] uploading {onnx_dir}", flush=True)
        # bounded + few attempts: a retry re-uploads the whole part (~0.5-1 GB)
        model = retry(lambda: hub.upload_model(str(onnx_dir)), f"upload {part_name}", timeout=4 * 3600, attempts=3)
        state.set(key("model"), model.model_id)
    if not state.get(key("link")):
        model = retry(lambda: hub.get_model(state.get(key("model"))), "get model")
        cjobs, ljob = retry(lambda: hub.submit_compile_and_link_jobs(
            models=model, device=device, name=f"qwen3-reranker-{part_name}-fp16",
            compile_options="--quantize_full_type float16"), "submit compile+link")
        state.set(key("compile"), [j.job_id for j in cjobs])
        state.set(key("link"), ljob.job_id)
    for j in state.get(key("compile")):
        if not wait(j, f"{part_name} compile"):
            return False
    if not wait(state.get(key("link")), f"{part_name} link"):
        return False
    if not state.get(key("context_model")):
        ctx = retry(lambda: hub.get_job(state.get(key("link"))).get_target_model(), "get context model")
        state.set(key("context_model"), ctx.model_id)
    if profile:
        if not state.get(key("profile")):
            job = retry(lambda: hub.submit_profile_job(model=hub.get_model(state.get(key("context_model"))),
                                                       device=device, name=f"qwen3-reranker-{part_name}-profile"),
                        "submit profile")
            state.set(key("profile"), job.job_id)
        if wait(state.get(key("profile")), f"{part_name} profile"):
            p = retry(lambda: hub.get_job(state.get(key("profile"))).download_profile(), "download profile",
                      timeout=600)
            t = sorted(p["execution_summary"]["all_inference_times"])
            units = {}
            for op in p.get("execution_detail", []):
                units[op.get("compute_unit")] = units.get(op.get("compute_unit"), 0) + 1
            print(f"[{part_name}] on-device median {t[len(t) // 2] / 1000:.0f} ms, compute units {units}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx-dir", required=True, help="output dir of export_onnx.py")
    ap.add_argument("--state", required=True, help="json file recording AI Hub ids (created if missing)")
    ap.add_argument("--device", default="QCS8550 (Proxy)")
    ap.add_argument("--no-profile", action="store_true")
    args = ap.parse_args()

    manifest = json.loads((Path(args.onnx_dir) / "manifest.json").read_text())
    state = State(args.state)
    state.set("device", args.device)
    state.set("seq_len", manifest["seq_len"])
    device = hub.Device(args.device)
    results = {}

    def worker(p):
        results[p["name"]] = run_part(p["name"], Path(args.onnx_dir) / p["onnx"], device, state, not args.no_profile)

    threads = [threading.Thread(target=worker, args=(p,)) for p in manifest["parts"]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ok = all(results.get(p["name"]) for p in manifest["parts"])
    print(f"[aihub] {'all parts compiled' if ok else 'FAILED: ' + str(results)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
