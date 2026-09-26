"""Optional: end-to-end accuracy of the compiled parts on a real device through AI Hub.

The host does the embedding lookup; part k runs as an AI Hub inference job whose outputs feed the
job of part k+1. The final logit margins are compared with the official fp32 PyTorch model.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import qai_hub as hub
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aihub_compile import retry  # noqa: E402
from prompt import SAMPLE_PAIRS, encode_pair, reference_margin  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--device", default=None, help="defaults to the device used for compilation")
    args = ap.parse_args()

    d = Path(args.onnx_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    state = json.loads(Path(args.state).read_text())
    device = hub.Device(args.device or state["device"])
    L = manifest["seq_len"]
    tok = AutoTokenizer.from_pretrained(args.model)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    ids, masks, refs = [], [], []
    for q, doc, _ in SAMPLE_PAIRS:
        i, m = encode_pair(tok, q, doc, L)
        ids.append(i)
        masks.append(m)
        refs.append(reference_margin(ref_model, tok, i, m))
    emb = np.load(d / "embed_tokens_fp16.npy")
    hidden = [emb[i[0]].astype(np.float32)[None] for i in ids]

    for p in manifest["parts"]:
        job = retry(lambda: hub.submit_inference_job(
            model=hub.get_model(state[f"{p['name']}.context_model"]), device=device,
            name=f"qwen3-reranker-eval-{p['name']}", inputs={"hidden_in": hidden, "attention_mask": masks}),
            f"submit {p['name']}", timeout=1800)  # uploads ~16 MB of hidden state per sample
        print(f"[eval] {p['name']}: inference job {job.job_id}", flush=True)
        while not (st := retry(lambda: hub.get_job(job.job_id).get_status(), "status")).finished:
            time.sleep(30)
        if not st.success:
            raise SystemExit(f"{p['name']} inference failed: {st.message}")
        out = retry(lambda: hub.get_job(job.job_id).download_output_data(), "download outputs", timeout=1800)
        first = out[sorted(out)[0]]  # compiled graphs rename outputs to output_0, output_1, ...
        if p is not manifest["parts"][-1]:
            hidden = [np.asarray(x, dtype=np.float32).reshape(1, L, -1) for x in first]
        else:
            logits = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1, 2) for x in first])

    dev = logits[:, 1] - logits[:, 0]
    for (q, _, _), m, r, v in zip(SAMPLE_PAIRS, masks, refs, dev):
        print(f"tokens={int(m.sum()):5d} fp32={r:9.4f} device={v:9.4f} diff={abs(v - r):.4f}  {q[:30]}")
    print(f"[eval] max |margin diff| = {np.max(np.abs(dev - np.array(refs))):.4f}")


if __name__ == "__main__":
    main()
