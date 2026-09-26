"""Step 2: run the exported ONNX parts in sequence (onnxruntime, CPU) and compare with the official
PyTorch model on the built-in sample pairs (short, Chinese, and a full-length document)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt import SAMPLE_PAIRS, encode_pair, reference_margin  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--onnx-dir", required=True, help="output dir of export_onnx.py")
    ap.add_argument("--tol", type=float, default=1e-3, help="max allowed |margin diff|")
    args = ap.parse_args()

    d = Path(args.onnx_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    L = manifest["seq_len"]
    tok = AutoTokenizer.from_pretrained(args.model)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    emb = np.load(d / "embed_tokens_fp16.npy")
    sessions = [ort.InferenceSession(str(d / p["onnx"] / "model.onnx"), providers=["CPUExecutionProvider"])
                for p in manifest["parts"]]

    worst = 0.0
    for q, doc, _ in SAMPLE_PAIRS:
        ids, mask = encode_pair(tok, q, doc, L)
        ref = reference_margin(ref_model, tok, ids, mask)
        t0 = time.perf_counter()
        x = emb[ids[0]].astype(np.float32)[None]  # host-side embedding lookup, as on the device
        for s in sessions:
            out = s.run(None, {"hidden_in": x, "attention_mask": mask})
            x = out[0]
        margin = float(out[0][0, 1] - out[0][0, 0])
        worst = max(worst, abs(margin - ref))
        print(f"tokens={int(mask.sum()):5d} ref={ref:9.4f} onnx={margin:9.4f} diff={abs(margin - ref):.1e} "
              f"({time.perf_counter() - t0:.1f}s)  {q[:30]}", flush=True)
    ok = worst < args.tol
    print(f"max |diff| = {worst:.2e} (tol {args.tol}) -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
