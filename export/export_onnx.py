"""Step 1: export Qwen3-Reranker as N static-shape ONNX parts + the host embedding table.

Output (in --out):
  part{i}of{N}.onnx/{model.onnx, model.data}   AI Hub "ONNX directory" format (external weights)
  embed_tokens_fp16.npy                        [vocab, hidden] fp16 embedding table for the host
  manifest.json                                seq_len, parts, layer ranges, IO names
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling import build_parts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model dir (Qwen/Qwen3-Reranker-0.6B)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--parts", type=int, default=4, help="number of sequential graphs")
    ap.add_argument("--attn-chunk", type=int, default=512, help="query block size (0 = plain attention)")
    ap.add_argument("--mask-value", type=float, default=-100.0)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, embed, parts, bounds = build_parts(args.model, args.seq_len, args.parts, args.attn_chunk,
                                                  args.mask_value)
    np.save(out / "embed_tokens_fp16.npy", embed.weight.detach().half().numpy())

    ids = torch.full((1, args.seq_len), tokenizer.pad_token_id, dtype=torch.int32)
    mask = torch.zeros((1, args.seq_len), dtype=torch.int32)
    ids[0, -8:] = 100
    mask[0, -8:] = 1
    with torch.no_grad():
        x = embed(ids)
    manifest = {"seq_len": args.seq_len, "attn_chunk": args.attn_chunk, "mask_value": args.mask_value,
                "pad_token_id": tokenizer.pad_token_id, "hidden_size": int(x.shape[-1]), "parts": []}
    for i, (part, (s, e)) in enumerate(zip(parts, bounds), 1):
        name = f"part{i}of{args.parts}"
        tmp = out / f"_tmp_{name}"
        tmp.mkdir(exist_ok=True)
        outputs = ["logits", "score"] if part.is_last else ["hidden_out"]
        with torch.no_grad():
            torch.onnx.export(part, (x, mask), str(tmp / "model.onnx"), input_names=["hidden_in", "attention_mask"],
                              output_names=outputs, opset_version=args.opset, dynamo=False, external_data=True)
            next_x = None if part.is_last else part(x, mask)
        # repack the scattered external tensors into one model.data file
        dst = out / f"{name}.onnx"
        dst.mkdir(exist_ok=True)
        onnx.save_model(onnx.load(str(tmp / "model.onnx")), str(dst / "model.onnx"), save_as_external_data=True,
                        all_tensors_to_one_file=True, location="model.data", size_threshold=1024)
        (dst / "model.data").chmod(0o644)
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
        manifest["parts"].append({"name": name, "onnx": dst.name, "layers": [s, e - 1],
                                  "inputs": ["hidden_in", "attention_mask"], "outputs": outputs})
        print(f"[export] {name}: layers {s}-{e - 1} -> {dst}", flush=True)
        x = next_x
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[export] done: {out}/manifest.json")


if __name__ == "__main__":
    main()
