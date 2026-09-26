"""Step 5: assemble the on-device package.

Copies the device code, the context binaries, the embedding table and the tokenizer, writes
variants.json and the selftest reference, and copies the QNN runtime pieces from YOUR local QAIRT SDK
(those files are Qualcomm-licensed and are not part of this repository).

QNN host libraries must come from the SDK build that supports your SoC; see the "supported
Snapdragon devices" table in the QAIRT docs. For QCS8550 on Linux that is aarch64-oe-linux-gcc11.2.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "export"))
from prompt import SAMPLE_PAIRS, encode_pair, reference_margin  # noqa: E402

HOST_LIBS = ["libQnnHtp.so", "libQnnSystem.so", "libQnnHtp{arch}Stub.so"]
DSP_LIBS = ["libQnnHtp{arch}Skel.so", "libqnnhtp{arch_l}.cat"]
TOOLS = {"bin": ["qnn-platform-validator", "qnn-net-run"],
         "lib": ["libPlatformValidatorShared.so", "libQnnHtp{arch}CalculatorStub.so", "libcalculator.so"],
         "dsp": ["libCalculator_skel.so"]}


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model dir (tokenizer + fp32 reference)")
    ap.add_argument("--onnx-dir", required=True, help="output dir of export_onnx.py (manifest + embedding)")
    ap.add_argument("--binaries", required=True, help="dir with part*.bin from aihub_download.py")
    ap.add_argument("--qairt-sdk", required=True, help="QAIRT SDK root, e.g. /opt/qairt/2.50.0.xxxxxx")
    ap.add_argument("--qnn-target", default="aarch64-oe-linux-gcc11.2", help="SDK lib/bin subdir for the board")
    ap.add_argument("--htp-arch", default="v73", help="Hexagon arch of the NPU (QCS8550/SM8550: v73)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cross-compile", action="store_true",
                    help="also build device/libqnn_reranker.so for aarch64 with `python -m ziglang`")
    ap.add_argument("--no-reference", action="store_true", help="skip computing the selftest reference")
    args = ap.parse_args()

    sdk, out = Path(args.qairt_sdk), Path(args.out)
    manifest = json.loads((Path(args.onnx_dir) / "manifest.json").read_text())
    arch, arch_l = args.htp_arch.upper(), args.htp_arch.lower()
    fmt = lambda s: s.format(arch=arch, arch_l=arch_l)  # noqa: E731

    # device code (+ its pip requirements, referenced by the README's board instructions)
    shutil.copytree(REPO / "device", out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    copy(REPO / "requirements-device.txt", out / "requirements-device.txt")

    # models + tokenizer
    variant = f"L{manifest['seq_len']}"
    contexts = []
    for p in manifest["parts"]:
        src = Path(args.binaries) / f"{p['name']}.bin"
        dst = out / "models" / f"qwen3_reranker_{variant}_fp16_{p['name']}.bin"
        if not dst.exists():
            copy(src, dst)
        contexts.append(f"models/{dst.name}")
    if not (out / "models" / "embed_tokens_fp16.npy").exists():
        copy(Path(args.onnx_dir) / "embed_tokens_fp16.npy", out / "models" / "embed_tokens_fp16.npy")
    copy(Path(args.model) / "tokenizer.json", out / "tokenizer" / "tokenizer.json")
    cfg = {"tokenizer": "tokenizer/tokenizer.json", "pad_token_id": manifest["pad_token_id"],
           "htp_arch": arch_l, "qnn_target": args.qnn_target,
           "variants": {variant: {"seq_len": manifest["seq_len"], "contexts": contexts,
                                  "host_embedding": "models/embed_tokens_fp16.npy"}}}
    (out / "variants.json").write_text(json.dumps(cfg, indent=2))

    # QNN runtime from the local SDK (not redistributed with this repository)
    for name in map(fmt, HOST_LIBS):
        copy(sdk / "lib" / args.qnn_target / name, out / "qnn_libs" / name)
    for name in map(fmt, DSP_LIBS):
        src = sdk / "lib" / f"hexagon-{arch_l}" / "unsigned" / name
        if src.exists():
            copy(src, out / "qnn_libs" / name)
    shutil.copytree(sdk / "include" / "QNN", out / "include" / "QNN", dirs_exist_ok=True)
    for name in TOOLS["bin"]:
        copy(sdk / "bin" / args.qnn_target / name, out / "tools" / name)
    for name in map(fmt, TOOLS["lib"]):
        copy(sdk / "lib" / args.qnn_target / name, out / "tools" / "lib" / name)
    for name in TOOLS["dsp"]:
        copy(sdk / "lib" / f"hexagon-{arch_l}" / "unsigned" / name, out / "tools" / "dsp" / name)

    if args.cross_compile:
        (out / "lib").mkdir(exist_ok=True)
        cmd = [sys.executable, "-m", "ziglang", "c++", "-target", "aarch64-linux-gnu.2.31", "-std=c++17", "-O2",
               "-fPIC", "-shared", "-s", f"-I{out / 'include' / 'QNN'}", str(out / "src" / "qnn_reranker.cpp"),
               "-o", str(out / "lib" / "libqnn_reranker.so"), "-ldl"]
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        print(f"[deploy] cross-compiled {out / 'lib' / 'libqnn_reranker.so'}")

    if not args.no_reference:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
        pairs = []
        for q, doc, label in SAMPLE_PAIRS:
            ids, mask = encode_pair(tok, q, doc, manifest["seq_len"])
            pairs.append({"query": q, "document": doc, "label": label,
                          "num_tokens": {variant: int(mask.sum())},
                          "ref_margin": {variant: round(reference_margin(model, tok, ids, mask), 5)}})
        (out / "tests").mkdir(exist_ok=True)
        (out / "tests" / "reference.json").write_text(json.dumps(
            {"note": "fp32 reference margins (logit(yes) - logit(no)) from the official PyTorch model",
             "pairs": pairs}, ensure_ascii=False, indent=1))
    print(f"[deploy] package ready: {out}")


if __name__ == "__main__":
    main()
