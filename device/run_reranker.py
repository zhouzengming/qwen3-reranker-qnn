#!/usr/bin/env python3
"""Command line entry for the QNN Qwen3-Reranker.

  python3 run_reranker.py selftest                    # compare against fp32 references (tests/reference.json)
  python3 run_reranker.py bench -n 10                 # latency, per part
  python3 run_reranker.py rerank -q "query" -d "doc 1" -d "doc 2"
  python3 run_reranker.py rerank -q "query" --docs-file docs.txt   # one doc per line
  python3 run_reranker.py info                        # IO tensors of each context binary
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VARIANTS = list(json.loads((HERE / "variants.json").read_text())["variants"]) if (HERE / "variants.json").exists() else []
from qwen3_reranker import Qwen3Reranker  # noqa: E402

# Qnn_DataType_t values (QnnTypes.h)
DTYPES = {"0x32": "int32", "0x64": "int64", "0x132": "uint32", "0x216": "float16", "0x232": "float32"}


def load(args):
    t0 = time.perf_counter()
    rr = Qwen3Reranker(HERE, variant=args.variant, burst=not args.no_burst, log_level=args.log_level,
                       lib_path=args.lib, qnn_lib_dir=args.qnn_libs)
    print(f"[load] variant={args.variant} parts={rr.rt.num_parts} seq_len={rr.seq_len} "
          f"({time.perf_counter() - t0:.2f}s)", flush=True)
    return rr


def cmd_info(args):
    rr = load(args)
    mode = {2: "createFromBinaryListAsync + shareResources", 1: "spill-fill group registration",
            0: "none (standalone contexts)"}[rr.rt.lib.qr_contexts_grouped(rr.rt.handle)]
    print(f"context memory sharing: {mode}")
    for p in range(rr.rt.num_parts):
        spill = rr.rt.lib.qr_spill_fill_bytes(rr.rt.handle, p) / 2**20
        print(f"part{p + 1} graph={rr.rt.lib.qr_graph_name(rr.rt.handle, p).decode()} "
              f"spill-fill={spill:.1f} MiB")
        for io, output in (("in ", False), ("out", True)):
            for name, dt, dims in rr.rt.tensors(p, output):
                print(f"part{p + 1} {io} {name:24s} {DTYPES.get(dt, dt):8s} {dims}")
    rr.close()


def cmd_selftest(args):
    ref = json.loads((HERE / "tests" / "reference.json").read_text())
    rr = load(args)
    key = args.variant
    diffs = []
    print(f"{'tokens':>6} {'ref_margin':>11} {'npu_margin':>11} {'diff':>7}  query")
    for item in ref["pairs"]:
        m, _ = rr.score(item["query"], item["document"])
        r = item["ref_margin"][key]
        diffs.append(abs(m - r))
        print(f"{item['num_tokens'][key]:6d} {r:11.4f} {m:11.4f} {abs(m - r):7.4f}  {item['query'][:30]}")
    rr.close()
    worst = max(diffs)
    ok = worst < args.tol
    print(f"max |diff| = {worst:.4f} (tolerance {args.tol}) -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_bench(args):
    rr = load(args)
    doc = "Beijing is the capital of China. It has a history of over three thousand years. " * 400
    ids, mask, n, _ = rr.encode_pair("What is the capital of China?", doc)
    rr.score_ids(ids, mask)  # warm-up
    totals, parts = [], []
    for _ in range(args.n):
        t0 = time.perf_counter()
        rr.score_ids(ids, mask)
        totals.append((time.perf_counter() - t0) * 1000)
        parts.append(rr.last_part_ms.copy())
    rr.close()
    parts = np.array(parts)
    print(f"[bench] variant={args.variant} tokens={n} runs={args.n}")
    print(f"  end-to-end (incl. host embedding/copies): median {np.median(totals):.1f} ms, "
          f"min {np.min(totals):.1f}, max {np.max(totals):.1f}")
    for k in range(parts.shape[1]):
        print(f"  part{k + 1} execute: median {np.median(parts[:, k]):.1f} ms")


def cmd_rerank(args):
    docs = list(args.doc or [])
    if args.docs_file:
        docs += [l.rstrip("\n") for l in open(args.docs_file, encoding="utf-8") if l.strip()]
    if not docs:
        sys.exit("no documents given (-d / --docs-file)")
    rr = load(args)
    t0 = time.perf_counter()
    results = rr.rerank(args.query, docs, instruction=args.instruction, top_k=args.top_k)
    dt = time.perf_counter() - t0
    rr.close()
    for rank, r in enumerate(results, 1):
        flag = " (truncated)" if r.truncated else ""
        print(f"{rank:3d}. margin={r.margin:8.4f} score={r.score:.4f} doc#{r.index} tokens={r.num_tokens}{flag}  "
              f"{r.text[:80]}")
    print(f"[rerank] {len(docs)} docs in {dt:.2f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("info", "selftest", "bench", "rerank"):
        p = sub.add_parser(name)
        p.add_argument("--variant", default=VARIANTS[0] if VARIANTS else None, choices=VARIANTS or None)
        p.add_argument("--no-burst", action="store_true", help="do not vote for max HTP clocks")
        p.add_argument("--log-level", type=int, default=1, help="QNN log level 0-3")
        p.add_argument("--lib", default=None, help="override path to libqnn_reranker.so")
        p.add_argument("--qnn-libs", default=None, help="override QNN runtime library directory")
        if name == "selftest":
            p.add_argument("--tol", type=float, default=0.15, help="max allowed |margin diff| vs fp32")
        if name == "bench":
            p.add_argument("-n", type=int, default=10)
        if name == "rerank":
            p.add_argument("-q", "--query", required=True)
            p.add_argument("-d", "--doc", action="append")
            p.add_argument("--docs-file")
            p.add_argument("--instruction", default=None)
            p.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()
    rc = {"info": cmd_info, "selftest": cmd_selftest, "bench": cmd_bench, "rerank": cmd_rerank}[args.cmd](args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
