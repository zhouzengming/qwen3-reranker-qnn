"""Qwen3-Reranker-0.6B on the Qualcomm HTP (QNN context binaries), driven from Python.

Tokenization / prompt building / padding happen here (tokenizers + numpy); graph execution happens
in libqnn_reranker.so through ctypes. Variants are described in variants.json:

    L4096 : four context binaries run in sequence, input = hidden states; the token-embedding
            lookup runs on the CPU from embed_tokens_fp16.npy (memory-mapped).
    (Single-graph variants that take token ids directly are also supported by the runtime.)

Rank with `margin` = logit(yes) - logit(no). `score` = sigmoid(margin) = P(yes); it saturates near 1
in fp16, so ties there are expected - use margin for ordering.

    from qwen3_reranker import Qwen3Reranker
    rr = Qwen3Reranker("/path/to/deploy")
    for r in rr.rerank("What is the capital of China?", ["Beijing is the capital.", "Gravity ..."]):
        print(r.margin, r.score, r.text)
"""
from __future__ import annotations

import ctypes
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from tokenizers import Tokenizer

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
          "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


@dataclass
class RerankResult:
    index: int          # position in the input document list
    margin: float       # logit(yes) - logit(no); use this for ranking
    score: float        # P(yes) = sigmoid(margin)
    text: str
    num_tokens: int     # prompt tokens actually scored (after truncation)
    truncated: bool


class _Runtime:
    """Thin ctypes wrapper over libqnn_reranker.so."""

    def __init__(self, lib_path: Path, backend: Path, system: Path, contexts: Sequence[Path],
                 burst: bool = True, log_level: int = 1, graph_order: Sequence[str] | None = None):
        self.lib = ctypes.CDLL(str(lib_path))
        L = self.lib
        L.qr_create2.restype = ctypes.c_void_p
        L.qr_create2.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.qr_graph_name.restype = ctypes.c_char_p
        L.qr_graph_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.qr_contexts_grouped.argtypes = [ctypes.c_void_p]
        L.qr_destroy.argtypes = [ctypes.c_void_p]
        L.qr_last_error.restype = ctypes.c_char_p
        L.qr_num_parts.argtypes = [ctypes.c_void_p]
        L.qr_spill_fill_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.qr_spill_fill_bytes.restype = ctypes.c_ulonglong
        L.qr_num_tensors.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        L.qr_tensor_info.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                     ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint32),
                                     ctypes.POINTER(ctypes.c_int)]
        f32p, i32p = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32)
        L.qr_run.argtypes = [ctypes.c_void_p, i32p, f32p, i32p, f32p, ctypes.POINTER(ctypes.c_double)]
        L.qr_run.restype = ctypes.c_int

        paths = (ctypes.c_char_p * len(contexts))(*[str(p).encode() for p in contexts])
        order = list(graph_order or [])
        order_arr = (ctypes.c_char_p * max(len(order), 1))(*[g.encode() for g in order])
        self.handle = L.qr_create2(str(backend).encode(), str(system).encode(), paths, len(contexts),
                                   order_arr if order else None, len(order), 1 if burst else 0, log_level)
        if not self.handle:
            raise RuntimeError(f"qr_create failed: {L.qr_last_error().decode()}")
        self.num_parts = L.qr_num_parts(self.handle)

    def tensors(self, part: int, output: bool):
        out = []
        for i in range(self.lib.qr_num_tensors(self.handle, part, int(output))):
            name = ctypes.create_string_buffer(256)
            dtype, rank = ctypes.c_int(), ctypes.c_int()
            dims = (ctypes.c_uint32 * 8)()
            self.lib.qr_tensor_info(self.handle, part, int(output), i, name, 256, ctypes.byref(dtype), dims,
                                    ctypes.byref(rank))
            out.append((name.value.decode(), hex(dtype.value), list(dims[: rank.value])))
        return out

    def run(self, input_ids: np.ndarray | None, hidden: np.ndarray | None, mask: np.ndarray):
        logits = np.zeros(2, dtype=np.float32)
        times = np.zeros(self.num_parts, dtype=np.float64)
        ids_p = input_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if input_ids is not None else None
        hid_p = hidden.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if hidden is not None else None
        rc = self.lib.qr_run(self.handle, ids_p, hid_p, mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                             logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                             times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if rc != 0:
            raise RuntimeError(f"qr_run failed: {self.lib.qr_last_error().decode()}")
        return logits, times

    def close(self):
        if self.handle:
            self.lib.qr_destroy(self.handle)
            self.handle = None


class Qwen3Reranker:
    def __init__(self, deploy_dir: str | os.PathLike, variant: str | None = None, burst: bool = True,
                 log_level: int = 1, lib_path: str | os.PathLike | None = None,
                 qnn_lib_dir: str | os.PathLike | None = None):
        root = Path(deploy_dir)
        cfg = json.loads((root / "variants.json").read_text())
        variant = variant or next(iter(cfg["variants"]))
        if variant not in cfg["variants"]:
            raise ValueError(f"unknown variant {variant}; available: {list(cfg['variants'])}")
        v = cfg["variants"][variant]
        self.variant, self.seq_len = variant, v["seq_len"]
        self.pad_id = cfg["pad_token_id"]

        self.tok = Tokenizer.from_file(str(root / cfg["tokenizer"]))
        self.prefix_ids = self._encode(PREFIX)
        self.suffix_ids = self._encode(SUFFIX)

        self.embedding = None
        if v.get("host_embedding"):
            # memory-mapped: only rows of tokens actually used are paged in
            self.embedding = np.load(root / v["host_embedding"], mmap_mode="r")

        qnn_dir = Path(qnn_lib_dir) if qnn_lib_dir else root / "qnn_libs"
        self.rt = _Runtime(Path(lib_path) if lib_path else root / "lib" / "libqnn_reranker.so",
                           qnn_dir / "libQnnHtp.so", qnn_dir / "libQnnSystem.so",
                           [root / p for p in v["contexts"]], burst=burst, log_level=log_level,
                           graph_order=v.get("graph_order"))
        self.last_part_ms: np.ndarray | None = None

    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_pair(self, query: str, doc: str, instruction: str | None = None):
        """Left-padded (input_ids, attention_mask) int32 [seq_len], plus (num_tokens, truncated)."""
        instruction = instruction or DEFAULT_INSTRUCTION
        body = self._encode(f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}")
        room = self.seq_len - len(self.prefix_ids) - len(self.suffix_ids)
        truncated = len(body) > room
        ids = self.prefix_ids + body[:room] + self.suffix_ids  # truncation drops the document tail
        input_ids = np.full(self.seq_len, self.pad_id, dtype=np.int32)
        mask = np.zeros(self.seq_len, dtype=np.int32)
        input_ids[self.seq_len - len(ids):] = ids
        mask[self.seq_len - len(ids):] = 1
        return input_ids, mask, len(ids), truncated

    def score_ids(self, input_ids: np.ndarray, mask: np.ndarray) -> float:
        """Margin logit(yes) - logit(no) for one encoded sample."""
        if self.embedding is not None:
            hidden = np.ascontiguousarray(self.embedding[input_ids], dtype=np.float32)
            logits, times = self.rt.run(None, hidden, mask)
        else:
            logits, times = self.rt.run(input_ids, None, mask)
        self.last_part_ms = times
        return float(logits[1] - logits[0])

    def score(self, query: str, doc: str, instruction: str | None = None) -> tuple[float, float]:
        ids, mask, _, _ = self.encode_pair(query, doc, instruction)
        m = self.score_ids(ids, mask)
        return m, 1.0 / (1.0 + math.exp(-m))

    def rerank(self, query: str, docs: Iterable[str], instruction: str | None = None,
               top_k: int | None = None) -> list[RerankResult]:
        results = []
        for i, d in enumerate(docs):
            ids, mask, n, trunc = self.encode_pair(query, d, instruction)
            m = self.score_ids(ids, mask)
            results.append(RerankResult(i, m, 1.0 / (1.0 + math.exp(-m)), d, n, trunc))
        results.sort(key=lambda r: r.margin, reverse=True)
        return results[:top_k] if top_k else results

    def close(self):
        self.rt.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
