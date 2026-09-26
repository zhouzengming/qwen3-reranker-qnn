"""Qwen3-Reranker prompt template and fixed-length encoding (must match device/qwen3_reranker.py)."""
import numpy as np

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
          "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def encode_pair(tokenizer, query, doc, seq_len, instruction=DEFAULT_INSTRUCTION):
    """Left-padded int32 (input_ids, attention_mask), each shaped (1, seq_len).

    The document tail is truncated when the prompt does not fit.
    """
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    body = tokenizer.encode(f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}",
                            add_special_tokens=False)
    ids = prefix + body[: seq_len - len(prefix) - len(suffix)] + suffix
    input_ids = np.full((1, seq_len), tokenizer.pad_token_id, dtype=np.int32)
    attention_mask = np.zeros((1, seq_len), dtype=np.int32)
    input_ids[0, seq_len - len(ids):] = ids
    attention_mask[0, seq_len - len(ids):] = 1
    return input_ids, attention_mask


def reference_margin(model, tokenizer, input_ids, attention_mask):
    """logit(yes) - logit(no) from the official PyTorch model, run on the unpadded tokens."""
    import torch

    n = int(attention_mask.sum())
    with torch.no_grad():
        logits = model(input_ids=torch.tensor(input_ids[:, -n:], dtype=torch.long)).logits[0, -1]
    yes, no = tokenizer.convert_tokens_to_ids("yes"), tokenizer.convert_tokens_to_ids("no")
    return float(logits[yes] - logits[no])


# Built-in sample pairs (English, Chinese, relevant / irrelevant, one long document) used for
# verification and for the on-device selftest reference.
SAMPLE_PAIRS = [
    ("What is the capital of China?", "The capital of China is Beijing.", 1),
    ("What is the capital of China?", "Gravity is a force that attracts two bodies towards each other.", 0),
    ("Explain gravity", "Gravity is a force that attracts two bodies towards each other. It gives weight to "
     "physical objects and is responsible for the movement of planets around the sun.", 1),
    ("中国的首都是哪里？", "北京是中华人民共和国的首都。", 1),
    ("中国的首都是哪里？", "今天天气很好，适合出去散步。", 0),
    ("What is the capital of China?",
     "Beijing is the capital of China. It has a history of over three thousand years and hosted the 2008 "
     "Olympics. " * 200, 1),  # ~4000 tokens: exercises the full sequence length
]
