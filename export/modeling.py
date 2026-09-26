"""Qwen3-Reranker rewritten into static, NPU-friendly graphs.

Changes versus the Hugging Face model (all numerically equivalent, verified < 1e-6 vs. fp32):
  * static shape [1, L], left padding, int32 inputs
  * token embedding is looked up on the host: every graph takes hidden states, so no embedding
    Gather sits in the NPU graph (at L=4096 graphs containing it fail HTP graph prepare)
  * decoder layers are split into N sequential parts (one QNN context binary each)
  * the additive attention mask is built in-graph from `attention_mask` with elementwise ops
    (arange comparison, no L x L constant), finite mask value for fp16 friendliness
  * query-chunked attention: queries are processed in blocks and each block only attends to the
    causally visible key prefix, capping the largest attention tensor at heads x block x L
    (needed to fit L=4096 into HTP memory)
  * the last part keeps only the "yes"/"no" rows of lm_head: outputs logits [1, 2] and P(yes) [1]
"""
import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

_ATTN_CHUNK = 512


def chunked_q_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Drop-in replacement for eager attention: same math, computed per query block."""
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    seq_len = query.shape[2]
    outs = []
    for start in range(0, seq_len, _ATTN_CHUNK):
        end = min(start + _ATTN_CHUNK, seq_len)
        # keys after `end` are causally masked for every query of this block: skip them
        w = torch.matmul(query[:, :, start:end], key[:, :, :end].transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask[:, :, start:end, :end]
        w = torch.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        outs.append(torch.matmul(w, value[:, :, :end]))
    return torch.cat(outs, dim=2).transpose(1, 2).contiguous(), None


AttentionInterface.register("chunked_q", chunked_q_attention)


class RerankerPart(torch.nn.Module):
    """Decoder layers [layer_start, layer_end) of the reranker; the last part adds norm + yes/no head."""

    def __init__(self, base, lm_head_rows, layer_start, layer_end, is_last, seq_len, mask_value):
        super().__init__()
        self.is_last = is_last
        self.layers = torch.nn.ModuleList(base.layers[layer_start:layer_end])
        self.rotary_emb = base.rotary_emb
        self.norm = base.norm if is_last else None
        if is_last:
            self.register_buffer("yes_no_weight", lm_head_rows.clone())
        self.mask_value = mask_value
        self.register_buffer("pos_f", torch.arange(seq_len, dtype=torch.float32))
        # transformers uses arange positions regardless of padding for this model
        self.register_buffer("position_ids", torch.arange(seq_len).view(1, seq_len))

    def forward(self, hidden, attention_mask):
        pad = attention_mask.to(hidden.dtype).view(1, 1, 1, -1)
        causal = (self.pos_f.view(1, 1, 1, -1) <= self.pos_f.view(1, 1, -1, 1)).to(hidden.dtype)
        bias = (1.0 - causal * pad) * self.mask_value
        pos = self.rotary_emb(hidden, self.position_ids)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=bias, position_embeddings=pos, position_ids=self.position_ids)
        if not self.is_last:
            return hidden
        last = self.norm(hidden)[:, -1, :]  # left padding: the last position is the real last token
        logits = last @ self.yes_no_weight.T
        return logits, torch.softmax(logits, dim=-1)[:, 1]


def split_points(num_layers, num_parts):
    base, extra = divmod(num_layers, num_parts)
    bounds, start = [], 0
    for i in range(num_parts):
        end = start + base + (1 if i < extra else 0)
        bounds.append((start, end))
        start = end
    return bounds


def load_reranker(model_dir, attn_chunk):
    """Tokenizer + fp32 causal LM using chunked attention (attn_chunk=0: plain eager attention)."""
    global _ATTN_CHUNK
    if attn_chunk:
        _ATTN_CHUNK = attn_chunk
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="chunked_q" if attn_chunk else "eager").eval()
    return tokenizer, model


def build_parts(model_dir, seq_len, num_parts, attn_chunk=512, mask_value=-100.0):
    tokenizer, model = load_reranker(model_dir, attn_chunk)
    no_id, yes_id = tokenizer.convert_tokens_to_ids("no"), tokenizer.convert_tokens_to_ids("yes")
    rows = model.lm_head.weight[[no_id, yes_id]].detach()
    bounds = split_points(len(model.model.layers), num_parts)
    parts = [RerankerPart(model.model, rows, s, e, i == num_parts - 1, seq_len, mask_value).eval()
             for i, (s, e) in enumerate(bounds)]
    return tokenizer, model.model.embed_tokens, parts, bounds
