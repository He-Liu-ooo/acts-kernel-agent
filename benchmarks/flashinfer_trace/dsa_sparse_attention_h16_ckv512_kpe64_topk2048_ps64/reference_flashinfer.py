import math
import torch
import flashinfer

_ws_cache = {}


def _ws(device):
    k = str(device)
    b = _ws_cache.get(k)
    if b is None:
        b = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
        _ws_cache[k] = b
    return b


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    # Shapes
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    device = q_nope.device

    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.item())
    else:
        sm_scale = float(sm_scale)

    # Flatten paged KV to token level; treat each KV token as a page_size=1 page.
    ckv_flat = ckv_cache.reshape(num_pages * page_size, 1, head_dim_ckv)
    kpe_flat = kpe_cache.reshape(num_pages * page_size, 1, head_dim_kpe)

    # Build ragged paged layout from the per-query sparse index lists.
    # Each query attends to its valid (non -1) gathered token ids; page_size=1
    # means kv_indices entries ARE token ids.
    valid = sparse_indices != -1                      # [T, topk]
    kv_lens = valid.sum(dim=1).to(torch.int32)        # [T]
    q_indptr = torch.arange(0, num_tokens + 1, device=device, dtype=torch.int32)
    kv_indptr = torch.zeros(num_tokens + 1, device=device, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(kv_lens, dim=0)
    kv_indices = sparse_indices[valid].to(torch.int32)  # flattened valid token ids

    out = torch.empty(num_tokens, num_qo_heads, head_dim_ckv, dtype=q_nope.dtype, device=device)
    lse = torch.empty(num_tokens, num_qo_heads, dtype=torch.float32, device=device)

    W = flashinfer.mla.BatchMLAPagedAttentionWrapper(_ws(device), backend="fa2")
    W.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_qo_heads,
        head_dim_ckv,
        head_dim_kpe,
        1,             # page_size (token-level)
        False,         # causal
        sm_scale,
        q_nope.dtype,
        ckv_flat.dtype,
    )
    o, l = W.run(q_nope, q_pe, ckv_flat, kpe_flat, return_lse=True)
    out.copy_(o)
    # fa2 already returns base-2 LSE matching the reference (which divides natural
    # logsumexp by ln2) — no further conversion needed.
    lse.copy_(l)
    return out, lse
