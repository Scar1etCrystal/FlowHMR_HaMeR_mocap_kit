"""Experimental query-batched attention for the existing FlowHMR model.

Main blocks keep their original local visibility rules, QKV projections, RoPE,
gates, residuals and FFNs. The token refiner keeps GLOBAL attention and its
original global context mean. Both paths use PyTorch SDPA's FP32 MATH backend.

Usage inside a configured FlowHMR environment (the model must already be loaded):

    with install(network, local=True, refiner=True, chunk=256):
        output = pipeline.generate(...)

This is an inference-only, temporary module-level patch. Use one inference at a
time per process. Leaving the context restores the old
implementation, including on exceptions.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType

import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


@dataclass
class GlobalPolicy:
    """Refiner: batch queries, but keep every valid key/value available."""

    valid: object  # Boolean tensor [batch, time], or None for all-valid input.
    chunk: int


@dataclass
class LocalPolicy:
    """Main-block visibility, expressed without a full sequence-square mask.

    mm/mc/cm/cc mean QUERY stream -> KEY stream:
        m = motion, c = video condition.
    Nonnegative radii are in frames; -1 means global, -2 means disabled.
    Motion and condition tokens each use their own zero-based frame timeline.
    """

    motion: int
    condition: int
    mm: int
    mc: int
    cm: int
    cc: int
    valid_motion: object
    valid_condition: object
    chunk: int


def batched_attention(q, k, v, *, attn_mask, **kwargs):
    """Drop-in attention dispatch for the two compact policies above."""
    if (
        kwargs.get('return_attn')
        or kwargs.get('training')
        or kwargs.get('drop_rate', 0)
    ):
        raise ValueError(
            'Experimental batching supports eval / dropout0 '
            'without attention weights only'
        )

    policy = attn_mask

    # [batch, tokens, heads, dim] -> [batch, heads, tokens, dim].
    # Q/K already carry the ORIGINAL full-sequence RoPE positions.
    Q, K, V = (x.transpose(1, 2) for x in (q, k, v))
    out = torch.empty_like(Q)

    if isinstance(policy, GlobalPolicy):
        N = q.shape[1]
        valid = policy.valid
        if valid is None:
            valid = torch.ones(
                (q.shape[0], N), dtype=torch.bool, device=q.device
            )

        for start in range(0, N, policy.chunk):
            end = min(start + policy.chunk, N)
            allowed = valid[:, start:end, None] & valid[:, None, :]

            # Preserve refiner padding semantics: a query with no valid keys
            # may attend only to its own diagonal entry, even if it is padding.
            diag = (
                torch.arange(start, end, device=q.device)[:, None]
                == torch.arange(N, device=q.device)[None, :]
            )
            allowed = torch.where(
                allowed.any(-1, keepdim=True), allowed, diag[None]
            )

            # Only Q is sliced. K/V still cover the ENTIRE video.
            # SDPA bool masks use True = visible.
            with sdpa_kernel(SDPBackend.MATH):
                out[:, :, start:end] = F.scaled_dot_product_attention(
                    Q[:, :, start:end],
                    K,
                    V,
                    attn_mask=allowed[:, None],
                    dropout_p=0.,
                    is_causal=False,
                )

    elif isinstance(policy, LocalPolicy):
        p = policy
        M, C = p.motion, p.condition

        def key_range(start, end, radius, total):
            if radius < 0:
                return 0, total
            return max(0, start - radius), min(total, end + radius)

        # Storage is [all motion tokens, all condition tokens]. offset maps a
        # stream-local frame index to its position in the concatenated Q/K/V.
        for stream, N, offset in [('motion', M, 0), ('condition', C, M)]:
            for start in range(0, N, p.chunk):
                end = min(start + p.chunk, N)
                indices = []
                times = []
                kinds = []
                radii = [p.mm, p.mc] if stream == 'motion' else [p.cm, p.cc]

                # Gather the UNION of keys visible to this query batch.
                # Different rows will then get their own small exact mask.
                for kind, (radius, total, base) in enumerate(
                    zip(radii, [M, C], [0, M])
                ):
                    if radius == -2:
                        continue  # E.g. condition queries cannot see motion keys.
                    lo, hi = key_range(start, end, radius, total)
                    ids = torch.arange(lo, hi, device=q.device)
                    indices.append(ids + base)
                    times.append(ids)
                    kinds.append(torch.full_like(ids, kind))

                ids = torch.cat(indices)
                kt = torch.cat(times)
                kind = torch.cat(kinds)
                qt = torch.arange(start, end, device=q.device)
                allow = torch.zeros(
                    (end - start, len(ids)), dtype=torch.bool, device=q.device
                )

                # The union includes extra neighbors for some rows. Mask those
                # out so each query retains exactly its original window.
                for keykind, radius in enumerate(radii):
                    if radius == -2:
                        continue
                    distance = (qt[:, None] - kt[None, :]).abs()
                    allow |= (kind[None, :] == keykind) & (
                        (distance <= radius)
                        if radius >= 0
                        else torch.ones_like(distance, dtype=torch.bool)
                    )

                allvalid = torch.cat([p.valid_motion, p.valid_condition], dim=1)
                allowed = allow[None] & allvalid[:, None, ids]

                # Main attention's original fully-masked-row fallback differs
                # from the refiner: such a row sees ALL keys. Keep that behavior
                # for padded batches; normal all-valid video inputs do not hit it.
                bad = ~allowed.any(-1)
                with sdpa_kernel(SDPBackend.MATH):
                    # Motion + condition keys MUST share ONE softmax. Computing
                    # each stream separately and adding would change the model.
                    value = F.scaled_dot_product_attention(
                        Q[:, :, offset + start:offset + end],
                        K.index_select(2, ids),
                        V.index_select(2, ids),
                        attn_mask=allowed[:, None],
                        dropout_p=0.,
                        is_causal=False,
                    )
                    if bad.any():
                        fallback = F.scaled_dot_product_attention(
                            Q[:, :, offset + start:offset + end],
                            K,
                            V,
                            dropout_p=0.,
                            is_causal=False,
                        )
                        value = torch.where(
                            bad[:, None, :, None], fallback, value
                        )
                out[:, :, offset + start:offset + end] = value
    else:
        raise TypeError(type(policy))

    return out.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1)


def individual_forward(self, x, c, mask=None):
    """Replace only the refiner's dense mask construction and attention calls."""
    # Its parent SingleTokenRefiner still computes the original global mean.
    policy = GlobalPolicy(mask, self._query_chunk)
    for block in self.blocks:
        x = block(x, c, policy)
    return x


def network_forward(
    self,
    x,
    ctxt_input,
    vtxt_input,
    timesteps,
    x_mask_temporal,
    ctxt_mask_temporal,
    cond_mask_prob=0.,
    **kwargs,
):
    """Original network flow, with a compact policy instead of dense masks."""
    from flowhmr.core.framework.type_converter import get_module_device
    import torch.nn as nn

    if (
        self.training
        or self.insert_start_token
        or self.mask_mode != 'narrowband_v2m'
    ):
        raise ValueError(
            'Experimental path is eval narrowband_v2m without start token only'
        )

    # Preserve original encoders and timestep/global conditioning.
    device = get_module_device(self)
    origin_feat = self.input_encoder(x)
    motion_feat = origin_feat
    timestep_feat = self.timestep_encoder(timesteps)
    if vtxt_input is None:
        vtxt_input = torch.zeros(
            (x.shape[0], 1, self.vtxt_input_dim), device=device
        )
    adapter = timestep_feat + self.vtxt_encoder(vtxt_input.float())

    if isinstance(self.ctxt_encoder, nn.ModuleDict):
        ctxt_feat = 0
        for key in self.ctxt_encoder:
            ctxt_feat = ctxt_feat + self.ctxt_encoder[key](ctxt_input[key])
    else:
        ctxt_feat = self.ctxt_encoder(ctxt_input.float())

    if hasattr(self, 'text_refiner'):
        ctxt_feat = self.text_refiner(
            x=ctxt_feat, t=timesteps, mask=ctxt_mask_temporal.to(device)
        )

    # Read the model's actual radii; do not assume all four windows are equal.
    # In particular cm=0 in the checkpoint means DISABLED, not self-frame access.
    M, C = motion_feat.shape[1], ctxt_feat.shape[1]
    mc = (
        int(round(self.narrowband_v2m_length))
        if self.narrowband_v2m_length > 0
        else -1
    )
    cm = (
        -2
        if self.narrowband_tm_length == 0
        else int(round(self.narrowband_tm_length))
        if self.narrowband_tm_length > 0
        else -1
    )
    ccval = (
        self.narrowband_tt_length
        if self.narrowband_tt_length is not None
        else self.narrowband_v2m_length
    )
    cc = int(round(ccval)) if ccval > 0 else -1
    policy = LocalPolicy(
        M,
        C,
        int(round(self.narrowband_length)),
        mc,
        cm,
        cc,
        x_mask_temporal.to(device),
        ctxt_mask_temporal.to(device),
        self._query_chunk,
    )

    # Existing blocks still compute QKV, RoPE, gates, residuals and FFNs.
    # Each attention call finishes ALL query batches before the next layer runs;
    for block in self.double_blocks:
        motion_feat, ctxt_feat = block(
            motion_feat=motion_feat,
            text_feat=ctxt_feat,
            adapter=adapter,
            attn_mask=policy,
        )
    joined = torch.cat([motion_feat, ctxt_feat], dim=1)
    for block in self.single_blocks:
        joined = block(x=joined, split_len=M, adapter=adapter, attn_mask=policy)
    x = joined[:, :M]
    if self.with_long_skip_connection:
        x = self.long_skip_net(origin_feat, timestep_feat) + x
    return self.final_layer(x, adapter)


@contextmanager
def install(network, local=True, refiner=True, chunk=256):
    """Temporarily enable either/both paths; always restore original methods."""
    import flowhmr.network.hymotion_mmdit_for_v2m as mainmod
    import flowhmr.network.token_refiner as refmod

    orig_main = mainmod.attention
    orig_ref = refmod.attention
    orig_forward = network.forward
    individual = network.text_refiner.individual_token_refiner
    orig_individual = individual.forward

    def route(original):
        def f(q, k, v, **kw):
            # Only compact policies take the new path. Ordinary attention calls
            # keep their original implementation, allowing separate ablations.
            if isinstance(kw.get('attn_mask'), (GlobalPolicy, LocalPolicy)):
                return batched_attention(q, k, v, **kw)
            return original(q, k, v, **kw)
        return f

    mainmod.attention = route(orig_main)
    refmod.attention = route(orig_ref)
    network._query_chunk = chunk
    individual._query_chunk = chunk
    if local:
        network.forward = MethodType(network_forward, network)
    if refiner:
        individual.forward = MethodType(individual_forward, individual)
    try:
        yield
    finally:
        network.forward = orig_forward
        individual.forward = orig_individual
        mainmod.attention = orig_main
        refmod.attention = orig_ref
        del network._query_chunk
        del individual._query_chunk
