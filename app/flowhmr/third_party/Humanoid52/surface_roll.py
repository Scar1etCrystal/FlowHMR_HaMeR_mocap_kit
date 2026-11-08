"""Approved paired surface roll, independent of inference/contact correction.

Frozen numerical method from paired-surface-refine-2026-10-24. Only the eight
outgoing limb frames change; no joint, distal frame or source skin is refitted.
Finite centred filters and synchronous iterations permit bounded chunk halos.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from .bone_overlay import reference_frame

LIMBS = [('Left knee',1,4,7), ('Right knee',2,5,8),
         ('Left elbow',16,18,20), ('Right elbow',17,19,21)]
ACTIVE = [1,4,2,5,16,18,17,19]
def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

def smooth(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)

def evidence(y, hx, hz, f, s):
    """Constrained one-angle Wahba evidence around a fixed outgoing shaft.
 Both observed directions are independently measured. Projection magnitude
 weights observability. Side-consistent front fades when folded joint skin
 points along/opposite the outgoing segment's front. No per-frame sign swap.
 """
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    fp = f - y * np.sum(f * y, axis=-1, keepdims=True)
    sp = s - y * np.sum(s * y, axis=-1, keepdims=True)
    nf = np.linalg.norm(fp, axis=-1)
    ns = np.linalg.norm(sp, axis=-1)
    agreement = np.sum(unit(fp) * unit(np.cross(sp, y)), axis=-1)
    wf = nf * smooth(agreement, -0.1, 0.5)
    ws = ns
    c = wf * np.sum(hz * fp, axis=-1) + ws * np.sum(hx * sp, axis=-1)
    t = wf * np.sum(y * np.cross(hz, fp), axis=-1) + ws * np.sum(y * np.cross(hx, sp), axis=-1)
    strength = wf * nf + ws * ns
    return (c, t, strength, ns, agreement)

def common_axis(points, prior_targets, fps):
    targets = np.zeros_like(prior_targets)
    weights = []
    agreements = []
    bends = []
    turns = []
    for (k, (u, m, e)) in enumerate(((1, 4, 7), (2, 5, 8), (16, 18, 20), (17, 19, 21))):
        y = unit(points[:, m] - points[:, u])
        lower = unit(points[:, e] - points[:, m])
        px = unit(np.cross(y, prior_targets[:, k, 0]))
        cross = np.cross(y, lower)
        bend = np.degrees(np.arctan2(np.linalg.norm(cross, axis=-1), np.sum(y * lower, axis=-1)))
        normal = unit(cross) * (-1 if k < 2 else 1)
        agreement = np.sum(px * normal, axis=-1)
        discrepancy = np.degrees(np.arccos(np.clip(agreement, -1, 1)))
        observable = smooth(bend, 35.0, 95.0) * (1 - smooth(bend, 160.0, 175.0)) * smooth(agreement, -0.2, 0.3)
        confidence = np.exp(-(discrepancy / 45.0) ** 2 * (1 - smooth(bend, 70.0, 110.0)))
        weight = observable * confidence
        common = unit((1 - weight[:, None]) * px + weight[:, None] * normal)
        raw = np.arctan2(np.sum(y * np.cross(px, common), axis=-1), np.sum(px * common, axis=-1))
        sigma = max(0.5, float(fps) * 0.1)
        filtered = np.arctan2(gaussian_filter1d(np.sin(raw), sigma, mode='nearest', truncate=3), gaussian_filter1d(np.cos(raw), sigma, mode='nearest', truncate=3))
        strong = smooth(bend, 70.0, 110.0) * (1 - smooth(bend, 160.0, 175.0)) * smooth(agreement, -0.2, 0.3)
        turn = filtered + strong * np.arctan2(np.sin(raw - filtered), np.cos(raw - filtered))
        turn *= smooth(bend, 25.0, 40.0)
        common = px * np.cos(turn[:, None]) + np.cross(y, px) * np.sin(turn[:, None])
        for (s, axis) in enumerate((y, lower)):
            x = common - axis * np.sum(common * axis, axis=-1, keepdims=True)
            norm = np.linalg.norm(x, axis=-1)
            fallback = unit(np.cross(axis, prior_targets[:, k, s]))
            candidate = unit(x)
            theta = np.arctan2(np.sum(axis * np.cross(fallback, candidate), axis=-1), np.sum(fallback * candidate, axis=-1)) * smooth(norm, 0.02, 0.2)
            x = fallback * np.cos(theta[:, None]) + np.cross(axis, fallback) * np.sin(theta[:, None])
            targets[:, k, s] = unit(np.cross(x, axis))
        weights.append(weight)
        agreements.append(agreement)
        bends.append(bend)
        turns.append(turn)
    return targets

def confidence_solve(points, world, r, endpoints, fps):
    method = 'side_upper_shared'
    n = len(points)
    front = unit(endpoints[:, :4, 0] - endpoints[:, :4, 1])
    side = unit(endpoints[:, :4, 2] - endpoints[:, :4, 3])
    upfront = unit(endpoints[:, 4:, 0] - endpoints[:, 4:, 1])
    upside = unit(endpoints[:, 4:, 2] - endpoints[:, 4:, 3])
    out = world.copy()
    turns = []
    diagnostics = []
    shared = method.endswith('_shared')
    prior_targets = np.empty((n, 4, 2, 3))
    for (k, (_, u, m, e)) in enumerate(LIMBS):
        upper = unit(points[:, m] - points[:, u])
        lower = unit(points[:, e] - points[:, m])
        for (b, ch) in ((u, m), (m, e)):
            y = unit(points[:, ch] - points[:, b])
            D = reference_frame(r[ch] - r[b])
            hx = world[:, b] @ D[:, 0]
            hz = world[:, b] @ D[:, 2]
            (c, t, strength, side_norm, agreement) = evidence(y, hx, hz, front[:, k], side[:, k])
            if method in ('joint_side_upper', 'side_upper_shared'):
                pf = upfront[:, k]
                ps = upside[:, k]
                if b == m:
                    cross = np.cross(upper, lower)
                    dot = np.sum(upper * lower, axis=-1)

                    def transport(a):
                        return a + np.cross(cross, a) + np.cross(cross, np.cross(cross, a)) / np.maximum(1 + dot[:, None], 1e-06)
                    pf = transport(pf)
                    ps = transport(ps)
                (pc, pt, pstrength, _, _) = evidence(y, hx, hz, pf, ps)
                alpha = 0.65 * (1 - smooth(side_norm, 0.3, 0.8)) * smooth(agreement, -0.2, 0.5)
                if b == m:
                    alpha *= smooth(1 + dot, 0.005, 0.08)
                c = (1 - alpha) * c + alpha * pc
                t = (1 - alpha) * t + alpha * pt
                strength = (1 - alpha) * strength + alpha * pstrength
            fc = c if shared else gaussian_filter1d(c, fps * 0.1, mode='nearest', truncate=3)
            ft = t if shared else gaussian_filter1d(t, fps * 0.1, mode='nearest', truncate=3)
            norm = np.hypot(fc, ft)
            theta = np.arctan2(ft, fc)
            theta = theta * smooth(norm, 0.005, 0.08)
            out[:, b] = Rotation.from_rotvec(y * theta[:, None]).as_matrix() @ out[:, b]
            prior_targets[:, k, 0 if b == u else 1] = out[:, b] @ D[:, 2]
            turns.append(theta)
    if shared:
        for (k, (_, u, m, e)) in enumerate(LIMBS):
            for s in range(2):
                relative = np.einsum('tji,tj->ti', world[:, u], prior_targets[:, k, s])
                prior_targets[:, k, s] = unit(np.einsum('tij,tj->ti', world[:, u], gaussian_filter1d(relative, max(0.5, fps * 0.035), axis=0, mode='nearest', truncate=3)))
        target = common_axis(points, prior_targets, fps)
        out = world.copy()
        turns = []
        for (k, (_, u, m, e)) in enumerate(LIMBS):
            for (s, (b, ch)) in enumerate(((u, m), (m, e))):
                y = unit(points[:, ch] - points[:, b])
                D = reference_frame(r[ch] - r[b])
                have = world[:, b] @ D[:, 2]
                desired = target[:, k, s] - y * np.sum(target[:, k, s] * y, axis=-1, keepdims=True)
                condition = np.linalg.norm(desired, axis=-1)
                desired = unit(desired)
                theta = np.arctan2(np.sum(y * np.cross(have, desired), axis=-1), np.sum(have * desired, axis=-1))
                theta = theta * smooth(condition, 0.02, 0.2)
                theta = np.arctan2(gaussian_filter1d(np.sin(theta), fps * 0.1, mode='nearest', truncate=3), gaussian_filter1d(np.cos(theta), fps * 0.1, mode='nearest', truncate=3))
                out[:, b] = Rotation.from_rotvec(y * theta[:, None]).as_matrix() @ world[:, b]
                turns.append(theta)
    return (out, np.array(turns).T, diagnostics)

def paired(points, world, rest, fps, source=None):
    """Two roll DOFs per limb: signed axis agreement + surface prior + finite temporal support.
 Fixed synchronous iterations avoid an unbounded history dependency. Only roll changes.
 """
    out = world.copy()
    corrections = []
    for (_, u, m, e) in LIMBS:
        ys = np.stack([unit(points[:, m] - points[:, u]), unit(points[:, e] - points[:, m])], 1)
        xs = np.stack([world[:, u] @ reference_frame(rest[m] - rest[u])[:, 0], world[:, m] @ reference_frame(rest[e] - rest[m])[:, 0]], 1)
        delta = np.zeros((len(points), 2))
        lam = 4.0
        tau = 3.0
        sigma = max(0.5, fps * 0.06)
        for iteration in range(12):
            x = xs * np.cos(delta[:, :, None]) + np.cross(ys, xs) * np.sin(delta[:, :, None])
            dx = np.cross(ys, x)
            res = x[:, 0] - x[:, 1]
            temporal = gaussian_filter1d(delta, sigma, axis=0, mode='nearest', truncate=3)
            g0 = delta[:, 0] + tau * (delta[:, 0] - temporal[:, 0]) + lam * np.sum(dx[:, 0] * res, axis=-1)
            g1 = delta[:, 1] + tau * (delta[:, 1] - temporal[:, 1]) - lam * np.sum(dx[:, 1] * res, axis=-1)
            h0 = 1 + tau + lam * np.sum(dx[:, 0] ** 2, axis=-1)
            h1 = 1 + tau + lam * np.sum(dx[:, 1] ** 2, axis=-1)
            h01 = -lam * np.sum(dx[:, 0] * dx[:, 1], axis=-1)
            det = h0 * h1 - h01 * h01
            step = np.stack([(h1 * g0 - h01 * g1) / det, (h0 * g1 - h01 * g0) / det], 1)
            delta -= 0.8 * np.clip(step, -0.15, 0.15)
        if source is not None:
            delta = np.radians(5.0) * np.tanh(delta / np.radians(5.0))
            for (s, (b, ch)) in enumerate(((u, m), (m, e))):
                D = reference_frame(rest[ch] - rest[b])
                have = source[:, b] @ D[:, 2]
                now = world[:, b] @ D[:, 2]
                beta = np.arctan2(np.sum(ys[:, s] * np.cross(have, now), axis=-1), np.sum(have * now, axis=-1))
                total = beta + delta[:, s]
                filtered = np.arctan2(gaussian_filter1d(np.sin(total), max(0.5, fps * 0.1), mode='nearest', truncate=3), gaussian_filter1d(np.cos(total), max(0.5, fps * 0.1), mode='nearest', truncate=3))
                out[:, b] = Rotation.from_rotvec(ys[:, s] * filtered[:, None]).as_matrix() @ source[:, b]
        else:
            for (s, b) in enumerate((u, m)):
                out[:, b] = Rotation.from_rotvec(ys[:, s] * delta[:, s, None]).as_matrix() @ world[:, b]
        corrections.append(delta)
    return (out, np.concatenate(corrections, axis=1))
