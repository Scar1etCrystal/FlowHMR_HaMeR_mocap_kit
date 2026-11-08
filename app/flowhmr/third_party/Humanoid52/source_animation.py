"""Source-faithful animation rig, separate from the contact solver's rig.

Straight rest limbs retain the actor's lengths. Constant source bases transport
the original animation; only outgoing limb roll is stabilised. Joint positions,
root/trunk animation, distal world frames and the original surface are retained.
This deliberately replaces the export fitter's hard Cartesian hinge projection.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from .mixamo_rig import HINGES, align, fk, unit

FIT_VERSION = 17


def temporal_halo(fps=30.):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid animation frame rate')
    # Sum the exact finite radii along the longest dependency path: v16
    # baseline, confidence, common axis, confidence roll, 12 paired iterations,
    # and final source-relative roll. No hidden dependence on previous chunks.
    radius = lambda seconds: int(3 * max(.5, fps * seconds) + .5)
    return radius(.05) + radius(.035) + 3 * radius(.10) + 12 * radius(.06)


def source_rig(original, source_rest):
    """Keep registered torso/pivots and fingers; straighten only rest limbs."""
    rig = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in original.items()}
    old = rig['rest'].copy()
    rest = rig['rest'] = np.asarray(rig['rest'], dtype=float).copy()
    source_rest = np.asarray(source_rest, dtype=float)
    rest[:22] = source_rest[:22] - source_rest[0]
    for hand, ids in ((20, range(22, 37)), (21, range(37, 52))):
        rest[list(ids)] += rest[hand] - old[hand]
    source = rest.copy()
    basis = np.tile(np.eye(3), (22, 1, 1))
    for upper, middle, end, _ in HINGES:
        direction = np.array([0., -1., 0.]) if upper < 3 else np.array([1. if upper == 16 else -1., 0., 0.])
        for joint, child in ((upper, middle), (middle, end)):
            delta = source[child] - source[joint]
            if np.linalg.norm(delta) < 1e-6:
                raise ValueError('Registered limb has zero length')
            rest[child] = rest[joint] + direction * np.linalg.norm(delta)
            basis[joint] = align(direction, delta)
        descendants = [end + 3] if end in (7, 8) else list(range(22, 37) if end == 20 else range(37, 52))
        rest[descendants] += rest[end] - source[end]
    rig['alignment'] = basis
    rig['source_animation_version'] = FIT_VERSION
    return rig


def _frame(direction, pole, fallback):
    """A proper outgoing frame, including degenerate or opposed pole hints."""
    y = unit(direction)
    x = pole - y * np.sum(pole * y, axis=-1, keepdims=True)
    alt = fallback - y * np.sum(fallback * y, axis=-1, keepdims=True)
    x = np.where(np.linalg.norm(x, axis=-1, keepdims=True) > 1e-8, x, alt)
    # The calibrated source axis is perpendicular in normal inputs. Handle a
    # malformed/parallel fallback deterministically without temporal state.
    cardinal = np.eye(3)[np.argmin(abs(y), axis=-1)]
    alt = cardinal - y * np.sum(cardinal * y, axis=-1, keepdims=True)
    x = unit(np.where(np.linalg.norm(x, axis=-1, keepdims=True) > 1e-8, x, alt))
    return np.stack((x, y, unit(np.cross(x, y))), axis=-1)


def validate(local):
    """Validate real matrices; report off-axis residual, never a false hinge pass."""
    local = np.asarray(local)
    if not np.isfinite(local).all() or np.max(abs(local.swapaxes(-1, -2) @ local - np.eye(3))) > 1e-4 or np.linalg.det(local).min() < .9999:
        raise ValueError('Invalid source animation rotation frame')
    return max(float(np.linalg.norm(local[:, m] @ np.asarray(axis) - axis, axis=-1).max()) for _, m, _, axis in HINGES)


def fit(rig, roots, source_world, goals, *, fps=30., surface_endpoints=None, **unused):
    temporal_halo(fps)
    source = np.asarray(source_world, dtype=float)
    if not np.isfinite(source).all() or np.max(abs(source.swapaxes(-1, -2) @ source - np.eye(3))) > 2e-4:
        raise ValueError('Invalid inferred rotation frames')
    source = Rotation.from_matrix(source.reshape(-1, 3, 3)).as_matrix().reshape(source.shape)
    world = np.tile(np.eye(3), (len(source), 52, 1, 1))
    world[:, :22] = source @ rig['alignment']
    # Match the legacy neutral finger transport until full hand poses are added.
    for j in range(22, 52):
        world[:, j] = world[:, rig['parents'][j]]
    local = world.copy()
    for j, parent in enumerate(rig['parents']):
        if parent >= 0:
            local[:, j] = world[:, parent].swapaxes(-1, -2) @ world[:, j]
    points, _ = fk(rig, roots, local)
    reference = world.copy()
    for upper, middle, end, axis in HINGES:
        axis = np.asarray(axis, dtype=float)
        a = unit(points[:, middle] - points[:, upper])
        b = unit(points[:, end] - points[:, middle])
        bend = np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))
        prior = reference[:, upper] @ axis
        observed = unit(np.cross(a, b))
        t = np.clip((bend - 20.) / 25., 0, 1)
        weight = t * t * (3 - 2 * t)
        pole = unit((1 - weight[:, None]) * prior + weight[:, None] * observed)
        for joint, child, direction in ((upper, middle, a), (middle, end, b)):
            rest_frame = _frame(rig['rest'][child] - rig['rest'][joint], axis, axis)
            # A shared bend-plane target can unnecessarily twist the shin or
            # forearm away from its own source frame, especially with curved
            # neutral limbs. Keep each segment's source frame as its baseline.
            # Only a small, temporally regularised outgoing-axis adjustment is
            # allowed; positions and every distal world frame stay exact.
            source_axis = reference[:, joint] @ rest_frame[:, 0]
            target_axis = _frame(direction, pole, prior)[:, :, 0]
            sine = np.sum(direction * np.cross(source_axis, target_axis), axis=-1)
            cosine = np.sum(source_axis * target_axis, axis=-1)
            sigma = max(.5, fps * .05)
            sine = gaussian_filter1d(sine, sigma=sigma, mode='nearest', truncate=3)
            cosine = gaussian_filter1d(cosine, sigma=sigma, mode='nearest', truncate=3)
            angle = np.arctan2(sine, cosine)
            # An ambiguous circular mean has no reliable correction direction.
            angle = np.where(np.hypot(sine, cosine) > 1e-6, angle, 0.)
            limit = np.deg2rad(5.)
            # Sine fades opposed hints back to the source rather than jumping
            # between +/- the limit at the atan2 branch cut.
            angle = limit * np.tanh(.25 * np.sin(angle) / limit)
            adjustment = Rotation.from_rotvec(direction * angle[:, None]).as_matrix()
            world[:, joint] = adjustment @ reference[:, joint]
    for j, parent in enumerate(rig['parents']):
        local[:, j] = world[:, j] if parent < 0 else world[:, parent].swapaxes(-1, -2) @ world[:, j]
    result, world = fk(rig, roots, local)
    validate(local)
    if np.max(abs(result - points)) > 1e-7:
        raise ValueError('Roll stabilisation moved a source joint')
    if np.max(np.linalg.norm(result[:, :22] - goals, axis=-1)) > .00025:
        raise ValueError('Source shape/segment lengths changed; cannot preserve this pose with a fixed-length rig')
    if surface_endpoints is not None:
        from .bvh_bind import bind_pose
        from .surface_roll import confidence_solve, paired
        surface_endpoints = np.asarray(surface_endpoints, dtype=float)
        if surface_endpoints.shape != (len(source), 8, 4, 3) or not np.isfinite(surface_endpoints).all():
            raise ValueError('Invalid source surface direction samples')
        binding = bind_pose(rig)
        exported = world @ binding['basis']
        confidence = confidence_solve(result, exported, binding['rest'], surface_endpoints, fps)[0]
        adjusted = paired(result, confidence, binding['rest'], fps, exported)[0]
        world = adjusted @ binding['basis'].swapaxes(-1, -2)
        for j, parent in enumerate(rig['parents']):
            local[:, j] = world[:, j] if parent < 0 else world[:, parent].swapaxes(-1, -2) @ world[:, j]
        after, world = fk(rig, roots, local)
        if np.max(abs(after - result)) > 1e-7:
            raise ValueError('Surface roll moved a source joint')
        validate(local)
        result = after
    return local, result, world


def policy(fps):
    return dict(fit_version=FIT_VERSION,
        method='Paired joint front/side surface roll with proximal confidence fallback',
        hinge_reference='Surface-aligned bind and animation; multi-axis residuals retained to preserve source positions',
        anatomical_limits_deg=None,
        temporal_fit=dict(halo_frames=temporal_halo(fps), source_rotation_filtering=False,
                          roll_filter_seconds=.10, paired_trust_region_deg=5.,
                          paired_iterations=12, joint_position_budget_m=.00025))
