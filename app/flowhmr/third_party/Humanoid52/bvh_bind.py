"""Pose-preserving BVH reference frames, in metres for Blender import.

The rounded hand/head directions were measured from the supplied Taunt FBX
bind pose. Source-animation limbs and feet retain SMPL neutral geometry.
No FBX geometry, animation, or Blender implementation is embedded.
The inference/fitting rig stays unchanged: this is a constant change of basis.
"""
from __future__ import annotations
import numpy as np

VERSION = 6


def bind_pose(rig):
    from .mixamo_rig import align,unit
    rest=np.asarray(rig['rest'],dtype=float);ends=np.asarray(rig['ends'],dtype=float)
    parents=np.asarray(rig['parents']);names=[name.rsplit(':',1)[-1] for name in rig['names']]
    if rest.shape!=(52,3) or ends.shape!=(52,3) or parents.shape!=(52,) or len(names)!=52:
        raise ValueError('Expected a complete 52-bone export skeleton')
    if not np.isfinite(rest).all() or not np.isfinite(ends).all() or parents[0]!=-1 or any(not 0<=parents[j]<j for j in range(1,52)):
        raise ValueError('Invalid export reference skeleton')
    if len(set(names))!=52 or any(not name.isascii() or not name or any(c.isspace() or c in '{}:' for c in name) for name in names):
        raise ValueError('Invalid export bone names')
    basis=np.tile(np.eye(3),(52,1,1))
    if rig.get('source_animation_version',0)>=15:
        # The internal fitter uses straight limbs. Its constant alignment maps
        # each straight segment into the registered source-neutral direction.
        # Undo that basis at export, preserving every animated point and the
        # existing roll stabilisation rather than solving the motion again.
        alignment=np.asarray(rig['alignment'],dtype=float)
        if alignment.shape!=(22,3,3) or not np.isfinite(alignment).all() or np.max(abs(alignment.swapaxes(-1,-2)@alignment-np.eye(3)))>1e-6 or np.linalg.det(alignment).min()<.999999:
            raise ValueError('Invalid source-neutral limb alignment')
        for j in (1,2,4,5,16,17,18,19):
            basis[j]=alignment[j].T
    for side,hand,sign in (('Left',20,1),('Right',21,-1)):
        middle=names.index(side+'HandMiddle1')
        palm=rest[middle]-rest[hand]
        if np.linalg.norm(palm)>1e-8:
            basis[hand]=align([sign,0,0],palm)
        for finger in ('Thumb','Index','Middle','Ring','Pinky'):
            ids=[names.index(side+'Hand'+finger+str(k)) for k in (1,2,3)]
            direction=unit([sign*.769,-.447,.458]) if finger=='Thumb' else np.array([sign,0.,0.])
            for k,j in enumerate(ids):
                outgoing=rest[ids[k+1]]-rest[j] if k<2 else ends[j]
                if np.linalg.norm(outgoing)<1e-8:raise ValueError('A finger reference segment has zero length')
                # Open-hand reference, including the Taunt thumb direction.
                # Constant frame transport preserves the animated source hand.
                basis[j]=align(direction,outgoing)
    # Source-animation rigs already contain the actor's registered neutral
    # ankle-to-toe vectors. Keep those instead of imposing the FBX foot pitch.
    # Legacy/contact rigs retain their reference conversion.
    if rig.get('source_animation_version',0)<15:
        for foot,toe in ((7,10),(8,11)):
            basis[foot]=align([0.,-.632,.775],rest[toe]-rest[foot])
    if np.linalg.norm(ends[15])>1e-8:basis[15]=align([0.,.993,-.121],ends[15])
    if rig.get('source_animation_version',0)>=17:
        from .bone_overlay import reference_frame
        # Match the approved surface-aligned bind. Roll and curvature are
        # transported through one constant per-segment basis; no pose is added.
        for upper,middle,end,target_direction in ((1,4,7,[0,-1,0]),(2,5,8,[0,-1,0]),
                                                   (16,18,20,[1,0,0]),(17,19,21,[-1,0,0])):
            for j,ch in ((upper,middle),(middle,end)):
                neutral_delta=basis[j].T@(rest[ch]-rest[j])
                conversion=reference_frame(neutral_delta)@reference_frame(np.asarray(target_direction,float)).T
                basis[j]=basis[j]@conversion
    target=rest.copy()
    for j,p in enumerate(parents):
        if p>=0:target[j]=target[p]+basis[p].T@(rest[j]-rest[p])
    tips=np.einsum('jik,jk->ji',basis.swapaxes(-1,-2),ends)
    # Hips is the root (there is no extra ground/root bone). The reference
    # feet/toe joints stand at Y=0, centred between the ankle pivots in X/Z.
    # This offset is REST ONLY: animated roots remain the original world path.
    origin=np.array([target[[7,8],0].mean(),min(target[10,1],target[11,1]),target[[7,8],2].mean()])
    target-=origin
    return dict(names=names,parents=parents.copy(),rest=target,ends=tips,
        basis=basis,origin=origin,scale=rig.get('scale',1.),bind_version=VERSION)


def local_frames(local,binding):
    """O(frames*52), independently chunkable constant basis conversion."""
    basis=binding['basis'];parents=binding['parents'];result=np.asarray(local)@basis
    for j,p in enumerate(parents):
        if p>=0:result[...,j,:,:]=basis[p].T@result[...,j,:,:]
    return result


def asset_frames(packed,binding):
    """Transport local display vertices/normals with their bone basis."""
    packed=np.asarray(packed).copy();ids=packed[:,6].astype(int)
    inverse=binding['basis'][ids].swapaxes(-1,-2)
    packed[:,:3]=np.einsum('vij,vj->vi',inverse,packed[:,:3])
    packed[:,3:6]=np.einsum('vij,vj->vi',inverse,packed[:,3:6])
    return packed
