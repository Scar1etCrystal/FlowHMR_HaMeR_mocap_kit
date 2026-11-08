"""Source-proportioned Mixamo hierarchy, FK and a rigid beveled mannequin.

No inference or file I/O here. The first 22 joints keep the source body ordering;
the thirty finger joints are real channels, not display-only embellishments.
"""
from __future__ import annotations
import numpy as np

VERSION = 10
BODY_NAMES = ('Hips','LeftUpLeg','RightUpLeg','Spine','LeftLeg','RightLeg','Spine1',
    'LeftFoot','RightFoot','Spine2','LeftToeBase','RightToeBase','Neck','LeftShoulder',
    'RightShoulder','Head','LeftArm','RightArm','LeftForeArm','RightForeArm','LeftHand','RightHand')
BODY_PARENTS = (-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19)
HINGES = ((1,4,7,(1,0,0)), (2,5,8,(1,0,0)),
          (16,18,20,(0,-1,0)), (17,19,21,(0,1,0)))
# Rounded sagittal directions from the user's Taunt Mixamo bind pose. The
# frontal centerline is straight; each actor retains its own segment lengths.
SPINE_PROFILE = ((0,3,.013),(3,6,-.149),(6,9,-.149),(9,12,-.122),(12,15,.175))
TRUNK_POSE_JOINTS = (0,3,6,9,12,13,14,15)


def unit(v, fallback=(0,1,0)):
    v=np.asarray(v,dtype=float);n=np.linalg.norm(v,axis=-1,keepdims=True)
    return np.where(n>1e-10,v/np.maximum(n,1e-10),np.asarray(fallback))


def perpendicular(vector,direction,fallback):
    projected=np.asarray(vector)-direction*(np.asarray(vector)@direction)
    if np.linalg.norm(projected)<1e-8:
        projected=np.asarray(fallback)-direction*(np.asarray(fallback)@direction)
    if np.linalg.norm(projected)<1e-8:
        basis=np.eye(3)[np.argmin(np.abs(direction))];projected=basis-direction*(basis@direction)
    return unit(projected)


def align(a,b):
    """Proper shortest rotation; deterministic at the antiparallel singularity."""
    a=unit(a);b=unit(b);v=np.cross(a,b);c=float(np.clip(a@b,-1,1))
    if c < -1+1e-8:
        axis=unit(np.cross(a,np.eye(3)[np.argmin(np.abs(a))]))
        return 2*np.outer(axis,axis)-np.eye(3)
    skew=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+skew+skew@skew/max(1+c,1e-10)


def fit_frame(reference,desired):
    """Proper rigid frame from several anatomical vectors; no scale change."""
    covariance=np.einsum('vi,tvj->tij',reference,desired)
    u,_,vt=np.linalg.svd(covariance);v=vt.transpose(0,2,1)
    v[:,:,2]*=np.linalg.det(v@u.transpose(0,2,1))[:,None]
    return v@u.transpose(0,2,1)


def skeleton(source_rest,source_joints=None,source_anchors=False):
    rest=np.asarray(source_rest,dtype=float)
    if rest.shape!=(22,3) or not np.isfinite(rest).all():raise ValueError('Invalid source rest skeleton')
    target=rest.copy();alignment=np.tile(np.eye(3),(22,1,1))
    for parent,child,slope in SPINE_PROFILE:
        original=rest[child]-rest[parent];direction=unit([0.,1.,slope])
        target[child]=target[parent]+direction*np.linalg.norm(original)
    # A crooked source spine must not roll otherwise neutral hips/collars.
    # Translate branches with their canonical anchor, preserving their offsets.
    for collar in (13,14):target[collar]=target[9]+rest[collar]-rest[9]
    for collar,arm in ((13,16),(14,17)):
        target[arm]=target[collar]+rest[arm]-rest[collar]
    if source_anchors:
        # Canonical neutral spine must not translate both shoulders away from
        # their inferred pivots and make a bent source arm falsely unreachable.
        target[[13,14,16,17]]=rest[[13,14,16,17]]
    # A true target T-pose. Rest alignment belongs in animation, never in a
    # renderer-only roll adjustment. All target limb lengths equal the source.
    for upper,middle,end,axis in HINGES:
        direction=np.array([0.,-1.,0.]) if upper<3 else np.array([1. if upper==16 else -1.,0.,0.])
        for parent,child in ((upper,middle),(middle,end)):
            original=rest[child]-rest[parent]
            target[child]=target[parent]+direction*np.linalg.norm(original)
            alignment[parent]=align(direction,original)
        if upper<3:
            toe=10 if end==7 else 11
            target[toe]=target[end]+rest[toe]-rest[end]
    names=list(BODY_NAMES);parents=list(BODY_PARENTS);ends=np.zeros((52,3))
    h=sum(np.linalg.norm(rest[b]-rest[a]) for a,b in ((0,3),(3,6),(6,9),(9,12)))
    scale=max(.4,h/.49)
    positions=list(target)
    # Mixamo uses three animated phalanges and a terminal tip, not three
    # equal rods. Reference proportions are rounded from the user's Taunt bind
    # pose; registered body anatomy supplies each actor's actual hand lengths.
    finger_profile=(('Thumb',.027,.025,.026,(.042,.034,.026),37,.76),
        ('Index',.023,.091,-.005,(.037,.029,.028),25,.97),
        ('Middle',0,.095,0,(.037,.030,.030),28,1.),
        ('Ring',-.019,.091,0,(.032,.030,.026),34,.89),
        ('Pinky',-.038,.081,-.005,(.036,.021,.021),31,1.01))
    for side,hand,sign in (('Left',20,1),('Right',21,-1)):
        for finger_number,(finger,z,x,y,default,index,tip_ratio) in enumerate(finger_profile):
            direction=unit([sign*.77,-.45,.46]) if finger=='Thumb' else np.array([sign,0.,0.])
            first=target[hand]+np.array([sign*x,y,z])*scale
            lengths=np.array(default)*scale
            chain=None
            if source_joints is not None:
                source=np.asarray(source_joints);index+=15 if hand==21 else 0
                first=target[hand]+source[index]-source[hand]
                lengths[:2]=np.linalg.norm(np.diff(source[index:index+3],axis=0),axis=1)
                lengths[2]=lengths[1]*tip_ratio
                # Raw SMPL-X already has a fixed relaxed hand mean. Bake that
                # static shape into the target's offsets, instead of replacing
                # it with straight fingers. Neutral channels still mean no
                # finger tracking; all three source joint positions now match.
                chain=target[hand]+source[index:index+3]-source[hand]
                direction=unit(chain[2]-chain[1])
            parent=hand;point=first
            for segment in range(3):
                index=len(names);names.append(side+'Hand'+finger+str(segment+1));parents.append(parent)
                if chain is not None:point=chain[segment]
                positions.append(point.copy());ends[index]=(chain[segment+1]-point) if chain is not None and segment<2 else direction*lengths[segment]
                if chain is not None and segment==2 and len(source)>75:
                    tip=66+finger_number+(5 if hand==21 else 0)
                    ends[index]=source[tip]-source[hand]+target[hand]-point
                parent=index;point=point+direction*lengths[segment]
    ends[15]=[0,.14*scale,0]
    ends[10]=ends[11]=[0,0,.075*scale]
    return dict(names=names,parents=np.asarray(parents,np.int32),
        rest=np.asarray(positions,np.float32),alignment=alignment,ends=ends.astype(np.float32),scale=scale)


def fk(rig,root,local):
    root=np.asarray(root);local=np.asarray(local)
    rotations=local.copy();positions=np.empty((*local.shape[:-3],len(rig['parents']),3))
    positions[...,0,:]=root
    for j,p in enumerate(rig['parents']):
        if p<0:continue
        positions[...,j,:]=positions[...,p,:]+np.einsum('...ij,j->...i',rotations[...,p,:,:],rig['rest'][j]-rig['rest'][p])
        rotations[...,j,:,:]=rotations[...,p,:,:]@local[...,j,:,:]
    return positions,rotations


def points_from_world(rig,root,world):
    positions=np.empty((len(root),len(rig['parents']),3));positions[:,0]=root
    for j,p in enumerate(rig['parents']):
        if p>=0:positions[:,j]=positions[:,p]+np.einsum('tij,j->ti',world[:,p],rig['rest'][j]-rig['rest'][p])
    return positions


def from_source(rig,root,world,pole_state=None,constrain=True,landmarks=None,anatomical_legs=False,anatomical_arms=False):
    n=len(root);g=np.tile(np.eye(3),(n,52,1,1))
    g[:,:22]=np.asarray(world)@rig['alignment']
    # Transfer torso pose relative to neutral, rather than aiming back at the
    # source's staggered rest joints. Zero source pose must remain zero target
    # torso pose; otherwise straight BVH offsets alone merely hide the defect.
    g[:,TRUNK_POSE_JOINTS]=np.asarray(world)[:,TRUNK_POSE_JOINTS]
    for j in range(22,52):g[:,j]=g[:,rig['parents'][j]]
    points=points_from_world(rig,root,g)
    local=g.copy()
    for j,p in enumerate(rig['parents']):
        if p>=0:local[:,j]=g[:,p].transpose(0,2,1)@g[:,j]
    poles=g[...,0][:,[1,2]].copy() if anatomical_legs else None
    arm_poles=np.stack((-g[:,16,:,1],g[:,17,:,1]),axis=1) if anatomical_arms else None
    return constrain_hinges(rig,root,points,g,pole_state,landmarks,leg_poles=poles,arm_poles=arm_poles) if constrain else (local,points,g)


def transport_leg_poles(reference,points):
    """Minimal leg swing transports the pre-correction hinge, without roll.

    reference: N x 2 legs x (hip-ankle direction, anatomical hinge normal) x 3.
    The knee's local X alone is insufficient: re-rolling its thigh can pass a
    local hinge check while turning an added knee bend sideways in the body.
    """
    directions=unit(points[:,[7,8]]-points[:,[1,2]])
    result=np.empty_like(directions,dtype=float)
    for t in range(len(points)):
        for foot in range(2):
            result[t,foot]=align(reference[t,foot,0],directions[t,foot])@reference[t,foot,1]
    return result


def constrain_hinges(rig,root,points,rotations,pole_state=None,landmarks=None,leg_poles=None,arm_poles=None,preserve_arms=False):
    n=len(root);rotations=np.array(rotations,copy=True);local=rotations.copy()
    if leg_poles is not None and (np.shape(leg_poles)!=(n,2,3) or not np.isfinite(leg_poles).all()):
        raise ValueError('Invalid anatomical leg poles')
    if arm_poles is not None and (np.shape(arm_poles)!=(n,2,3) or not np.isfinite(arm_poles).all()):
        raise ValueError('Invalid anatomical arm poles')
    state=pole_state if pole_state is not None else {}
    # A fitted target has explicit hinge frames. Axial rotation belongs at the
    # limb root and wrist/ankle, not an independently rolled elbow/knee box.
    for upper,middle,end,axis in HINGES:
        # Contact correction does not solve arms. Re-estimating their planes
        # from rounded/near-collinear endpoints can add a second, spurious roll.
        if preserve_arms and upper in (16,17):continue
        u0=unit(rig['rest'][middle]-rig['rest'][upper]);l0=unit(rig['rest'][end]-rig['rest'][middle])
        a=float(np.linalg.norm((rig['rest'][middle]-rig['rest'][upper]).astype(float)))
        b=float(np.linalg.norm((rig['rest'][end]-rig['rest'][middle]).astype(float)))
        # Never infer an anatomical axis from tiny rest-pose curvature: SMPL
        # knees often curve sideways, so that cross product defines a sideways
        # hinge and visibly rolls the cuboid cross-sections. Canonical straight
        # limbs share one explicit signed axis in BOTH target joint frames.
        normal0=np.asarray(axis,dtype=float)
        for t in range(n):
            observed_points=points if landmarks is None else landmarks
            u=unit(observed_points[t,middle]-points[t,upper]);v=unit(observed_points[t,end]-observed_points[t,middle])
            observed=np.cross(u,v);amount=np.linalg.norm(observed)
            delta=observed_points[t,end]-points[t,upper];direction=unit(delta)
            observed_height=np.linalg.norm(np.cross(observed_points[t,middle]-points[t,upper],direction))
            hint=perpendicular(rotations[t,upper]@normal0,u,rotations[t,upper]@np.cross(u0,normal0))
            # Plane estimates become ill-conditioned well before exact zero.
            # Transport the previous pole with the proximal segment's measured
            # motion. This state is baked once, never reconstructed on seek.
            previous=state.get(upper)
            if previous is not None:
                old_normal,old_frame,old_axis=previous
                transported=rotations[t,upper]@old_frame.T@old_normal
                transported=perpendicular(transported,u,hint)
                hint=unit(.8*transported+.2*hint)
            candidate=unit(observed)
            # Confidence is a positional quantity, not a broad bend-angle
            # threshold. Outside the near-straight uncertainty zone, preserve
            # the measured middle joint plane instead of inventing a new one.
            low=(a+b)*.01;high=(a+b)*.03
            weight=float(np.clip((observed_height-low)/(high-low),0,1));weight=weight*weight*(3-2*weight)
            if candidate@hint<0 and observed_height<high:weight=0.
            normal=perpendicular((1-weight)*hint+weight*candidate,u,hint)
            if previous is not None:
                theta=float(np.arccos(np.clip(normal@transported,-1,1)))
                axis_step=float(np.arccos(np.clip(u@old_axis,-1,1)))
                if theta>np.deg2rad(25) and axis_step<np.deg2rad(30) and observed_height<high:
                    normal=unit(transported+unit(normal-transported*(normal@transported))*np.tan(np.deg2rad(25)))
            if leg_poles is not None and upper in (1,2):
                # Source SMPL-X rest curvature / knee YZ motion is not a valid
                # reason to roll the thigh so the knee can bend sideways.
                # Preserve the anatomical femur frame before correction, then
                # only swing that frame as the foot target moves.
                normal=perpendicular(leg_poles[t,upper-1],direction,hint)
            if arm_poles is not None and upper in (16,17):
                # A local elbow hinge is insufficient if its whole upper arm
                # was rolled to follow a noisy observed elbow plane. Preserve
                # the calibrated proximal anatomical pole, as for the legs.
                # Transport by the shortest shoulder-to-goal swing, rather
                # than projecting a pole that can become parallel to a goal.
                # In-plane elbow flexion leaves this normal unchanged.
                normal=align(rotations[t,upper]@u0,direction)@arm_poles[t,upper-16]
            # Reconstruct the lower axis in the one allowed bend plane.
            # Never preserve a noisy endpoint by introducing impossible roll.
            # Preserve the wrist/ankle endpoint on EVERY path, including GVHMR
            # and the final post-correction projection. Rotating the lower axis
            # around an unadjusted upper direction used to move that endpoint.
            distance=float(np.clip(np.linalg.norm(delta),abs(a-b)+1e-7,a+b-1e-7))
            normal=perpendicular(normal,direction,hint)
            cosine=float(np.clip((a*a+distance*distance-b*b)/(2*a*distance),-1,1))
            u=direction*cosine-np.cross(normal,direction)*np.sqrt(max(0,1-cosine*cosine))
            angle=float(np.arccos(np.clip((distance*distance-a*a-b*b)/(2*a*b),-1,1)))
            v=u*np.cos(angle)+np.cross(normal,u)*np.sin(angle)
            state[upper]=(normal.copy(),rotations[t,upper].copy(),u.copy())
            for j,direction,reference in ((upper,u,u0),(middle,v,l0)):
                normal_j=unit(normal-direction*(normal@direction))
                rest_normal=unit(normal0-reference*(normal0@reference))
                now=np.column_stack((direction,normal_j,np.cross(direction,normal_j)))
                before=np.column_stack((reference,rest_normal,np.cross(reference,rest_normal)))
                rotations[t,j]=now@before.T
    for j,p in enumerate(rig['parents']):
        local[:,j]=rotations[:,j] if p<0 else rotations[:,p].transpose(0,2,1)@rotations[:,j]
    points,rotations=fk(rig,root,local)
    return local,points,rotations


def rounded_box(size,radius=None):
    """Six sampled rounded faces, analytic normals, fixed two-step bevels."""
    half=np.asarray(size,dtype=float)*.5
    radius=float(min(half)*.30 if radius is None else min(radius,min(half)*.8))
    inner=half-radius;vertices=[];normals=[];faces=[]
    grid=[np.array([-h,-h+radius*.293,-h+radius,h-radius,h-radius*.293,h]) for h in half]
    for axis in range(3):
        u=(axis+1)%3;v=(axis+2)%3
        for sign in (-1,1):
            base=len(vertices)
            for a in grid[u]:
                for b in grid[v]:
                    p=np.zeros(3);p[axis]=sign*half[axis];p[u]=a;p[v]=b
                    center=np.clip(p,-inner,inner);normal=unit(p-center)
                    vertices.append(center+radius*normal);normals.append(normal)
            for a in range(5):
                for b in range(5):
                    i=base+a*6+b
                    triangles=((i,i+6,i+7),(i,i+7,i+1))
                    faces.extend(triangles if sign>0 else [tuple(reversed(t)) for t in triangles])
    return np.asarray(vertices),np.asarray(normals),np.asarray(faces,np.uint32)


def mannequin(rig,sole_y=None,anatomy=None):
    rest=rig['rest'];s=rig['scale'];parts=[];probes=[];regions={}
    if anatomy is not None:
        vertices,joints,skin=(np.asarray(anatomy[k]) for k in ('vertices','joints','skin'))
        dominant=skin.argmax(1)
        for j in range(22):
            region=vertices[dominant==j]-joints[j]
            if len(region)>=6:regions[j]=region@rig['alignment'][j]
    def bounds(j):
        region=regions.get(j)
        if region is None:return None
        low,high=np.quantile(region,[.01,.99],axis=0)
        return (low+high)*.5,np.maximum(high-low,.008*s)
    def box(j,center,size,basis=np.eye(3),contact=False):
        vertices,normals,faces=rounded_box(size)
        parts.append((vertices@basis.T+center,normals@basis.T,faces,j,contact))
    def segment(j,end,width,depth):
        delta=np.asarray(end);length=np.linalg.norm(delta)
        if length<1e-5:return
        # SMPL's curved rest-spine chords are not the body surface's axes.
        # Aiming a cuboid along those short chords produced apparent torso
        # twists even with only a few degrees of actual joint rotation.
        torso=j in (0,3,6,9)
        basis=np.eye(3) if torso else align([0,1,0],delta)
        extent=abs(delta[1]) if torso else length
        center=delta*.5
        if j in regions and torso:
            low,high=np.quantile(regions[j]@basis,[.02,.98],axis=0)
            width,depth=float(high[0]-low[0]),float(high[2]-low[2])
            cross=(low+high)*.5;cross[1]=0;center=center+basis@cross
            if torso:
                center[1]=(low[1]+high[1])*.5
                extent=max(.015*s,float(high[1]-low[1])-.004*s)
        if torso:depth=.10*s
        box(j,center,[max(.008*s,width),max(.006,extent*.97),max(.008*s,depth)],basis)
    # Slim hip connectors reveal the actual three-way pelvis joint instead of
    # burying its pivots in a broad solid block. Body boxes remain substantial.
    segment(0,rest[3]-rest[0],.16*s,.10*s)
    for child in (1,2):
        delta=rest[child]-rest[0];box(0,delta*.5,[.026*s,np.linalg.norm(delta)*.97,.026*s],align([0,1,0],delta))
    for j,child,width,depth in ((3,6,.19,.10),(6,9,.23,.10),(9,12,.30,.10),(12,15,.027,.027),
            (13,16,.026,.026),(14,17,.026,.026),(16,18,.032,.032),(17,19,.032,.032),
            (18,20,.026,.026),(19,21,.026,.026),(1,4,.044,.044),(2,5,.044,.044),(4,7,.034,.034),(5,8,.034,.034)):
        segment(j,rest[child]-rest[j],width*s,depth*s)
    head=bounds(15)
    size=head[1].copy() if head else np.array([.17*s,.215*s,.14*s]);size[2]=.14*s
    box(15,head[0] if head else [0,.060*s,0],size)
    for hand,sign in ((20,1),(21,-1)):
        palm=bounds(hand)
        size=palm[1].copy() if palm else np.array([.09*s,.022*s,.083*s]);size[1]=min(size[1],.022*s)
        center=palm[0] if palm else np.array([sign*.045*s,0,0])
        segment(hand,center,.015*s,.015*s)
        box(hand,center,size)
    for j in range(22,52):segment(j,rig['ends'][j],.0075*s,.009*s)
    for foot,toe in ((7,10),(8,11)):
        # Foot sole is defined by the displayed geometry, not the ankle joint.
        delta=rest[toe]-rest[foot];length=max(.16*s,float(abs(delta[2]))+.065*s)
        bottom=float(sole_y[foot] if sole_y is not None else rest[foot,1]-.065*s)
        center=np.array([0,bottom-rest[foot,1]+.027*s,.025*s])
        size=np.array([.083*s,.054*s,length])
        measured=bounds(foot);toe_measured=bounds(toe)
        if measured is not None:
            center,size=measured
            bottom=rest[foot,1]+center[1]-size[1]*.5
            # Flat sole at the measured rest surface, not at the ankle pivot.
            if toe_measured is not None:bottom=min(bottom,float(rest[toe,1]+toe_measured[0][1]-toe_measured[1][1]*.5))
            size[1]=max(.012*s,float(center[1]+size[1]*.5-(bottom-rest[foot,1])))
            center[1]=bottom-rest[foot,1]+size[1]*.5
        old_bottom=center[1]-size[1]*.5;size[1]=min(size[1],.032*s);center[1]=old_bottom+size[1]*.5
        top=center[1]+size[1]*.5
        if top<-.001*s:segment(foot,np.array([0,top,0]),.022*s,.022*s)
        box(foot,center,size,contact=True)
        for z in (-.28,.28):probes.append((foot,center+np.array([0,-size[1]*.5,z*length])))
        # A modest toe section, no decorative front/back blocks.
        if toe_measured is not None:
            c,z=toe_measured;z[1]=max(.012*s,float(c[1]+z[1]*.5-(bottom-rest[toe,1])));c[1]=bottom-rest[toe,1]+z[1]*.5
            old_bottom=c[1]-z[1]*.5;z[1]=min(z[1],.025*s);c[1]=old_bottom+z[1]*.5
            box(toe,c,z,contact=True)
            rig['ends'][toe]=[0,0,max(.025*s,float(c[2]+z[2]*.5))]
        else:box(toe,[0,bottom-rest[toe,1]+.025*s,.025*s],[.083*s,.046*s,.085*s],contact=True)
    vertices=[];normals=[];faces=[];bones=[];contact_vertices=[]
    for v,n,f,j,contact in parts:
        faces.append(f+sum(len(x) for x in vertices));vertices.append(v);normals.append(n);bones.extend([j]*len(v));contact_vertices.extend([contact]*len(v))
    packed=np.column_stack((np.concatenate(vertices),np.concatenate(normals),np.asarray(bones))).astype(np.float32)
    sole_ids=[];sole_groups=[]
    eligible=np.asarray(contact_vertices,dtype=bool)
    for foot,toe in ((7,10),(8,11)):
        middle=float(np.median(packed[(packed[:,6]==foot)&eligible,2]))
        for front in (False,True):
            ids=np.flatnonzero(eligible&(((packed[:,6]==foot)&((packed[:,2]>=middle) if front else (packed[:,2]<=middle))) |
                ((packed[:,6]==toe)&front)))
            sole_ids.extend(ids);sole_groups.append(len(sole_ids))
    return dict(packed=packed,faces=np.concatenate(faces),vertex_count=len(packed),
        probe_bones=np.array([p[0] for p in probes]),probe_local=np.array([p[1] for p in probes],np.float32),
        sole_ids=np.asarray(sole_ids,np.int32),sole_groups=np.asarray(sole_groups,np.int32))


def sole_samples(asset,positions,rotations):
    vertices=asset['packed'][asset['sole_ids'],:3]
    bones=asset['packed'][asset['sole_ids'],6].astype(int)
    points=positions[:,bones]+np.einsum('tvij,vj->tvi',rotations[:,bones],vertices)
    result=[];first=0
    for end in asset['sole_groups']:
        patch=points[:,first:end];value=patch.mean(1);value[:,1]=patch[:,:,1].min(1)
        result.append(value);first=end
    return np.stack(result,axis=1)


def write_bvh(path,rig,root,local,fps):
    from scipy.spatial.transform import Rotation
    from .bvh_bind import bind_pose,local_frames
    import warnings
    if not np.isfinite(fps) or fps<=0 or not len(root) or local.shape!=(len(root),52,3,3) or not np.isfinite(local).all() or not np.isfinite(root).all():raise ValueError('Invalid skeleton animation')
    binding=bind_pose(rig);local=local_frames(local,binding);rig=binding
    # A native Blender bone always points along its local Y. Both canonical
    # knee and elbow hinges therefore import as local X flexion. Putting X in
    # the middle of its native Euler order causes an importer singularity at
    # 90 degrees, even when the BVH contains an exact one-axis rotation.
    # Keep ordinary joints/root in ZXY and true hinges in ZYX (X last).
    pure_hinges={}
    for _,middle,_,axis in HINGES:
        rv=Rotation.from_matrix(local[:,middle]).as_rotvec();axis=np.asarray(axis)
        if np.linalg.norm(rv-(rv@axis)[:,None]*axis,axis=1).max()<1e-6:
            pure_hinges[middle]=rv
    order=[];lines=['HIERARCHY']
    def visit(j,depth):
        pad='  '*depth;order.append(j);parent=rig['parents'][j]
        lines.extend([pad+('ROOT ' if parent<0 else 'JOINT ')+rig['names'][j],pad+'{'])
        delta=rig['rest'][j] if parent<0 else rig['rest'][j]-rig['rest'][parent]
        lines.append(pad+'  OFFSET '+' '.join(f'{v:.9f}' for v in delta))
        rotation_channels='Zrotation Yrotation Xrotation' if j in pure_hinges else 'Zrotation Xrotation Yrotation'
        lines.append(pad+'  CHANNELS '+('6 Xposition Yposition Zposition ' if parent<0 else '3 ')+rotation_channels)
        children=np.flatnonzero(rig['parents']==j)
        for child in children:visit(int(child),depth+1)
        if not len(children):lines.extend([pad+'  End Site',pad+'  {',pad+'    OFFSET '+' '.join(f'{v:.9f}' for v in rig['ends'][j]),pad+'  }'])
        lines.append(pad+'}')
    visit(0,0)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Gimbal lock detected')
        angles=Rotation.from_matrix(local.reshape(-1,3,3)).as_euler('ZXY').reshape(-1,52,3)
    # A hinge beyond 90 degrees must not become Z=180 / X=70 / Y=180.
    # Those are equivalent keyframes, but interpolating the three channels
    # invents a large twist between samples. Keep true hinges on one channel.
    for middle,rv in pure_hinges.items():
        # Pure rotations need no equivalent-chart choice or Euler decomposition.
        axis=next(axis for _,joint,_,axis in HINGES if joint==middle)
        xyz=int(np.argmax(np.abs(axis)));channel={0:2,1:1,2:0}[xyz]
        angles[:,middle]=0.;angles[:,middle,channel]=rv[:,xyz]
    # Choose equivalent ZXY charts continuously for free wrists, ankles and
    # body joints too, instead of merely unwrapping each independent channel.
    previous=np.zeros((52,3))
    for t in range(len(angles)):
        a=angles[t].copy();b=a.copy();b[:,0]+=np.pi;b[:,1]=np.pi-b[:,1];b[:,2]+=np.pi
        a+=2*np.pi*np.round((previous-a)/(2*np.pi));b+=2*np.pi*np.round((previous-b)/(2*np.pi))
        use_b=np.sum((b-previous)**2,axis=1)<np.sum((a-previous)**2,axis=1)
        for middle in pure_hinges:
            a[middle]=angles[t,middle];use_b[middle]=False
        angles[t]=np.where(use_b[:,None],b,a);previous=angles[t]
    angles=np.rad2deg(angles)
    lines.extend(['MOTION',f'Frames: {len(root)}',f'Frame Time: {1/fps:.10f}'])
    with open(path,'w',encoding='ascii') as out:
        out.write('\n'.join(lines)+'\n')
        # Blender's importer interprets root translation channels as absolute
        # positions and subtracts the root REST offset itself. Do not subtract
        # it here a second time or change the animated world-space trajectory.
        for r,a in zip(root,angles):out.write(' '.join(f'{v:.9f}' for v in np.r_[r,a[order].reshape(-1)])+'\n')
