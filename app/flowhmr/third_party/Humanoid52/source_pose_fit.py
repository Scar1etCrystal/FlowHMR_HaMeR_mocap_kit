"""Source fitting with actual Cartesian elbow and knee hinges.

Solve limb swing, one shared axial rotation and a signed hinge together. No
independent elbow/knee twist is introduced; distal world frames stay unchanged.
Small batched normal equations keep worker memory/time independent of duration.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .mixamo_rig import HINGES,unit,fk,points_from_world,align


ROLL_LIMIT_DEG=60.
FIT_VERSION=14
POSITION_PRIORITY=16.
EXTENSION_ROLL_LIMIT_DEG=10.
EXTENSION_ROLL_PRIOR_SCALE=64.
ELBOW_WRIST_BUDGET_M=.001
ELBOW_AXIS_SLACK_DEG=.01
# Neutral source alignment guides only the proximal frame. Never bake a
# carrying/varus correction into the exported elbow/knee's local rotation.
# Canonical straight limbs use signed +X knees and -Y/+Y elbows; Blender's
# imported bone frames express these as -X knees and +X elbows.
ROTATION_PRIOR_SCALE=.3


def temporal_support(fps=30.):
    """Finite support, so bounded workers and arbitrary seeks agree exactly."""
    if not np.isfinite(fps) or fps<=0:raise ValueError('Invalid source fitting frame rate')
    branch_radius=max(1,min(24,round(.2*fps)))
    slow_sigma=max(.5,min(6.,.1*fps))
    fast_sigma=max(.5,min(2.,fps/30.))
    filter_radius=int(np.ceil(3*max(slow_sigma,fast_sigma)))
    return branch_radius,fast_sigma,slow_sigma,filter_radius


def temporal_halo(fps=30.):
    radius,_,_,filter_radius=temporal_support(fps)
    return radius+filter_radius


def continuous_candidate(values,costs,confidence,fps):
    """Choose the centre of a local minimum-cost path, with no persistent state.

    Costs combine the original fitting objective with changes in *added* upper
    swing/roll coordinates, never changes in the source animation. Each centre
    has the same finite context regardless of worker chunking. This is a local
    continuity preference, not an unbounded full-sequence optimisation.
    """
    n,k,_=values.shape
    radius,_,_,_=temporal_support(fps)
    centres=np.arange(n)
    def path_cost(direction):
        indices=np.clip(centres+direction*radius,0,n-1)
        score=costs[indices].copy();previous=values[indices]
        previous_confidence=confidence[indices]
        for offset in range(radius-1,-1,-1):
            indices=np.clip(centres+direction*offset,0,n-1)
            current=values[indices]
            delta=current[:,None,:,:3]-previous[:,:,None,:3]
            # Axial ambiguity dominates near extension; do not suppress real
            # source swing or use a lower joint's twist to select upper roll.
            delta[...,:2]*=.25
            reliability=np.minimum(confidence[indices],previous_confidence)
            weight=(.03+.06*(1-reliability))*(fps/30.)**.5
            transition=(delta*delta).sum(-1)*weight[:,None,None]**2
            score=costs[indices]+np.min(score[:,:,None]+transition,axis=1)
            previous=current;previous_confidence=confidence[indices]
        return score
    score=path_cost(-1)+path_cost(1)-costs
    return values[centres,np.argmin(score,axis=1)].copy()


def stabilise_roll(roll,confidence,features,fps):
    """Smooth added twist more where bend direction is poorly observable."""
    _,fast,slow,radius=temporal_support(fps)
    sigma=slow+(fast-slow)*confidence
    offsets=np.arange(-radius,radius+1)
    indices=np.clip(np.arange(len(roll))[:,None]+offsets,0,len(roll)-1)
    weights=np.exp(-.5*(offsets[None]/sigma[:,None])**2)
    weights*=np.abs(offsets)[None]<=np.ceil(3*sigma[:,None])
    # Similar poses may share the correction; fast changes of the selected
    # source must not be blurred across an extension/flexion transition.
    difference=features[indices]-features[:,None]
    weights*=np.exp(-.5*np.sum(difference*difference,axis=-1)/.12**2)
    return (weights*roll[indices]).sum(1)/weights.sum(1)


def fit(rig,roots,source_world,goals,iterations=18,roll_degrees=ROLL_LIMIT_DEG,end_weight=4.,smooth_roll=False,leg_end_weight=None,fps=30.,elbow_match=True):
    roots=np.asarray(roots,dtype=float);goals=np.asarray(goals,dtype=float)
    if not np.isfinite(goals).all() or not np.isfinite(source_world).all():
        raise ValueError('Source pose fitting requires finite source frames and joints')
    n=len(roots);world=np.tile(np.eye(3),(n,52,1,1))
    world[:,:22]=np.asarray(source_world)@rig['alignment']
    from .mixamo_rig import TRUNK_POSE_JOINTS
    world[:,TRUNK_POSE_JOINTS]=np.asarray(source_world)[:,TRUNK_POSE_JOINTS]
    for j in range(22,52):world[:,j]=world[:,rig['parents'][j]]
    points=points_from_world(rig,roots,world)
    for upper,middle,end,axis in HINGES:
        distal_weight=end_weight if upper>=16 or leg_end_weight is None else leg_end_weight
        middle_weight=POSITION_PRIORITY
        distal_weight*=POSITION_PRIORITY
        pole_weight=.12
        axis=np.asarray(axis,float)
        neutral=np.eye(3)
        u0=unit(rig['rest'][middle]-rig['rest'][upper]);v0=unit(rig['rest'][end]-rig['rest'][middle])
        a=np.linalg.norm(rig['rest'][middle]-rig['rest'][upper]);b=np.linalg.norm(rig['rest'][end]-rig['rest'][middle])
        base=world[:,upper].copy();lower=world[:,middle].copy();pivot=points[:,upper].copy()
        pole_normal=base@axis;normal=base@unit(axis-u0*(u0@axis));other=np.cross(base@u0,normal)
        target_mid=goals[:,middle];target_end=goals[:,end]
        # The actual exported local rotation is a signed Cartesian hinge.
        # Source alignment remains a proximal orientation prior, never a
        # pre/post rotation hidden inside the elbow or knee channels.
        max_bend=np.deg2rad(160 if upper<3 else 150)
        limit=np.deg2rad(roll_degrees)
        lower_direction=lower@v0
        relative_direction=np.einsum('tij,tj->ti',base.swapaxes(-1,-2),lower_direction)
        reference=neutral@v0
        observed=relative_direction-(relative_direction@axis)[:,None]*axis
        reference=reference-axis*(reference@axis)
        bend_angle=np.arctan2(np.cross(reference,observed)@axis,observed@reference)
        x=np.zeros((n,4));x[:,3]=np.clip(bend_angle,0,max_bend)
        original=x.copy()
        # A calibrated outgoing lower direction measures hinge observability.
        # A source rest-joint cross product is NOT a new anatomical pole.
        middle_direction=unit(target_mid-pivot)
        end_direction=unit(target_end-target_mid)
        # Observability comes from source flexion with its fixed neutral shape
        # removed, not from the target's straight rest. This is only a fitting
        # confidence: the exported local hinge below remains fully Cartesian.
        # Direction-only evidence is invariant to lower-joint axial rotation.
        source_axis=rig['alignment'][upper].T@axis
        source_reference=rig['alignment'][upper].T@rig['alignment'][middle]@v0
        source_reference-=source_axis*(source_reference@source_axis)
        source_observed=relative_direction-(relative_direction@source_axis)[:,None]*source_axis
        source_bend=np.arctan2(np.cross(source_reference,source_observed)@source_axis,source_observed@source_reference)
        confidence=np.clip(np.sin(np.clip(source_bend,0,max_bend))/np.sin(np.deg2rad(35.)),0,1)**2
        features=np.concatenate((middle_direction,end_direction),axis=1)
        # Near extension, lateral neutral joint-centre curvature is not evidence
        # of a sideways anatomical pole. Bound only the *added* roll; actual
        # source upper-limb rotation remains fully present in base. Observable
        # flexion gradually restores the user's 60-degree additional ceiling.
        extension_limit=min(limit,np.deg2rad(EXTENSION_ROLL_LIMIT_DEG))
        limit=extension_limit+(limit-extension_limit)*confidence
        bounds=np.tile([.75,.75,0.,max_bend],(n,1));bounds[:,2]=limit
        lower_bounds=-bounds.copy();lower_bounds[:,3]=0
        def evaluate(value,output=False):
            swing=Rotation.from_rotvec(normal*value[:,0,None]+other*value[:,1,None]).as_matrix()
            roll=Rotation.from_rotvec(u0*value[:,2,None]).as_matrix()
            upper_frame=swing@base@roll
            hinge=Rotation.from_rotvec(axis*value[:,3,None]).as_matrix()@neutral
            lower_frame=upper_frame@hinge
            mid=pivot+np.einsum('tij,j->ti',upper_frame,u0*a)
            endpoint=mid+np.einsum('tij,j->ti',lower_frame,v0*b)
            if output:return mid,endpoint,upper_frame,lower_frame
            return np.concatenate(((mid-target_mid)*middle_weight,(endpoint-target_end)*distal_weight,
                ((upper_frame-base)*(.030*ROTATION_PRIOR_SCALE)).reshape(n,9),
                (upper_frame@axis-pole_normal)*(pole_weight*(1-.9*confidence[:,None])),
                (lower_frame@v0-lower_direction)*(.0212132034356*ROTATION_PRIOR_SCALE),
                value[:,2,None]*(.008+.072*EXTENSION_ROLL_PRIOR_SCALE*(1-confidence[:,None])**4)),axis=1)
        residual=evaluate(x);cost=(residual*residual).sum(1);damping=np.full(n,1e-5)
        # Give all limbs a reachable endpoint seed, including uncorrected
        # elbows. A signed hinge can otherwise start on a wrong branch.
        # Axial lower-joint twist must never choose incoming upper roll.
        seed=x.copy();delta=target_end-pivot;distance=np.linalg.norm(delta,axis=1)
        reference=neutral@v0
        k=(u0@axis)*(reference@axis)
        c=u0@reference-k;s=u0@np.cross(axis,reference)
        phase=np.arctan2(s,c)
        arc=np.arccos(np.clip(((distance**2-a*a-b*b)/(2*a*b)-k)/max(np.hypot(c,s),1e-12),-1,1))
        options=np.clip(np.column_stack((phase+arc,phase-arc,np.zeros(n),np.full(n,max_bend),np.full(n,np.clip(phase,0,max_bend)))),0,max_bend)
        distances=np.sqrt(np.maximum(0,a*a+b*b+2*a*b*(k+c*np.cos(options)+s*np.sin(options))))
        seed[:,3]=options[np.arange(n),np.argmin(abs(distances-distance[:,None]),axis=1)]
        bend=Rotation.from_rotvec(axis*seed[:,3,None]).as_matrix()@neutral
        reach=unit(a*u0+b*np.einsum('tij,j->ti',bend,v0))
        for t in range(n):
            desired=align(base[t]@reach[t],delta[t])@base[t]
            swing=align(base[t]@u0,desired@u0)
            rv=Rotation.from_matrix(swing).as_rotvec()
            seed[t,:2]=[rv@normal[t],rv@other[t]]
            roll=base[t].T@swing.T@desired
            seed[t,2]=Rotation.from_matrix(roll).as_rotvec()@u0
        seed=np.clip(seed,lower_bounds,bounds)
        candidate=evaluate(seed);seed_cost=(candidate*candidate).sum(1);choose=seed_cost<cost
        x[choose]=seed[choose];residual[choose]=candidate[choose];cost[choose]=seed_cost[choose]
        def optimise(steps,lock_roll=False):
            nonlocal residual,cost,damping
            for _ in range(steps):
                jac=[]
                for parameter in range(4):
                    bumped=x.copy();bumped[:,parameter]+=1e-5
                    jac.append((evaluate(bumped)-residual)/1e-5)
                jac=np.stack(jac,axis=-1)
                normal_matrix=jac.swapaxes(1,2)@jac
                normal_matrix+=np.eye(4)[None]*damping[:,None,None]
                gradient=np.einsum('nri,nr->ni',jac,residual)
                # Once an anatomical bound is active, solve the remaining free
                # variables rather than repeatedly proposing an impossible roll.
                active=((x<=lower_bounds+1e-8)&(gradient>0))|((x>=bounds-1e-8)&(gradient<0))
                active|=(bounds-lower_bounds)<1e-12
                if lock_roll:active[:,2]=True
                normal_matrix=np.where(active[:,:,None]|active[:,None,:],0,normal_matrix)
                normal_matrix+=np.eye(4)[None]*active[:,None,:]
                gradient=np.where(active,0,gradient)
                step=np.linalg.solve(normal_matrix,-gradient)
                step=np.clip(step,-.2,.2)
                proposed=np.clip(x+step,lower_bounds,bounds)
                candidate=evaluate(proposed);new_cost=(candidate*candidate).sum(1)
                accepted=new_cost<cost
                x[accepted]=proposed[accepted];residual[accepted]=candidate[accepted];cost[accepted]=new_cost[accepted]
                damping=np.where(accepted,np.maximum(1e-7,damping*.5),np.minimum(1.,damping*4))
        optimise(iterations)
        candidates=[x.copy()];candidate_costs=[cost.copy()]
        negative=seed.copy();negative[:,2]=-limit
        positive=seed.copy();positive[:,2]=limit
        # Fully optimise hypotheses before selecting: comparing only their
        # initial errors can choose a collapsed hinge or another local branch.
        for initial in (original,negative,positive):
            x=initial.copy();residual=evaluate(x);cost=(residual*residual).sum(1);damping[:]=1e-5
            optimise(iterations)
            candidates.append(x.copy());candidate_costs.append(cost.copy())
        if upper>=16 and elbow_match:
            # The legacy objective favours the wrist four to one. Compare
            # additional elbow/axis hypotheses against that exact reference;
            # neither a better wrist nor a legal local hinge justifies moving
            # the elbow farther away or rolling its anatomical axis sideways.
            legacy_values=np.stack(candidates,axis=1)
            legacy_costs=np.stack(candidate_costs,axis=1)
            baseline=legacy_values[np.arange(n),np.argmin(legacy_costs,axis=1)].copy()
            reference_mid,reference_end,reference_up,_=evaluate(baseline,True)
            reference_middle=np.linalg.norm(reference_mid-target_mid,axis=1)
            reference_distal=np.linalg.norm(reference_end-target_end,axis=1)
            def axis_error(frame):
                pole=frame@axis
                return np.arctan2(np.linalg.norm(np.cross(pole,pole_normal),axis=1),
                                 np.sum(pole*pole_normal,axis=1))
            reference_axis=axis_error(reference_up)
            def admissible(value):
                mid_point,end_point,frame,_=evaluate(value,True)
                middle_error=np.linalg.norm(mid_point-target_mid,axis=1)
                distal_error=np.linalg.norm(end_point-target_end,axis=1)
                facing=axis_error(frame)
                valid=((middle_error<=reference_middle+1e-7)&
                    (distal_error<=reference_distal+ELBOW_WRIST_BUDGET_M+1e-10)&
                    (np.linalg.norm(end_point-reference_end,axis=1)<=ELBOW_WRIST_BUDGET_M+1e-10)&
                    (facing<=reference_axis+np.deg2rad(ELBOW_AXIS_SLACK_DEG)))
                score=((middle_error/np.maximum(reference_middle,.001))**2+
                    (facing/np.maximum(reference_axis,np.deg2rad(1.)))**2+
                    .05*(distal_error/np.maximum(reference_distal+ELBOW_WRIST_BUDGET_M,.005))**2)*.0001
                return valid,score
            proposals=[baseline.copy()];scores=[admissible(baseline)[1]]
            for middle_weight,distal_weight,pole_weight in ((2.,2.,.3),(2.,2.,.6),(4.,2.,.3)):
                x=baseline.copy();residual=evaluate(x);cost=(residual*residual).sum(1);damping[:]=1e-5
                optimise(iterations)
                trial=x.copy()
                for amount in (.125,.25,.5,.75,1.):
                    value=baseline+(trial-baseline)*amount
                    valid,score=admissible(value)
                    proposals.append(value);scores.append(np.where(valid,score,1e6))
            candidates=proposals;candidate_costs=scores
        values=np.stack(candidates,axis=1);costs=np.stack(candidate_costs,axis=1)
        best=values[np.arange(n),np.argmin(costs,axis=1)].copy()
        def joint_errors(value):
            middle_point,end_point,_,_=evaluate(value,True)
            return np.stack((np.linalg.norm(middle_point-target_mid,axis=1),
                             np.linalg.norm(end_point-target_end,axis=1)),axis=1)
        # Continuity may choose between similarly accurate solutions, but must
        # not purchase smoothness with a large middle-joint or contact error.
        error_budget=joint_errors(best)+np.array([.002,.00025])
        for k in range(len(candidates)):
            valid=(joint_errors(values[:,k])<=error_budget).all(axis=1)
            costs[~valid,k]+=1.e6
        if smooth_roll and n>1:
            x=continuous_candidate(values,costs,confidence,fps)
        else:x=values[np.arange(n),np.argmin(costs,axis=1)].copy()
        if smooth_roll and n>1:
            # Only the extra shared axial correction is regularised. Original
            # inference, hinge timing, source rotations and distal frames are
            # not filtered. temporal_halo() accounts for both finite passes.
            unfiltered=x.copy()
            x[:,2]=stabilise_roll(x[:,2],confidence,features,fps)
            residual=evaluate(x);cost=(residual*residual).sum(1);damping[:]=1e-5
            optimise(8,lock_roll=True)
            # Keep the greatest tested stabilisation amount within the same
            # source-position budget. Interpolating bounded fit parameters
            # retains signed hinges and fixed segment lengths.
            stabilised=x.copy();selected=unfiltered.copy()
            for amount in (.125,.25,.5,.75,1.):
                trial=unfiltered+(stabilised-unfiltered)*amount
                valid=(joint_errors(trial)<=error_budget+1e-12).all(axis=1)
                if upper>=16 and elbow_match:valid&=admissible(trial)[0]
                selected[valid]=trial[valid]
            x=selected
        mid,endpoint,up,down=evaluate(x,True)
        world[:,upper]=up;world[:,middle]=down
        points[:,middle]=mid;points[:,end]=endpoint
    # Keep all distal world orientations, source pelvis and trunk. Rebuilding
    # locals/FK also transports all thirty neutral finger bones from the wrist.
    for j in range(22,52):world[:,j]=world[:,rig['parents'][j]]
    local=world.copy()
    for j,p in enumerate(rig['parents']):
        if p>=0:local[:,j]=world[:,p].swapaxes(-1,-2)@world[:,j]
    points,world=fk(rig,roots,local)
    return local,points,world


def hinge_motion(rig,local,upper,middle):
    """The complete local rotation, including every constant animated offset.

    A calibration-removed rotation is insufficient for validating the actual
    bone's fixed local axis. Do not accept that older proxy here.
    """
    return np.asarray(local)[...,middle,:,:]


def anatomical_axes(rig,world):
    """Actual fixed hinge axes, independently of octahedral corner geometry."""
    return np.stack([np.asarray(world)[:,upper] @ np.asarray(axis)
                     for upper,middle,end,axis in HINGES],axis=1)
