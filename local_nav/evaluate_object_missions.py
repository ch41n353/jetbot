"""Repeatable mission, recovery and fault scenarios; no hardware access."""
import argparse
import json
from simulate_object_mission import run_case


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    scenarios=[]
    for seed in (1,2,3):
        for name,target in [('aligned',(0,100)),('angled',(-17,60))]:
            scenarios.append((name+'-'+str(seed),dict(speed=10+seed,coast=.03+.02*seed,seed=seed,pivot_cm=5+seed),dict(target=target)))
    for name,parameters,options in [
            ('temporary_occlusion',{},dict(occlude=(1.,1.5))),
            ('persistent_occlusion',{},dict(occlude=(1.,20.))),
            ('camera_loss',dict(frame_fault_after=1.),{}),
            ('slow_camera',dict(camera_hz=5),{}),
            ('stall',dict(stall=True),{}),
            ('cancel_during_recovery',dict(cancel_after=1.2),dict(occlude=(1.,1.5))),
            ('excess_processing',{},dict(perception_compute=.15)),
            ('retained_obstacle',{},dict(target=(20,80),obstacles=[[-8,30,8,40]]))]:
        scenarios.append((name,dict(dict(speed=12,coast=.05,seed=3,pivot_cm=7.5),**parameters),options))
    reports=[]
    for name,parameters,options in scenarios:
        run=run_case(parameters,**options)
        reports.append(dict(name=name,parameters=parameters,options=options,run=run))
        with open(args.output,'w') as out:json.dump(dict(complete=len(reports)==len(scenarios),cases=reports),out,indent=2)
        print(json.dumps(dict(name=name,outcome=run['result']['outcome'],reason=run['result'].get('reason'),
                              seconds=run['result']['elapsed_seconds'],range_cm=run['true_target_range_cm'],
                              clearance_violations=len(run['clearance_violations']),motors=run['final_motor_output'])),flush=True)


if __name__=='__main__':main()
