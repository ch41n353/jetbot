#!/usr/bin/env python3
"""Offline comparison of fixed-corridor and measured-map local navigation."""
import argparse
import json
import os
import time
import numpy as np
from simulate_spatial import run_spatial_case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seeds',type=int,default=12)
    args=parser.parse_args()
    if not 1<=args.seeds<=100:parser.error('seeds must be1 to100')
    rng=np.random.RandomState(20260914)
    cases=[]
    scenarios=[dict(name='offset_return',goal=(0,0),free=[-18,-30,18,34],
                    initial_pose=[2.4597813589,20.8447391402,1.607047403]),
               dict(name='narrow_out_and_back',goals=[[0,20],[0,0]],free=[-18,-30,18,34]),
               dict(name='turn_and_drive',goal=(30,30))]
    for seed in range(args.seeds):
        parameters=dict(speed=float(rng.uniform(8,16)),coast=float(rng.uniform(.035,.12)),
                        camera_hz=float(rng.uniform(10,15)),latency=float(rng.uniform(.005,.035)),
                        compute=.045,seed=seed+1,pivot_cm=float(rng.uniform(4.5,10.5)),
                        drive_yaw_bias_dps=float(rng.uniform(-4,4)),height_scale=float(rng.uniform(.98,1.02)))
        for scenario in scenarios:
            options={k:v for k,v in scenario.items() if k!='name'}
            for candidate in (False,True):
                started=time.perf_counter()
                run=run_spatial_case(parameters,measured_map_drive=candidate,**options)
                result=run['result']
                summary=dict(seed=seed+1,scenario=scenario['name'],measured_map_drive=candidate,
                             parameters=parameters,outcome=result['outcome'],reason=result.get('reason'),
                             true_goal_error_cm=run['true_goal_error_cm'],
                             true_final_pose=run['true_final_pose'],
                             estimated_final_pose=result.get('last_verified_pose'),
                             clearance_violations=run['clearance_violations'],
                             intermediate_model_calls=result['intermediate_model_calls'],
                             simulated_seconds=result['elapsed_seconds'],
                             simulation_wall_seconds=time.perf_counter()-started,
                             local_planning_seconds=result['local_planning_seconds'],
                             local_searches=result.get('local_searches'),
                             actions=[dict(kind=a['plan']['kind'],value=a['plan']['value'],
                                           outcome=a['outcome'],reason=a.get('reason')) for a in result['actions']],
                             final_motor_output=run['final_motor_output'])
                cases.append(summary)
                with open(args.output+'.tmp','w') as output:
                    json.dump(dict(complete=False,cases=cases),output,indent=2)
                os.replace(args.output+'.tmp',args.output)
                print(json.dumps(dict(case=len(cases),scenario=scenario['name'],candidate=candidate,
                                      outcome=result['outcome'],reason=result.get('reason'),
                                      error_cm=run['true_goal_error_cm'],violations=len(run['clearance_violations']))),flush=True)
    with open(args.output,'w') as output:json.dump(dict(complete=True,cases=cases),output,indent=2)


if __name__=='__main__':main()
