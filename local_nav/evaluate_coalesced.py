"""Offline paired timing comparison for joined straight drives."""
import argparse
import json
from simulate_spatial import run_spatial_case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    runs=[]
    for seed in range(1,5):
        for name,options in [('forward30',dict(goal=(0,30))),
                             ('reverse30',dict(goal=(0,-30))),
                             ('out_back',dict(goals=[[0,20],[0,0]])),
                             ('turn_drive',dict(goal=(30,30)))]:
            for joined in (False,True):
                run=run_spatial_case(dict(speed=10+seed,coast=.03+.02*seed,seed=seed,
                                         pivot_cm=5+seed,drive_yaw_bias_dps=seed-2),
                                     measured_map_drive=True,coalesce_drives=joined,**options)
                result=run['result']
                summary=dict(scenario=name,seed=seed,joined=joined,outcome=result['outcome'],
                             reason=result.get('reason'),seconds=result['elapsed_seconds'],
                             actions=len(result['actions']),true_error_cm=run['true_goal_error_cm'],
                             clearance_violations=run['clearance_violations'],
                             final_motor_output=run['final_motor_output'],watchdog_stops=run['watchdog_stops'])
                runs.append(summary)
                with open(args.output,'w') as output:json.dump(runs,output,indent=2)
                print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
