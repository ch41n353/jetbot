#!/usr/bin/env python3
"""Paired saved-frame API evaluation; no service connection or motor access."""
import copy
import json
import os
import sys
import time
import cv2
sys.path.insert(0, os.path.dirname(__file__))
import fetch

BASE = '/mnt/robotlogs/goals/'
CASES = [
    ('visible_advil', 'search-20260920-011303/view-03.jpg',
     'Approach the visible Advil bottle, blue cap and yellow label left of center. Stop short. This is the only destination.', None),
    ('near_advil', 'astra-advil-arrival-20260920.jpg',
     'Approach the Advil bottle directly ahead. Stop short. This is the only destination.', None),
    ('bin_fills_view', 'search-20260920-004821/view-06.jpg',
     'Approach the pink bin at its left end. Stop before touching it. This is the only destination.',
     {'route_pixels':[{'x':320,'y':420},{'x':300,'y':350}], 'already_reached':[],
      'target_was':{'still_in_view':True,'pixel':{'x':300,'y':350},'where':'ahead','bearing_degrees':0}}),
    ('unseen_no_prior', 'search-20260920-011303/initial.jpg',
     'Approach the Advil bottle. This is the only destination.', None),
    ('visible_bin_end', 'search-20260920-004821/view-05.jpg',
     'Approach the visible pink bin at its left-front corner. Stop short. This is the only destination.', None),
    ('close_edge_obstacles', 'astra-advil-correction-20260920.jpg',
     'Approach the yellow block near image center. Avoid the close red object clipped at the left edge and blue block on the right. This is the only destination.',None),
]

def main():
    out=sys.argv[1]
    if not os.path.ismount('/mnt/robotlogs'):raise RuntimeError('USB unavailable')
    os.makedirs(out,exist_ok=True)
    baseline=open(os.path.join(os.path.dirname(__file__),'prompts/trajectory-20260920-baseline.txt')).read()
    candidate=fetch.PROMPT
    schema=copy.deepcopy(fetch.SCHEMA)
    results=[]
    for repeat in range(2):
        for name,path,instruction,prior in CASES:
            img=cv2.imread(BASE+path)
            if img is None:raise RuntimeError('Missing '+path)
            # Alternate ordering to reduce simple run-order effects.
            for version in (['baseline','candidate'] if repeat==0 else ['candidate','baseline']):
                fetch.PROMPT=baseline if version=='baseline' else candidate
                fetch.SCHEMA=copy.deepcopy(schema)
                if version=='baseline':
                    fetch.SCHEMA['required'].remove('motion');del fetch.SCHEMA['properties']['motion']
                started=time.monotonic();error=None
                try:answer=fetch.recognize(img,instruction,prior)
                except fetch.PlannerHold as exc:answer=exc.answer
                except Exception as exc:answer={};error=str(exc)
                row=dict(case=name,image=BASE+path,instruction=instruction,prior=prior,
                         version=version,repeat=repeat,seconds=time.monotonic()-started,
                         answer=answer,error=error)
                results.append(row)
                with open(out+'/results.jsonl','a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
    with open(out+'/results.json','w') as f:json.dump(results,f,indent=2)

if __name__=='__main__':main()
