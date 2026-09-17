"""Deterministic multi-room graph experiments. No API, hardware or camera imports.

Models graph events, localization outages, doors and resumable coverage. It does
not simulate vision, wheel contact, obstacle sensing, or real SLAM performance.
"""
import argparse
import collections
import json
import os
import statistics
import tempfile
import time
from search_graph import SearchGraph


def run_case(rooms=10,missing_target=False,closed_door=False,lost_pose=False,
             restart=False,dynamic_discovery=False,blocked_frontier=False):
    graph=SearchGraph('other robot')
    def room(i):
        key='room_%05d'%i
        graph.place(key,key,kind='frontier' if dynamic_discovery and i<rooms-1 else 'viewpoint')
        return key
    def connect(a,b,cost=100.):
        graph.connect(a+'>'+b,a,b,cost,dict(source_revision=graph.data['nodes'][a]['revision'],
            target_revision=graph.data['nodes'][b]['revision'],evidence='simulation_only'))
    if dynamic_discovery:room(0)
    else:
        for i in range(rooms):room(i)
        for i in range(rooms-1):
            a,b='room_%05d'%i,'room_%05d'%(i+1)
            connect(a,b);connect(b,a)
        if closed_door and rooms>2:connect('room_00000','room_00002',250.)
    current='room_00000';sim_time=0.;pose_valid=True
    counters=collections.Counter();events=[];observed=set();latencies=[]
    maximum_action_ms=0.;duplicate_observations=0;unsafe_edges=0
    restarted=False;door_changed=False;outage_seen=False;stale_rejections=0
    started=time.monotonic()
    with tempfile.TemporaryDirectory() as directory:
        for step in range(rooms*25+50):
            node=graph.data['nodes'][current]
            localization=dict(status='tracked' if pose_valid else 'lost',place=current,
                              submap=node['submap'],revision=node['revision'],time=sim_time)
            began=time.monotonic();action=graph.next_action(current,localization,sim_time)
            elapsed=1000*(time.monotonic()-began);latencies.append(elapsed)
            maximum_action_ms=max(maximum_action_ms,elapsed)
            kind=action['kind'];counters[kind]+=1
            if len(events)<40:events.append(dict(step=step,place=current,action=kind))
            if kind=='observe_heading':
                key=(current,action['heading_degrees'],node['revision'])
                if key in observed:duplicate_observations+=1
                observed.add(key)
                graph.observe(current,action['heading_degrees'],'simulated-frame-%d'%step,node['revision'])
                sim_time+=.2
                if current=='room_%05d'%(rooms-1) and action['heading_degrees']==90 and not missing_target:
                    graph.identify('SIMULATED-RECOGNITION',[300,200,340,240],'simulated_recognizer')
                if lost_pose and not outage_seen and len(observed)==7:
                    pose_valid=False;outage_seen=True
                if restart and not restarted and len(observed)==5:
                    path=os.path.join(directory,'checkpoint.json');graph.save(path);graph=SearchGraph.load(path)
                    restarted=True
            elif kind=='relocalize':
                # Simulated backend reacquisition, not resetting live odometry.
                sim_time+=2.;pose_valid=True
            elif kind=='map_frontier':
                if blocked_frontier:
                    graph.defer_frontier(current,'simulated occluded doorway');continue
                i=int(current.split('_')[1]);graph.place(current,current,kind='viewpoint')
                if i+1<rooms:
                    other=room(i+1);connect(current,other);connect(other,current)
                sim_time+=1.
            elif kind=='navigate_graph':
                if closed_door and not door_changed and 'room_00000>room_00001' in action['edges']:
                    graph.block('room_00000>room_00001');door_changed=True
                    if not graph.validate_route(action,current):stale_rejections+=1;continue
                if not graph.validate_route(action,current):unsafe_edges+=1;break
                for edge_id in action['edges']:
                    edge=graph.data['edges'][edge_id]
                    if edge['source']!=current or not graph.usable(edge):unsafe_edges+=1;break
                    current=edge['target'];sim_time+=edge['cost']/10.
                    counters['edges_traversed']+=1
            elif kind in ('object_found','known_map_searched','map_required'):
                break
            else:raise RuntimeError('Unhandled simulated action '+kind)
        else:kind='simulation_step_budget_exhausted'
        checkpoint_bytes=len(json.dumps(graph.data))
    return dict(rooms=rooms,outcome=kind,actions=dict(counters),observations=len(observed),
        duplicate_observations=duplicate_observations,unsafe_edge_traversals=unsafe_edges,
        stale_routes_rejected=stale_rejections,checkpoint_resumed=restarted,
        simulated_seconds=sim_time,wall_seconds=time.monotonic()-started,
        median_planner_ms=statistics.median(latencies),maximum_planner_ms=maximum_action_ms,
        checkpoint_bytes=checkpoint_bytes,known_places=len(graph.data['nodes']),events=events,
        limitations='Topological event simulation; no camera, dynamics, live mapping, motor or API calls')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cases=[('two_rooms',dict(rooms=2)),('hundred_rooms',dict(rooms=100)),
           ('thousand_rooms',dict(rooms=1000)),('closed_door_detour',dict(rooms=10,closed_door=True)),
           ('lost_localization',dict(rooms=10,lost_pose=True)),('restart',dict(rooms=10,restart=True)),
           ('absent_target',dict(rooms=10,missing_target=True)),
           ('discovered_rooms',dict(rooms=100,dynamic_discovery=True)),
           ('blocked_frontier',dict(rooms=10,dynamic_discovery=True,blocked_frontier=True)),
           ('combined',dict(rooms=100,closed_door=True,lost_pose=True,restart=True))]
    results={}
    for name,parameters in cases:
        results[name]=run_case(**parameters)
        print(json.dumps(dict(case=name,**{k:v for k,v in results[name].items() if k not in ('events','limitations')})),flush=True)
    with open(args.output,'w') as output:json.dump(results,output,indent=2)


if __name__=='__main__':main()
