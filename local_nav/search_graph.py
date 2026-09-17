"""Persistent multi-room search scheduling; no sensors, model calls, or motors.

Mapper supplies places, doorway connections, frontiers, and versioned corridor
certificates. Executor must recheck each local sweep. This is not a SLAM backend.
"""
import argparse
import copy
import heapq
import json
import math
import os
import tempfile
import uuid


class SearchGraph:
    def __init__(self,target,snapshot=None):
        self.data=copy.deepcopy(snapshot) if snapshot is not None else dict(
            version=1,target=target,nodes={},edges={},coverage={},found=None)
        if self.data.get('version')!=1 or self.data.get('target')!=target:
            raise ValueError('Search checkpoint version or target mismatch')
        self.data.setdefault('mission_id',uuid.uuid4().hex)
        self.data.setdefault('pending_recognition',{})

    def new_search(self,target):
        """Reuse the map, never another target's negative search coverage."""
        if not isinstance(target,str) or not target.strip():raise ValueError('Target required')
        graph=SearchGraph(target)
        graph.data['nodes']=copy.deepcopy(self.data['nodes'])
        graph.data['edges']=copy.deepcopy(self.data['edges'])
        graph.data['coverage']={key:{} for key in graph.data['nodes']}
        return graph

    def recognition_ticket(self,image):
        if not image:raise ValueError('Recognition requires a captured image')
        if len(self.data['pending_recognition'])>=2:
            raise ValueError('Recognition queue full; avoid an unbounded stale-image backlog')
        ticket=dict(id=uuid.uuid4().hex,mission_id=self.data['mission_id'],
                    target=self.data['target'],image=image)
        self.data['pending_recognition'][ticket['id']]=copy.deepcopy(ticket)
        return ticket

    def accept_recognition(self,ticket,box,model):
        expected=self.data['pending_recognition'].get(ticket.get('id'))
        if expected is None or ticket!=expected or ticket.get('mission_id')!=self.data['mission_id']:
            return False
        # Null means a negative detection; acknowledge it without claiming success.
        if box is not None:self.identify(ticket['image'],box,model)
        del self.data['pending_recognition'][ticket['id']]
        return True

    def place(self,key,submap,revision=0,kind='viewpoint',headings=None,
              position_cm=None,metric_frame=None):
        if not key or not submap or type(revision) is not int or revision<0:
            raise ValueError('Place requires stable IDs and map revision')
        if kind not in ('viewpoint','doorway','frontier'):
            raise ValueError('Unknown place kind')
        headings=list(range(0,360,30)) if headings is None else list(headings)
        if any(type(h) not in (int,float) or not math.isfinite(h) for h in headings):
            raise ValueError('Invalid observation headings')
        if position_cm is not None:
            if (not metric_frame or len(position_cm)!=2 or
                    any(type(v) not in (int,float) or not math.isfinite(v) for v in position_cm)):
                raise ValueError('Metric position requires a shared frame and finite coordinates')
        old=self.data['nodes'].get(key)
        if old is not None and revision<old['revision']:
            raise ValueError('Cannot roll back a map revision')
        if old is not None and old['submap']!=submap:
            raise ValueError('Place frame changes require explicit map migration')
        self.data['nodes'][key]=dict(submap=submap,revision=revision,kind=kind,
                                    headings=sorted(set(float(h)%360 for h in headings)))
        if position_cm is not None:
            self.data['nodes'][key].update(position_cm=list(position_cm),metric_frame=metric_frame)
        if old is not None and old['revision']==revision and old.get('deferred'):
            self.data['nodes'][key]['deferred']=old['deferred']
        if old is None or old['revision']!=revision:
            self.data['coverage'][key]={}

    def connect(self,key,source,target,cost,certificate=None):
        if source not in self.data['nodes'] or target not in self.data['nodes']:
            raise ValueError('Connection refers to missing place')
        if type(cost) not in (int,float) or not math.isfinite(cost) or cost<=0:
            raise ValueError('Connection requires positive finite path cost')
        # A recognizer's doorway label alone is never a traversability certificate.
        if certificate is not None:
            if not certificate.get('evidence') or certificate.get('source_revision')!=self.data['nodes'][source]['revision'] or certificate.get('target_revision')!=self.data['nodes'][target]['revision']:
                raise ValueError('Missing or stale corridor certificate')
        self.data['edges'][key]=dict(source=source,target=target,cost=float(cost),
                                    certificate=copy.deepcopy(certificate),blocked=False)

    def block(self,key):
        self.data['edges'][key]['blocked']=True

    def defer_frontier(self,key,reason):
        if self.data['nodes'][key]['kind']!='frontier' or not reason:
            raise ValueError('Only unresolved frontiers may be deferred with a reason')
        self.data['nodes'][key]['deferred']=str(reason)

    def usable(self,edge):
        proof=edge['certificate']
        return bool(not edge['blocked'] and proof and proof.get('evidence') and
            proof.get('source_revision')==self.data['nodes'][edge['source']]['revision'] and
            proof.get('target_revision')==self.data['nodes'][edge['target']]['revision'])

    def observe(self,place,heading,image,revision):
        node=self.data['nodes'][place]
        if revision!=node['revision']:raise ValueError('Observation belongs to stale map frame')
        if not image or type(heading) not in (int,float) or not math.isfinite(heading):
            raise ValueError('Observation requires an image and finite heading')
        # Record only the matching view; do not claim whole-room coverage.
        for requested in node['headings']:
            if abs((heading-requested+180)%360-180)<=2:
                self.data['coverage'][place][str(requested)]=image

    def identify(self,image,box,model):
        if not image or not model or len(box)!=4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError('Recognition requires image, model, and valid box')
        x0,y0,x1,y1=box
        if not (0<=x0<x1<=640 and 0<=y0<y1<=480):raise ValueError('Invalid recognition box')
        # Image identity does not assert localization or physical arrival.
        self.data['found']=dict(image=image,box=list(box),model=model,target=self.data['target'])

    def missing(self,place):
        return [h for h in self.data['nodes'][place]['headings']
                if str(h) not in self.data['coverage'][place]]

    def shortest_paths(self,start):
        """A* with h=0 (Dijkstra): no invented global metric between submaps."""
        adjacency={key:[] for key in self.data['nodes']}
        for key,edge in sorted(self.data['edges'].items()):
            if self.usable(edge):adjacency[edge['source']].append((key,edge))
        distance={start:0.};parent={};heap=[(0.,start)]
        while heap:
            cost,node=heapq.heappop(heap)
            if cost!=distance[node]:continue
            for key,edge in adjacency[node]:
                candidate=cost+edge['cost'];other=edge['target']
                if candidate<distance.get(other,float('inf')):
                    distance[other]=candidate;parent[other]=(node,key)
                    heapq.heappush(heap,(candidate,other))
        return distance,parent

    def astar(self,start,goals):
        """Multi-goal A*: nearest reachable observation place, with path evidence.

        Use metric distance only within a shared mapped frame. Scale by the
        cheapest edge-cost/distance ratio so mixed cost units and shortcuts
        cannot overestimate. Unregistered submaps fall back to h=0.
        """
        goals=set(goals)&set(self.data['nodes'])
        if start not in self.data['nodes'] or not goals:return None
        nodes=self.data['nodes'];adjacency={key:[] for key in nodes}
        edges=[(key,e) for key,e in sorted(self.data['edges'].items()) if self.usable(e)]
        frame=nodes[start].get('metric_frame')
        metric=bool(frame) and all(n.get('metric_frame')==frame and 'position_cm' in n
                                   for n in nodes.values())
        scale=float('inf')
        for key,edge in edges:
            adjacency[edge['source']].append((key,edge))
            if metric:
                a,b=(nodes[edge[k]]['position_cm'] for k in ('source','target'))
                distance=math.hypot(a[0]-b[0],a[1]-b[1])
                if distance:scale=min(scale,edge['cost']/distance)
        if not metric or not math.isfinite(scale):scale=0.
        def heuristic(key):
            if not scale:return 0.
            x,z=nodes[key]['position_cm']
            return scale*min(math.hypot(x-nodes[g]['position_cm'][0],z-nodes[g]['position_cm'][1])
                             for g in goals)
        costs={start:0.};parents={};queue=[(heuristic(start),0.,start)];expanded=0
        while queue:
            _,cost,node=heapq.heappop(queue)
            if cost!=costs[node]:continue
            expanded+=1
            if node in goals:
                route=[];cursor=node
                while cursor!=start:
                    cursor,edge=parents[cursor];route.append(edge)
                route.reverse()
                return dict(goal=node,edges=route,cost=cost,expanded_nodes=expanded,
                            heuristic='metric_lower_bound' if scale else 'zero')
            for key,edge in adjacency[node]:
                other=edge['target'];candidate=cost+edge['cost']
                if candidate<costs.get(other,float('inf')):
                    costs[other]=candidate;parents[other]=(node,key)
                    heapq.heappush(queue,(candidate+heuristic(other),candidate,other))
        return None

    def next_action(self,current,localization,now):
        if self.data['found'] is not None:
            return dict(kind='object_found',recognition=copy.deepcopy(self.data['found']),motion_authorized=False)
        if current not in self.data['nodes']:return dict(kind='map_required',motion_authorized=False)
        node=self.data['nodes'][current]
        stamp=localization.get('time')
        if not (localization.get('status')=='tracked' and localization.get('place')==current and
                localization.get('submap')==node['submap'] and localization.get('revision')==node['revision'] and
                type(stamp) in (int,float) and type(now) in (int,float) and
                math.isfinite(stamp) and math.isfinite(now) and 0<=now-stamp<=.5):
            return dict(kind='relocalize',submap=node['submap'],motion_authorized=False)
        missing=self.missing(current)
        if missing:
            return dict(kind='observe_heading',place=current,submap=node['submap'],revision=node['revision'],
                        heading_degrees=missing[0],requires_local_sweep_check=True,motion_authorized=False)
        if node['kind']=='frontier' and not node.get('deferred'):
            return dict(kind='map_frontier',place=current,submap=node['submap'],motion_authorized=False)
        pending=[key for key in self.data['nodes'] if self.missing(key) or self.data['nodes'][key]['kind']=='frontier']
        eligible=[key for key in pending if key!=current and not self.data['nodes'][key].get('deferred')]
        result=self.astar(current,eligible)
        if result is None:
            return dict(kind='map_required' if pending else 'known_map_searched',
                        unreachable_places=sorted(pending),motion_authorized=False)
        route=result['edges']
        return dict(kind='navigate_graph',goal=result['goal'],edges=route,cost=result['cost'],
                    expanded_nodes=result['expanded_nodes'],heuristic=result['heuristic'],
                    certificates=[copy.deepcopy(self.data['edges'][e]['certificate']) for e in route],
                    requires_local_sweep_check=True,motion_authorized=False)

    def validate_route(self,action,current):
        """Recheck graph topology immediately before each executor handoff.

This validates graph evidence only, never fresh localization or a metric sweep.
"""
        if action.get('kind')!='navigate_graph':return False
        route=action.get('edges',[]);certificates=action.get('certificates',[])
        if not route or len(route)!=len(certificates):return False
        cursor=current
        for key,proof in zip(route,certificates):
            edge=self.data['edges'].get(key)
            if edge is None or edge['source']!=cursor or not self.usable(edge) or edge['certificate']!=proof:
                return False
            cursor=edge['target']
        return cursor==action.get('goal')

    def save(self,path):
        directory=os.path.dirname(os.path.abspath(path))
        fd,temporary=tempfile.mkstemp(prefix='.search-',dir=directory)
        try:
            with os.fdopen(fd,'w') as output:
                json.dump(self.data,output,allow_nan=False,sort_keys=True)
                output.flush();os.fsync(output.fileno())
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)

    @classmethod
    def load(cls,path):
        with open(path) as source:snapshot=json.load(source)
        return cls(snapshot['target'],snapshot)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint');parser.add_argument('--current',required=True)
    args=parser.parse_args()
    graph=SearchGraph.load(args.checkpoint)
    # CLI is an inspection tool: no fabricated tracked pose or motor permission.
    print(json.dumps(graph.next_action(args.current,{},0.),indent=2))


if __name__=='__main__':main()
