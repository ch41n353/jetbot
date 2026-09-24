# Search architecture — 2026-09-15

The goal is deterministic, persistent multi-room object search with Sol for
recognition and an independent local controller. A room rectangle is a bounded
test fixture, not the representation for the whole environment.

## Components and ownership

| Component | Responsibility | Current state |
| --- | --- | --- |
| Mapper/localizer | Camera/IMU keyframes, local metric submaps, loop closure, relocalization, traversability evidence | No live SLAM integration in this navigation stack yet |
| Persistent search graph | Places, rooms/submaps, doorway edges, unexplored frontiers, observation coverage, resumable target search | `local_nav/search_graph.py`, motor-free tests |
| Deterministic scheduler | Initial scan; select reachable unobserved places; traverse doorway graph; request more mapping at frontiers | Graph scheduler implemented; local rectangle/perimeter harness remains a bounded live test adapter |
| Sol | Recognize the requested object in sampled RGB frames; never invent traversability or claim arrival | Existing API integration, real image evidence |
| Local planner/controller | Metric A*, swept chassis clearance, target tracking, battery monitoring, watchdog and stop | Existing experimental robot tools |
| Astra | Development, unusual unresolved failures and user interaction | Not in the normal per-action decision path |

Global routes are topological; metric geometry and pose uncertainty belong to
local submaps. Submaps are not simply fresh odometry origins: a mapping backend
must establish and maintain their relationships. No resetting uncertainty on a
room change. Relocalization must produce evidence tied to a map revision before
movement resumes. Recognizing an object from a stopped image does not require
claiming an accurate global pose.

## Mapper/executor contract

1. Publish stable place and submap IDs, map revisions, and local observation
   headings. Room names are optional labels; no rectangular-room assumption.
2. Publish directed doorway/corridor edges with costs and versioned clearance
   evidence. A door recognized in an RGB image is not yet an executable edge.
3. Mark frontiers as known safe observation places bordering unknown space.
   Looking beyond them can expand the graph only after mapping verifies geometry.
4. Return a fresh localized place/submap/revision. Lost or stale localization
   yields `relocalize`, preserving mission coverage; it does not erase history.
5. Recheck each planned local trajectory and its certificate before execution.
   Graph planning alone never authorizes motors. Map revisions invalidate stale
   edge certificates and local coverage; obstacles remain when out of view.
6. Feed completed observations back into coverage and persist checkpoints.
   Closed doors block edges and trigger deterministic route recomputation.

The graph uses A* with zero heuristic (Dijkstra) because unrelated submaps do not
yet have a trustworthy shared Euclidean coordinate system. Local metric A*
continues to compute physical trajectories. The graph has no room-count or
total-distance cutoff; physical action/energy/time budgets remain executor
constraints. `known_map_searched` refers only to represented coverage, never
proof that the target is absent from an unknown building.

## Immediate failure fix and evidence

The previous live search completed six turns (~182 degrees), then reached
4.15 degrees estimated yaw sigma, over the unchanged4-degree limit. Position
sigma was0.77cm. The loop rejected movement before even asking Sol to inspect
the resulting image. That final view contained the other robot.

`sol_search.py` now captures and recognizes while stopped even when pose
uncertainty prevents more motion. A search-only task can return `object_found`
with `found_pose_valid:false` and no claimed global location. An approach remains
blocked until localization is restored. Cancellation and sensor checks still
precede recognition. Final event histories are deep-copied so later coverage
updates cannot rewrite earlier events.

14 harness tests pass, including recognized/absent targets after excessive pose
uncertainty, prevention of further motors/approach, and cancellation during API
inference. Seven graph tests pass, including100 synthetic connected rooms,
closed-door detours, map revision invalidation and persistent resume. The recorded
live turn/recognition replay is `recorded-failure-replay.json`. No new motion was
performed for these fixes. Neither the replay nor the100-room graph test is a
hardware multi-room navigation result.

## Remaining implementation and acceptance gate

The graph is not wired to a live SLAM backend or autonomous doorway traversal.
The existing `sol_search.py` remains a bounded static-map controller, not an
implementation of the graph's mapper contract. Do not label it multi-room-ready.
The graph is a tested scheduler boundary; it does not calculate new room maps or
recover lost physical localization itself.

ORB-SLAM3 is a candidate to evaluate because its upstream implementation supports
monocular/fisheye visual-inertial and multiple-map operation:
https://github.com/UZ-SLAMLab/ORB_SLAM3 . This is a candidate, not a selected or
installed backend. Camera/IMU timing, full extrinsics, IMU noise parameters,
initialization, computational cost on this Jetson, and usable obstacle/free-space
mapping need measurement. Sparse SLAM landmarks alone are not a navigation
occupancy map.

Next acceptance test should be a recorded camera/IMU passage between two rooms
and back: preserve identity of the starting place, detect tracking loss, recover
without manual pose resets, and measure loop-closure error. Only then connect
the graph's routes to guarded doorway traversal. This replaces repeated special
cases for a single room with a mapping/localization interface that can grow.

Robot remains stopped and service closed. Changes are local; nothing pushed.
