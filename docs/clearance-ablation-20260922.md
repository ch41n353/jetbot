# Carton clearance and prior-trajectory experiment — 2026-09-22

Source run: `/mnt/robotlogs/goals/advil-flow-20260922-060659`.
Robot motion is not part of this experiment.

## Correction to previous evaluation

The single-contact-point metric reported call 4 improving from 2.8 to 28.2 cm.
That does not establish safe clearance. Reviewing the full visible carton
silhouette shows that BOTH its original and larger-buffer routes intersect it.
Call 6's larger-buffer route also intersects the silhouette. Its original route
misses the silhouette but enters a 12-pixel dilation of it. This image dilation
is a diagnostic, not a metric margin. Neither metric models the full 3D scene,
the executed trajectory, or motion-estimation uncertainty.

## Prepared API experiment

Five variants on calls 4 and 6, three samples each (30 calls):

1. Existing larger-buffer prompt and original annotated image/text prior.
2. Same prompt and clean image, retaining the text prior.
3. Same prompt and clean image, removing the entire `last_time` context.
4. Stronger footprint-first/partial-path instructions with original reference.
5. Stronger instructions with clean image and no `last_time` context.

Model, reasoning effort, schema, target instruction, output budget, and image
detail remain unchanged. Clean images were saved from the same original frame;
no scene recapture, retouching, or invented image content is used. The stronger
prompt permits stopping a partial plan before a risky corner while retaining
the true target identity and contact. It overrides the forced target endpoint.

Manual carton polygons for calls 4 and 6 are evaluation labels only and are
never sent to GPT. Check every segment, including an approximate connector from
the image's bottom centre. Report crossings, near misses, holds and useful
progress separately; returning hold everywhere is not a successful navigator.

This is a fixed-frame ablation: the same old context is used for each replay;
it cannot establish performance of a closed-loop run after changed movement.
Calls 4 and 6 are tuning cases, not an independent validation set. Evaluate the
selected variant on the other 12 saved frames before deployment and report
regressions. Only deploy if supported; otherwise retain the experimental files.

Runner: `scripts/ablate_clearance.py`. Candidate instructions:
`local_nav/prompts/clearance-candidate-20260922.txt`. Every outbound body and raw
response is preserved alongside derived results and a self-contained gallery.

## Completed results

All 30 API calls returned parsed responses. Each row below contains six outputs:
calls 4 and 6, three repeats each. A near pass includes an actual crossing.

| Variant | Silhouette crossings | Within 12px image buffer | Holds | Mean latency |
|---|---:|---:|---:|---:|
| Larger buffer, original reference | 5/6 | 6/6 | 0/6 | 5.05 s |
| Larger buffer, clean image, text prior retained | 3/6 | 6/6 | 0/6 | 5.18 s |
| Larger buffer, clean image, no prior | 2/6 | 6/6 | 0/6 | 4.09 s |
| Footprint-first partial paths, original reference | 0/6 | 1/6 | 0/6 | 4.51 s |
| Footprint-first partial paths, clean image, no prior | 0/6 | 0/6 | 0/6 | 4.82 s |

Removing reference information helped but did not solve this failure on its own.
The combined footprint-first/partial-path prompt was the stronger improvement.
This experiment does not isolate which part of that combined prompt mattered.
The best tested variant keeps the true target contact but often ends its route
early, before the corner. Six follow outputs are not six completed approaches.
No hypothesis test or broad reliability claim is justified by six outputs on
two tuning frames. Runs were interleaved with two concurrent requests; latency
is descriptive, not a deployment benchmark. Evaluation polygons were not sent.

An additional motor-free projection/`avoid` check (goal=None diagnostic) found
all six candidate paths project to usable floor. The local path repair leaves
call 4 unchanged, but modifies all three call 6 paths. Thus these visual gains
must not be equated with the actual controller's final geometry. No robot drive,
production prompt replacement, or controller safety claim was made. The remaining
12 frames and closed-loop continuation have not been tested with this candidate.

Artifacts: `/mnt/robotlogs/goals/advil-flow-20260922-060659/clearance-ablation-02/`.
Gallery: `http://192.168.86.158:8772/clearance-ablation`.
