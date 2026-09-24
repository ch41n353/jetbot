# Completed comparison — 2026-09-15

The run started with rates checked on 2026-09-14 and finished on 2026-09-15.
All 196 requests completed and were scored/priced. No API or scoring failures;
no candidate drove the robot. A read-only battery check recorded 12.436V pack
and 5.04V input during the run.

## Recommendation from these results

GPT-5.6 Sol with reasoning disabled is the strongest fast visual-perception
candidate: 12/12 target boxes and contact points met the frozen criteria,
45/46 annotated obstacles matched, median request latency 4.24s. It made two
unsafe approvals across the 70 repeated synthetic clearance checks, so it
must not replace deterministic geometric validation.

Sol low retained 12/12 target/contact matches and 45/46 obstacles, with zero
unsafe synthetic approvals in this small set. It took 10.36s median. Choose
this as a candidate when model-side constraint reasoning matters more than
latency; zero observed errors is not proof of safe autonomous navigation.

Luna low is the budget candidate: $0.001099/request observed mean, 12/12 target
and contact checks, but only 42/46 annotated obstacles. Its 9.45s median is
similar to larger models at low reasoning. Terra did not provide a better
quality/latency/cost combination in this particular set. GPT-5.5 low found all
46 obstacles, but cost $0.030006/request and had 9/12 contact-point passes.

The matched Astra-high control took 34.14s median and cost $0.090326/request.
It matched 11/12 target boxes, 9/12 target/contact checks and 46/46 obstacles.
Do not infer broad model superiority from differences on these few correlated
room photos and manually annotated contact points.

## Cost interpretation

Estimated total for this run: **$3.970076**, before credits, taxes and account
discounts. Average prices use actual input/output usage, include reasoning
tokens once, and apply reported cache reads/writes at the verified rates.

For a fresh request with the same measured output length and entirely fresh
input, including documented cache-write pricing, Sol-off averages **$0.012908**
instead of the observed **$0.009636**. Sol-low is **$0.020981** fresh versus
**$0.017709** observed; Luna-low **$0.001263** fresh versus **$0.001099** observed.
Changing image count, prompt length or generated reasoning changes these costs.
These are list-price estimates, not a billing statement or a promise that
complimentary credits apply.

All 14 configurations and their sources are in [RESULTS.md](RESULTS.md).
Per-request raw usage/responses are in [responses.jsonl](responses.jsonl), and
aggregates with cost ranges and fresh-input estimates in [summary.json](summary.json).
The unchanged image hashes, improved pixel prompt and price snapshot are in
[manifest.json](manifest.json).

Reproduce with `local_nav/compare_vlm_cost.py`. The run used four simultaneous
requests for every configuration, interleaved randomly, rather than the two
used in earlier benchmarks. The new 5.4/Astra controls used that same protocol.
Nothing was pushed to GitHub or deployed to the motion controller.
