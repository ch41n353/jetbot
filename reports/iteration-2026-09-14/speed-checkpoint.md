# Speed checkpoint — 2026-09-14

Implemented padded target optical-flow crops and cropped floor processing.
139 tests passed in98.489seconds. Robot remained stopped throughout; no powered
performance claim is made.

Recorded target replay: median39.43→11.72ms, p9556.53→16.16ms, zero losses on
37frames repeated3times. Floor replay: median62.22→44.97ms across111pairs,
zero outcome mismatches and maximum0.0036cm translation difference. These are
separate component benchmarks, not a measured complete-loop latency.

The baseline angled simulation used7.475s, of which3.450s commanded zero output;
the aligned case used8.625s with7.475s continuous driving. Reducing modeled
target processing from45ms to25ms produced7.220s and8.550s respectively, both
within tolerance with no audited clearance violations. That scenario predates
the final floor crop and is not a hardware-speed guarantee.

An earlier waypoint handoff increased target loss. Direct initial alignment
caused repeated turn/settle cycles until the uncertainty guard stopped it.
Both were reverted; retained reports identify failures. Reusing stationary
history did not improve the paired timing and was also reverted.

Next substantial motion bottleneck: turn execution and target recovery, rather
than scheduled pauses on the aligned approach. Safety thresholds and motor
power are unchanged. Skill revision12 records tools, limits and benchmarks.
