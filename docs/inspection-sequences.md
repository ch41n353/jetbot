# Maneuver-and-capture inspection batches

Implemented 2026-09-20. Not loaded into the live robot server or tested with powered motion.

POST `/api/inspection` starts a bounded local worker. GET `/api/inspection` returns its status (`running`, `complete`, `failed`, `cancelled`), measured maneuver events, timestamped original JPEGs as `jpeg_base64`, and `contact_sheet_base64`. GET `/api/progress` reports the worker as running. POST `/api/halt` cancels through the existing motor stop. Partial images survive a failure. The next batch replaces the in-memory result; callers must save it if persistence is required. No automatic file writes or recording are added by this endpoint.

Four views in one request:

```json
{"preset":"four_way"}
```

Captures at the starting heading, then after each of three +90-degree turns: four images spanning the room, ending approximately 270 degrees from the start. Labels use measured relative heading, not requested heading. It does not make a fourth turn back to the starting orientation.

Custom sequence:

```json
{"steps":[
  {"action":"capture"},
  {"action":"turn","degrees":-45},
  {"action":"capture"},
  {"action":"route","points_cm":[[0,20]],"obstacles_cm":[{"label":"block","point_cm":[25,30]}]},
  {"action":"capture"}
]}
```

Route and obstacle coordinates are centimeters in the robot frame **at the start of that route step**: x right, second coordinate forward. They must come from reviewed current geometry, accounting for earlier maneuvers. The batch does not infer clearance, discover new obstacles, or synthesize routes. Routes use the existing avoidance and following functions; an endpoint miss over 5 cm fails the batch instead of silently continuing. Each route is capped at 100 cm; batches at 16 actions and 360 degrees of commanded turns. At least one capture is required. Turn errors over 8 degrees fail the batch. Sensor/power checks run before every action; existing controller checks remain active during motion.

This is a local orchestration tool for the high-level assistant or a planner harness, not a new GPT prompt instruction. There are no GPT calls between captures. Review the contact sheet once and inspect original images for small objects. A four-view scan requires clearance for the whole swept robot footprint; current code does not independently certify that sweep. Recording remains the caller's responsibility under the task lifecycle rules.

## Validation

Motor-free tests exercise the actual localhost HTTP endpoint and worker, four-view ordering, measured headings, invalid input, cancellation with partial images, and route failure/success. Existing planner-page tests also pass. No powered acceptance test and no live-server restart were performed because the user prohibited motion.
