const $ = (id) => document.getElementById(id);
let lastImage = null, lastPreview = null, busy = false;

function esc(s) {
  return String(s ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}
function timeLabel(seconds) {
  return new Date(seconds * 1000).toLocaleTimeString([], {hour12:false, hour:'2-digit', minute:'2-digit', second:'2-digit'});
}
function number(value, digits=2) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}
async function command(path, body={}) {
  busy = true; renderButtons(); $('error').style.display = 'none';
  try {
    const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed');
    render(data);
  } catch (error) {
    $('error').textContent = error.message; $('error').style.display = 'block';
  } finally { busy = false; renderButtons(); }
}
function renderButtons(state=window.currentState || {phase:'idle',running:false,can_reset:true}) {
  $('start').disabled = busy || !state.motion_enabled || !['idle','complete','stopped','error'].includes(state.phase);
  $('stop').disabled = busy || !state.running;
  $('reset').disabled = busy || !state.can_reset;
  $('target').disabled = busy || state.running;
}
function renderEvents(events) {
  $('eventCount').textContent = `${events.length} EVENT${events.length===1?'':'S'}`;
  if (!events.length) return $('events').innerHTML = '<div class="empty-line">No mission events yet.</div>';
  $('events').innerHTML = events.slice().reverse().map(e => `<article class="event ${esc(e.kind)}">
    <time>${timeLabel(e.time)}</time><span class="event-type">${esc(e.kind)}</span><p>${esc(e.message)}</p>
    ${e.detail ? `<details><summary>RAW EVENT</summary><pre>${esc(JSON.stringify(e.detail,null,2))}</pre></details>` : ''}</article>`).join('');
}
const OVERLAY = {target:'#7cff8a', contact:'#ff4d5e', obstacle:'#ffb03a', route:'#43d4ff'};

function box(b, colour, label, dash) {
  if (!b) return '';
  const x = Math.min(b.x0,b.x1), y = Math.min(b.y0,b.y1);
  const w = Math.abs(b.x1-b.x0), h = Math.abs(b.y1-b.y0);
  const text = label ? `<text x="${x+2}" y="${y>12?y-4:y+h+12}" fill="${colour}"
    font-family="ui-monospace,monospace" font-size="13">${esc(label)}</text>` : '';
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="${colour}"
    stroke-width="2" ${dash?`stroke-dasharray="${dash}"`:''}/>${text}`;
}
function renderOverlay(state) {
  const svg = $('overlay'), tag = $('overlayState');
  // A stale cached index.html has no overlay element. Without this guard the
  // throw aborted render() before the camera and preview panels were updated,
  // so one new panel silently blanked two working ones.
  if (!svg || !tag) return;
  const recognition = state.recognition;
  if (!recognition || !state.artifacts?.image) {
    svg.innerHTML = ''; tag.textContent = 'NO RECOGNITION YET'; return;
  }
  const a = recognition.answer || {};
  let parts = [];
  for (const o of a.obstacles || []) parts.push(box(o.box, OVERLAY.obstacle, o.label, '6 4'));
  const route = (a.route_pixels || []).filter(p => Number.isFinite(p?.x) && Number.isFinite(p?.y));
  if (route.length) {
    parts.push(`<polyline points="${route.map(p=>`${p.x},${p.y}`).join(' ')}" fill="none"
      stroke="${OVERLAY.route}" stroke-width="2.5" stroke-dasharray="9 5"/>`);
    route.forEach((p,i) => parts.push(`<circle cx="${p.x}" cy="${p.y}" r="5" fill="${OVERLAY.route}"/>
      <text x="${p.x+8}" y="${p.y+4}" fill="${OVERLAY.route}" font-family="ui-monospace,monospace"
      font-size="12">${i+1}</text>`));
  }
  if (a.target_visible && a.target_box) parts.push(box(a.target_box, OVERLAY.target, 'TARGET'));
  if (a.contact_pixel) {
    const {x,y} = a.contact_pixel;
    parts.push(`<circle cx="${x}" cy="${y}" r="6" fill="none" stroke="${OVERLAY.contact}" stroke-width="2.5"/>
      <line x1="${x-11}" y1="${y}" x2="${x+11}" y2="${y}" stroke="${OVERLAY.contact}" stroke-width="1.5"/>
      <line x1="${x}" y1="${y-11}" x2="${x}" y2="${y+11}" stroke="${OVERLAY.contact}" stroke-width="1.5"/>`);
  }
  svg.innerHTML = parts.join('');
  const counts = [a.target_visible ? 'TARGET VISIBLE' : 'TARGET ABSENT',
                  `${(a.obstacles||[]).length} OBSTACLES`];
  if (route.length) counts.push(`${route.length}-POINT ROUTE`);
  // The frame shown is the one the model was given, so the boxes always match
  // the pixels. Say so when the robot has since moved on.
  counts.push(recognition.current_frame ? 'CURRENT FRAME' : 'FRAME AT LAST RECOGNITION');
  tag.textContent = counts.join(' · ');
}
function renderHealth(health) {
  const power = health.power || {}, motor = health.motor || {};
  const pack = power.pack_voltage_v;
  const rail = power.voltage_v;
  $('pack').textContent = number(pack); $('rail').textContent = number(rail);
  $('cameraAge').textContent = number((health.camera_age ?? health.camera?.age_seconds) * 1000, 0);
  $('imuAge').textContent = number((health.imu_age ?? health.imu?.age_seconds) * 1000, 0);
  const output = motor.output || [0,0]; $('motors').textContent = `${number(output[0])} / ${number(output[1])}`;
  $('guard').textContent = power.motion_allowed ? 'ARMED' : 'BLOCKED';
  $('health').textContent = health.healthy === true ? 'HEALTHY' : health.healthy === false ? 'FAULT' : 'UNKNOWN';
}
function renderResult(state) {
  const result = state.result, checkpoint = state.checkpoint;
  if (result) {
    const outcome = result.outcome || 'finished';
    const arrived = outcome.includes('reached');
    const found = outcome === 'object_found';
    // A bare glyph told the operator nothing. Name the three outcomes that
    // differ in what the robot actually did: arrived, recognised but did not
    // move, or stopped on a guard.
    const icon = arrived ? '✓' : found ? '👁' : '✕';
    const kind = arrived ? 'REACHED' : found ? 'SEEN, NOT REACHED' : 'STOPPED';
    const reason = result.reason ? ` — ${result.reason}` : '';
    $('result').innerHTML = `<span class="result-icon ${arrived?'good':found?'seen':'halt'}">${icon}</span>
      <p><strong>${esc(kind)}</strong> · ${esc(outcome.split('_').join(' '))}${esc(reason)}</p>`;
  } else $('result').innerHTML = '<span class="result-icon">—</span><p>Waiting for a mission.</p>';
  $('checkpoint').innerHTML = checkpoint ? `Phase ${esc(checkpoint.phase)} · ${checkpoint.sol_calls ?? 0} Sol calls · ${checkpoint.action_count ?? 0} local actions<br>Auto-resume: disabled` : '';
}
function section(name, run) {
  // Panels are independent: a failure in one must not blank the others. Before
  // this, an exception anywhere above the artifact blocks stopped the camera
  // frame and the trajectory preview from ever being set.
  try { run(); } catch (error) { console.error('panel failed: ' + name, error); }
}
function render(state) {
  window.currentState = state;
  $('phase').textContent = state.phase.toUpperCase();
  $('liveDot').classList.toggle('live', state.running);
  if (state.target) $('target').value = state.target;
  section('buttons', () => renderButtons(state));
  section('events', () => renderEvents(state.timeline || []));
  section('health', () => renderHealth(state.health || {}));
  section('result', () => renderResult(state));
  section('overlay', () => renderOverlay(state));
  $('modeNotice').textContent = state.motion_enabled ?
    'MOTION MODE ENABLED — Start Search will capture, plan, and execute.' :
    'PREVIEW MODE — Start Search is disabled. Restart the dashboard with --enable-motion after charging.';
  $('modeNotice').classList.toggle('armed', state.motion_enabled);
  section('camera', () => {
    if (!state.artifacts?.image) return;
    // Refetch only when the file itself changed. Reloading on every poll cost
    // the robot its IMU continuity, not just bandwidth.
    const token = state.artifacts.image_token || state.updated_at;
    if (lastImage !== token) { $('camera').src = `/api/image?t=${token}`; lastImage = token; }
    $('camera').style.display='block'; $('cameraEmpty').style.display='none';
    $('imageState').textContent='LIVE RGB';
  });
  section('preview', () => {
    if (!state.artifacts?.preview) return;
    // Make the frame visible FIRST. Setting src and then touching an element an
    // older cached page lacks left the preview loaded but display:none.
    $('preview').style.display='block'; $('previewEmpty').style.display='none';
    const token = state.artifacts.preview_token || state.updated_at;
    if (lastPreview !== token) { $('preview').src = `/api/preview?t=${token}`; lastPreview = token; }
    const kind = state.artifacts.preview_kind;
    const open = $('previewOpen'), note = $('previewKind'), tag = $('previewState');
    if (open) open.href = `/api/preview?t=${state.updated_at}`;
    if (tag) tag.textContent = kind ? String(kind).split('_').join(' ').toUpperCase() : 'CURRENT';
    if (note) note.textContent = kind === 'approach'
      ? 'APPROACH CORRIDOR · PLAN, NOT A COLLISION SENSOR'
      : 'SWEPT CORRIDOR, NOT A COLLISION SENSOR';
  });
}
async function poll() {
  try { const r = await fetch('/api/state', {cache:'no-store'}); if (r.ok) render(await r.json()); } catch (_) {}
}
$('goalForm').addEventListener('submit', e => { e.preventDefault(); command('/api/start', {target:$('target').value}); });
$('stop').addEventListener('click', () => command('/api/stop'));
$('reset').addEventListener('click', () => { lastImage=lastPreview=null; $('camera').style.display='none'; $('preview').style.display='none'; $('cameraEmpty').style.display='grid'; $('previewEmpty').style.display='grid'; command('/api/reset'); });
setInterval(() => $('clock').textContent = new Date().toLocaleTimeString([], {hour12:false}), 1000);
setInterval(poll, 750); poll();
