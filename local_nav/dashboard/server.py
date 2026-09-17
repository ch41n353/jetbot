#!/usr/bin/env python3
"""Authenticated dashboard for the autonomous Sol search stack.

The dashboard owns one maneuver_session subprocess.  It never opens the motor
socket itself and never exposes the process environment to HTTP clients.
"""
import argparse
import hmac
import json
import mimetypes
import os
import pathlib
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from http.cookies import SimpleCookie
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

ROOT = pathlib.Path(__file__).resolve().parents[2]
STATIC = pathlib.Path(__file__).resolve().parent / "static"
DEFAULT_OUTPUT = ROOT / "local_nav/goals/dashboard"
DEFAULT_PREVIEWS = ROOT / "local_nav/goals/dashboard/previews"
TARGET_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,160}$")
TERMINAL_EVENTS = {"route_finished", "stopped", "session_closed", "command_rejected"}

#: Shell wrapped around the preview fragment so it renders inside the console's
#: iframe instead of only as a full-size page. Standards mode, dark ground to
#: match the panel, and the camera SVG scaled to fit the frame's width.
PREVIEW_HEAD = (b"<!doctype html><html><head><meta charset=\"utf-8\">"
                b"<style>html,body{margin:0;padding:0;background:#0a0d10;"
                b"color:#8e9aa3;font:11px ui-monospace,SFMono-Regular,Menlo,monospace}"
                b"#turn-sweep-preview,#spatial-route-preview{padding:6px 8px}"
                b"strong{display:block;color:#d6ff3f;font-weight:600;"
                b"letter-spacing:.06em;margin-bottom:4px}"
                b"svg{max-width:100%;height:auto;display:block}"
                b".viz-row{display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start}"
                b"</style></head><body>")
PREVIEW_TAIL = b"</body></html>"


def _safe_json(path):
    try:
        with open(path, encoding="utf-8") as source:
            return json.load(source)
    except (OSError, ValueError, TypeError):
        return None


class Supervisor:
    """Own and summarize one maneuver_session process."""

    def __init__(self, output=DEFAULT_OUTPUT, previews=DEFAULT_PREVIEWS,
                 motion_enabled=False, popen=subprocess.Popen):
        self.output = pathlib.Path(output).resolve()
        self.previews = pathlib.Path(previews).resolve()
        self.motion_enabled = bool(motion_enabled)
        self._popen = popen
        self._lock = threading.RLock()
        self._process = None
        self._reader = None
        self._state = self._empty_state()
        self._plan_path = None
        self._result_path = None
        self._checkpoint_path = None
        self._image_path = None
        self._preview_path = None
        self._preview_kind = None
        self._recognition = None
        self._recognition_path = None
        self._route_event_offset = 0
        self._route_seen = set()
        self._watch_stop = threading.Event()
        self._watcher = None

    def _empty_state(self):
        return {
            "phase": "idle", "running": False, "can_reset": True,
            "motion_enabled": self.motion_enabled, "target": None,
            "timeline": [], "health": {}, "result": None,
            "checkpoint": None, "updated_at": time.time(),
        }

    def _append(self, kind, message, detail=None):
        row = {"id": uuid.uuid4().hex, "time": time.time(),
               "kind": kind, "message": message}
        if detail is not None:
            row["detail"] = detail
        self._state["timeline"].append(row)
        self._state["timeline"] = self._state["timeline"][-250:]
        self._state["updated_at"] = time.time()

    def _send(self, command):
        process = self._process
        if process is None or process.poll() is not None:
            raise RuntimeError("Navigation supervisor is not running")
        process.stdin.write(json.dumps(command) + "\n")
        process.stdin.flush()

    def _spawn(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.previews.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, str(ROOT / "local_nav/maneuver_session.py"),
                "--output-directory", str(self.output),
                "--preview-directory", str(self.previews)]
        if self.motion_enabled:
            argv.append("--enable-motion")
        # Inherit OPENAI_API_KEY for sol_search; never serialize the environment.
        self._process = self._popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def start(self, target):
        target = str(target).strip()
        if not TARGET_RE.fullmatch(target):
            raise ValueError("Target must be 1–160 printable characters")
        with self._lock:
            if self._state["phase"] not in ("idle", "complete", "stopped", "error"):
                raise RuntimeError("A navigation run is already active")
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("Reset the stopped session before starting again")
            self._state = self._empty_state()
            self._state.update(phase="starting", running=True, can_reset=False,
                               target=target)
            self._append("session", "Starting guarded navigation supervisor")
            self._spawn()
        return self.snapshot()

    def stop(self):
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._state.update(phase="stopped", running=False, can_reset=True)
                return self.snapshot()
            self._state.update(phase="stopping", can_reset=False)
            self._append("operator", "Stop requested")
            self._send({"command": "stop"})
        return self.snapshot()

    def reset(self):
        with self._lock:
            if self._state["running"] or self._state["phase"] in ("starting", "stopping"):
                raise RuntimeError("Reset is available only while stopped")
            process = self._process
            if process is not None and process.poll() is None:
                self._send({"command": "shutdown"})
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    raise RuntimeError("Supervisor did not shut down; reset refused")
            self._process = None
            self._plan_path = self._result_path = self._checkpoint_path = None
            self._image_path = self._preview_path = None
            self._preview_kind = self._recognition = None
            self._recognition_path = None
            self._state = self._empty_state()
        return self.snapshot()

    def close(self):
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    self._send({"command": "stop"})
                    self._send({"command": "shutdown"})
                    process.wait(timeout=8)
                except Exception:
                    process.terminate()

    def _write_plan(self, capture):
        self.output.mkdir(parents=True, exist_ok=True)
        self.previews.mkdir(parents=True, exist_ok=True)
        plan_id = "dashboard-search-" + uuid.uuid4().hex
        plan_path = self.output / (plan_id + ".plan.json")
        preview_path = self.previews / (plan_id + ".html")
        checkpoint_path = self.output / (plan_id + ".checkpoint.json")
        plan = {
            key: capture[key] for key in
            ("image_path", "captured_monotonic", "session_id", "control_epoch")
        }
        plan.update({
            "target_label": self._state["target"], "standoff_cm": 25,
            "target_radius_cm": 7,
            # The initial 35 cm turn envelope is user-supplied clearance. The
            # current forward image also covers the approach/braking corridor.
            "inspected_free_rectangle_cm": [-35, -35, 35, 45],
            "obstacle_rectangles_cm": [], "search_stations_cm": [],
            # Post-turn pose is confirmed by tracking carpet features between the
            # pre- and post-turn frames, and Lucas-Kanade cannot match a patch
            # rotated 30 degrees. Ten-degree steps keep that match reliable;
            # recognising every third step preserves a Sol call each 30 degrees,
            # so 360 degrees costs 36 turns and 12 calls, inside both budgets.
            "scan_step_degrees": 10, "scan_direction": 1,
            "max_recognition_calls": 20, "max_search_actions": 72,
            "recognition_every_actions": 3, "search_turn_power": 0.14,
            "search_turn_tolerance_degrees": 3.0,
            "search_turn_brake_margin_degrees": 1.0,
            "scan_turn_controller": "continuous",
            "preview_path": str(preview_path),
            "checkpoint_path": str(checkpoint_path),
        })
        temporary = plan_path.with_suffix(".tmp")
        with open(temporary, "w", encoding="utf-8") as output:
            json.dump(plan, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, plan_path)
        self._plan_path = plan_path
        self._preview_path = preview_path
        self._checkpoint_path = checkpoint_path
        return plan_path

    def _event_message(self, event):
        name = event.get("event", "event")
        if name == "sol_prefetched":
            return "Sent current observation to GPT-5.6 Sol"
        if name == "sol_result":
            answer = event.get("response", {}).get("answer", {})
            return "Sol sees target" if answer.get("target_visible") else "Sol did not see target"
        if name == "search_decision":
            choice = event.get("choice", {})
            if choice.get("kind") == "relocate":
                return "Policy chose local A* relocation"
            return "Policy chose " + str(choice.get("kind", "hold"))
        if name == "trajectory_preview":
            return "Updated trajectory preview"
        if name == "local_action_complete":
            return "Controller completed local action"
        if name == "route_finished":
            result = event.get("result") or {}
            outcome = result.get("outcome")
            if outcome in ("object_reached_estimate", "target_reached_estimate",
                           "object_found"):
                return "Navigation reached its goal"
            reason = result.get("reason") or outcome or "unknown failure"
            return "Navigation stopped: " + str(reason)
        return name.replace("_", " ").capitalize()

    def _latest_step_image(self):
        if self._result_path is None:
            return
        pattern = self._result_path.name + ".step-*.jpg"
        images = sorted(self._result_path.parent.glob(pattern))
        if images:
            self._image_path = images[-1].resolve()

    def _drain_route_events(self):
        """Read complete worker event lines once, without waiting for completion."""
        if self._result_path is None:
            return 0
        event_path = pathlib.Path(str(self._result_path) + ".events.jsonl")
        if not event_path.is_file():
            self._latest_step_image()
            return 0
        count = 0
        with open(str(event_path), "rb") as source:
            source.seek(self._route_event_offset)
            while True:
                start = source.tell()
                line = source.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    source.seek(start)
                    break
                self._route_event_offset = source.tell()
                try:
                    event = json.loads(line.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue
                signature = (event.get("event"), event.get("elapsed_seconds"),
                             self._route_event_offset)
                if signature in self._route_seen:
                    continue
                self._route_seen.add(signature)
                self._handle_route_event(event)
                count += 1
        self._latest_step_image()
        return count

    def _handle_route_event(self, event):
        name = event.get("event")
        category = {
            "sol_prefetched": "sol", "sol_result": "sol",
            "search_decision": "policy", "recognition_skipped": "policy",
            "trajectory_preview": "planner", "local_action_complete": "controller",
            "search_stopped": "controller", "object_found": "result",
            "intervention_required": "result", "localization_required": "result",
        }.get(name, "controller")
        self._append(category, self._event_message(event), event)
        if name == "sol_result":
            # Pin the answer to the frame the model actually saw. Boxes drawn
            # over a later frame would be wrong by however far the robot moved.
            self._latest_step_image()
            answer = (event.get("response") or {}).get("answer")
            if isinstance(answer, dict):
                self._recognition_path = self._image_path
                self._recognition = dict(
                    answer=answer,
                    frame=self._image_path.name if self._image_path else None,
                    elapsed_seconds=event.get("elapsed_seconds"))
        if name == "trajectory_preview" and event.get("path"):
            self._preview_path = pathlib.Path(event["path"]).resolve()
            self._preview_kind = event.get("kind")

    def _watch_route(self):
        while not self._watch_stop.wait(.15):
            with self._lock:
                self._drain_route_events()
        with self._lock:
            self._drain_route_events()

    def _read_loop(self):
        process = self._process
        try:
            for raw in process.stdout:
                try:
                    event = json.loads(raw)
                except ValueError:
                    with self._lock:
                        self._append("diagnostic", "Supervisor output", raw.strip()[:500])
                    continue
                with self._lock:
                    self._handle_event(event)
        finally:
            with self._lock:
                code = process.poll()
                if self._state["running"]:
                    self._state.update(phase="error" if code else "stopped",
                                       running=False, can_reset=True)
                    self._append("session", "Supervisor exited", {"exit_code": code})

    def _handle_event(self, event):
        name = event.get("event")
        if name == "status":
            # snapshot() polls status roughly once a second for as long as a
            # browser is open. It belongs in the health panel only: appending it
            # to the timeline buried the decision history and pushed real events
            # out of the 250-row buffer in about four minutes, which is shorter
            # than a full search.
            self._state["health"] = event.get("status", {})
            self._state["updated_at"] = time.time()
            return
        category = {
            "sol_prefetched": "sol", "sol_result": "sol",
            "search_decision": "policy", "recognition_skipped": "policy",
            "trajectory_preview": "planner", "local_action_complete": "controller",
            "route_started": "controller", "route_finished": "result",
        }.get(name, "system")
        self._append(category, self._event_message(event), event)
        if name == "ready":
            self._state.update(phase="capturing", motion_enabled=event.get("motion_enabled", False))
            self._send({"command": "capture"})
        elif name == "captured":
            self._image_path = pathlib.Path(event["image_path"]).resolve()
            plan = self._write_plan(event)
            self._state["phase"] = "executing"
            self._append("planner", "Created bounded deterministic search plan",
                         {"plan": str(plan), "clearance_cm": 35})
            self._send({"command": "execute", "plan": str(plan), "search": True})
        elif name in ("observed",):
            self._image_path = pathlib.Path(event["image_path"]).resolve()
        elif name == "route_started":
            self._result_path = pathlib.Path(event["result_path"]).resolve()
            self._route_event_offset = 0
            self._route_seen = set()
            self._watch_stop = threading.Event()
            self._watcher = threading.Thread(target=self._watch_route, daemon=True)
            self._watcher.start()
        elif name == "route_finished":
            self._drain_route_events()
            self._watch_stop.set()
            self._state["result"] = event.get("result")
            if self._state["phase"] != "stopped":
                result = event.get("result") or {}
                succeeded = result.get("outcome") in (
                    "object_reached_estimate", "target_reached_estimate", "object_found")
                self._state.update(phase="complete" if succeeded else "error",
                                   running=False, can_reset=True)
            try:
                self._send({"command": "status"})
            except RuntimeError:
                pass
        elif name == "stopped":
            self._state.update(phase="stopped", running=False, can_reset=True)
            try:
                self._send({"command": "status"})
            except RuntimeError:
                pass
        elif name == "command_rejected":
            self._state.update(phase="error", running=False, can_reset=True,
                               result={"outcome": "command_rejected", "reason": event.get("reason")})

    def snapshot(self):
        with self._lock:
            state = json.loads(json.dumps(self._state))
            if self._checkpoint_path:
                state["checkpoint"] = _safe_json(self._checkpoint_path)
            if self._result_path:
                state["result"] = _safe_json(self._result_path) or state.get("result")
            # Tokens change only when the underlying file does. The console used
            # to re-fetch both artifacts on every 750 ms poll while a run was
            # live -- a 78 KB frame plus a 118 KB preview, decoded and re-drawn
            # three times abreast of the controller. On this board that starved
            # the IMU sampling loop until the estimator's 80 ms continuity check
            # ended the run, so simply having the dashboard open broke searches.
            def token(path):
                try:
                    return "%s-%d" % (path.name, path.stat().st_mtime_ns)
                except (OSError, AttributeError):
                    return None
            state["artifacts"] = {
                "image": bool(self._image_path and self._image_path.is_file()),
                "preview": bool(self._preview_path and self._preview_path.is_file()),
                "preview_kind": self._preview_kind,
                "image_name": self._image_path.name if self._image_path else None,
                "image_token": token(self._image_path),
                "preview_token": token(self._preview_path),
            }
            # The overlay is only honest on the frame the model was shown, so
            # say which frame that was and let the client decide to draw it.
            if self._recognition:
                current = (self._image_path is not None and
                           self._recognition.get("frame") == self._image_path.name)
                state["recognition"] = dict(self._recognition, current_frame=current)
            # Status is requested through the owner process, never the motor socket.
            process = self._process
            if process is not None and process.poll() is None:
                now = time.monotonic()
                if now - getattr(self, "_last_status", 0) > 1:
                    try:
                        self._send({"command": "status"})
                        self._last_status = now
                    except RuntimeError:
                        pass
            return state

    def _overlay_svg(self):
        """Recognizer geometry as SVG in the 640x480 frame both views use."""
        if not self._recognition:
            return ""
        answer = self._recognition.get("answer") or {}
        out = []
        for obstacle in answer.get("obstacles") or []:
            box = obstacle.get("box") or {}
            try:
                x0, y0 = float(box["x0"]), float(box["y0"])
                x1, y1 = float(box["x1"]), float(box["y1"])
            except (KeyError, TypeError, ValueError):
                continue
            label = str(obstacle.get("label", ""))[:40].replace("&", "&amp;")
            label = label.replace("<", "&lt;").replace(">", "&gt;")
            out.append('<rect x="%g" y="%g" width="%g" height="%g" fill="none" '
                       'stroke="#ffb03a" stroke-width="2" stroke-dasharray="6 4"/>'
                       '<text x="%g" y="%g" fill="#ffb03a" font-family="monospace" '
                       'font-size="12">%s</text>'
                       % (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0),
                          min(x0, x1) + 2, min(y0, y1) - 4 if y0 > 14 else max(y0, y1) + 13,
                          label))
        route = []
        for point in answer.get("route_pixels") or []:
            try:
                route.append((float(point["x"]), float(point["y"])))
            except (KeyError, TypeError, ValueError):
                continue
        if route:
            out.append('<polyline points="%s" fill="none" stroke="#43d4ff" '
                       'stroke-width="2.5" stroke-dasharray="9 5"/>'
                       % " ".join("%g,%g" % p for p in route))
            for index, (x, y) in enumerate(route):
                out.append('<circle cx="%g" cy="%g" r="5" fill="#43d4ff"/>'
                           '<text x="%g" y="%g" fill="#43d4ff" font-family="monospace" '
                           'font-size="11">%d</text>' % (x, y, x + 8, y + 4, index + 1))
        box = answer.get("target_box")
        if answer.get("target_visible") and isinstance(box, dict):
            try:
                x0, y0 = float(box["x0"]), float(box["y0"])
                x1, y1 = float(box["x1"]), float(box["y1"])
                out.append('<rect x="%g" y="%g" width="%g" height="%g" fill="none" '
                           'stroke="#7cff8a" stroke-width="2.5"/>'
                           '<text x="%g" y="%g" fill="#7cff8a" font-family="monospace" '
                           'font-size="12">TARGET</text>'
                           % (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0),
                              min(x0, x1), min(y0, y1) - 5))
            except (KeyError, TypeError, ValueError):
                pass
        contact = answer.get("contact_pixel")
        if isinstance(contact, dict):
            try:
                x, y = float(contact["x"]), float(contact["y"])
                out.append('<circle cx="%g" cy="%g" r="6" fill="none" stroke="#ff4d5e" '
                           'stroke-width="2.5"/><line x1="%g" y1="%g" x2="%g" y2="%g" '
                           'stroke="#ff4d5e" stroke-width="1.5"/><line x1="%g" y1="%g" '
                           'x2="%g" y2="%g" stroke="#ff4d5e" stroke-width="1.5"/>'
                           % (x, y, x - 11, y, x + 11, y, x, y - 11, x, y + 11))
            except (KeyError, TypeError, ValueError):
                pass
        return "".join(out)

    def artifact(self, kind):
        with self._lock:
            if kind == "image":
                # Show the frame the recognizer was given, so the overlay and the
                # pixels underneath it always describe the same moment.
                path = self._recognition_path or self._image_path
                if path is None or not path.is_file():
                    return None
                return path.read_bytes()
            path = self._preview_path
            if path is None or not path.is_file():
                return None
            body = path.read_bytes()
            overlay = self._overlay_svg()
            if overlay:
                # The preview draws its corridor over the same 640x480 camera
                # SVG, so the recognizer geometry drops straight into it.
                at = body.find(b"</svg>")
                if at >= 0:
                    body = body[:at] + overlay.encode("utf-8") + body[at:]
            # The writers emit a bare fragment. Served straight into an iframe
            # that renders it in quirks mode on a white page, taller than the
            # panel, so the operator saw an empty strip and had to open it full
            # size. Wrap it in a real document that fits the frame and matches
            # the console, without touching the files the writers produce.
            return PREVIEW_HEAD + body + PREVIEW_TAIL


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "JetBotDashboard/1"

    def _local_request(self):
        host = self.headers.get("Host", "").split(":", 1)[0]
        return host in ("127.0.0.1", "localhost", "[::1]")

    def _authorized(self):
        token = self.server.access_token
        if token is None:
            return self.server.allow_remote_unauthenticated or self._local_request()
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
            supplied = cookie["jetbot_access"].value
        except (KeyError, AttributeError):
            return False
        return hmac.compare_digest(supplied, token)

    def _accept_token_link(self, parsed):
        """Exchange a one-time URL parameter for a same-origin cookie."""
        token = self.server.access_token
        if token is None or parsed.path != "/":
            return False
        supplied = parse_qs(parsed.query).get("token", [""])[0]
        if not supplied or not hmac.compare_digest(supplied, token):
            return False
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header(
            "Set-Cookie",
            "jetbot_access=%s; Path=/; HttpOnly; SameSite=Strict" % token,
        )
        self.send_header("Location", "/")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        return True

    def _headers(self, status=200, content_type="application/json", length=None,
                 preview=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        policy = "sandbox allow-scripts" if preview else "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'"
        self.send_header("Content-Security-Policy", policy)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.end_headers()

    def _json(self, payload, status=200):
        body = json.dumps(payload, separators=(",", ":")).encode()
        self._headers(status, length=len(body))
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if self._accept_token_link(parsed):
            return
        if not self._authorized():
            return self._json({"error": "dashboard access token required"}, 403)
        path = parsed.path
        if path == "/api/state":
            return self._json(self.server.supervisor.snapshot())
        if path in ("/api/image", "/api/preview"):
            kind = path.rsplit("/", 1)[-1]
            body = self.server.supervisor.artifact(kind)
            if body is None:
                return self._json({"error": "artifact unavailable"}, 404)
            content = "image/jpeg" if kind == "image" else "text/html; charset=utf-8"
            self._headers(200, content, len(body), preview=kind == "preview")
            return self.wfile.write(body)
        if path == "/":
            path = "/index.html"
        name = path.lstrip("/")
        if name not in ("index.html", "app.js", "styles.css"):
            return self._json({"error": "not found"}, 404)
        body = (STATIC / name).read_bytes()
        content = mimetypes.guess_type(name)[0] or "application/octet-stream"
        self._headers(200, content, len(body))
        self.wfile.write(body)

    def do_POST(self):
        if not self._authorized():
            return self._json({"error": "dashboard access token required"}, 403)
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            return self._json({"error": "application/json required"}, 415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            path = urlparse(self.path).path
            if path == "/api/start":
                result = self.server.supervisor.start(payload.get("target", ""))
            elif path == "/api/stop":
                result = self.server.supervisor.stop()
            elif path == "/api/reset":
                result = self.server.supervisor.reset()
            else:
                return self._json({"error": "not found"}, 404)
            self._json(result)
        except (ValueError, RuntimeError, TypeError) as exc:
            self._json({"error": str(exc)}, 409)

    def log_message(self, fmt, *args):
        sys.stderr.write("dashboard: " + fmt % args + "\n")


class DashboardServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, supervisor, access_token=None):
        super().__init__(address, DashboardHandler)
        self.supervisor = supervisor
        self.access_token = access_token
        self.allow_remote_unauthenticated = address[0] == "0.0.0.0"


def lan_address():
    """Return the preferred outbound LAN address without sending a packet."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1",
                        choices=("127.0.0.1", "localhost", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--access-token",
                        help="optional access token for LAN clients")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--preview-directory", default=str(DEFAULT_PREVIEWS))
    args = parser.parse_args()
    if args.access_token and len(args.access_token) < 16:
        parser.error("--access-token must contain at least 16 characters")
    access_token = args.access_token
    supervisor = Supervisor(args.output_directory, args.preview_directory, args.enable_motion)
    server = DashboardServer((args.host, args.port), supervisor, access_token)
    display_host = lan_address() if args.host == "0.0.0.0" else args.host
    url = "http://%s:%d/" % (display_host, server.server_address[1])
    if access_token:
        url += "?token=" + access_token
    print("JetBot dashboard: " + url, flush=True)
    if args.host == "0.0.0.0":
        print("LAN mode is visible to every device on this network.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        supervisor.close()
        server.server_close()


if __name__ == "__main__":
    main()
