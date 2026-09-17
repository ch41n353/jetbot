import json
import http.client
import math
import os
import pathlib
import sys
import tempfile
import threading
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from local_nav.dashboard.server import DashboardHandler, DashboardServer, Supervisor


class FakeSupervisor:
    def __init__(self):
        self.calls = []
        self.state = {"phase": "idle", "running": False, "can_reset": True,
                      "timeline": [], "health": {}, "result": None,
                      "checkpoint": None, "updated_at": 1,
                      "artifacts": {"image": False, "preview": False}}

    def snapshot(self):
        return self.state

    def start(self, target):
        self.calls.append(("start", target))
        self.state.update(phase="starting", running=True, can_reset=False, target=target)
        return self.state

    def stop(self):
        self.calls.append(("stop",))
        return self.state

    def reset(self):
        self.calls.append(("reset",))
        return self.state

    def artifact(self, kind):
        return None


class DashboardTests(unittest.TestCase):
    def test_worker_spawn_uses_python_36_compatible_text_mode(self):
        captured = {}

        class Process:
            stdout = ()

            def poll(self):
                return 0

        def popen(argv, **kwargs):
            captured.update(kwargs)
            return Process()

        supervisor = Supervisor(popen=popen)
        supervisor._spawn()
        supervisor._reader.join(timeout=1)
        self.assertTrue(captured["universal_newlines"])
        self.assertNotIn("text", captured)

    def test_plan_is_bounded_and_contains_no_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            supervisor = Supervisor(pathlib.Path(directory) / "runs",
                                    pathlib.Path(directory) / "previews")
            supervisor._state["target"] = "Advil bottle"
            capture = {"image_path": "/tmp/frame.jpg", "captured_monotonic": 12.5,
                       "session_id": "session", "control_epoch": 3}
            path = supervisor._write_plan(capture)
            plan = json.loads(path.read_text())
            self.assertEqual(plan["target_label"], "Advil bottle")
            self.assertEqual(plan["inspected_free_rectangle_cm"], [-35, -35, 35, 45])
            # Small enough that carpet features still match across one step.
            self.assertEqual(plan["scan_step_degrees"], 10)
            self.assertEqual(plan["recognition_every_actions"], 3)
            # A full scan must stay inside both the call and action budgets.
            steps = math.ceil(360 / plan["scan_step_degrees"])
            self.assertLessEqual(steps, plan["max_search_actions"])
            self.assertLessEqual(1 + steps // plan["recognition_every_actions"],
                                 plan["max_recognition_calls"])
            self.assertEqual(plan["search_turn_tolerance_degrees"], 3.0)
            self.assertEqual(plan["search_turn_brake_margin_degrees"], 1.0)
            self.assertNotIn("OPENAI_API_KEY", path.read_text())

    def test_reset_refused_while_running(self):
        supervisor = Supervisor()
        supervisor._state.update(phase="executing", running=True, can_reset=False)
        with self.assertRaisesRegex(RuntimeError, "only while stopped"):
            supervisor.reset()

    def test_target_validation(self):
        supervisor = Supervisor(popen=lambda *args, **kwargs: None)
        with self.assertRaisesRegex(ValueError, "printable"):
            supervisor.start("bad\ntarget")

    def test_frontend_uses_service_voltage_field_names(self):
        source = (pathlib.Path(__file__).parents[1] /
                  "local_nav/dashboard/static/app.js").read_text()
        self.assertIn("power.pack_voltage_v", source)
        self.assertIn("power.voltage_v", source)

    def test_stop_state_is_not_overwritten_by_late_route_finished(self):
        supervisor = Supervisor()
        supervisor._state.update(phase="stopped", running=False, can_reset=True)
        supervisor._process = type("P", (), {"poll": lambda self: 0})()
        supervisor._handle_event({"event": "route_finished", "result": {"outcome": "stopped"}})
        self.assertEqual(supervisor._state["phase"], "stopped")

    def test_status_updates_health_without_flooding_the_timeline(self):
        supervisor = Supervisor()
        for _ in range(300):
            supervisor._handle_event({"event": "status",
                                      "status": {"healthy": True, "motor": {"output": [0, 0]}}})
        self.assertEqual(supervisor._state["timeline"], [])
        self.assertTrue(supervisor._state["health"]["healthy"])
        # Decisions must survive a long run's worth of status polling.
        supervisor._handle_event({"event": "search_decision",
                                  "choice": {"kind": "turn", "degrees": 10.0}})
        for _ in range(300):
            supervisor._handle_event({"event": "status", "status": {"healthy": True}})
        kinds = [row["kind"] for row in supervisor._state["timeline"]]
        self.assertEqual(kinds, ["policy"])

    def test_recognition_is_pinned_to_the_frame_the_model_was_shown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            supervisor = Supervisor(output=str(root), previews=str(root))
            supervisor._result_path = root / "route-01.json"
            answer = {"target_visible": True,
                      "target_box": {"x0": 1, "y0": 2, "x1": 3, "y1": 4},
                      "obstacles": [], "route_pixels": []}
            (root / "route-01.json.step-00.jpg").write_bytes(b"x")
            supervisor._handle_route_event({"event": "sol_result",
                                            "response": {"answer": answer}})
            state = supervisor.snapshot()
            self.assertEqual(state["recognition"]["frame"], "route-01.json.step-00.jpg")
            self.assertTrue(state["recognition"]["current_frame"])
            # A later frame must mark the overlay stale rather than drawing
            # last recognition's boxes over pixels the robot has moved past.
            (root / "route-01.json.step-01.jpg").write_bytes(b"x")
            supervisor._latest_step_image()
            self.assertFalse(supervisor.snapshot()["recognition"]["current_frame"])

    def test_preview_kind_is_reported_for_the_route_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            preview = pathlib.Path(directory) / "preview.html"
            preview.write_text("<html></html>")
            supervisor = Supervisor(output=directory, previews=directory)
            supervisor._handle_route_event({"event": "trajectory_preview",
                                            "path": str(preview), "kind": "approach"})
            self.assertEqual(supervisor.snapshot()["artifacts"]["preview_kind"], "approach")

    def test_failed_route_is_reported_as_error_with_reason(self):
        supervisor = Supervisor()
        supervisor._state.update(phase="executing", running=True, can_reset=False)
        supervisor._handle_event({
            "event": "route_finished",
            "result": {"outcome": "stopped", "reason": "no_checked_route"},
        })
        self.assertEqual(supervisor._state["phase"], "error")
        self.assertFalse(supervisor._state["running"])
        self.assertIn("no_checked_route", supervisor._state["timeline"][-1]["message"])

    def test_reached_route_is_the_only_completed_route(self):
        supervisor = Supervisor()
        supervisor._state.update(phase="executing", running=True, can_reset=False)
        supervisor._handle_event({
            "event": "route_finished",
            "result": {"outcome": "object_reached_estimate"},
        })
        self.assertEqual(supervisor._state["phase"], "complete")

    def test_live_worker_events_are_ingested_before_route_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            supervisor = Supervisor(directory, pathlib.Path(directory) / "previews")
            result = pathlib.Path(directory) / "route-01.json"
            supervisor._result_path = result
            event_path = pathlib.Path(str(result) + ".events.jsonl")
            event_path.write_text(
                json.dumps({"event": "sol_result", "elapsed_seconds": 1.2,
                            "response": {"answer": {"target_visible": False}}}) + "\n" +
                json.dumps({"event": "search_decision", "elapsed_seconds": 1.3,
                            "choice": {"kind": "turn"}}) + "\n")
            self.assertEqual(supervisor._drain_route_events(), 2)
            kinds = [row["kind"] for row in supervisor._state["timeline"]]
            self.assertEqual(kinds, ["sol", "policy"])
            self.assertIsNone(supervisor._state["result"])
            self.assertEqual(supervisor._drain_route_events(), 0)

    def test_http_host_guard_accepts_only_loopback_names(self):
        handler = object.__new__(DashboardHandler)
        for host in ("127.0.0.1:8765", "localhost:8765"):
            handler.headers = {"Host": host}
            self.assertTrue(handler._local_request())
        handler.headers = {"Host": "evil.example"}
        self.assertFalse(handler._local_request())

    def test_explicit_lan_binding_can_run_without_token(self):
        handler = object.__new__(DashboardHandler)
        handler.headers = {"Host": "192.168.86.158:8765"}
        handler.server = types.SimpleNamespace(
            access_token=None, allow_remote_unauthenticated=True)
        self.assertTrue(handler._authorized())

    def test_lan_mode_requires_token_then_sets_cookie(self):
        token = "0123456789abcdef0123456789abcdef"
        server = DashboardServer(("127.0.0.1", 0), FakeSupervisor(), token)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        try:
            connection.request("GET", "/api/state")
            response = connection.getresponse()
            self.assertEqual(response.status, 403)
            response.read()

            connection.request("GET", "/?token=" + token)
            response = connection.getresponse()
            self.assertEqual(response.status, 303)
            cookie = response.getheader("Set-Cookie").split(";", 1)[0]
            response.read()

            connection.request("GET", "/api/state", headers={"Cookie": cookie})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["phase"], "idle")
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
