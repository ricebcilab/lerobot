import json
import threading
import time
import urllib.error
import urllib.request

from live_view import LiveView
from teleop import KeyboardReader


def make_view() -> LiveView:
    return LiveView(0, KeyboardReader(), lambda: {})  # port 0: the OS picks one when started


def wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.01)
    assert predicate()


def test_submit_without_an_offer_is_rejected():
    view = make_view()
    assert view.offered_commands() == []
    assert view.submit_command("start") is False


def test_wait_command_returns_the_operators_choice():
    view = make_view()
    out: list[str] = []
    thread = threading.Thread(target=lambda: out.append(view.wait_command(("start", "skip"))))
    thread.start()
    wait_until(lambda: view.offered_commands() == ["start", "skip"])
    assert view.submit_command("quit") is False  # not on offer
    assert view.submit_command("skip") is True
    thread.join(2)
    assert out == ["skip"]
    assert view.offered_commands() == []  # the offer is withdrawn once taken
    assert view.submit_command("skip") is False  # a late click does not queue up for the next wait


def test_http_control_roundtrip():
    view = make_view()
    view.start()
    base = f"http://127.0.0.1:{view._server.server_address[1]}"

    def post(command):
        req = urllib.request.Request(
            f"{base}/control",
            data=json.dumps({"command": command}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status
        except urllib.error.HTTPError as e:
            return e.code

    def status():
        with urllib.request.urlopen(f"{base}/status", timeout=2) as resp:
            return json.load(resp)

    try:
        assert status()["controls"] == []
        assert post("start") == 409  # nothing on offer yet

        out: list[str] = []
        thread = threading.Thread(target=lambda: out.append(view.wait_command(("start", "skip"))))
        thread.start()
        wait_until(lambda: status()["controls"] == ["start", "skip"])
        assert post("bogus") == 409
        assert post("start") == 204
        thread.join(2)
        assert out == ["start"]
        assert status()["controls"] == []
    finally:
        view.close()
