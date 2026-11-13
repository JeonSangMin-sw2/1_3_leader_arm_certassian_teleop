#!/usr/bin/env python3
"""정지 절차 검사 (시뮬레이터, 가짜 리더암): Ctrl+C, 리더암 장애(소프트/즉시)에서 main()이 예제 35 순서로 멈추는지.

가짜 리더암은 레디자세에 머물며, stop(soft)이 불린 시각과 soft 여부를 기록한다.
각 경우 뒤에 컨트롤 매니저 상태와 12V·전체 전원 상태를 보고, ensure_ready로 다시 켜지는지 확인한다.

사용: .venv/bin/python tests/shutdown_sim.py
"""
import os
import signal
import sys
import threading
import time

import rby1_sdk as rby

import _sim
import leader_arm
import teleop
from robot import Robot


class FakeLeader:
    calls = []
    fail_after = None  # 이 시간 [s] 뒤부터 콜백이 끊긴 것처럼 alive가 False

    def __init__(self, cfg):
        self.t0 = None

    def start(self):
        self.t0 = time.monotonic()

    def alive(self, timeout):
        return self.fail_after is None or time.monotonic() - self.t0 < self.fail_after

    def align(self, target_q, duration, stop_event):  # 움직이지 않고 호출만 기록한다
        FakeLeader.calls.append(("leader.align", len(target_q), round(time.monotonic() - self.t0, 2)))

    def latest(self):
        return _sim.both(_sim.READY), (False, False), (0.0, 0.0)

    def stop(self, soft=True):
        FakeLeader.calls.append(("leader.stop", soft, round(time.monotonic() - self.t0, 2)))


def state(r):
    cm = r.get_control_manager_state()
    names = [p.name for p in r.get_robot_info().power_infos]
    power = {n: r.is_power_on(n) for n in names}
    return f"cm={cm.state.name}/{cm.control_state.name} power={power}"


def run_main(extra_args, fail_after=None, sigint_after=None):
    FakeLeader.calls, FakeLeader.fail_after = [], fail_after
    leader_arm.LeaderArm = FakeLeader
    teleop.LOG_DIR = _sim.Path(_sim.tempfile.mkdtemp())
    sys.argv = ["teleop.py", "--robot_address", _sim.SIM, "--ready_minimum_time_s", "3.0"] + extra_args
    if sigint_after:
        threading.Timer(sigint_after, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    t0 = time.monotonic()
    err = None
    try:
        teleop.main()
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return time.monotonic() - t0, err


def main():
    r = rby.create_robot(_sim.SIM, "m")
    assert r.connect()
    cases = [
        ("Ctrl+C", [], None, 6.0),
        ("리더암 장애, fault_soft_stop true", [], 6.0, None),
        ("리더암 장애, fault_soft_stop false", ["--fault_soft_stop", "false"], 6.0, None),
    ]
    for name, args, fail_after, sigint_after in cases:
        print(f"=== {name}", flush=True)
        dur, err = run_main(args, fail_after, sigint_after)
        print(f"  main 종료까지 {dur:.1f} s, 예외: {err}")
        print(f"  leader.stop 호출: {FakeLeader.calls}")
        print(f"  끝난 뒤 {state(r)}")
        Robot(_sim.make_cfg()).ensure_ready()
        print(f"  ensure_ready 뒤 {state(r)}", flush=True)


if __name__ == "__main__":
    main()
