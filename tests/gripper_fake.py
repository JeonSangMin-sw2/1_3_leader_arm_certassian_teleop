#!/usr/bin/env python3
"""그리퍼 오프라인 검사 (장치 없이 가짜 Dynamixel 버스). homing 범위, 트리거→목표 방향, 반전, 시간 초과를 본다.

사용: .venv/bin/python tests/gripper_fake.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gripper  # noqa: E402

LO, HI = 0.2, 1.8  # 가짜 손가락 엔코더 이동 범위


class FakeBus:
    stuck = False  # True면 끝에 닿지 않고 계속 움직인다 (시간 초과 확인)
    CurrentControlMode = gripper.rby.DynamixelBus.CurrentControlMode
    CurrentBasedPositionControlMode = gripper.rby.DynamixelBus.CurrentBasedPositionControlMode

    def __init__(self, name):
        self.q = np.array([1.0, 1.0])
        self.torque = np.zeros(2)
        self.positions = []

    def open_port(self): return True
    def set_baud_rate(self, b): return True
    def set_torque_constant(self, c): pass
    def ping(self, i): return True
    def group_sync_write_torque_enable(self, pairs): pass
    def group_sync_write_operating_mode(self, pairs): pass

    def group_sync_write_send_torque(self, pairs):
        for i, t in pairs:
            self.torque[i] = t

    def group_fast_sync_read_encoder(self, ids):
        step = 0.05 * np.sign(self.torque)
        self.q = self.q + step if self.stuck else np.clip(self.q + step, LO, HI)
        return [(i, float(self.q[i])) for i in ids]

    def group_sync_write_send_position(self, pairs):
        self.positions.append(dict(pairs))


def make(**cfg):
    gripper.rby.DynamixelBus = FakeBus
    gripper.time.sleep = lambda s: None  # homing을 빨리 돌린다
    return gripper.Gripper({"gripper_max_open": 1000, "gripper_invert_trigger": False, **cfg})


def main():
    real_sleep = time.sleep
    g = make()
    g.connect()
    assert np.allclose(g._min_q, LO) and np.allclose(g._max_q, HI), (g._min_q, g._max_q)
    checks = []
    g.set_trigger((0, 1000)); checks.append(("트리거 0 → 열림(min), 1000 → 닫힘(max)", g._target_q, [LO, HI]))
    g.set_trigger((500, 500)); checks.append(("트리거 500 → 가운데", g._target_q, [1.0, 1.0]))
    g.disconnect()

    g = make(gripper_max_open=500)
    g.connect()
    g.set_trigger((0, 0)); checks.append(("max_open 500, 트리거 0 → 절반만 염", g._target_q, [1.0, 1.0]))
    g.disconnect()

    g = make(gripper_invert_trigger=True)
    g.connect()
    g.set_trigger((0, 1000)); checks.append(("반전, 트리거 0 → 닫힘, 1000 → 열림", g._target_q, [HI, LO]))
    gripper.time.sleep = real_sleep
    real_sleep(0.35)  # 루프가 위치를 쓰는지
    written = len(g._bus.positions)
    g.disconnect()

    ok = True
    for name, got, want in checks:
        good = np.allclose(got, want)
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {name}: {np.round(got, 3).tolist()} (기대 {want})")
    print(f"  {'OK ' if written >= 2 else 'BAD'} 루프 위치 쓰기 {written}회 / 0.35 s")
    ok &= written >= 2

    FakeBus.stuck = True
    g = make()
    try:
        g.connect()
        print("  BAD 끝에 안 닿는데 homing이 끝남"); ok = False
    except RuntimeError as e:
        print(f"  OK  homing 시간 초과 오류: {str(e)[:40]}...")
    print("통과" if ok else "실패")


if __name__ == "__main__":
    main()
