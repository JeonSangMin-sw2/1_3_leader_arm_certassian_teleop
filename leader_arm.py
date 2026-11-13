"""리더암 읽기. 버튼을 누르고 있는 팔은 중력 보상으로 자유롭게, 놓은 팔은 그 자리에 고정한다.

참고: SDK 예제 35 (버튼·토크 동작과 토크 값).
"""

import threading

import numpy as np
import rby1_sdk as rby

SIDES = (slice(0, 7), slice(7, 14))


class LeaderArm:
    def __init__(self, cfg):
        device = cfg["leader_device"] or rby.upc.resolve_leader_arm_device_name()
        rby.upc.initialize_device(device)
        self.arm = rby.upc.LeaderArm(device)
        self.arm.set_model_path(cfg["leader_urdf"])
        self.arm.set_control_period(cfg["leader_period_s"])
        ids = self.arm.initialize(verbose=False)
        if len(ids) != rby.upc.LeaderArm.DeviceCount:
            raise RuntimeError(f"리더암 장치 수가 맞지 않습니다: {ids} (필요 {rby.upc.LeaderArm.DeviceCount}개)")
        self.torque_limit = np.array(cfg["leader_torque_limit"])
        self.viscous_gain = np.array(cfg["leader_viscous_gain"])
        self.gravity_gain = cfg["leader_gravity_gain"]
        self._lock = threading.Lock()
        self._latest = None
        self._hold_q = None

    def _control(self, state):
        q = np.array(state.q_joint)
        if self._hold_q is None:
            self._hold_q = q.copy()
        torque = np.clip(np.array(state.gravity_term) + self.viscous_gain * np.array(state.qvel_joint),
                         -self.torque_limit, self.torque_limit)
        pressed = (state.button_right.button == 1, state.button_left.button == 1)
        inp = rby.upc.LeaderArm.ControlInput()
        for on, s in zip(pressed, SIDES):
            if on:
                inp.target_operating_mode[s].fill(rby.DynamixelBus.CurrentControlMode)
                inp.target_torque[s] = torque[s] * self.gravity_gain
                self._hold_q[s] = q[s]
            else:
                inp.target_operating_mode[s].fill(rby.DynamixelBus.CurrentBasedPositionControlMode)
                inp.target_torque[s] = self.torque_limit[s]
                inp.target_position[s] = self._hold_q[s]
        triggers = (float(state.button_right.trigger), float(state.button_left.trigger))
        with self._lock:
            self._latest = (q, pressed, triggers)
        return inp

    def start(self):
        if not self.arm.start_control(self._control):
            raise RuntimeError("리더암 제어를 시작하지 못했습니다")

    def latest(self):
        """(관절값 14개, (오른쪽 버튼, 왼쪽 버튼), (오른쪽 트리거, 왼쪽 트리거) 0~1000). 첫 샘플 전에는 None."""
        with self._lock:
            return self._latest

    def stop(self):
        self.arm.stop_control(torque_disable=True)
