"""리더암 읽기. 버튼을 누르고 있는 팔은 중력 보상으로 자유롭게, 놓은 팔은 그 자리에 고정한다.
시작할 때 align()으로 리더암을 로봇 팔 관절값으로 옮긴다.

참고: SDK 예제 35 (버튼·토크 동작과 토크 값, 소프트 정지, 리더암 자세 정렬 move_leader_arm_to_pose).
SDK의 C++ LeaderArm은 통신 장애를 알려 주지 않아서, 제어 콜백이 마지막으로 불린 시각으로 응답 끊김을 판단한다.
"""

import threading
import time

import numpy as np
import rby1_sdk as rby

SIDES = (slice(0, 7), slice(7, 14))
SOFT_STOP_STEPS = 10  # 예제 35: 3 s 동안 10단계로 토크를 줄인다


class LeaderArmFault(RuntimeError):
    """리더암 제어 콜백이 leader_timeout_s 넘게 불리지 않았다."""


class LeaderArm:
    def __init__(self, cfg):
        device = cfg["leader_device"] or rby.upc.resolve_leader_arm_device_name()
        rby.upc.initialize_device(device)
        self.arm = rby.upc.LeaderArm(device)
        self.arm.set_model_path(cfg["leader_urdf"])
        self.arm.set_control_period(cfg["leader_period_s"])
        ids = self.arm.initialize(verbose=False)
        if len(ids) != rby.upc.LeaderArm.DeviceCount:
            missing = sorted((set(range(14)) | {rby.upc.LeaderArm.RightToolId, rby.upc.LeaderArm.LeftToolId}) - set(ids))
            raise RuntimeError(f"리더암 장치 수가 맞지 않습니다: {len(ids)}/{rby.upc.LeaderArm.DeviceCount}, "
                               f"응답 없는 ID {missing}. 커넥터나 모터를 확인하세요")
        self.torque_limit = np.array(cfg["leader_torque_limit"])
        self.viscous_gain = np.array(cfg["leader_viscous_gain"])
        self.gravity_gain = cfg["leader_gravity_gain"]
        self.soft_stop_time = cfg["soft_stop_time_s"]
        self.timeout = cfg["leader_timeout_s"]
        self._lock = threading.Lock()
        self._latest = None
        self._hold_q = None
        self._started = None
        self._last_callback = None
        self._torque = np.zeros(14)   # 마지막 측정 토크 (소프트 정지 시작값)
        self._gravity = np.zeros(14)
        self._ramp = None             # 소프트 정지 중이면 (시작 시각, 시작 토크)
        self._align = None            # 정렬 중이면 (시작 관절값, 목표 관절값, 시작 시각, 시간)

    def _control(self, state):
        now = time.monotonic()
        self._last_callback = now
        self._torque = np.array(state.torque_joint)
        self._gravity = np.array(state.gravity_term)
        inp = rby.upc.LeaderArm.ControlInput()
        if self._ramp is not None:
            t0, tau0 = self._ramp
            k = min(int((now - t0) / (self.soft_stop_time / SOFT_STOP_STEPS)) + 1, SOFT_STOP_STEPS)
            inp.target_operating_mode.fill(rby.DynamixelBus.CurrentControlMode)
            inp.target_torque[:] = tau0 * (1.0 - k / SOFT_STOP_STEPS)
            return inp
        q = np.array(state.q_joint)
        if self._hold_q is None:
            self._hold_q = q.copy()
        torque = np.clip(self._gravity + self.viscous_gain * np.array(state.qvel_joint),
                         -self.torque_limit, self.torque_limit)
        pressed = (state.button_right.button == 1, state.button_left.button == 1)
        align = self._align
        if align is not None:  # 정렬 중에는 버튼과 상관없이 14축 모두 위치 제어 (예제 35)
            q0, q1, t0, duration = align
            ratio = min((now - t0) / duration, 1.0)
            inp.target_operating_mode.fill(rby.DynamixelBus.CurrentBasedPositionControlMode)
            inp.target_torque[:] = self.torque_limit
            inp.target_position[:] = q0 + 0.5 * (1.0 - np.cos(np.pi * ratio)) * (q1 - q0)
            pressed = (False, False)
        else:
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
        self._started = time.monotonic()
        if not self.arm.start_control(self._control):
            raise RuntimeError("리더암 제어를 시작하지 못했습니다")

    def alive(self, timeout):
        """제어 콜백이 timeout [s] 안에 불렸는지 (시작 직후에는 시작 시각부터 잰다)."""
        last = self._last_callback or self._started
        return last is not None and time.monotonic() - last < timeout

    def align(self, target_q, duration, stop_event):
        """리더암을 지금 자세에서 target_q(관절값 14개, 오른팔 7 + 왼팔 7)로 duration초 동안 옮기고 그 자세에 고정한다.

        SDK 예제 35처럼 코사인 S-커브로 보간하고 전류 한도는 leader_torque_limit이다. 끝날 때까지 기다린다.
        duration이 0 이하면 건너뛴다. stop_event가 서면 그 자리에 멈춘다.
        콜백이 leader_timeout_s 넘게 없으면 LeaderArmFault (텔레옵 루프와 같은 장애 판단).
        """
        target_q = np.array(target_q, dtype=float)
        if target_q.shape != (14,):
            raise ValueError(f"리더암 정렬 목표는 관절값 14개여야 합니다: {target_q.shape}")
        if duration <= 0:
            return
        while self.latest() is None:  # start() 뒤 첫 샘플을 기다린다
            if not self.alive(self.timeout):
                raise LeaderArmFault(f"리더암 정렬 전 첫 샘플이 {self.timeout} s 안에 오지 않았습니다 (통신 장애)")
            time.sleep(0.005)
        print(f"[리더암] 로봇 팔 자세로 정렬 ({duration:.1f}초)...", flush=True)
        t0 = time.monotonic()
        self._align = (self.latest()[0].copy(), target_q, t0, duration)
        while time.monotonic() - t0 < duration and not stop_event.wait(0.01):
            if not self.alive(self.timeout):
                raise LeaderArmFault(f"리더암 정렬 중 제어 콜백이 {self.timeout} s 넘게 없습니다 (통신 장애)")
        # 정렬 중에는 콜백이 _hold_q를 건드리지 않으므로 먼저 바꾼 뒤 정렬을 끝낸다
        self._hold_q = self.latest()[0].copy() if stop_event.is_set() else target_q.copy()
        self._align = None
        print("[리더암] 정렬 완료" if not stop_event.is_set() else "[리더암] 정렬 중단", flush=True)

    def latest(self):
        """(관절값 14개, (오른쪽 버튼, 왼쪽 버튼), (오른쪽 트리거, 왼쪽 트리거) 0~1000). 첫 샘플 전에는 None."""
        with self._lock:
            return self._latest

    def stop(self, soft=True):
        """soft면 측정 토크(0이면 중력 토크)에서 soft_stop_time_s 동안 10단계로 줄인 뒤 토크를 끈다 (리더암이 툭 떨어지지 않게).

        콜백이 이미 끊긴 장애 상태면 줄일 수 없어 바로 토크를 끈다.
        """
        if soft and self._started is not None and self.alive(0.2):
            tau0 = self._torque if np.any(np.abs(self._torque) > 1e-4) else self._gravity
            self._ramp = (time.monotonic(), np.clip(tau0, -self.torque_limit, self.torque_limit))
            time.sleep(self.soft_stop_time)
        self.arm.stop_control(torque_disable=True)
