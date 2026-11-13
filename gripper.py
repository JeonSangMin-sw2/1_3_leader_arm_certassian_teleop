"""RB-Y1 그리퍼. 리더암 트리거 값(0~1000)을 닫힘 정도로 써서 양쪽 그리퍼를 움직인다.

참고: SDK 예제 35의 Gripper, rby1_3d_mouse_teleop의 gripper.py (실물에서 보강한 homing 시간 제한·방향).
/dev/rby1_gripper는 UPC에만 있다. 시작할 때 homing으로 손가락이 양 끝까지 움직인다.
"""

import threading
import time

import numpy as np
import rby1_sdk as rby

IDS = [0, 1]  # 오른쪽, 왼쪽
BAUD_RATE = 2_000_000
HOMING_TORQUE = 0.3
HOLD_TORQUE = 5.0
LOOP_PERIOD_S = 0.1
HOMING_STILL_COUNT = 30  # 엔코더가 멈춘 주기가 이만큼 쌓이면 끝으로 본다 (예제 35 값). 끝에서 흔들리면 연속으로는 안 쌓여서 누적으로 센다
HOMING_TIMEOUT_S = 15.0


class Gripper:
    def __init__(self, cfg):
        self.max_open = cfg["gripper_max_open"] / 1000.0
        self.invert = cfg["gripper_invert_trigger"]
        self._bus = rby.DynamixelBus(rby.upc.GripperDeviceName)
        self._min_q = np.full(len(IDS), np.inf)
        self._max_q = np.full(len(IDS), -np.inf)
        self._target_q = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def connect(self):
        if not self._bus.open_port():
            raise RuntimeError(f"그리퍼 포트를 열지 못했습니다: {rby.upc.GripperDeviceName}")
        if not self._bus.set_baud_rate(BAUD_RATE):
            raise RuntimeError("그리퍼 baud rate를 설정하지 못했습니다")
        self._bus.set_torque_constant([1.0] * len(IDS))
        for dev_id in IDS:
            if not self._bus.ping(dev_id):
                raise RuntimeError(f"그리퍼 Dynamixel ID {dev_id}가 응답하지 않습니다")
        self._bus.group_sync_write_torque_enable([(i, 1) for i in IDS])
        self._home()
        self._set_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        self._bus.group_sync_write_send_torque([(i, HOLD_TORQUE) for i in IDS])
        self._set_closing(np.zeros(len(IDS)))  # 처음에는 열어 둔다
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _set_mode(self, mode):
        self._bus.group_sync_write_torque_enable([(i, 0) for i in IDS])
        self._bus.group_sync_write_operating_mode([(i, mode) for i in IDS])
        self._bus.group_sync_write_torque_enable([(i, 1) for i in IDS])

    def _home(self):
        """양 방향으로 밀어 엔코더가 멈춘 곳을 이동 범위 끝으로 삼는다."""
        self._set_mode(rby.DynamixelBus.CurrentControlMode)
        q = np.zeros(len(IDS))
        for direction in (1.0, -1.0):
            previous, still = None, 0
            deadline = time.monotonic() + HOMING_TIMEOUT_S
            while still < HOMING_STILL_COUNT:
                if time.monotonic() > deadline:
                    self._bus.group_sync_write_send_torque([(i, 0.0) for i in IDS])
                    raise RuntimeError(f"그리퍼 homing이 {HOMING_TIMEOUT_S:.0f}초 안에 끝나지 않았습니다 "
                                       f"(방향 {direction:+.0f}, 엔코더 {q.round(3).tolist()})")
                self._bus.group_sync_write_send_torque([(i, HOMING_TORQUE * direction) for i in IDS])
                read = self._bus.group_fast_sync_read_encoder(IDS)
                if read is not None:
                    for dev_id, encoder in read:
                        q[IDS.index(dev_id)] = encoder
                self._min_q = np.minimum(self._min_q, q)
                self._max_q = np.maximum(self._max_q, q)
                if previous is not None and np.array_equal(previous, q):
                    still += 1
                previous = q.copy()
                time.sleep(0.1)
        self._bus.group_sync_write_send_torque([(i, 0.0) for i in IDS])
        if not np.all(self._max_q > self._min_q):
            raise RuntimeError("그리퍼 homing에서 이동 범위를 얻지 못했습니다")

    def set_trigger(self, triggers):
        """리더암 트리거 (오른쪽, 왼쪽) 0~1000. 클수록 닫는다 (gripper_invert_trigger면 반대).

        실물은 닫힘이 엔코더 최댓값이다 (rby1_3d_mouse_teleop에서 확인. 예제 35의 GRIPPER_DIRECTION=False는 반대로 움직였다).
        열 때는 gripper_max_open만큼만 벌린다.
        """
        closing = np.clip(np.asarray(triggers, dtype=float) / 1000.0, 0.0, 1.0)
        self._set_closing(1.0 - closing if self.invert else closing)

    def _set_closing(self, closing):
        opening = (1.0 - closing) * self.max_open
        with self._lock:
            self._target_q = self._max_q - opening * (self._max_q - self._min_q)

    def _loop(self):
        while self._running:
            with self._lock:
                target = self._target_q
            self._bus.group_sync_write_send_position([(i, q) for i, q in zip(IDS, target.tolist())])
            time.sleep(LOOP_PERIOD_S)

    def disconnect(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._bus.group_sync_write_torque_enable([(i, 0) for i in IDS])  # SDK 0.10 DynamixelBus에는 close_port가 없다
