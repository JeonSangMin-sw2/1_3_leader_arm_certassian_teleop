#!/usr/bin/env python3
"""실물 확인용 단순 텔레옵: 매핑으로 얻은 base→ee 4×4를 안전 처리 없이 그대로 set_body_command로 보낸다.

teleop.py에서 뺀 것: torso 비율·허용영역·밀기, 팔 길이·머리 높이 한도, 충돌 보호, 관절 IK, 리더암 통신 장애 감시.
남긴 것: 시작 절차(전원·서보·레디자세·그리퍼·리더암 정렬), 버튼 클러치 매핑(teleop.map_hand), 정지 절차(teleop.shutdown).

  use_torso true : 몸통 Cartesian 명령 하나에 base→ee_right, base→ee_left (torso는 제어기가 푼다)
  use_torso false: torso는 시작 관절값으로 고정, 팔마다 base→ee Cartesian + 리더암 팔꿈치(arm_3) 각도를 그대로 관절 목표로

사용:
  python teleop_direct.py --use_torso false
"""

import csv
import signal
import threading
import time
from datetime import datetime

import numpy as np
import yaml

import settings
from leader_arm import LeaderArmFault
from robot import ARM_L, ARM_R, TORSO, Robot
from so3 import log_so3
from teleop import LOG_DIR, ArmFK, map_hand, shutdown

ELBOW = 3  # 팔 관절 안에서 arm_3(팔꿈치) 순번. 리더암 관절값은 오른팔 0~6, 왼팔 7~13


def run(cfg, robot, leader, gripper, stop_event):
    fk = ArmFK(cfg["fk_urdf"])
    period = cfg["command_period_s"]
    robot.start_state_update()
    q, T_bt, T_be, _ = robot.kinematics()
    torso_q = None if cfg["use_torso"] else q[robot.torso_idx].copy()
    elbow_idx = [robot.model.right_arm_idx[ELBOW], robot.model.left_arm_idx[ELBOW]]
    target = [np.array(T_be[0]), np.array(T_be[1])]  # 누르기 전에는 지금 손 자세 유지
    elbow = [float(q[j]) for j in elbow_idx]          # 누르기 전에는 지금 팔꿈치 각도 유지
    engaged = [None, None]                            # 팔마다 누른 순간의 (T_base_torso5, T_torso5_ee, 리더암 FK)

    log_file = None
    if cfg["log_enabled"]:
        LOG_DIR.mkdir(exist_ok=True)
        log_path = LOG_DIR / f"direct_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(log_path.with_suffix(".yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        log_file = open(log_path, "w", newline="")
        log = csv.writer(log_file)
        log.writerow(["t", "btn_r", "btn_l"] + [f"qL{i}" for i in range(14)]
                     + [f"tgt_{s}_{a}" for s in "rl" for a in ("x", "y", "z", "wx", "wy", "wz")]
                     + ["elbow_tgt_r", "elbow_tgt_l"] + [f"q_{n}" for n in TORSO + ARM_R + ARM_L])
        print(f"log: {log_path}", flush=True)

    robot.open_stream()
    t0 = time.time()
    try:
        while not stop_event.is_set():
            tick = time.time()
            q, T_bt, T_be, _ = robot.kinematics()
            sample = leader.latest()
            if sample is not None:
                qL, pressed, triggers = sample
                if gripper is not None:
                    gripper.set_trigger(triggers)
                T_L = fk(qL)
                for i in range(2):
                    if not pressed[i]:
                        engaged[i] = None
                        continue
                    if engaged[i] is None:
                        engaged[i] = (np.array(T_bt), np.linalg.inv(T_bt) @ T_be[i], T_L[i])
                    T_bt0, T_tee0, T_L0 = engaged[i]
                    target[i] = T_bt0 @ map_hand(T_L[i], T_L0, T_tee0, cfg["scale"])
                    elbow[i] = float(qL[7 * i + ELBOW])

            if cfg["use_torso"]:
                robot.send(target[0], target[1])
            else:
                robot.send_arms_cartesian(torso_q, target[0], target[1], elbow)

            if log_file and sample is not None:
                log.writerow([f"{tick - t0:.3f}", int(pressed[0]), int(pressed[1])] + list(np.round(qL, 5))
                             + [round(float(v), 5) for T in target for v in np.r_[T[:3, 3], log_so3(T[:3, :3])]]
                             + [round(e, 5) for e in elbow]
                             + list(np.round(q[robot.torso_idx + robot.arm_idx], 5)))
            time.sleep(max(0.0, period - (time.time() - tick)))
    finally:
        robot.close_stream()
        robot.stop_state_update()
        if log_file:
            log_file.close()


def main():
    cfg = settings.parse(description=__doc__)
    from leader_arm import LeaderArm

    robot = Robot(cfg)
    print(f"[direct] robot {cfg['robot_address']}  use_torso={cfg['use_torso']}  scale={cfg['scale']}  "
          "안전 처리 없음", flush=True)
    stop_event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop_event.set())
    leader = gripper = None
    soft = True
    robot.ensure_ready()
    try:
        if cfg["move_to_ready"]:
            robot.move_ready()
        if cfg["use_gripper"]:
            from gripper import Gripper
            robot.ensure_tool_flange_12v()
            gripper = Gripper(cfg)
            gripper.connect()
        leader = LeaderArm(cfg)
        leader.start()
        leader.align(robot.arm_q(), cfg["leader_align_duration_s"], stop_event)
        run(cfg, robot, leader, gripper, stop_event)
    except LeaderArmFault as e:
        soft = cfg["fault_soft_stop"]
        print(f"[장애] {e}", flush=True)
        raise
    finally:
        shutdown(robot, leader, gripper, soft)


if __name__ == "__main__":
    main()
