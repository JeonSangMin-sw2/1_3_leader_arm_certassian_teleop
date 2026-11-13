#!/usr/bin/env python3
"""리더암 → RB-Y1 몸통 Cartesian 텔레옵 (torso와 양팔을 한 명령으로 푼다).

팔마다 버튼을 누르고 있는 동안:
  리더암 관절값 → v1.2 팔 FK (link_torso_5 → ee) → 누른 순간 대비 변화량 (위치 × scale, 회전 1:1)
  → 누른 순간의 torso_5 자세로 base 기준 손 목표를 만든다.
torso_mode=region이면 torso_5 목표 = 기준 + 축별 비율 × 손 이동량 (회전도 축별 비율)을
허용영역(base xy 사각형)·높이·회전 한도 안으로 자른다. 손 목표는 base 기준이라 torso가 맡지 않은 나머지를 팔이 맡는다.
버튼을 놓으면 리더암은 그 자리에 고정되고 로봇의 손과 torso는 마지막 목표에 머문다.

사용:
  python teleop.py                                     # settings.yaml 값으로 실행
  python teleop.py --scale 1.3 --torso_ratio_xyz "[0.5, 0.2, 0.3]"
"""

import csv
import signal
import threading
import time
from datetime import datetime

import numpy as np
import rby1_sdk as rby
import yaml

import settings
from arm_ik import ArmIK
from leader_arm import LeaderArmFault
from robot import ARM_L, ARM_R, TORSO, Robot
from so3 import exp_so3, log_so3

LOG_DIR = settings.PROJECT_DIR / "logs"


class ArmFK:
    """리더암 관절값 → RB-Y1 v1.2 팔의 link_torso_5 → ee 변환 (리더암과 관절 순서·축·0점이 같다)."""

    def __init__(self, urdf):
        self.dyn = rby.dynamics.Robot(rby.dynamics.load_robot_from_urdf(urdf, "base"))
        self.st = self.dyn.make_state(["link_torso_5", "ee_right", "ee_left"], ARM_R + ARM_L)
        self.n = len(self.st.get_q())

    def __call__(self, q14):
        x = np.zeros(self.n)
        x[:14] = q14  # make_state는 지정한 관절을 앞에 둔다. 나머지 관절은 이 체인에 없다
        self.st.set_q(x)
        self.dyn.compute_forward_kinematics(self.st)
        return (self.dyn.compute_transformation(self.st, 0, 1).copy(),
                self.dyn.compute_transformation(self.st, 0, 2).copy())


def softsat(x, hi, m):
    """x를 상한 hi로 부드럽게 수렴시킨다. hi - m까지는 그대로, 그 뒤는 지수로 hi에 점근 (접점에서 기울기 연속). m=0이면 min."""
    if m <= 0:
        return min(x, hi)
    knee = hi - m
    return x if x <= knee else hi - m * np.exp(-(x - knee) / m)


def softsat2(x, lo, hi, m):
    return -softsat(-softsat(x, hi, m), -lo, m)


def map_hand(T_L, T_L0, T_R0, scale):
    """누른 순간의 torso_5 좌표계에서 변화량 매핑: 위치는 scale배, 회전은 1:1."""
    T = np.eye(4)
    T[:3, :3] = T_L[:3, :3] @ T_L0[:3, :3].T @ T_R0[:3, :3]
    T[:3, 3] = T_R0[:3, 3] + scale * (T_L[:3, 3] - T_L0[:3, 3])
    return T


class TorsoRegion:
    """torso_5 허용영역: base xy 볼록 사각형 × z 범위."""

    def __init__(self, xy, z_limits):
        self.poly = np.array(xy, dtype=float)
        if self.poly.shape != (4, 2):
            raise ValueError(f"torso_region_xy는 꼭짓점 4개 [[x, y], ...]여야 합니다: {xy}")
        edges = np.roll(self.poly, -1, axis=0) - self.poly
        cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        if not (np.all(cross > 0) or np.all(cross < 0)):
            raise ValueError(f"torso_region_xy가 순서대로 놓인 볼록 사각형이 아닙니다: {xy}")
        self.sign = np.sign(cross[0])
        self.z = np.array(z_limits, dtype=float)
        if self.z.shape != (2,) or self.z[0] >= self.z[1]:
            raise ValueError(f"torso_z_limits는 [하한, 상한]이어야 합니다: {z_limits}")

    def contains(self, p):
        e = np.roll(self.poly, -1, axis=0) - self.poly
        d = p[:2] - self.poly
        inside_xy = np.all(self.sign * (e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0]) >= 0)
        return inside_xy and self.z[0] <= p[2] <= self.z[1]

    def soft_clamp(self, p, c, m):
        """c(영역 안의 기준점)에서 p로 가는 방향의 경계까지 거리로 부드럽게 자른다. 경계 앞 m부터 감속하고 넘지 않는다."""
        if m <= 0:
            return self.clamp(p)
        q = p.copy()
        q[2] = softsat2(q[2], self.z[0], self.z[1], m)
        u = p[:2] - c[:2]
        t = np.linalg.norm(u)
        if t > 1e-9:
            u /= t
            q[:2] = c[:2] + softsat(t, self._ray_exit(c[:2], u), m) * u
        return q

    def _ray_exit(self, c, u):
        """c에서 u 방향으로 나갈 때 사각형 경계까지 거리."""
        best = np.inf
        for a, b in zip(self.poly, np.roll(self.poly, -1, axis=0)):
            e = b - a
            n = self.sign * np.array([e[1], -e[0]])  # 바깥쪽 법선
            nu = n @ u
            if nu > 1e-12:
                best = min(best, n @ (a - c) / nu)
        return best

    def margin_from(self, c):
        """c에서 가장 가까운 경계(xy 변, z 한계)까지 거리."""
        d = [self.sign * (np.array([b[1] - a[1], -(b[0] - a[0])]) @ (a - c[:2])) / np.linalg.norm(b - a)
             for a, b in zip(self.poly, np.roll(self.poly, -1, axis=0))]
        return min(min(d), c[2] - self.z[0], self.z[1] - c[2])

    def clamp(self, p):
        q = p.copy()
        q[2] = np.clip(q[2], *self.z)
        if not self.contains(np.r_[q[:2], self.z[0]]):  # xy가 밖이면 가장 가까운 변 위의 점
            best = None
            for a, b in zip(self.poly, np.roll(self.poly, -1, axis=0)):
                t = np.clip(np.dot(q[:2] - a, b - a) / np.dot(b - a, b - a), 0, 1)
                c = a + t * (b - a)
                if best is None or np.linalg.norm(q[:2] - c) < np.linalg.norm(q[:2] - best):
                    best = c
            q[:2] = best
        return q


def validate(cfg):
    """설정 불변식을 로봇을 움직이기 전에 검사한다."""
    if cfg["control_mode"] not in ("joint_ik", "cartesian"):
        raise ValueError(f"control_mode는 joint_ik 또는 cartesian: {cfg['control_mode']}")
    if cfg["torso_mode"] not in ("region", "free"):
        raise ValueError(f"torso_mode는 region 또는 free: {cfg['torso_mode']}")
    if cfg["use_torso"] and cfg["torso_mode"] == "free" and cfg["control_mode"] != "cartesian":
        raise ValueError("torso_mode free는 control_mode cartesian에서만 쓸 수 있습니다 (제어기가 torso를 푼다)")
    if cfg["impedance_enabled"] and cfg["control_mode"] != "joint_ik":
        raise ValueError("impedance_enabled는 control_mode joint_ik에서만 쓸 수 있습니다 (팔 관절 임피던스)")
    for key in ("ik_posture_weight", "arm_stiffness", "arm_torque_limit"):
        if len(cfg[key]) != 7:
            raise ValueError(f"{key}는 값 7개여야 합니다")
    if cfg["hand_reference"] not in ("torso", "base"):
        raise ValueError(f"hand_reference는 torso 또는 base: {cfg['hand_reference']}")
    for key in ("torso_ratio_xyz", "torso_ratio_rot"):
        r = np.array(cfg[key], dtype=float)
        if r.shape != (3,) or np.any(r < 0) or np.any(r > 1):
            raise ValueError(f"{key}는 0~1 값 3개여야 합니다: {cfg[key]}")
    lim = np.array(cfg["torso_rot_limit_deg"], dtype=float)
    if lim.shape != (3,) or np.any(lim < 0):
        raise ValueError(f"torso_rot_limit_deg는 0 이상 값 3개여야 합니다: {cfg['torso_rot_limit_deg']}")
    if not 0 <= cfg["gripper_max_open"] <= 1000:
        raise ValueError(f"gripper_max_open은 0~1000이어야 합니다: {cfg['gripper_max_open']}")
    if cfg["soft_margin_m"] < 0:
        raise ValueError(f"soft_margin_m은 0 이상이어야 합니다: {cfg['soft_margin_m']}")
    if not 0 < cfg["arm_reach_ratio"] < 1:
        raise ValueError(f"arm_reach_ratio는 0과 1 사이여야 합니다: {cfg['arm_reach_ratio']}")
    if cfg["scale"] <= 0:
        raise ValueError(f"scale은 0보다 커야 합니다: {cfg['scale']}")
    for key, n in (("leader_torque_limit", 14), ("leader_viscous_gain", 14)):
        if len(cfg[key]) != n:
            raise ValueError(f"{key}는 값 {n}개여야 합니다")
    TorsoRegion(cfg["torso_region_xy"], cfg["torso_z_limits"])
    lim = np.array(cfg["torso_joint_limits_deg"], dtype=float)
    if lim.shape != (6, 2) or np.any(lim[:, 0] >= lim[:, 1]):
        raise ValueError(f"torso_joint_limits_deg는 [하한, 상한] 6쌍이어야 합니다: {cfg['torso_joint_limits_deg']}")


class Teleop:
    def __init__(self, cfg, robot, leader, gripper=None):
        validate(cfg)
        self.cfg, self.robot, self.leader, self.gripper = cfg, robot, leader, gripper
        self.fk = ArmFK(cfg["fk_urdf"])
        self.region = TorsoRegion(cfg["torso_region_xy"], cfg["torso_z_limits"])
        self.joint_mode = cfg["control_mode"] == "joint_ik"
        self.region_mode = cfg["use_torso"] and cfg["torso_mode"] == "region"
        self.free_mode = cfg["use_torso"] and cfg["torso_mode"] == "free"
        self.ratio_xyz = np.array(cfg["torso_ratio_xyz"], dtype=float)
        self.ratio_rot = np.array(cfg["torso_ratio_rot"], dtype=float)
        self.rot_limit = np.deg2rad(cfg["torso_rot_limit_deg"])
        self.margin = cfg["soft_margin_m"]

    def _region_clamp(self, p):
        return self.region.soft_clamp(p, self.region_center, self.margin)

    def _reset_anchor(self):
        """누르거나 놓을 때마다 torso 비율 계산의 기준을 현재 목표로 옮긴다 (목표가 튀지 않게)."""
        self.torso_anchor = self.torso_base.copy()
        self.hand_anchor = [T.copy() for T in self.hand_raw]

    def _torso_target(self):
        eng = [i for i in range(2) if self.engaged[i] is not None]
        dp = np.mean([self.hand_raw[i][:3, 3] - self.hand_anchor[i][:3, 3] for i in eng], axis=0)
        dw = np.mean([log_so3(self.hand_raw[i][:3, :3] @ self.hand_anchor[i][:3, :3].T) for i in eng], axis=0)
        T = np.eye(4)
        raw = self.torso_anchor[:3, 3] + self.ratio_xyz * dp
        T[:3, 3] = self._region_clamp(raw)
        R_raw = exp_so3(self.ratio_rot * dw) @ self.torso_anchor[:3, :3]
        psi = np.clip(log_so3(R_raw @ self.torso_ref_R.T), -self.rot_limit, self.rot_limit)
        T[:3, :3] = exp_so3(psi) @ self.torso_ref_R
        return T, bool(np.linalg.norm(T[:3, 3] - raw) > 1e-9)

    def _wrist(self, p_t, R_t, i):
        """torso_5 자세 (p_t, R_t)에서 i번 팔 원래 손 목표의 어깨→손목 거리와 방향."""
        d = (self.hand_raw[i] @ self.inv_6e[i])[:3, 3] - (p_t + R_t @ self.shoulder[i])
        n = np.linalg.norm(d)
        return n, d / n

    def _push_torso(self):
        """손목이 팔 길이 한도에 가까워지면 torso_5 목표를 손 쪽으로 옮긴다 (영역 안에서). 옮긴 거리를 돌려준다.

        r_max - 2m부터 밀기 시작해 손목 거리가 r_max - m에 점근하게 한다. 손 당기기(r_max - m부터)보다 먼저 걸린다.
        부드러운 함수는 제 출력에 거듭 적용하면 값이 계속 변하므로 매 주기 torso_base에서 새로 계산한다.
        """
        self.torso_tgt = self.torso_base.copy()
        R_t, p0 = self.torso_tgt[:3, :3], self.torso_tgt[:3, 3].copy()
        push = np.zeros(3)
        for i in range(2):
            n, u = self._wrist(p0, R_t, i)
            push += (n - softsat(n, self.r_max[i] - self.margin, self.margin)) * u
        p = self._region_clamp(p0 + push)
        self.torso_tgt[:3, 3] = p
        return float(np.linalg.norm(p - p0))

    def _clamp_hands(self, T_t):
        """보내는 손 목표를 정한다. 비트 1·2: 팔 길이 한도로 당김, 4·8: 머리 높이로 내림 (오른팔, 왼팔).

        팔 길이: torso로 다 못 줄인 팔은 손 목표를 r_max로 부드럽게 당긴다 (팔이 펴지는 특이점 회피).
        높이: 손 목표가 머리 회전 중심 - hand_head_clearance_m보다 위로 가지 않게 부드럽게 내린다 (머리 위는 카메라로 못 본다).
        클러치 기준은 한도 + m을 넘은 만큼만 옮긴다. 매 주기 당긴 만큼 다 옮기면 부드러운 구간에서 손이 계속 끌려온다.
        """
        flags = 0
        z_cap = (T_t[:3, :3] @ self.head + T_t[:3, 3])[2] - self.cfg["hand_head_clearance_m"]
        for i in range(2):
            self.hand_tgt[i] = self.hand_raw[i].copy()
            n, u = self._wrist(T_t[:3, 3], T_t[:3, :3], i)
            pull = n - softsat(n, self.r_max[i], self.margin)
            if pull > 1e-9:
                self.hand_tgt[i][:3, 3] -= pull * u
                flags |= 1 << i
            over = n - (self.r_max[i] + self.margin)
            if over > 0 and self.engaged[i] is not None:  # 목표가 한도 밖에 쌓이지 않게
                T_bt0, T_tee0 = self.engaged[i][:2]
                T_tee0[:3, 3] -= T_bt0[:3, :3].T @ (over * u)
            z = self.hand_tgt[i][2, 3]
            z_new = softsat(z, z_cap, self.margin)
            if z_new < z - 1e-9:
                self.hand_tgt[i][2, 3] = z_new
                flags |= 4 << i
            over = self.hand_raw[i][2, 3] - (z_cap + self.margin)
            if over > 0 and self.engaged[i] is not None:
                T_bt0, T_tee0 = self.engaged[i][:2]
                T_tee0[:3, 3] -= T_bt0[:3, :3].T @ np.array([0.0, 0.0, over])
        return flags

    def run(self, duration=None, stop_event=None):
        cfg, robot = self.cfg, self.robot
        stop_event = stop_event or threading.Event()
        period = cfg["command_period_s"]
        robot.start_state_update()
        q, T_bt, T_be, dist_prev = robot.kinematics()
        if self.region_mode and not self.region.contains(T_bt[:3, 3]):
            robot.stop_state_update()
            raise ValueError(f"시작 torso_5 위치 {np.round(T_bt[:3, 3], 3)}가 허용영역 밖입니다")
        # *_raw/*_base는 팔 길이 한도·밀기를 적용하기 전 목표, *_tgt는 보내는 목표. 누르기 전에는 지금 자세 유지
        self.hand_raw = [T_be[0].copy(), T_be[1].copy()]
        self.hand_tgt = [T.copy() for T in self.hand_raw]
        self.torso_base = T_bt.copy()
        self.torso_tgt = T_bt.copy()
        self.torso_ref_R = T_bt[:3, :3].copy()  # torso 회전 한도의 기준
        self.region_center = T_bt[:3, 3].copy()  # 부드러운 영역 자르기의 기준점
        if self.region_mode and self.region.margin_from(self.region_center) <= self.margin:
            robot.stop_state_update()
            raise ValueError(f"시작 torso_5 위치에서 영역 경계까지 {self.region.margin_from(self.region_center):.3f} m로 "
                             f"soft_margin_m {self.margin}보다 가깝습니다")
        geom = robot.arm_geometry()
        self.shoulder = [g[0] for g in geom]
        self.inv_6e = [np.linalg.inv(g[1]) for g in geom]
        self.r_max = [cfg["arm_reach_ratio"] * g[2] for g in geom]
        self.head = robot.head_offset()
        lim = np.deg2rad(np.array(cfg["torso_joint_limits_deg"], dtype=float))
        model_lo, model_hi = robot.torso_limits()
        if np.any(lim[:, 0] < model_lo - 1e-6) or np.any(lim[:, 1] > model_hi + 1e-6):
            robot.stop_state_update()
            raise ValueError(f"torso_joint_limits_deg가 로봇 모델 한계 {np.round(np.rad2deg(np.c_[model_lo, model_hi]), 1).tolist()}를 넘습니다")
        self.torso_lim = (lim[:, 0], lim[:, 1])
        self.torso_q = q[robot.torso_idx].copy()
        if self.region_mode and (np.any(self.torso_q < lim[:, 0] - 1e-3) or np.any(self.torso_q > lim[:, 1] + 1e-3)):
            robot.stop_state_update()
            raise ValueError(f"시작 torso 관절값 {np.round(np.rad2deg(self.torso_q), 1).tolist()}이 torso_joint_limits_deg 밖입니다")
        self.torso_fixed_q, self.torso_fixed_T = q[robot.torso_idx].copy(), T_bt.copy()  # use_torso false면 이 자세로 고정
        self.ik = ArmIK(robot, cfg) if self.joint_mode else None
        if self.ik:
            self.ik.reset(q)
        self.engaged = [None, None]       # 팔마다 [T_bt0, T_tee0, T_L0]
        need_release = [False, False]     # 충돌 보호로 멈춘 뒤에는 버튼을 놓았다 다시 눌러야 한다
        self._reset_anchor()
        clamped, hand_limited, pushed, projected = False, 0, 0.0, 0.0

        log_file, log_path = None, None
        if cfg["log_enabled"]:
            LOG_DIR.mkdir(exist_ok=True)
            log_path = LOG_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.csv"
            with open(log_path.with_suffix(".yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            log_file = open(log_path, "w", newline="")
            log = csv.writer(log_file)
            log.writerow(["t", "btn_r", "btn_l"] + [f"qL{i}" for i in range(14)]
                         + [f"hand_{s}_{a}" for s in "rl" for a in "xyz"]
                         + [f"torso_tgt_{a}" for a in "xyz"] + [f"torso_tgt_w{a}" for a in "xyz"]
                         + [f"torso_{a}" for a in "xyz"] + [f"q_{n}" for n in TORSO + ARM_R + ARM_L]
                         + ["min_dist", "torso_clamped", "hand_limited", "torso_pushed", "torso_projected",
                            "trigger_r", "trigger_l"])

        robot.open_stream()
        t0 = time.time()
        try:
            while not stop_event.is_set() and (duration is None or time.time() - t0 < duration):
                tick = time.time()
                q, T_bt, T_be, dist = robot.kinematics()
                alive = getattr(self.leader, "alive", None)  # 시험용 합성 리더암에는 없다
                if alive is not None and not alive(cfg["leader_timeout_s"]):
                    raise LeaderArmFault(f"리더암 제어 콜백이 {cfg['leader_timeout_s']} s 넘게 없습니다 (통신 장애)")
                sample = self.leader.latest()
                pressed = sample[1] if sample is not None else (False, False)
                if sample is not None:
                    if self.gripper is not None:  # 예제 35처럼 클러치와 상관없이 트리거를 따른다
                        self.gripper.set_trigger(sample[2])
                    T_L = self.fk(sample[0])
                    before = [e is not None for e in self.engaged]
                    for i in range(2):
                        if not pressed[i]:
                            self.engaged[i] = None
                            need_release[i] = False
                        elif self.engaged[i] is None and not need_release[i]:
                            self.engaged[i] = [T_bt.copy(), np.linalg.inv(T_bt) @ T_be[i], T_L[i]]
                            self.hand_raw[i] = T_be[i].copy()
                    if before != [e is not None for e in self.engaged]:
                        self._reset_anchor()
                    for i in range(2):
                        if self.engaged[i] is not None:
                            T_bt0, T_tee0, T_L0 = self.engaged[i]
                            self.hand_raw[i] = T_bt0 @ map_hand(T_L[i], T_L0, T_tee0, cfg["scale"])
                    if self.region_mode and any(e is not None for e in self.engaged):
                        self.torso_base, clamped = self._torso_target()
                    if self.region_mode:
                        pushed = self._push_torso()
                        T_proj, self.torso_q = robot.project_torso(self.torso_tgt, self.torso_q, *self.torso_lim)
                        projected = float(np.linalg.norm(T_proj[:3, 3] - self.torso_tgt[:3, 3]))
                        self.torso_tgt = T_proj
                    T_t = self.torso_tgt if self.region_mode else (T_bt if self.free_mode else self.torso_fixed_T)
                    hand_limited = self._clamp_hands(T_t)

                # 충돌 보호: 링크가 가까워지면서 계속 줄면 지금 자세에서 멈춘다
                if dist < cfg["collision_guard_m"] and dist < dist_prev - 1e-4 and any(
                        e is not None for e in self.engaged):
                    print(f"[충돌 보호] 가장 가까운 링크 거리 {dist * 1000:.0f} mm, 정지. "
                          "버튼을 놓았다 다시 누르면 이어서 움직입니다.", flush=True)
                    self.hand_raw = [T_be[0].copy(), T_be[1].copy()]
                    self.hand_tgt = [T.copy() for T in self.hand_raw]
                    self.torso_base = T_bt.copy()
                    self.torso_tgt = T_bt.copy()
                    if self.ik:
                        self.ik.reset(q)
                    self.engaged = [None, None]
                    need_release = [True, True]
                    self._reset_anchor()
                dist_prev = dist

                if self.joint_mode:
                    torso_cmd = self.torso_q if self.region_mode else self.torso_fixed_q
                    q_base = q.copy()
                    q_base[robot.torso_idx] = torso_cmd
                    self.ik.step(q_base, self.hand_tgt)
                    robot.send_joints(torso_cmd, *self.ik.q_cmd)
                elif not cfg["use_torso"]:
                    robot.send_arms_cartesian(self.torso_fixed_q, *self.hand_tgt)
                else:
                    robot.send(self.hand_tgt[0], self.hand_tgt[1], self.torso_tgt if self.region_mode else None)

                if log_file and sample is not None:
                    log.writerow([f"{tick - t0:.3f}", int(pressed[0]), int(pressed[1])]
                                 + list(np.round(sample[0], 5))
                                 + list(np.round(self.hand_tgt[0][:3, 3], 5)) + list(np.round(self.hand_tgt[1][:3, 3], 5))
                                 + list(np.round(self.torso_tgt[:3, 3], 5))
                                 + list(np.round(log_so3(self.torso_tgt[:3, :3] @ self.torso_ref_R.T), 5))
                                 + list(np.round(T_bt[:3, 3], 5)) + list(np.round(q[robot.torso_idx + robot.arm_idx], 5))
                                 + [round(dist, 4), int(clamped), hand_limited, round(pushed, 5), round(projected, 5)]
                                 + list(sample[2]))
                time.sleep(max(0.0, period - (time.time() - tick)))
        finally:
            robot.close_stream()
            robot.stop_state_update()
            if log_file:
                log_file.close()
        return log_path


def shutdown(robot, leader, gripper, soft):
    """예제 35와 같은 종료 순서. 한 단계가 실패해도 알리고 나머지 정지 단계는 계속한다.

    soft: 스트림 취소 → 리더암 토크를 soft_stop_time_s 동안 줄임 → 그리퍼 끔 → 제어 취소·disable → 12V 끔.
    아니면: 스트림 취소 → 리더암·그리퍼 토크 즉시 끔 → 전원 라인 모두 끔.
    """
    print(f"[종료] {'소프트 정지' if soft else '즉시 정지'} 시작", flush=True)
    steps = [("명령 스트림 취소", robot.close_stream)]
    if leader is not None:
        steps.append(("리더암 토크 끔", lambda: leader.stop(soft)))
    if gripper is not None:
        steps.append(("그리퍼 끔", gripper.disconnect))
    steps.append(("제어 취소·disable·12V 끔", robot.shutdown) if soft else ("전원 모두 끔", robot.power_off_all))
    for name, step in steps:
        try:
            step()
        except Exception as e:  # 정지 절차는 끝까지 간다
            print(f"[종료] {name} 실패: {e}", flush=True)
    print("[종료] 완료", flush=True)


def main():
    cfg = settings.parse(description=__doc__)
    validate(cfg)
    from leader_arm import LeaderArm

    robot = Robot(cfg)
    print(f"robot {cfg['robot_address']}  control={cfg['control_mode']}  use_torso={cfg['use_torso']}  "
          f"impedance={cfg['impedance_enabled']}  scale={cfg['scale']}")
    stop_event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop_event.set())
    leader = gripper = None
    soft = True  # Ctrl+C·정상 종료·시작 실패는 소프트 정지 (예제 35)
    robot.ensure_ready()
    try:
        if cfg["move_to_ready"]:
            robot.move_ready()
        if cfg["use_gripper"]:
            from gripper import Gripper
            robot.ensure_tool_flange_12v()  # 그리퍼 전원
            gripper = Gripper(cfg)
            gripper.connect()  # homing으로 손가락이 양 끝까지 움직인다
        leader = LeaderArm(cfg)
        leader.start()
        log_path = Teleop(cfg, robot, leader, gripper).run(stop_event=stop_event)
        if log_path:
            print(f"log: {log_path}")
    except LeaderArmFault as e:
        soft = cfg["fault_soft_stop"]
        print(f"[장애] {e}", flush=True)
        raise
    finally:
        shutdown(robot, leader, gripper, soft)


if __name__ == "__main__":
    main()
