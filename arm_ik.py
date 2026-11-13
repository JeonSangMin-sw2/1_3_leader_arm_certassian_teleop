"""팔 관절 IK: 손 목표 자세로 가는 다음 관절 각도를 팔마다 한 주기씩 푼다.

rby1_3d_mouse_teleop의 joint_ik.py에서 가져왔다 (pose_error, solve_step).
관절 명령은 실제 각도가 아니라 직전에 보낸 관절 명령에서 이어 푼다. 문제 꼴은 rby1-sdk OptimalControl과 같다:
관절 변화량을 변수로 손 오차와 자세 유지(관절 변화량)를 가중 최소제곱으로 줄이고 관절 위치·속도 한도를 상하한으로 건다.
팔 7축의 여유 자유도(팔꿈치 돌림)는 자세 유지 항이 붙잡는다.

리더암 팔꿈치 돌림각을 따르게 하는 목표는 넣었다가 뺐다 (사용자 결정, 2026-11-13). 팔이 펴진 자세에서 손 유지 오차가
0.2 mm에서 같은 최소제곱 가중치로 8.7 mm, 영공간 투영 16.7 mm, 손 고정 2단계 풀이(v1.3) 16.2 mm로 커졌다.
"""

import numpy as np
from scipy.optimize import lsq_linear

from so3 import log_so3

# rby1 ROS2 드라이버(kinematics.cpp)와 같은 가중치: 위치 [1/m], 회전 [1/rad]
POSITION_WEIGHT = 1000.0
ROTATION_WEIGHT = 100.0


def pose_error(target, current, max_position, max_rotation):
    """current에서 target까지의 [위치 m, 회전 벡터 rad] (base 기준). 한 주기에 갈 거리(max_*)를 넘으면 그만큼으로 줄인다."""
    parts = [target[:3, 3] - current[:3, 3], log_so3(target[:3, :3] @ current[:3, :3].T)]
    for part, limit in zip(parts, (max_position, max_rotation)):
        norm = np.linalg.norm(part)
        if norm > limit:
            part *= limit / norm
    return np.concatenate(parts)


def solve_step(jacobian, error, q, lower, upper, max_step, posture_weight):
    """error만큼 손을 옮기는 관절 변화량 dq [rad]. jacobian: 6×n ([선속도; 각속도], base 방향).

    q가 이미 한도 밖이면 안쪽으로만 움직인다.
    """
    weights = np.array([POSITION_WEIGHT] * 3 + [ROTATION_WEIGHT] * 3)
    matrix = np.vstack([weights[:, None] * jacobian, np.diag(posture_weight)])
    vector = np.concatenate([weights * error, np.zeros(len(q))])
    step_lower = np.maximum(np.minimum(lower - q, 0.0), -max_step)
    step_upper = np.minimum(np.maximum(upper - q, 0.0), max_step)
    return lsq_linear(matrix, vector, bounds=(step_lower, step_upper), method="bvls").x


class ArmIK:
    """양팔 관절 명령을 들고 매 주기 이어 푼다. 기구학은 접속한 로봇 자신의 모델(robot.dyn)로 계산한다."""

    def __init__(self, robot, cfg):
        self.cfg = cfg
        self.dyn = robot.dyn
        self.state = self.dyn.make_state(["base", "ee_right", "ee_left"], robot.model.robot_joint_names)
        self.arm_idx = [list(robot.model.right_arm_idx), list(robot.model.left_arm_idx)]
        lo, hi = self.dyn.get_limit_q_lower(self.state), self.dyn.get_limit_q_upper(self.state)
        qdot, qddot = self.dyn.get_limit_qdot_upper(self.state), self.dyn.get_limit_qddot_upper(self.state)
        self.lower = [lo[i] for i in self.arm_idx]
        self.upper = [hi[i] for i in self.arm_idx]
        self.vel_limit = [qdot[i] for i in self.arm_idx]
        self.acc_limit = [qddot[i] * cfg["acceleration_limit_scaling"] for i in self.arm_idx]
        self.posture_weight = np.array(cfg["ik_posture_weight"], dtype=float)
        self.q_cmd = [None, None]
        self.prev_step = [np.zeros(7), np.zeros(7)]

    def reset(self, q):
        self.q_cmd = [np.array(q[i]) for i in self.arm_idx]
        self.prev_step = [np.zeros(7), np.zeros(7)]

    def step(self, q_base, targets):
        """q_base(torso는 보낼 torso 명령)에서 양팔 관절 명령을 한 주기 갱신한다. targets: 팔별 base 기준 손 목표 4×4."""
        c = self.cfg
        dt = c["command_period_s"]
        q = np.array(q_base, dtype=float)
        for i in range(2):
            q[self.arm_idx[i]] = self.q_cmd[i]
        self.state.set_q(q)
        self.dyn.compute_forward_kinematics(self.state)
        for i in range(2):
            current = np.asarray(self.dyn.compute_transformation(self.state, 0, 1 + i))
            Jb = np.asarray(self.dyn.compute_body_jacobian(self.state, 0, 1 + i))[:, self.arm_idx[i]]  # [각속도; 선속도], 손 좌표계
            R = current[:3, :3]
            J = np.vstack([R @ Jb[3:], R @ Jb[:3]])
            error = pose_error(targets[i], current, c["hand_linear_velocity"] * dt, c["hand_angular_velocity"] * dt)
            step = solve_step(J, error, self.q_cmd[i], self.lower[i], self.upper[i], self.vel_limit[i] * dt,
                              self.posture_weight)
            # 관절 가속도 한도: 직전 변화량에서 한 주기에 바꿀 수 있는 만큼만 (특이점 근처에서 해가 튀는 것을 막는다)
            dv = self.acc_limit[i] * dt * dt
            step = np.clip(step, self.prev_step[i] - dv, self.prev_step[i] + dv)
            q_new = self.q_cmd[i] + step
            q_new = np.clip(q_new, np.minimum(self.lower[i], self.q_cmd[i]), np.maximum(self.upper[i], self.q_cmd[i]))
            self.prev_step[i] = q_new - self.q_cmd[i]
            self.q_cmd[i] = q_new
