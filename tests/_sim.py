"""시뮬레이터 테스트 공용: 합성 리더암, 실행, 로그 분석. 리더암 장치 없이 돈다."""

import csv
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import settings  # noqa: E402
import teleop  # noqa: E402
from robot import ARM_L, ARM_R, TORSO, Robot  # noqa: E402

SIM = "127.0.0.1:50051"
D = np.deg2rad
MIRROR = np.array([1, -1, -1, 1, -1, 1, -1])
READY = D([0, -5, 0, -120, 0, 70, 0])


def minjerk(s):
    s = np.clip(s, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def both(q_right, q_left=None):
    return np.r_[q_right, q_right * MIRROR if q_left is None else q_left]


class ScriptedLeader:
    """script(t) -> (관절값 14개, (오른쪽 버튼, 왼쪽 버튼)). t는 첫 latest() 호출부터 잰다. 트리거는 0으로 둔다."""

    def __init__(self, script):
        self.script, self.t0 = script, None

    def latest(self):
        if self.t0 is None:
            self.t0 = time.time()
        return (*self.script(time.time() - self.t0), (0.0, 0.0))


def make_cfg(**overrides):
    cfg = settings.parse([])
    cfg.update(overrides, robot_address=SIM, log_enabled=True)
    teleop.validate(cfg)
    return cfg


def connect():
    robot = Robot(make_cfg())
    robot.ensure_ready()
    return robot


def run(robot, cfg, script, duration):
    teleop.LOG_DIR = Path(tempfile.mkdtemp())  # 실제 logs/를 덮지 않게
    robot.cfg = cfg
    robot.move_ready(3.0)
    return teleop.Teleop(cfg, robot, ScriptedLeader(script)).run(duration=duration)


def analyze(robot, log_path):
    rows = list(csv.DictReader(open(log_path)))
    m = robot.model
    ds = robot.dyn.make_state(["base", "ee_right", "ee_left", "link_head_2"], m.robot_joint_names)
    idx = robot.torso_idx + robot.arm_idx
    names = TORSO + ARM_R + ARM_L
    f = lambda keys: np.array([[float(r[k]) for k in keys] for r in rows])
    q = f([f"q_{n}" for n in names])
    hand, head_z = [], []
    for qi in q:
        x = np.zeros(len(m.robot_joint_names))
        x[idx] = qi
        ds.set_q(x)
        robot.dyn.compute_forward_kinematics(ds)
        hand.append([robot.dyn.compute_transformation(ds, 0, 1)[:3, 3], robot.dyn.compute_transformation(ds, 0, 2)[:3, 3]])
        head_z.append(robot.dyn.compute_transformation(ds, 0, 3)[2, 3])
    return dict(
        t=f(["t"])[:, 0],
        hand=np.array(hand),
        head_z=np.array(head_z),
        hand_tgt=f([f"hand_{s}_{a}" for s in "rl" for a in "xyz"]).reshape(-1, 2, 3),
        torso=f([f"torso_{a}" for a in "xyz"]),
        torso_tgt=f([f"torso_tgt_{a}" for a in "xyz"]),
        torso_tgt_w=f([f"torso_tgt_w{a}" for a in "xyz"]),
        q=q,
        qL=f([f"qL{i}" for i in range(14)]),
        clamped=f(["torso_clamped"])[:, 0],
        hand_limited=f(["hand_limited"])[:, 0],
        min_dist=f(["min_dist"])[:, 0],
    )


def acc_peak(t, x, win=5):
    """x의 2차 미분 최대(크기). 상태 스트림과 명령 루프가 같은 50 Hz라 같은 상태가 두 번 찍히는 표본을 빼고 이동평균 후 미분한다."""
    keep = np.r_[True, np.any(np.abs(np.diff(x, axis=0)) > 0, axis=1)]
    t, x = t[keep], x[keep]
    k = np.ones(win) / win
    xs = np.stack([np.convolve(c, k, mode="valid") for c in x.T], axis=1)
    ts = np.convolve(t, k, mode="valid")
    a = np.gradient(np.gradient(xs, ts, axis=0), ts, axis=0)
    return np.abs(a).max(axis=0)


def window(a, t0, t1):
    return (a["t"] >= t0) & (a["t"] < t1)


def outside_region(cfg, p):
    """torso_5 위치가 허용영역 밖으로 나간 거리 [m] (안이면 0)."""
    r = teleop.TorsoRegion(cfg["torso_region_xy"], cfg["torso_z_limits"])
    return np.linalg.norm(r.clamp(p) - p)
