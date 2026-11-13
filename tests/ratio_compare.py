#!/usr/bin/env python3
"""torso 비율 비교 (시뮬레이터, 합성 리더암): 균일 비율 vs 축별 비율, 회전 비율 유무.

동작마다 레디에서 시작해 0.5 s에 양팔을 누르고 3 s 동안 목표 자세로 간 뒤 1.5 s 유지한다.
유지 구간 끝 0.5 s에서 손 추종 오차, torso 목표 추종 오차를 보고, 전체 구간에서 torso·팔 관절 이동량과
팔 길이 한도·영역에 걸린 비율을 본다.

사용: .venv/bin/python tests/ratio_compare.py [--scale 1.3]
"""
import sys

import numpy as np
import yaml

import _sim

D, R, MJ = _sim.D, _sim.READY, _sim.minjerk
READY_L = R * _sim.MIRROR
MOTIONS = {  # 오른팔 목표 (왼팔은 대칭). turn은 양팔 어깨 yaw를 같은 쪽으로 돌려 손이 함께 옆으로 돈다
    "forward": (D([-80, -22, 0, -15, 0, 0, 0]), None),
    "low": (D([-30, -10, 0, -10, 0, -30, 0]), None),
    "side": (D([-20, -80, 0, -15, 0, 0, 0]), None),
    "up": (D([-150, -41, 0, -15, 0, 0, 0]), None),
    "turn": (R + D([0, 0, 40, 0, 0, 0, 0]), READY_L + D([0, 0, 40, 0, 0, 0, 0])),
}
CONFIGS = {
    "free": dict(torso_mode="free"),
    "ratio 0": dict(torso_ratio_xyz=[0.0, 0.0, 0.0]),
    "ratio 0.3": dict(torso_ratio_xyz=[0.3, 0.3, 0.3]),
    "ratio 0.6": dict(torso_ratio_xyz=[0.6, 0.6, 0.6]),
    "xyz .6/.2/.4": dict(torso_ratio_xyz=[0.6, 0.2, 0.4]),
    "0.3+yaw 0.5": dict(torso_ratio_xyz=[0.3, 0.3, 0.3], torso_ratio_rot=[0.0, 0.0, 0.5]),
}


def make_script(q_r, q_l):
    q_l = q_r * _sim.MIRROR if q_l is None else q_l

    def script(t):
        s = MJ((t - 0.5) / 3)
        return np.r_[R + s * (q_r - R), READY_L + s * (q_l - READY_L)], (t >= 0.5, t >= 0.5)
    return script


def main():
    overrides = {k.lstrip("-"): yaml.safe_load(v) for k, v in zip(sys.argv[1::2], sys.argv[2::2])}
    overrides.setdefault("scale", 1.3)
    robot = _sim.connect()
    print(f"scale={overrides['scale']}")
    print(f"{'motion':8s} {'config':13s} {'hand err':>8s} {'torso err':>9s} {'torso dxyz mm':>18s} {'yaw':>5s} "
          f"{'arm |dq|':>8s} {'reach lim':>9s} {'region':>6s} {'min d':>5s}")
    for mname, (q_r, q_l) in MOTIONS.items():
        for cname, c in CONFIGS.items():
            cfg = _sim.make_cfg(**{**overrides, **c})
            try:
                a = _sim.analyze(robot, _sim.run(robot, cfg, make_script(q_r, q_l), 5.0))
            except RuntimeError as e:
                print(f"{mname:8s} {cname:13s} 실패: {e}")
                robot.ensure_ready()
                continue
            hold = _sim.window(a, 4.5, 5.0)
            herr = np.linalg.norm(a["hand"] - a["hand_tgt"], axis=2)[hold].max() * 1000
            terr = (np.linalg.norm(a["torso"] - a["torso_tgt"], axis=1)[hold].max() * 1000
                    if cfg["torso_mode"] == "region" else float("nan"))
            dt = (a["torso"][-1] - a["torso"][0]) * 1000
            yaw = np.rad2deg(a["q"][-1, 5] - a["q"][0, 5])
            arm = np.rad2deg(np.abs(a["q"][-1, 6:] - a["q"][0, 6:])).mean()
            print(f"{mname:8s} {cname:13s} {herr:8.1f} {terr:9.1f} {np.array2string(np.round(dt), separator=','):>18s} "
                  f"{yaw:5.1f} {arm:8.1f} {(a['hand_limited'] > 0).mean():9.0%} {a['clamped'].mean():6.0%} "
                  f"{a['min_dist'].min() * 1000:5.0f}", flush=True)


if __name__ == "__main__":
    main()
