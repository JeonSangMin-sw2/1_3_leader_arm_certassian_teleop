#!/usr/bin/env python3
"""한계에 닿을 때의 덜컥임 비교 (시뮬레이터, 합성 리더암): soft_margin_m 0(바로 자름) vs 부드러운 수렴.

동작마다 레디에서 0.5 s에 양팔을 누르고 3 s 동안 목표 자세로 간 뒤 1.5 s 유지한다.
명령한 torso 목표의 가속도 최대, 로봇 torso·팔 관절 가속도 최대(로그 관절값의 2차 차분)를 조건마다 3회 돌려 중앙값으로 본다.
손이 머리 회전 중심보다 위로 간 최대 높이(음수면 아래)도 본다. split은 왼팔 위, 오른팔 아래로 반대 방향 동작이다.

사용: .venv/bin/python tests/smooth_compare.py [--scale 1.3] [--torso_ratio_xyz "[0.6, 0.6, 0.6]"]
"""
import sys

import numpy as np
import yaml

import _sim
import ratio_compare as rc

D = _sim.D
MOTIONS = {name: rc.MOTIONS[name] for name in ("forward", "up", "low")}
MOTIONS["split"] = (D([-30, -10, 0, -10, 0, -30, 0]), D([-150, 41, 0, -15, 0, 0, 0]))  # 오른팔 아래 앞, 왼팔 위
MARGINS = [0.0, 0.03, 0.06]
REPEAT = 3


def main():
    overrides = {k.lstrip("-"): yaml.safe_load(v) for k, v in zip(sys.argv[1::2], sys.argv[2::2])}
    overrides.setdefault("scale", 1.3)
    overrides.setdefault("torso_ratio_xyz", [0.6, 0.6, 0.6])
    robot = _sim.connect()
    print(f"scale={overrides['scale']} ratio_xyz={overrides['torso_ratio_xyz']}")
    print(f"{'motion':8s} {'margin':>6s} {'torso tgt acc':>13s} {'torso q acc':>11s} {'arm q acc':>9s} "
          f"{'hand err R/L':>12s} {'hand-head':>9s} {'torso dxyz mm':>18s} {'limited':>7s} {'min d':>5s}")
    for mname, (q_r, q_l) in MOTIONS.items():
        for m in MARGINS:
            cfg = _sim.make_cfg(**{**overrides, "soft_margin_m": m})
            res = []
            for _ in range(REPEAT):
                try:
                    a = _sim.analyze(robot, _sim.run(robot, cfg, rc.make_script(q_r, q_l), 5.0))
                except RuntimeError as e:
                    print(f"{mname:8s} {m:6.2f} 실패: {e}")
                    robot.ensure_ready()
                    continue
                t = a["t"]
                hold = _sim.window(a, 4.5, 5.0)
                acc_q = np.rad2deg(_sim.acc_peak(t, a["q"]))
                res.append(dict(
                    acc_tt=np.linalg.norm(np.gradient(np.gradient(a["torso_tgt"], t, axis=0), t, axis=0), axis=1).max(),
                    acc_qt=acc_q[:6].max(), acc_qa=acc_q[6:].max(),
                    herr=np.linalg.norm(a["hand"] - a["hand_tgt"], axis=2)[hold].max(axis=0) * 1000,
                    above=(a["hand"][:, :, 2].max(axis=1) - a["head_z"]).max() * 1000,
                    dt=(a["torso"][-1] - a["torso"][0]) * 1000,
                    lim=(a["hand_limited"] > 0).mean(), mind=a["min_dist"].min() * 1000))
            if not res:
                continue
            med = lambda k: np.median([r[k] for r in res], axis=0)
            print(f"{mname:8s} {m:6.2f} {med('acc_tt'):10.2f} m/s2 {med('acc_qt'):8.0f} °/s2 {med('acc_qa'):6.0f} °/s2 "
                  f"{med('herr')[0]:5.1f}/{med('herr')[1]:5.1f} {med('above'):6.0f} mm "
                  f"{np.array2string(np.round(med('dt')), separator=','):>18s} {med('lim'):7.0%} {med('mind'):5.0f}", flush=True)


if __name__ == "__main__":
    main()
