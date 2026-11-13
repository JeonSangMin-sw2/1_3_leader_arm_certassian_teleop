#!/usr/bin/env python3
"""teleop 기능 검증 (시뮬레이터, 합성 리더암).

  0-1 s    놓음, 레디
  1-5 s    양팔 누름, 레디 -> 앞으로, 4-5 s 유지          A: 추종
  5-7 s    놓음, 리더암은 레디로 돌아감                   B: 로봇 손·torso가 멈춰 있는지
  7 s      다시 누름                                      C: 다시 잡을 때 튐
  7-11 s   레디 -> 아래 앞쪽, 10.5-11 s 유지              D: 추종
  전체     E: torso_5 실제 위치가 허용영역 밖으로 나간 최대 거리, H: torso_5 목표 추종 오차
두 번째 실행 (레디자세에서 새로 시작. 클러치를 거듭하면 이동량이 쌓여 손 목표가 닿지 않는 곳까지 갈 수 있어서 분리)
  0-1 s    놓음
  1-5 s    오른팔만 누름, 오른팔 앞으로                   F: 놓아 둔 왼손이 고정인지, G: 오른손 추종

사용: .venv/bin/python tests/sim_synthetic.py [--scale 1.3] [--torso_mode free] ...  (settings 키 덮어쓰기)
"""
import sys

import numpy as np
import yaml

import _sim

FORWARD = _sim.D([-80, -22, 0, -15, 0, 0, 0])  # 어깨 Pitch 축이 20° 기울어 있어 롤로 손 폭을 유지
LOW = _sim.D([-30, -10, 0, -10, 0, -30, 0])
R, MJ = _sim.READY, _sim.minjerk


def script(t):
    if t < 1: return _sim.both(R), (False, False)
    if t < 5: return _sim.both(R + MJ((t - 1) / 3) * (FORWARD - R)), (True, True)
    if t < 7: return _sim.both(FORWARD + MJ((t - 5) / 2) * (R - FORWARD)), (False, False)
    return _sim.both(R + MJ((t - 7) / 3) * (LOW - R)), (True, True)


def script_one_arm(t):
    if t < 1: return _sim.both(R), (False, False)
    return _sim.both(R + MJ((t - 1) / 3) * (FORWARD - R), R * _sim.MIRROR), (True, False)


def main():
    overrides = {}
    for k, v in zip(sys.argv[1::2], sys.argv[2::2]):
        overrides[k.lstrip("-")] = yaml.safe_load(v)
    cfg = _sim.make_cfg(**overrides)
    robot = _sim.connect()
    a = _sim.analyze(robot, _sim.run(robot, cfg, script, 11.0))
    b = _sim.analyze(robot, _sim.run(robot, cfg, script_one_arm, 5.0))
    w = lambda t0, t1: _sim.window(a, t0, t1)
    err = np.linalg.norm(a["hand"] - a["hand_tgt"], axis=2) * 1000
    held, torso_held = a["hand"][w(5.3, 7.0)], a["torso"][w(5.3, 7.0)]
    pre, post = a["hand"][w(6.8, 7.0)][-1], a["hand"][w(7.0, 7.1)]
    left = b["hand"][_sim.window(b, 1.0, 5.0), 1]
    err_b = np.linalg.norm(b["hand"] - b["hand_tgt"], axis=2) * 1000
    region = cfg["torso_mode"] == "region"
    torso_err = np.linalg.norm(np.r_[a["torso"] - a["torso_tgt"], b["torso"] - b["torso_tgt"]], axis=1) * 1000
    res = {
        "A 추종(앞으로 유지)": err[w(4.5, 5.0)].max(),
        "B 놓았을 때 손 흐름": np.linalg.norm(held - held[0], axis=2).max() * 1000,
        "B 놓았을 때 torso 흐름": np.linalg.norm(torso_held - torso_held[0], axis=1).max() * 1000,
        "C 다시 잡을 때 튐": np.linalg.norm(post - pre, axis=2).max() * 1000,
        "D 추종(아래 유지)": err[w(10.5, 11.0)].max(),
        "E torso 영역 밖 최대": max(_sim.outside_region(cfg, p) for p in np.r_[a["torso"], b["torso"]]) * 1000,
        "F 놓아 둔 왼손 흐름": np.linalg.norm(left - left[0], axis=1).max() * 1000,
        "G 오른손 추종(한 팔 유지)": err_b[_sim.window(b, 4.5, 5.0), 0].max(),
    }
    if region:
        res["H torso 목표 추종 최대"] = torso_err.max()
    lim = np.deg2rad(np.array(cfg["torso_joint_limits_deg"]))
    qt = np.r_[a["q"][:, :6], b["q"][:, :6]]
    over = np.rad2deg(np.maximum(lim[:, 0] - qt, qt - lim[:, 1]).max(axis=0))
    print(f"scale={cfg['scale']} torso={cfg['torso_mode']} hand_ref={cfg['hand_reference']} ratio_xyz={cfg['torso_ratio_xyz']} "
          f"ratio_rot={cfg['torso_ratio_rot']}  samples={len(a['t'])}  "
          f"min link dist={a['min_dist'].min() * 1000:.0f} mm  torso clamped {a['clamped'].mean():.0%}")
    for k, v in res.items():
        print(f"  {k:22s} {v:7.1f} mm")
    print(f"  I torso 관절 한계 밖 최대(deg, 음수면 안쪽 여유) {np.round(over, 1).tolist()}")
    print(f"  torso_1 범위 (deg) [{np.rad2deg(qt[:, 1].min()):.1f}, {np.rad2deg(qt[:, 1].max()):.1f}]")
    print(f"  torso 이동 최대(xyz mm)  {np.round(np.abs(a['torso'] - a['torso'][0]).max(axis=0) * 1000).tolist()}")


if __name__ == "__main__":
    main()
