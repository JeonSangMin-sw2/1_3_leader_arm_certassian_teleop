#!/usr/bin/env python3
"""스트림 실시간 추종 확인 (시뮬레이터, 합성 리더암): 목표가 계속 바뀔 때 로봇이 따라오는지, 50 Hz와 100 Hz 비교.

0.5 s에 양팔을 누르고 1 s 동안 시작한 뒤 어깨 Pitch를 ±15° 사인파로 흔든다 (손이 앞뒤·위아래로 약 ±0.15 m).
실제 루프 주기, 목표 대비 지연(교차상관), 움직이는 동안 손 추종 오차를 본다.

사용: .venv/bin/python tests/stream_follow.py
"""
import numpy as np

import _sim

D, R = _sim.D, _sim.READY
AMP = D(15)


def make_script(freq):
    def script(t):
        s = np.clip((t - 0.5) / 1.0, 0, 1)  # 1 s 동안 진폭을 키운다
        dq = np.zeros(7)
        dq[0] = -s * AMP * np.sin(2 * np.pi * freq * max(t - 0.5, 0))
        return _sim.both(R + dq), (t >= 0.5, t >= 0.5)
    return script


def lag_ms(t, target, actual):
    """target을 몇 초 늦춰야 actual과 가장 잘 맞는지 (1 ms 격자로 보간)."""
    tt = np.arange(t[0], t[-1], 0.001)
    a, b = np.interp(tt, t, target), np.interp(tt, t, actual)
    a, b = a - a.mean(), b - b.mean()
    shifts = range(0, 400)
    err = [np.mean((a[:len(a) - k] - b[k:]) ** 2) for k in shifts]
    return float(np.argmin(err))


def main():
    robot = _sim.connect()
    print(f"{'period':>6s} {'torso':>6s} {'freq':>5s} {'loop period ms (mean/p99/max)':>30s} {'lag ms':>6s} "
          f"{'err rms/max mm':>15s} {'target ±mm':>10s}")
    for period in (0.02, 0.01):
        for mode in ("free", "region"):
            for freq in (0.5, 1.0):
                cfg = _sim.make_cfg(command_period_s=period, torso_mode=mode)
                a = _sim.analyze(robot, _sim.run(robot, cfg, make_script(freq), 6.0))
                t = a["t"]
                dt = np.diff(t) * 1000
                w = _sim.window(a, 2.0, 6.0)
                tgt, act = a["hand_tgt"][w, 0], a["hand"][w, 0]
                err = np.linalg.norm(act - tgt, axis=1) * 1000
                axis = int(np.argmax(tgt.std(axis=0)))
                lag = lag_ms(t[w], tgt[:, axis], act[:, axis])
                print(f"{period * 1000:4.0f}ms {mode:>6s} {freq:4.1f}Hz {dt.mean():9.1f} / {np.percentile(dt, 99):5.1f} / {dt.max():5.1f}"
                      f"      {lag:6.0f} {np.sqrt((err ** 2).mean()):7.1f} / {err.max():5.1f} {np.ptp(tgt[:, axis]) * 500:9.0f}",
                      flush=True)


if __name__ == "__main__":
    main()
