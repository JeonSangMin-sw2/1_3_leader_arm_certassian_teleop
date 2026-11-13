# RB-Y1 v1.3 리더암 텔레옵

리더암 관절값을 RB-Y1 v1.2 팔 모델로 풀어 손 자세(4×4)를 얻고, 버튼을 누른 순간 대비 변화량을 v1.3 로봇의 base→ee 목표로 바꿔 100 Hz 스트림으로 보낸다.
리더암과 v1.2 팔은 관절 순서·축·0점이 같고 링크만 약 1.43배 길다. 그래서 리더암 엔코더 → v1.2 FK가 곧 "로봇 크기로 키운 리더암"이 된다.

```
리더암 엔코더 → v1.2 FK (torso_5→ee 4×4) → 누른 순간 대비 변화량 (위치 × scale, 회전 1:1)
  → v1.3 base→ee 손 목표 → (use_torso면 torso_5 목표 분배·허용영역) → 팔 길이·머리 높이 한도
  → joint_ik: 관절 IK로 관절 명령   /   cartesian: 4×4 Cartesian 명령 (제어기가 IK)
```

## 1. 설치 (처음 한 번)

Python 3.10 이상. 로봇에 붙어 있는 PC(UPC, 리더암·그리퍼 USB가 연결된 곳)에서 한다.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt`는 `rby1-sdk==0.10.1`로 고정되어 있다. 로봇 펌웨어와 SDK 버전이 맞지 않으면 SDK 버전을 맞춘다.

## 2. 실행

### 시뮬레이터 (리더암 없이 기능 확인)

```bash
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -p 50051:50051 rainbowroboticsofficial/rby1-sim:0.10.6-m_v1.3
.venv/bin/python tests/sim_synthetic.py
```

`teleop.py`는 리더암 장치가 있어야 시작한다. 리더암 없이 보려면 `tests/`의 합성 리더암 테스트를 쓴다.

### 실물

1. `settings.yaml`에서 `robot_address`를 로봇 주소로 바꾼다 (UPC에서는 `192.168.30.1:50051`). 그리퍼를 쓰면 `use_gripper: true`.
2. 실행한다.

```bash
.venv/bin/python teleop.py
```

settings.yaml의 어떤 값이든 명령줄 `--키 값`으로 덮어쓸 수 있다 (값은 yaml 문법).

```bash
.venv/bin/python teleop.py --robot_address 192.168.30.1:50051 --use_torso false --scale 1.0
```

### 실행하면 일어나는 일

1. 로봇에 접속하고, 전원·서보가 꺼져 있으면 켠 뒤 컨트롤 매니저를 활성화한다.
2. **로봇이 레디자세로 움직인다 (5초).** `move_to_ready: false`면 건너뛴다.
3. `use_gripper`면 그리퍼가 homing으로 양 끝까지 움직인다.
4. 리더암이 켜지고, 버튼을 누르지 않은 팔은 그 자리에 고정된다.

### 조작

| 입력 | 동작 |
|---|---|
| 리더암 버튼 누르고 있기 | 그 팔을 로봇이 따라온다 (누른 순간 위치에서 시작) |
| 버튼 놓기 | 로봇 팔이 그 자리에 멈추고 리더암도 고정된다. 리더암을 옮긴 뒤 다시 누르면 그 자리에서 이어진다 (클러치) |
| 트리거 | 그리퍼 닫힘 정도 (0~1000). 버튼과 상관없이 따른다 |
| Ctrl+C | 스트림 취소, 리더암·그리퍼 토크 끔 |

링크 사이 거리가 `collision_guard_m`(20 mm)보다 작아지면서 계속 줄면 멈춘다. 버튼을 놓았다 다시 누르면 이어진다.
실행 기록은 `logs/run_날짜_시각.csv`와 그 실행의 설정 `.yaml`로 남는다.

## 3. 실물 첫 시험 순서 (권장)

비상정지 버튼을 손에 두고, 로봇 팔 반경 안에 사람이 없게 한다. 한 단계씩 확인하고 다음으로 넘어간다.

| 단계 | 덧붙일 옵션 | 확인할 것 |
|---|---|---|
| 1. 팔만, 느리게 | `--use_torso false --hand_linear_velocity 0.3 --hand_angular_velocity 1.0` | 레디자세 이동, 버튼 클러치, 손이 리더암 방향대로 움직이는지 |
| 2. 팔만, 기본 속도 | `--use_torso false` | 추종 지연, 팔 출렁임 |
| 3. 그리퍼 | `--use_torso false --use_gripper true` | 트리거를 당길 때 닫히는지. 반대면 `--gripper_invert_trigger true` |
| 4. 임피던스 | `--use_torso false --impedance_enabled true` | 접촉 시 부드러운지, 손 처짐 |
| 5. torso 사용 | (기본값 `use_torso: true`) | torso가 허용영역 안에서만 움직이는지, 무게중심 이동 시 흔들림 |
| (비교) Cartesian | `--control_mode cartesian` | 제어기 IK와 관절 IK 비교 |

## 4. 주요 설정

| 키 | 기본값 | 내용 |
|---|---|---|
| `control_mode` | `joint_ik` | `joint_ik`: 관절 IK → 관절 명령 (여유 자유도는 자세 유지로 붙잡음). `cartesian`: 4×4 Cartesian 명령, 제어기가 IK |
| `use_torso` | `true` | `false`면 torso를 시작 자세에 고정하고 팔만 쓴다 |
| `impedance_enabled` | `false` | 팔을 관절 임피던스로 (`joint_ik`에서만). 강성 `arm_stiffness`, 토크 한도 `arm_torque_limit` |
| `scale` | `1.0` | v1.2 FK 위에 곱하는 추가 위치 배율 (리더암 대비 실제 배율 약 1.43 × scale) |
| `command_period_s` | `0.01` | 100 Hz |
| `hand_linear_velocity` | `1.0` | 손 목표가 갈 수 있는 속도 [m/s] |
| `torso_ratio_xyz` | `[0.3, 0.3, 0.3]` | 손 이동량 중 torso가 맡는 비율 (base x, y, z) |
| `torso_region_xy`, `torso_z_limits` | 사각형, `[1.00, 1.20]` | torso_5 허용영역 (base 기준) |
| `soft_margin_m` | `0.03` | 한계 앞 이 구간부터 부드럽게 감속 (영역, 팔 길이, 머리 높이) |
| `hand_head_clearance_m` | `0.0` | 손이 머리 회전 중심보다 위로 못 가게 |
| `use_gripper` | `false` | 시뮬레이터에는 그리퍼 장치가 없다 |

## 5. 알려진 한계 (시뮬레이터에서 확인, 실물 미확인)

- **추종 지연**: 손을 ±9 cm로 흔들 때 0.5 Hz에서 joint_ik 90~106 ms, cartesian 122~143 ms. 1 Hz로 빠르게 흔들면 팔만 쓸 때(joint_ik) 관절 속도 한도(arm_0 180°/s)에 걸려 지연이 400 ms까지 커진다. torso를 쓰면 150 ms.
- **v1.3 레디자세**: SDK 예제 35의 레디자세는 손목 70°라 v1.3(arm_5 ±50°)이 거부한다. 손목만 45°로 바꿨다. 레디자세가 관절 한계 밖이면 시작할 때 오류로 멈춘다.
- **v1.3 손목**: arm_5가 ±50°라 리더암 손목을 50° 넘게 꺾으면 손 방향을 다 못 따라간다.
- **임피던스**: 강성 150에서 한 팔만 움직일 때 놓아 둔 손이 최대 27 mm 흔들렸다.
- **리더암 토크 값**(SDK 예제 35 기준)과 **트리거 방향**은 실물에서 확인하지 않았다.
- 팔꿈치 돌림각을 리더암에 맞추는 기능은 손 추종을 나쁘게 해서 넣지 않았다 (`arm_ik.py` 첫 주석).

## 6. 테스트 (시뮬레이터 127.0.0.1:50051)

```bash
.venv/bin/python tests/sim_synthetic.py                       # 기능: 추종, 클러치, 영역, 한 팔
.venv/bin/python tests/sim_synthetic.py --use_torso false     # settings 키 덮어쓰기 가능
.venv/bin/python tests/stream_follow.py                       # 실시간 추종 지연
.venv/bin/python tests/smooth_compare.py                      # 한계에서 덜컥임
.venv/bin/python tests/ratio_compare.py                       # torso 비율 비교
.venv/bin/python tests/gripper_fake.py                        # 그리퍼 오프라인 (장치 없이)
```

## 파일

| 파일 | 내용 |
|---|---|
| `teleop.py` | 매핑, torso 비율·허용영역·팔 길이·머리 높이 한도, 텔레옵 루프 |
| `robot.py` | 로봇 접속·준비·상태·명령 스트림 (Cartesian, 관절), torso 관절 한계 투영 (`Robot`) |
| `arm_ik.py` | 팔 관절 IK (`ArmIK`). rby1_3d_mouse_teleop의 joint_ik.py에서 가져옴 |
| `leader_arm.py` | 리더암 중력 보상·버튼 클러치·트리거 (`LeaderArm`) |
| `gripper.py` | 그리퍼 homing·트리거 추종 (`Gripper`). `/dev/rby1_gripper`는 UPC에만 있다 |
| `so3.py` | 회전 벡터 ↔ 회전 행렬 |
| `settings.py`, `settings.yaml` | 설정 읽기 (키 검사, 명령줄 덮어쓰기), 기본값 |
| `models/` | SDK 0.10.1 저장소 `models/leader_arm/model.urdf`, `models/rby1m/urdf/model_v1.2.urdf` 복사본 (수정하지 않음). 로봇 쪽 기구학은 접속한 로봇의 모델을 쓴다 |
| `tests/` | 시뮬레이터·오프라인 검증 스크립트. 합성 리더암으로 돈다 |
