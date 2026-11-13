"""설정 읽기. settings.yaml이 기본값이고, 명령줄 --키 값 이 있으면 그 값으로 덮어쓴다."""

import argparse
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = PROJECT_DIR / "settings.yaml"
PATH_KEYS = ("fk_urdf", "leader_urdf")  # 상대 경로면 프로젝트 폴더 기준

KEYS = (
    "robot_address", "robot_model", "power_regex", "servo_regex", "move_to_ready", "ready_pose_deg", "ready_minimum_time_s",
    "command_period_s", "command_minimum_time_s", "control_hold_time_s",
    "hand_linear_velocity", "hand_angular_velocity", "torso_linear_velocity", "torso_angular_velocity",
    "acceleration_limit_scaling", "collision_guard_m",
    "scale", "fk_urdf", "arm_reach_ratio", "soft_margin_m", "hand_head_clearance_m",
    "control_mode", "use_torso", "impedance_enabled", "arm_stiffness", "arm_torque_limit", "arm_damping_ratio",
    "ik_posture_weight",
    "torso_mode", "hand_reference", "torso_ratio_xyz", "torso_ratio_rot", "torso_region_xy", "torso_z_limits", "torso_rot_limit_deg",
    "torso_joint_limits_deg",
    "leader_device", "leader_urdf", "leader_period_s", "leader_torque_limit", "leader_viscous_gain",
    "leader_gravity_gain", "leader_timeout_s", "soft_stop_time_s", "fault_soft_stop",
    "use_gripper", "gripper_max_open", "gripper_invert_trigger",
    "log_enabled",
)


def load(path=SETTINGS_PATH):
    with open(path, encoding="utf-8") as f:
        values = yaml.safe_load(f)
    missing, unknown = set(KEYS) - set(values), set(values) - set(KEYS)
    if missing or unknown:
        raise ValueError(f"{path} 설정 키가 맞지 않습니다. 빠진 키: {sorted(missing)}, 모르는 키: {sorted(unknown)}")
    return values


def parse(argv=None, description=None):
    """settings.yaml을 읽고 명령줄 값으로 덮어쓴 설정 dict를 돌려준다.

    yaml의 모든 키가 --키 옵션이 된다. 값은 yaml 문법으로 읽는다 (예: --torso_ratio_xyz "[0.5, 0.2, 0.3]").
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--settings", default=str(SETTINGS_PATH))
    known, _ = pre.parse_known_args(argv)
    base = load(known.settings)

    p = argparse.ArgumentParser(description=description, parents=[pre])
    for key, value in base.items():
        p.add_argument(f"--{key}", type=yaml.safe_load, default=None, metavar=type(value).__name__.upper())
    args = vars(p.parse_args(argv))

    values = dict(base)
    for key in base:
        if args[key] is not None:
            if type(args[key]) is not type(base[key]) and not (isinstance(base[key], float) and isinstance(args[key], int)):
                raise ValueError(f"--{key}: {type(base[key]).__name__} 값이 필요합니다 (받은 값: {args[key]!r})")
            values[key] = args[key]
    for key in PATH_KEYS:
        values[key] = str((PROJECT_DIR / values[key]).resolve())
    return values
