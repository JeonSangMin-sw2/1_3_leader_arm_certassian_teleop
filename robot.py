"""RB-Y1 접속, 준비, 상태 스트림, 기구학, 몸통 Cartesian 명령 스트림.

참고: SDK 예제 04(상태 스트림), 28·33(Cartesian 명령), 35(전원·레디자세 절차).
"""

import time

import numpy as np
import rby1_sdk as rby

from so3 import log_so3

TORSO = [f"torso_{i}" for i in range(6)]
ARM_R = [f"right_arm_{i}" for i in range(7)]
ARM_L = [f"left_arm_{i}" for i in range(7)]


class Robot:
    def __init__(self, cfg):
        self.cfg = cfg
        self.robot = rby.create_robot(cfg["robot_address"], cfg["robot_model"])
        if not self.robot.connect():
            raise RuntimeError(f"로봇에 접속하지 못했습니다: {cfg['robot_address']}")
        self.model = self.robot.model()
        self.power_names = [p.name for p in self.robot.get_robot_info().power_infos]  # 실물 M: 12v 등, 시뮬레이터: main
        self.dyn = self.robot.get_dynamics()  # 접속한 로봇 자신의 모델 (v1.3이면 v1.3)
        self.dyn_state = self.dyn.make_state(["base", "link_torso_5", "ee_right", "ee_left"],
                                             self.model.robot_joint_names)
        self.torso_state = self.dyn.make_state(["base", "link_torso_5"], TORSO)
        self.torso_idx = list(self.model.torso_idx)
        self.arm_idx = list(self.model.right_arm_idx) + list(self.model.left_arm_idx)
        self._latest = {}
        self._stream = None
        self._state_streaming = False

    def ensure_ready(self):
        """전원·서보는 컨트롤 매니저가 Enabled가 아닐 때만 바꾼다 (Enabled 중에 바꾸면 MajorFault)."""
        faults = (rby.ControlManagerState.State.MinorFault, rby.ControlManagerState.State.MajorFault)
        if self.robot.get_control_manager_state().state in faults:
            for _ in range(20):  # 폴트 대응이 끝나기 전에는 리셋이 거부된다
                if self.robot.reset_fault_control_manager():
                    break
                time.sleep(0.5)
        # is_power_on·is_servo_on은 정규식에 맞는 것이 모두 켜져 있어야 True. 이미 켜진 쪽은 다시 켜지 않는다
        power, servo = self.cfg["power_regex"], self.cfg["servo_regex"]
        need_power = not self.robot.is_power_on(power)
        need_servo = not self.robot.is_servo_on(servo)
        if need_power or need_servo:
            self.robot.disable_control_manager()
            time.sleep(0.5)
            if need_power:
                if not self.robot.power_on(power):
                    raise RuntimeError(f"전원({power})을 켜지 못했습니다")
                time.sleep(1.5)  # 12V 안정화, 리더암 모터 부팅 (예제 35는 1.0 s)
            if self.robot.is_servo_on(servo):  # 전원을 켠 뒤 다시 본다 (12V만 켰으면 서보는 그대로 켜져 있다)
                print(f"[준비] 서보({servo})가 이미 켜져 있어 servo_on을 건너뜁니다", flush=True)
            else:
                if not self.robot.servo_on(servo):
                    raise RuntimeError(f"서보({servo})를 켜지 못했습니다")
                time.sleep(1.0)
            self.robot.reset_fault_control_manager()
        if not self.robot.enable_control_manager():
            raise RuntimeError("컨트롤 매니저를 활성화하지 못했습니다")

    def ensure_tool_flange_12v(self):
        """그리퍼 전원(툴 플랜지)을 12V로 맞춘다. 12V면 그대로, 꺼져 있으면 켜고, 다른 전압이면 끈 뒤 켠다.

        rby1_3d_mouse_teleop에서 가져왔다: 다른 패키지가 다른 전압으로 켜 둔 채 넘어오면 그리퍼 homing이 제대로 안 됐다.
        """
        voltage = lambda side: int(getattr(self.robot.get_state(), f"tool_flange_{side}").output_voltage)
        for side in ("right", "left"):
            if voltage(side) == 12:
                continue
            if voltage(side) != 0:
                if not self.robot.set_tool_flange_output_voltage(side, 0):
                    raise RuntimeError(f"{side} 툴 플랜지 전원을 끄지 못했습니다")
                time.sleep(0.5)
            if not self.robot.set_tool_flange_output_voltage(side, 12):
                raise RuntimeError(f"{side} 툴 플랜지를 12V로 켜지 못했습니다")
            deadline = time.monotonic() + 2.0
            while voltage(side) != 12:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{side} 툴 플랜지가 2초 안에 12V가 되지 않았습니다 (지금 {voltage(side)}V)")
                time.sleep(0.1)

    def move_ready(self, minimum_time=None):
        pose = self.cfg["ready_pose_deg"]
        t = minimum_time or self.cfg["ready_minimum_time_s"]
        lo, hi = self.dyn.get_limit_q_lower(self.dyn_state), self.dyn.get_limit_q_upper(self.dyn_state)
        for part, idx in (("torso", self.torso_idx), ("right_arm", self.model.right_arm_idx), ("left_arm", self.model.left_arm_idx)):
            q = np.deg2rad(pose[part])
            bad = [(i, round(float(np.rad2deg(q[k])), 1), round(float(np.rad2deg(lo[j])), 1), round(float(np.rad2deg(hi[j])), 1))
                   for k, j in enumerate(idx) if not lo[j] <= q[k] <= hi[j]]
            if bad:  # v1.3은 arm_5가 ±50°라 v1.2용 레디자세(손목 70°)를 거부한다
                raise ValueError(f"ready_pose_deg.{part}가 로봇 관절 한계 밖입니다 (관절 순번, 값, 하한, 상한): {bad}")
        rc = rby.BodyComponentBasedCommandBuilder()
        for setter, q in ((rc.set_torso_command, pose["torso"]), (rc.set_right_arm_command, pose["right_arm"]),
                          (rc.set_left_arm_command, pose["left_arm"])):
            setter(rby.JointPositionCommandBuilder().set_minimum_time(t).set_position(np.deg2rad(q)))
        fb = self.robot.send_command(rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(rc)), 1).get()
        if fb.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            raise RuntimeError(f"레디자세 이동 실패: {fb.finish_code}")

    def start_state_update(self):
        self._latest.clear()  # 이전 실행의 상태를 새 상태처럼 쓰지 않게
        self.robot.start_state_update(lambda rs, cm: self._latest.update(rs=rs, cm=cm),
                                      1.0 / self.cfg["command_period_s"])
        self._state_streaming = True
        while "rs" not in self._latest:
            time.sleep(0.01)

    def stop_state_update(self):
        if self._state_streaming:
            self.robot.stop_state_update()
            self._state_streaming = False

    def shutdown(self):
        """소프트 정지의 로봇 쪽 (예제 35): 동작 중인 제어 취소 → 컨트롤 매니저 disable → 12V(리더암·툴 플랜지) 끔."""
        self.close_stream()
        self.stop_state_update()
        cm = self.robot.get_control_manager_state()
        if cm.state == rby.ControlManagerState.State.Enabled:
            if cm.control_state != rby.ControlManagerState.ControlState.Idle:
                self.robot.cancel_control()
                time.sleep(0.1)
            self.robot.disable_control_manager()
            time.sleep(0.2)
        if "12v" not in self.power_names:
            print(f"[종료] 12v 전원 장치가 없어 끄지 않습니다 (전원 장치: {self.power_names})", flush=True)
        elif not self.robot.power_off("12v"):
            raise RuntimeError("12V 전원을 끄지 못했습니다")

    def power_off_all(self):
        """즉시 정지 (예제 35 USE_SOFT_STOP=False): 전원 라인을 모두 끈다."""
        self.close_stream()
        self.stop_state_update()
        if not self.robot.power_off(".*"):
            raise RuntimeError("전원을 끄지 못했습니다")

    def arm_q(self):
        """지금 팔 관절값 14개 (오른팔 7, 왼팔 7) [rad]. 상태 스트림 없이 한 번 읽는다."""
        return np.array(self.robot.get_state().position)[self.arm_idx]

    def kinematics(self):
        """최신 상태에서 (q, T_base_torso5, (T_base_ee_right, T_base_ee_left), 가장 가까운 링크 거리)."""
        rs = self._latest["rs"]
        self.dyn_state.set_q(rs.position)
        self.dyn.compute_forward_kinematics(self.dyn_state)
        T = [self.dyn.compute_transformation(self.dyn_state, 0, i) for i in (1, 2, 3)]
        dist = min((c.distance for c in rs.collisions), default=np.inf)
        return np.array(rs.position), T[0], (T[1], T[2]), dist

    def arm_geometry(self):
        """팔마다 (torso_5 기준 어깨 중심, link_arm_6→ee 변환, 팔을 다 편 어깨-손목 거리).

        arm_0~2 축이 link_arm_0 원점에서, arm_4~6 축이 link_arm_6 원점에서 만나므로 두 점을 어깨·손목 중심으로 쓴다.
        """
        links = ["link_torso_5", "link_right_arm_0", "link_right_arm_6", "ee_right",
                 "link_left_arm_0", "link_left_arm_6", "ee_left"]
        st = self.dyn.make_state(links, self.model.robot_joint_names)
        st.set_q(np.zeros(len(self.model.robot_joint_names)))  # 팔 관절 0 = 팔을 아래로 곧게 편 자세
        self.dyn.compute_forward_kinematics(st)
        out = []
        for k in (1, 4):
            T_t_sh = self.dyn.compute_transformation(st, 0, k)
            T_6_ee = self.dyn.compute_transformation(st, k + 1, k + 2)
            T_sh_6 = self.dyn.compute_transformation(st, k, k + 1)
            out.append((T_t_sh[:3, 3].copy(), T_6_ee.copy(), float(np.linalg.norm(T_sh_6[:3, 3]))))
        return out

    def head_offset(self):
        """torso_5 기준 머리 회전 중심 (link_head_2 원점). 머리 관절이 돌아도 이 점은 그대로다."""
        st = self.dyn.make_state(["link_torso_5", "link_head_2"], self.model.robot_joint_names)
        st.set_q(np.zeros(len(self.model.robot_joint_names)))
        self.dyn.compute_forward_kinematics(st)
        return self.dyn.compute_transformation(st, 0, 1)[:3, 3].copy()

    def torso_limits(self):
        """로봇 모델의 torso 관절 한계 (lower, upper) [rad]."""
        return (self.dyn.get_limit_q_lower(self.torso_state)[:6].copy(),
                self.dyn.get_limit_q_upper(self.torso_state)[:6].copy())

    def project_torso(self, T_target, q0, lower, upper, iterations=30):
        """관절 한계 안에서 T_target에 가장 가까운 torso_5 자세와 그 관절값 (한계로 자르는 감쇠 최소제곱 IK)."""
        st = self.torso_state
        x = np.zeros(len(st.get_q()))  # make_state가 torso 관절을 앞에 둔다

        def fk(q):
            x[:6] = q
            st.set_q(x)
            self.dyn.compute_forward_kinematics(st)
            return self.dyn.compute_transformation(st, 0, 1)

        q = np.clip(q0, lower, upper)
        for _ in range(iterations):
            T = fk(q)
            e = np.r_[log_so3(T[:3, :3].T @ T_target[:3, :3]), T[:3, :3].T @ (T_target[:3, 3] - T[:3, 3])]
            if np.linalg.norm(e) < 1e-7:
                break
            J = self.dyn.compute_body_jacobian(st, 0, 1)[:, :6]  # [각속도; 선속도], torso_5 좌표계
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), lower, upper)
        return fk(q).copy(), q

    def open_stream(self):
        self._stream = self.robot.create_command_stream()

    def send(self, T_right, T_left, T_torso=None):
        """양손(그리고 torso_5) 목표를 몸통 Cartesian 명령 하나로 보낸다. 목표는 모두 base 기준 행렬로 받는다.

        hand_reference가 torso면 손 목표를 torso_5 목표 기준으로 바꿔 보낸다. 다 풀리지 않을 때
        base 기준은 torso가 목표를 벗어나고, torso 기준은 손이 목표를 벗어난다.
        """
        c = self.cfg
        ref = "base"
        if T_torso is not None and c["hand_reference"] == "torso":
            ref, inv = "link_torso_5", np.linalg.inv(T_torso)
            T_right, T_left = inv @ T_right, inv @ T_left
        b = (rby.CartesianCommandBuilder()
             .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(c["control_hold_time_s"]))
             .add_target(ref, "ee_right", T_right, c["hand_linear_velocity"], c["hand_angular_velocity"],
                         c["acceleration_limit_scaling"])
             .add_target(ref, "ee_left", T_left, c["hand_linear_velocity"], c["hand_angular_velocity"],
                         c["acceleration_limit_scaling"]))
        if T_torso is not None:
            b.add_target("base", "link_torso_5", T_torso, c["torso_linear_velocity"], c["torso_angular_velocity"],
                         c["acceleration_limit_scaling"])
        b.set_minimum_time(c["command_minimum_time_s"])
        self._stream.send_command(rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(b)))

    def _header(self):
        return rby.CommandHeaderBuilder().set_control_hold_time(self.cfg["control_hold_time_s"])

    def _send_body(self, rc):
        self._stream.send_command(rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(rc)))

    def send_joints(self, torso_q, q_right, q_left):
        """torso와 양팔 관절 명령. 임피던스면 팔은 관절 임피던스, torso는 위치 명령 (rby1_3d_mouse_teleop과 같은 구성)."""
        c, t = self.cfg, self.cfg["command_minimum_time_s"]
        rc = rby.BodyComponentBasedCommandBuilder().set_torso_command(
            rby.JointPositionCommandBuilder().set_command_header(self._header()).set_minimum_time(t).set_position(torso_q))
        for setter, q in ((rc.set_right_arm_command, q_right), (rc.set_left_arm_command, q_left)):
            if c["impedance_enabled"]:
                b = (rby.JointImpedanceControlCommandBuilder().set_command_header(self._header()).set_minimum_time(t)
                     .set_position(q).set_stiffness(c["arm_stiffness"]).set_torque_limit(c["arm_torque_limit"])
                     .set_damping_ratio(c["arm_damping_ratio"]))
            else:
                b = rby.JointPositionCommandBuilder().set_command_header(self._header()).set_minimum_time(t).set_position(q)
            setter(b)
        self._send_body(rc)

    def send_arms_cartesian(self, torso_q, T_right, T_left, arm2_q=None):
        """torso는 관절 위치로 붙잡고 팔만 base→ee Cartesian 명령 (use_torso false).

        arm2_q (오른팔, 왼팔) [rad]가 있으면 팔마다 arm_2(어깨 yaw, 위팔 돌림) 관절 목표를 같은 Cartesian 명령에 더한다.
        """
        c, t = self.cfg, self.cfg["command_minimum_time_s"]
        rc = rby.BodyComponentBasedCommandBuilder().set_torso_command(
            rby.JointPositionCommandBuilder().set_command_header(self._header()).set_minimum_time(t).set_position(torso_q))
        for i, (setter, side, T) in enumerate(((rc.set_right_arm_command, "right", T_right),
                                               (rc.set_left_arm_command, "left", T_left))):
            b = (rby.CartesianCommandBuilder().set_command_header(self._header()).set_minimum_time(t)
                 .add_target("base", f"ee_{side}", T, c["hand_linear_velocity"], c["hand_angular_velocity"],
                             c["acceleration_limit_scaling"]))
            if arm2_q is not None:
                b.add_joint_position_target(f"{side}_arm_2", float(arm2_q[i]))
            setter(b)
        self._send_body(rc)

    def close_stream(self):
        if self._stream is not None:
            try:
                self._stream.cancel()
            except RuntimeError:
                pass  # 이미 끝난 스트림
            self._stream = None
            time.sleep(0.5)  # 취소 직후 보낸 명령은 Canceled로 거부된 적이 있다
