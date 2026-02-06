"""MuJoCo robot simulator with position control for ALOHA dual-arm manipulator.

This simulator is designed for the ALOHA robot which has:
- Two fixed arms (left and right) mounted on a table
- Each arm has 6 DOF (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate)
- Each arm has a 2-finger parallel gripper

Currently implemented: Left arm only (single arm control)
"""

import time
import numpy as np
import mujoco, mujoco.viewer
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple


class RobotConfig:
    """Robot simulation configuration constants for ALOHA."""

    # Left arm joints: [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
    LEFT_ARM_JOINT_NAMES = [
        "left/waist",
        "left/shoulder",
        "left/elbow",
        "left/forearm_roll",
        "left/wrist_angle",
        "left/wrist_rotate"
    ]

    LEFT_ARM_ACTUATOR_NAMES = [
        "left/waist",
        "left/shoulder",
        "left/elbow",
        "left/forearm_roll",
        "left/wrist_angle",
        "left/wrist_rotate"
    ]

    # Right arm joints: [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
    RIGHT_ARM_JOINT_NAMES = [
        "right/waist",
        "right/shoulder",
        "right/elbow",
        "right/forearm_roll",
        "right/wrist_angle",
        "right/wrist_rotate"
    ]

    RIGHT_ARM_ACTUATOR_NAMES = [
        "right/waist",
        "right/shoulder",
        "right/elbow",
        "right/forearm_roll",
        "right/wrist_angle",
        "right/wrist_rotate"
    ]

    # Default to left arm for backward compatibility
    ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES
    ARM_ACTUATOR_NAMES = LEFT_ARM_ACTUATOR_NAMES

    # End effector site name
    EE_SITE_NAME = "left/gripper"

    # Gripper actuator/joint names
    LEFT_GRIPPER_ACTUATOR_NAME = "left/gripper"
    LEFT_GRIPPER_JOINT_NAME = "left/left_finger"
    RIGHT_GRIPPER_ACTUATOR_NAME = "right/gripper"
    RIGHT_GRIPPER_JOINT_NAME = "right/left_finger"

    # Default to left gripper for backward compatibility
    GRIPPER_ACTUATOR_NAME = LEFT_GRIPPER_ACTUATOR_NAME
    GRIPPER_JOINT_NAME = LEFT_GRIPPER_JOINT_NAME

    # Arm PID controller gains for position control (6 joints)
    ARM_KP = np.array([2.0, 2.0, 2.0, 1.5, 1.0, 1.0])
    ARM_KI = np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
    ARM_I_LIMIT = np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
    ARM_KD = np.array([0.05, 0.05, 0.05, 0.01, 0.01, 0.01])
    
    # Joint limits from aloha.xml
    ARM_JOINT_LIMITS = np.array([
        [-3.14158, 3.14158],   # waist
        [-1.85005, 1.25664],   # shoulder
        [-1.76278, 1.6057],    # elbow
        [-3.14158, 3.14158],   # forearm_roll
        [-1.8675, 2.23402],    # wrist_angle
        [-3.14158, 3.14158]    # wrist_rotate
    ])

    # IK solver parameters
    IK_MAX_ITERATIONS = 100
    IK_POSITION_TOLERANCE = 0.001  # 1mm
    IK_ORIENTATION_TOLERANCE = 0.01  # ~0.57 degrees
    IK_DAMPING = 0.01  # Damped Least Squares damping factor
    IK_STEP_SIZE = 0.5  # Step size for joint updates

    # Camera settings
    CAM_LOOKAT = [0.0, -0.1, 0.2]
    CAM_DISTANCE = 1.5
    CAM_AZIMUTH = 90
    CAM_ELEVATION = -20

    # Initial positions
    ARM_INIT_POSITION = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
    GRIPPER_INIT_WIDTH = 0.037  # Fully open (max is 0.037)

    # Maximum arm joint speed in rad/s (for smoothing)
    MAX_ARM_SPEED = 2.0  # Limits the rate of change of target position


class MujocoSimulator:
    """MuJoCo simulator with position control for ALOHA robot."""

    def __init__(self, xml_path: str = "../model/aloha/scene.xml") -> None:
        """Initialize simulator with MuJoCo model and control indices."""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._arm_target_joint = RobotConfig.ARM_INIT_POSITION.copy()
        # Internal target for smoothing (current smoothed target)
        self._current_target_joint = RobotConfig.ARM_INIT_POSITION.copy()
        
        self._gripper_target_width = RobotConfig.GRIPPER_INIT_WIDTH
        self.dt = self.model.opt.timestep
        self._arm_error_integral = np.zeros(6,)

        # Resolve arm joint IDs and actuator IDs
        self.arm_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                              for name in RobotConfig.ARM_JOINT_NAMES]
        self.arm_actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                                 for name in RobotConfig.ARM_ACTUATOR_NAMES]
        
        # Build DOF indices for arm joints
        self.arm_dof_indices = []
        for joint_id in self.arm_joint_ids:
            dof_adr = self.model.jnt_dofadr[joint_id]
            dof_num = self._get_joint_dof_count(joint_id)
            self.arm_dof_indices.extend(range(dof_adr, dof_adr + dof_num))

        # Resolve end effector site ID
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, RobotConfig.EE_SITE_NAME)

        # Resolve gripper actuator ID
        self.gripper_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 
                                                      RobotConfig.GRIPPER_ACTUATOR_NAME)
        self.gripper_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                                   RobotConfig.GRIPPER_JOINT_NAME)

        # Set initial joint positions (qpos) and actuator targets (ctrl) for LEFT arm
        for i, (joint_id, actuator_id) in enumerate(zip(self.arm_joint_ids, self.arm_actuator_ids)):
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr] = RobotConfig.ARM_INIT_POSITION[i]
            self.data.ctrl[actuator_id] = RobotConfig.ARM_INIT_POSITION[i]

        # Set initial gripper position for LEFT arm
        gripper_qpos_adr = self.model.jnt_qposadr[self.gripper_joint_id]
        self.data.qpos[gripper_qpos_adr] = RobotConfig.GRIPPER_INIT_WIDTH
        self.data.ctrl[self.gripper_actuator_id] = RobotConfig.GRIPPER_INIT_WIDTH

        # Set initial joint positions for RIGHT arm (same as left arm for symmetry)
        right_arm_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                               for name in RobotConfig.RIGHT_ARM_JOINT_NAMES]
        right_arm_actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                                  for name in RobotConfig.RIGHT_ARM_ACTUATOR_NAMES]
        for i, (joint_id, actuator_id) in enumerate(zip(right_arm_joint_ids, right_arm_actuator_ids)):
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr] = RobotConfig.ARM_INIT_POSITION[i]
            self.data.ctrl[actuator_id] = RobotConfig.ARM_INIT_POSITION[i]

        # Set initial gripper position for RIGHT arm
        right_gripper_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                                    RobotConfig.RIGHT_GRIPPER_JOINT_NAME)
        right_gripper_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                                       RobotConfig.RIGHT_GRIPPER_ACTUATOR_NAME)
        right_gripper_qpos_adr = self.model.jnt_qposadr[right_gripper_joint_id]
        self.data.qpos[right_gripper_qpos_adr] = RobotConfig.GRIPPER_INIT_WIDTH
        self.data.ctrl[right_gripper_actuator_id] = RobotConfig.GRIPPER_INIT_WIDTH

        # Compute forward kinematics
        mujoco.mj_forward(self.model, self.data)

    def _get_joint_dof_count(self, joint_id: int) -> int:
        """Get the number of DOFs for a joint."""
        joint_type = self.model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return 6
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return 3
        if joint_type in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
            return 1
        raise ValueError(f"Unsupported joint type for joint_id {joint_id}")

    # ============================================================
    # Arm Joint Control Methods
    # ============================================================

    def get_arm_target_joint(self) -> np.ndarray:
        """Get current arm target joint positions [j1~j6] in radians."""
        return self._arm_target_joint

    def set_arm_target_joint(self, arm_target_joint: np.ndarray) -> None:
        """Set arm target joint positions [j1~j6] in radians."""
        self._arm_target_joint = np.array(arm_target_joint)
        self._arm_error_integral[:] = 0

    def get_arm_joint_position(self) -> np.ndarray:
        """Get current arm joint positions [j1~j6] from joint states."""
        positions = []
        for joint_id in self.arm_joint_ids:
            qpos_adr = self.model.jnt_qposadr[joint_id]
            positions.append(self.data.qpos[qpos_adr])
        return np.array(positions)

    def get_arm_joint_diff(self) -> np.ndarray:
        """Get arm position error [delta_j1~delta_j6] between target and current position."""
        return self._arm_target_joint - self.get_arm_joint_position()

    def get_arm_joint_velocity(self) -> np.ndarray:
        """Get current arm joint velocities [v1~v6] from joint velocities."""
        velocities = []
        for joint_id in self.arm_joint_ids:
            dof_adr = self.model.jnt_dofadr[joint_id]
            velocities.append(self.data.qvel[dof_adr])
        return np.array(velocities)

    def _compute_arm_control(self) -> np.ndarray:
        """Compute PID position control commands [j1~j6] for arm to reach target."""
        # 1. Smoothly update _current_target_joint towards _arm_target_joint
        diff = self._arm_target_joint - self._current_target_joint
        dist = np.linalg.norm(diff)
        
        if dist > 1e-5:
            # Calculate max step based on max speed
            max_step = RobotConfig.MAX_ARM_SPEED * self.dt
            if dist < max_step:
                self._current_target_joint = self._arm_target_joint.copy()
            else:
                self._current_target_joint += (diff / dist) * max_step
        
        # 2. Compute PID control based on _current_target_joint
        current_pos = self.get_arm_joint_position()
        current_vel = self.get_arm_joint_velocity()

        # Error is calculated against the SMOOTHED target
        pos_error = self._current_target_joint - current_pos

        # Update integral term with anti-windup
        self._arm_error_integral += pos_error * self.dt
        self._arm_error_integral = np.clip(
            self._arm_error_integral,
            -RobotConfig.ARM_I_LIMIT,
            RobotConfig.ARM_I_LIMIT
        )

        p_term = RobotConfig.ARM_KP * pos_error
        i_term = RobotConfig.ARM_KI * self._arm_error_integral
        d_term = RobotConfig.ARM_KD * current_vel

        return current_pos + p_term + i_term - d_term

    # ============================================================
    # End Effector Control Methods
    # ============================================================

    @staticmethod
    def _rotation_matrix_to_euler_xyz(rot: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to XYZ Euler angles [roll, pitch, yaw]."""
        return R.from_matrix(rot.reshape(3, 3)).as_euler("xyz")

    def get_ee_position(self, data: Optional[mujoco.MjData] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return current end effector position and orientation in world frame."""
        if data is None:
            data = self.data

        ee_pos = data.site_xpos[self.ee_site_id].copy()
        ee_rot = data.site_xmat[self.ee_site_id]
        ee_ori = self._rotation_matrix_to_euler_xyz(ee_rot)
        return ee_pos, ee_ori

    def _compute_ee_jacobian(self, data: Optional[mujoco.MjData] = None) -> np.ndarray:
        """Compute 6x6 Jacobian for the end effector site (arm joints only)."""
        if data is None:
            data = self.data

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.ee_site_id)

        jacp_arm = jacp[:, self.arm_dof_indices]
        jacr_arm = jacr[:, self.arm_dof_indices]
        return np.vstack([jacp_arm, jacr_arm])

    def _solve_ik_position(self, target_pos: np.ndarray, max_iterations: Optional[int] = None) -> Tuple[bool, np.ndarray]:
        """Solve IK for a target position (orientation is kept constant)."""
        if max_iterations is None:
            max_iterations = RobotConfig.IK_MAX_ITERATIONS

        q = self.get_arm_joint_position().copy()

        ik_data = mujoco.MjData(self.model)
        ik_data.qpos[:] = self.data.qpos[:]

        for _ in range(max_iterations):
            for i, joint_id in enumerate(self.arm_joint_ids):
                qpos_adr = self.model.jnt_qposadr[joint_id]
                ik_data.qpos[qpos_adr] = q[i]
            mujoco.mj_forward(self.model, ik_data)

            current_pos = ik_data.site_xpos[self.ee_site_id].copy()
            pos_error = target_pos - current_pos

            if np.linalg.norm(pos_error) < RobotConfig.IK_POSITION_TOLERANCE:
                return True, q

            jacobian = self._compute_ee_jacobian(ik_data)[:3, :]
            jjt = jacobian @ jacobian.T
            damping = (RobotConfig.IK_DAMPING ** 2) * np.eye(jacobian.shape[0])
            inv_term = np.linalg.inv(jjt + damping)
            dq = jacobian.T @ (inv_term @ pos_error)
            q += RobotConfig.IK_STEP_SIZE * dq
            q = np.clip(q, RobotConfig.ARM_JOINT_LIMITS[:, 0], RobotConfig.ARM_JOINT_LIMITS[:, 1])

        return False, q

    def set_ee_target_position(self, target_pos: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Set end effector target position in world frame."""
        success, joint_angles = self._solve_ik_position(target_pos)
        if success:
            self.set_arm_target_joint(joint_angles)
        return success, joint_angles

    # ============================================================
    # Gripper Control Methods
    # ============================================================

    def get_gripper_width(self) -> float:
        """Get current gripper width in meters."""
        gripper_qpos_adr = self.model.jnt_qposadr[self.gripper_joint_id]
        return 2.0 * self.data.qpos[gripper_qpos_adr]

    def set_target_gripper_width(self, width: float) -> None:
        """Set target gripper width in meters (0.004 = closed, 0.074 = fully open)."""
        # Control range is 0.002 to 0.037 for single finger
        self._gripper_target_width = np.clip(width / 2.0, 0.002, 0.037)

    def get_gripper_width_diff(self) -> float:
        """Get gripper width error between target and current position."""
        return self._gripper_target_width * 2.0 - self.get_gripper_width()

    def _compute_gripper_control(self) -> float:
        """Compute gripper control command."""
        return self._gripper_target_width

    # ============================================================
    # Pick & Place Methods
    # ============================================================

    def _wait_for_arm_convergence(self, timeout: float = 10.0) -> bool:
        """Wait for arm to converge to target position."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            pos_error = np.linalg.norm(self.get_arm_joint_diff())
            vel_error = np.linalg.norm(self.get_arm_joint_velocity())
            if pos_error < 0.1 and vel_error < 0.1:
                return True
            time.sleep(0.02)
        return False

    def pick_object(
        self,
        object_pos: np.ndarray,
        approach_height: float = 0.1,
        lift_height: float = 0.2,
        return_to_home: bool = True,
        timeout: float = 10.0,
        verbose: bool = False
    ) -> bool:
        """Pick up an object at the specified position."""
        if verbose:
            print(f"Starting pick sequence at position [{object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f}]")

        # Step 1: Open gripper
        if verbose:
            print("  Step 1: Opening gripper...")
        self.set_target_gripper_width(0.074)
        time.sleep(1.0)

        # Step 2: Move to approach position (above object)
        approach_pos = np.array([object_pos[0], object_pos[1], object_pos[2] + approach_height])
        if verbose:
            print(f"  Step 2: Moving to approach position (height: {approach_height:.3f}m above object)...")
        success, _ = self.set_ee_target_position(approach_pos)
        if not success:
            if verbose:
                print("  Failed to reach approach position")
            return False

        if not self._wait_for_arm_convergence(timeout):
            if verbose:
                print("  Timeout waiting for approach position")
            return False

        # Step 3: Lower to grasp position
        grasp_pos = np.array([object_pos[0], object_pos[1], object_pos[2]])
        if verbose:
            print(f"  Step 3: Lowering to grasp position...")
        success, _ = self.set_ee_target_position(grasp_pos)
        if not success:
            if verbose:
                print("  Failed to reach grasp position")
            return False

        if not self._wait_for_arm_convergence(timeout):
            if verbose:
                print("  Timeout waiting for grasp position")
            return False

        # Step 4: Close gripper to grasp
        if verbose:
            print("  Step 4: Closing gripper to grasp...")
        self.set_target_gripper_width(0.01)
        time.sleep(1.5)

        # Step 5: Lift object
        lift_pos = np.array([object_pos[0], object_pos[1], object_pos[2] + lift_height])
        if verbose:
            print(f"  Step 5: Lifting object (height: {lift_height:.3f}m above original position)...")
        success, _ = self.set_ee_target_position(lift_pos)
        if not success:
            if verbose:
                print("  Failed to lift object")
            return False

        if not self._wait_for_arm_convergence(timeout):
            if verbose:
                print("  Timeout waiting for lift position")
            return False

        # Step 6: Return to home position (optional)
        if return_to_home:
            if verbose:
                print("  Step 6: Returning arm to home position...")
            self.set_arm_target_joint(RobotConfig.ARM_INIT_POSITION)

            if not self._wait_for_arm_convergence(timeout):
                if verbose:
                    print("  Timeout waiting for home position")
                return False

        if verbose:
            print("  Pick sequence completed successfully!")
        return True

    def place_object(
        self,
        place_pos: np.ndarray,
        approach_height: float = 0.2,
        retract_height: float = 0.3,
        return_to_home: bool = True,
        timeout: float = 10.0,
        verbose: bool = False
    ) -> bool:
        """Place an object at the specified position."""
        if verbose:
            print(f"Starting place sequence at position [{place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f}]")

        # Step 1: Move to approach position (above placement location)
        approach_pos = np.array([place_pos[0], place_pos[1], place_pos[2] + approach_height])
        if verbose:
            print(f"  Step 1: Moving to approach position (height: {approach_height:.3f}m above target)...")
        success, _ = self.set_ee_target_position(approach_pos)
        if not success:
            if verbose:
                print("  Failed to reach approach position")
            return False

        if not self._wait_for_arm_convergence(timeout):
            if verbose:
                print("  Timeout waiting for approach position")
            return False

        # Step 2: Open gripper to release
        if verbose:
            print("  Step 2: Opening gripper to release object...")
        self.set_target_gripper_width(0.074)
        time.sleep(1.5)

        # Step 3: Retract upward
        retract_pos = np.array([place_pos[0], place_pos[1], place_pos[2] + retract_height])
        if verbose:
            print(f"  Step 3: Retracting (height: {retract_height:.3f}m above placement)...")
        success, _ = self.set_ee_target_position(retract_pos)
        if not success:
            if verbose:
                print("  Failed to retract")
            return False

        if not self._wait_for_arm_convergence(timeout):
            if verbose:
                print("  Timeout waiting for retract position")
            return False

        # Step 4: Return to home position (optional)
        if return_to_home:
            if verbose:
                print("  Step 4: Returning arm to home position...")
            self.set_arm_target_joint(RobotConfig.ARM_INIT_POSITION)

            if not self._wait_for_arm_convergence(timeout):
                if verbose:
                    print("  Timeout waiting for home position")
                return False

        if verbose:
            print("  Place sequence completed successfully!")
        return True

    # ============================================================
    # Object Interaction Methods
    # ============================================================

    @staticmethod
    def _rotation_matrix_to_euler_xyz(rot: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to XYZ Euler angles [roll, pitch, yaw]."""
        return R.from_matrix(rot.reshape(3, 3)).as_euler("xyz")

    def get_object_positions(self) -> dict:
        """Get list of object dictionaries with id, name, position and orientation in world frame.
        
        Returns:
            dict: Dictionary with object names as keys, each containing:
                - 'id': body id in MuJoCo model
                - 'pos': [x, y, z] position in meters
                - 'ori': [roll, pitch, yaw] orientation in radians
        """
        objects = {}
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith("object_"):
                objects[name] = {
                    'id': i,
                    'pos': self.data.xpos[i].copy(),
                    'ori': self._rotation_matrix_to_euler_xyz(self.data.xmat[i])
                }
        return objects

    # ============================================================
    # Simulation Loop
    # ============================================================

    def run(self) -> None:
        """Run simulation with 3D viewer and control loop (blocking)."""
        with mujoco.viewer.launch_passive(self.model, self.data) as v:
            # Camera setup
            v.cam.lookat[:] = RobotConfig.CAM_LOOKAT
            v.cam.distance = RobotConfig.CAM_DISTANCE
            v.cam.azimuth = RobotConfig.CAM_AZIMUTH
            v.cam.elevation = RobotConfig.CAM_ELEVATION

            # Hide debug visuals
            v.opt.geomgroup[0] = 0
            v.opt.sitegroup[0] = v.opt.sitegroup[1] = v.opt.sitegroup[2] = 0
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = False
            v.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
            v.opt.label = mujoco.mjtLabel.mjLABEL_NONE

            # Main loop
            while v.is_running():
                # Arm control
                arm_control = self._compute_arm_control()
                for i, actuator_id in enumerate(self.arm_actuator_ids):
                    self.data.ctrl[actuator_id] = arm_control[i]

                # Gripper control
                gripper_control = self._compute_gripper_control()
                self.data.ctrl[self.gripper_actuator_id] = gripper_control

                mujoco.mj_step(self.model, self.data)
                v.sync()
