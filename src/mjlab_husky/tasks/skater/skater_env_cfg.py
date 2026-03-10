import math
from mjlab_husky.envs import G1SkaterManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab_husky.tasks.skater import mdp
from mjlab_husky.tasks.skater.mdp import SkateUniformVelocityCommandCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_g1_skater_env_cfg() -> G1SkaterManagerBasedRlEnvCfg:
  ##
  # Observations
  ##

  policy_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "skate"},
      scale=(2.0, 1.0),
    ),
    "heading": ObservationTermCfg(
      func=mdp.heading,
      scale=1.0 / math.pi,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      scale=0.25,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      scale=0.05,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "phase": ObservationTermCfg(func=mdp.phase),
  }

  critic_terms = {
    **policy_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "heading_error": ObservationTermCfg(
      func=mdp.heading_error,
      params={"command_name": "skate"},
      scale=1.0 / math.pi,
    ),
    "skate_pose_local": ObservationTermCfg(func=mdp.skate_pose_local),
    "skate_vel_local": ObservationTermCfg(func=mdp.skate_vel_local),
    "skate_ang_vel_local": ObservationTermCfg(func=mdp.skate_ang_vel_local),
    "skateboard_roll": ObservationTermCfg(func=mdp.skateboard_roll),
    "trans_target_pos_b": ObservationTermCfg(func=mdp.trans_target_pos_b),
    "trans_target_quat_b": ObservationTermCfg(func=mdp.trans_target_quat_b),
    "l_foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "left_feet_ground_contact"},
    ),
    "r_foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "right_feet_ground_contact"},
    ),
    "l_foot_contact_forces_b": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "left_feet_board_contact"},
    ),
    "r_foot_contact_forces_b": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "right_feet_board_contact"},
    ),
    "contact_phase": ObservationTermCfg(func=mdp.contact_phase),
  }

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=5, 
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }


  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "skate": SkateUniformVelocityCommandCfg(
      resampling_time_range=(20.0, 20.0),
      rel_standing_envs=0.0,
      rel_heading_envs=1.0,
      heading_command=True,
      debug_vis=True,
      ranges=SkateUniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 1.5),
        heading=(-math.pi/4, math.pi/4),
      ),
    )
  }

  ##
  # Events
  ##

  events = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 10.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.01, 0.01),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link")),
        "operation": "add",
        "field": "body_ipos",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
    "skate_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("skateboard", body_names=("skateboard_deck")),
        "operation": "add",
        "field": "body_ipos",
        "ranges": {
          0: (-0.02, 0.02),
          1: (-0.02, 0.02),
          2: (-0.01, 0.01),
        },
      },
    ),
    "robot_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)), 
        "operation": "scale",
        "field": "geom_friction",
        "ranges": (0.3, 1.6),
      },
    ),
    "board_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("skateboard", geom_names=(".*_deck_collision",)), 
        "operation": "scale",
        "field": "geom_friction",
        "ranges": (0.8, 2.0),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(r"^(left|right)_foot[1-7]_collision$")),  # Set per-robot.
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.3, 1.8),
      },
    ),
    "wheel_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("skateboard", geom_names=(".*_wheel_collision",)), 
        "operation": "scale",
        "field": "geom_friction",
        "axes":[2],
        "ranges": (0.8, 1.6),
      },
    ),
  }

  ##
  # Rewards
  ##
  ### push phase rewards
  push_rewards = {
    "push_skateboard_lin_vel": RewardTermCfg(
      func=mdp.push_skateboard_lin_vel,
      weight=3.0,
      params={"asset_cfg": SceneEntityCfg("skateboard"),"command_name": "skate", "std": math.sqrt(0.25)},
    ),
    "push_yaw_align": RewardTermCfg(
      func=mdp.push_yaw_align,
      weight=1.0,
      params={"std": math.sqrt(0.25)},
    ),
    "push_air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=3.0,
      params={
        "sensor_name": "left_feet_ground_contact",
        "threshold_min": 0.1,
        "threshold_max": 0.5,
        "command_name": "skate",
        "command_threshold": 0.1,
      },
    ),
    "push_contact_ground_parallel": RewardTermCfg(func=mdp.push_contact_ground_parallel, weight=0.5),
  }
  ### steer phase rewards
  steer_rewards = {
    "steer_contact_num": RewardTermCfg(
      func=mdp.steer_contact_num,
      weight=3.0,
    ),
    "steer_joint_pos": RewardTermCfg(
      func=mdp.steer_joint_pos,
      weight=1.5,
      params={"std": math.sqrt(0.2)},
    ),
    "steer_feet_dis": RewardTermCfg(
      func=mdp.steer_feet_dis,
      weight=1.0,
      params={"std": math.sqrt(0.1)},
    ),
    "steer_track_heading": RewardTermCfg(
      func=mdp.steer_track_heading,
      weight=5.0,
      params={"command_name": "skate", "std": math.sqrt(0.02)},
    ),
    "steer_tilt_guide": RewardTermCfg(
      func=mdp.steer_tilt_guide,
      weight=4.0,
      params={"command_name": "skate", "std": math.sqrt(0.02)},
    ),
  }
  ### transition rewards
  transition_rewards = {
    "transition_body_pos_tracking": RewardTermCfg(
      func=mdp.transition_body_pos_tracking, 
      params={"std": math.sqrt(0.05)},
      weight=10.0,
    ),
    "transition_body_rot_tracking" : RewardTermCfg(
      func=mdp.transition_body_rot_tracking, 
      params={"std": math.sqrt(0.10)}, 
      weight=10.0,
    ),
    "transition_penalty_contact": RewardTermCfg(
      func=mdp.transition_penalty_contact, 
      params={"sensor_name": "left_feet_ground_contact"}, 
      weight=-0.5,
    ),
  }
  ### regularization rewards
  regularization_rewards = {
    "reg_wheel_contact_number": RewardTermCfg(func=mdp.reg_wheel_contact_number, weight=0.5),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.1),
    "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-3),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1e-6),
    "self_collisions": RewardTermCfg(func=mdp.self_collision_cost, weight=-10.0, params={"sensor_name": "robot_collision"}),
    "board_flat": RewardTermCfg(func=mdp.board_flat, weight=3.0, params={"std": math.sqrt(0.05)}),
    "stand_still": RewardTermCfg(func=mdp.stand_still, weight=1.0, params={"std": math.sqrt(0.1)}),
  }
  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}),
    "feet_off_board": TerminationTermCfg(func=mdp.bad_feet_off_board),
    "illegal_contact": TerminationTermCfg(func=mdp.illegal_contact, params={"sensor_name": "illegal_contact"}),
  }

  ##
  # Curriculum
  ##

  curriculum = {
  }

  ##
  # Assemble and return
  ##

  return G1SkaterManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(
        terrain_type="plane",
        terrain_generator=None,
      ),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    terminations=terminations,
    curriculum=curriculum,
    push_rewards=push_rewards,
    steer_rewards=steer_rewards,
    transition_rewards=transition_rewards,
    regularization_rewards=regularization_rewards,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=4.0,
      elevation=-10.0,
      azimuth=210.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
