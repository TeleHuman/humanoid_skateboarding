"""Unitree G1 skateboarding environment configurations."""

from mjlab_husky.asset_zoo.robots.skateboard.g1_skater_constants import (
  G1_23Dof_ACTION_SCALE,
  get_g1_23dof_robot_cfg,
  get_skateboard_cfg
)
from mjlab_husky.envs import G1SkaterManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab_husky.tasks.skater import mdp
from mjlab_husky.tasks.skater.mdp import SkateUniformVelocityCommandCfg
from mjlab_husky.tasks.skater.skater_env_cfg import make_g1_skater_env_cfg


def unitree_g1_skater_env_cfg(play: bool = False) -> G1SkaterManagerBasedRlEnvCfg:
  cfg = make_g1_skater_env_cfg()
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 55

  cfg.scene.entities = {"robot": get_g1_23dof_robot_cfg(), "skateboard": get_skateboard_cfg()}
  
  #########################################################
  ##### terrain #####
  #########################################################
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  #########################################################
  ##### contact sensors #####
  #########################################################
  left_feet_ground_cfg = ContactSensorCfg(
    name="left_feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  right_feet_ground_cfg = ContactSensorCfg(
    name="right_feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  left_feet_board_cfg = ContactSensorCfg(
    name="left_feet_board_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="geom", pattern="skateboard_marker_collision", entity="skateboard"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  right_feet_board_cfg = ContactSensorCfg(
    name="right_feet_board_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="geom", pattern="skateboard_deck_collision", entity="skateboard"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  robot_collision_cfg = ContactSensorCfg(
    name="robot_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  skateboard_collision_cfg = ContactSensorCfg(
    name="skateboard_collision",
    primary=ContactMatch(mode="geom", pattern=r".*_wheel_collision$", entity="skateboard"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found","force"),
    reduce="none",
    num_slots=1,
  )

  illegal_contact_cfg = ContactSensorCfg(
    name="illegal_contact",
    primary=ContactMatch(mode="geom", pattern=r".*_shin_collision|.*_linkage_brace_collision|.*_shoulder_yaw_collision|.*_elbow_yaw_collision|.*_wrist_collision|.*_hand_collision|pelvis_collision$", entity="robot"),
    # secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  cfg.scene.sensors = (robot_collision_cfg,skateboard_collision_cfg,left_feet_ground_cfg, right_feet_ground_cfg,left_feet_board_cfg, right_feet_board_cfg,illegal_contact_cfg)
  

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_23Dof_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  skate_cmd = cfg.commands["skate"]
  assert isinstance(skate_cmd, SkateUniformVelocityCommandCfg)
  skate_cmd.viz.z_offset = 1.15


  cfg.beizer_names = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    ]

  cfg.slerp_names = cfg.beizer_names
  cfg.phase_ratios = [0.0, 0.4, 0.5, 0.95, 1.0]
  cfg.steer_init_pos = [
    -0.15, 0.1, 0.05, 0.6, -0.42, 0.0, 
    -0.15, -0.1, 0.05, 0.6, -0.42, 0.0,
    0, 0, 0.1, 
    0, 0.55, -0.25, 0.55,
    0, -0.55, -0.25, 0.55
    ]

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(60.0)
    cfg.eval_mode = True
    cfg.observations["policy"].enable_corruption = False
    cfg.terminations = {
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }
    cfg.events.pop("push_robot", None)
    # cfg.commands["skate"].ranges.lin_vel_x = (1.0, 1.0)  # pyright: ignore[reportAttributeAccessIssue]
    # cfg.commands["skate"].ranges.heading = (0.7, 0.7)  # pyright: ignore[reportAttributeAccessIssue]
  return cfg