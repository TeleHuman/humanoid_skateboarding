from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from mjlab.utils.lab_api.math import (
  wrap_to_pi,
  quat_mul,
  quat_apply_inverse,
  quat_error_magnitude,
  euler_xyz_from_quat
)
if TYPE_CHECKING:
  from mjlab_husky.envs import G1SkaterManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


#### push phase rewards ####
def push_skateboard_lin_vel(
  env: G1SkaterManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  x_error = torch.square(command[:, 0] - actual[:, 0])
  y_error = torch.square(actual[:, 1])
  z_error = torch.square(actual[:, 2])
  lin_vel_error = x_error + y_error + z_error
  return torch.exp(-lin_vel_error / std**2)

def push_yaw_align(env: G1SkaterManagerBasedRlEnv, std: float) -> torch.Tensor:
  _, _, yaw = euler_xyz_from_quat(env.robot.data.root_link_quat_w)
  _, _, skateboard_yaw = euler_xyz_from_quat(env.skateboard.data.root_link_quat_w.squeeze(1))
  yaw_diff = torch.abs(yaw - skateboard_yaw)
  return torch.exp(-yaw_diff / std**2)

def feet_air_time(
  env: G1SkaterManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float,
  threshold_max: float,
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  push_envs = env.contact_phase[:,0] == 1.
  mean_air_time = torch.sum(current_air_time[push_envs] * in_air[push_envs].float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  command = env.command_manager.get_command(command_name)

  assert command is not None
  scale = (command[:, 0] > command_threshold).float()
  reward *= scale
  return reward

def push_contact_ground_parallel(env: G1SkaterManagerBasedRlEnv) -> torch.Tensor:
  left_ankle_pos = env.robot.data.site_pos_w[:, env.left_foot_site_ids, 2].clone() * 10
  var = left_ankle_pos.var(1)
  var = torch.mean(var.view(-1, 1), dim=-1)
  reward = var < 0.05
  left_feet_ground_contact = env._get_feet_contact_g()[:, 0]
  reward = reward * left_feet_ground_contact.float()
  return reward

#### steer phase rewards ####
def steer_contact_num(env: G1SkaterManagerBasedRlEnv) -> torch.Tensor:
  feet_contact_b = env._get_feet_contact_b()
  both_contact = torch.sum(feet_contact_b, dim=-1) == 2
  feet_contact_g = env._get_feet_contact_g()
  left_ground_contact = feet_contact_g[:,0]
  return 2* both_contact.float() - left_ground_contact.float()

def steer_joint_pos(env: G1SkaterManagerBasedRlEnv, std: float) -> torch.Tensor:
  dof_error = torch.mean(torch.square(env.robot.data.joint_pos - env.steer_init_pos), dim=1)
  return torch.exp(-dof_error / std**2)

def steer_feet_dis(env: G1SkaterManagerBasedRlEnv, std: float) -> torch.Tensor:
  dis = env._get_feet_marker_dis()
  skateb_contact_dis = torch.norm(dis, dim=-1).mean(dim=-1)
  reward = torch.exp(-skateb_contact_dis / std**2) 
  return reward

def steer_track_heading(env: G1SkaterManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  target_w = env.get_heading_target_w(command_name)
  heading_w = env.skateboard.data.heading_w
  if target_w is not None:
    error = wrap_to_pi(heading_w - target_w)
  else:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    error = wrap_to_pi(heading_w - command[:, 1])
  r = torch.exp(-torch.abs(error) / (std**2))
  in_steer = env.contact_phase[:, 1] == 1.0
  reward = torch.where(in_steer, r, torch.zeros_like(r))
  return reward

def steer_tilt_guide(env: G1SkaterManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  gamma = env.skateboard.data.joint_pos[:, 0]
  target_w = env.get_heading_target_w(command_name)
  heading_w = env.skateboard.data.heading_w
  if target_w is not None:
    delta_theta = wrap_to_pi(target_w - heading_w)
  else:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    delta_theta = wrap_to_pi(command[:, 1] - heading_w)
  vx = env.skateboard.data.root_link_lin_vel_b[:, 0]
  remaining_steps = env._steer_remaining_steps()
  delta_t = (remaining_steps * env.step_dt).clamp(min=0.5)
  lam = torch.deg2rad(torch.tensor(env.cfg.rake_angle, device=env.device, dtype=torch.float32))
  tan_sigma = (0.4 * delta_theta) / (vx * delta_t + 1e-6)
  sin_gamma = torch.clamp(tan_sigma / torch.tan(lam), -0.99, 0.99)
  gamma_ref = torch.clip(torch.asin(sin_gamma), -0.2, 0.2)
  steer_envs = env.contact_phase[:, 1] == 1.0
  diff_gamma = torch.abs(gamma - gamma_ref)
  reward = torch.exp(-diff_gamma / std**2)
  reward = torch.where(steer_envs, reward, torch.zeros_like(reward))
  return reward

#### transition rewards ####
def transition_body_pos_tracking(env: G1SkaterManagerBasedRlEnv, std: float) -> torch.Tensor:
  target_pos_b, _, in_transition = env._get_transition_target_b()
  
  body_pos_w = env.robot.data.body_link_pos_w[:, :, :3]
  root_pos_w = env.skateboard.data.root_link_pos_w[:, :3][:, None, :].repeat(1, env.robot.num_bodies, 1)
  root_quat_w = env.skateboard.data.root_link_quat_w[:, None, :].repeat(1, env.robot.num_bodies, 1)
  
  rel_pos_w = body_pos_w - root_pos_w
  current_body_pos_b = quat_apply_inverse(root_quat_w, rel_pos_w)
  
  pos_error = (current_body_pos_b - target_pos_b)[:, env.beizer_ids, :]
  pos_error_norm = torch.sum(torch.square(pos_error),dim=-1)
  reward = torch.exp(- pos_error_norm.mean(dim=-1) / std**2)
  reward = torch.where(in_transition, reward, torch.zeros_like(reward))
  return reward

def transition_body_rot_tracking(env: G1SkaterManagerBasedRlEnv, std: float) -> torch.Tensor:
  _, target_quat_b, in_transition = env._get_transition_target_b()
  
  body_quat_w = env.robot.data.body_link_quat_w[:, :, :4]
  root_quat_w = env.skateboard.data.root_link_quat_w[:, None, :].repeat(1, env.robot.num_bodies, 1)
  target_quat_w = quat_mul(root_quat_w, target_quat_b)

  quat_error = torch.square(quat_error_magnitude(target_quat_w[:, env.slerp_ids, :],body_quat_w[:, env.slerp_ids, :]))
  reward = torch.exp(- quat_error.mean(dim=-1) / std**2)
  reward = torch.where(in_transition, reward, torch.zeros_like(reward))
  return reward

def transition_penalty_contact(env: G1SkaterManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)

#### regularization rewards ####
def reg_wheel_contact_number(env: G1SkaterManagerBasedRlEnv) -> torch.Tensor:
  wheel_contact_number = torch.sum(env.wheel_contact_filt, dim=1)
  reward = wheel_contact_number == 4
  return reward

def self_collision_cost(env: G1SkaterManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Cost that returns the number of self-collisions detected by a sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)

def stand_still(env: G1SkaterManagerBasedRlEnv, std: float) -> torch.Tensor:
  still_envs = env.still.clone()
  dof_error = torch.mean(torch.square(env.robot.data.joint_pos - env.robot.data.default_joint_pos),dim=1)
  reward = torch.exp(-dof_error / std**2)
  return reward * still_envs.float()

def board_flat(env: G1SkaterManagerBasedRlEnv, std: float) -> torch.Tensor:
  gamma = env.skateboard.data.joint_pos[:, 0]
  non_steer = env.contact_phase[:, 1] != 1.0
  diff_gamma = torch.abs(gamma)
  reward = torch.exp(-diff_gamma / std**2)
  reward = torch.where(non_steer, reward, torch.zeros_like(reward))
  return reward
  