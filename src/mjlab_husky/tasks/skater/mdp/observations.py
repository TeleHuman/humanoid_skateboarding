from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab_husky.envs import G1SkaterManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_mul,
  quat_apply_inverse,
  quat_conjugate,
  wrap_to_pi,
)


def heading(env: G1SkaterManagerBasedRlEnv) -> torch.Tensor:
    return env.robot.data.heading_w.unsqueeze(-1)

def heading_error(env: G1SkaterManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    target_w = env.get_heading_target_w(command_name)
    heading_w = env.skateboard.data.heading_w
    if target_w is not None:
      error = wrap_to_pi(target_w - heading_w)
    else:
      command = env.command_manager.get_command(command_name)
      assert command is not None
      error = wrap_to_pi(command[:, 1] - heading_w)
    return error.unsqueeze(-1)

def contact_phase(env: G1SkaterManagerBasedRlEnv) -> torch.Tensor:
    return env.contact_phase.clone()

def phase(env: G1SkaterManagerBasedRlEnv) -> torch.Tensor:
    phase = env._get_phase().clone()
    return phase.unsqueeze(1)

def skate_pose_local(env: G1SkaterManagerBasedRlEnv) -> torch.Tensor:
    skate_pos = env.skateboard.data.root_link_pos_w.squeeze(1) - env.robot.data.root_link_pos_w
    skate_pos_local = quat_apply_inverse(env.robot.data.root_link_quat_w, skate_pos)

    skate_quat_local = quat_mul(quat_conjugate(env.robot.data.root_link_quat_w), env.skateboard.data.root_link_quat_w.squeeze(1))
    skate_rot_6d_local = quaternion_to_tangent_and_normal(skate_quat_local)

    skate_states = torch.cat([skate_pos_local, skate_rot_6d_local], dim=-1)
    return skate_states

def skate_vel_local(env: G1SkaterManagerBasedRlEnv) -> torch.Tensor:
    skate_vel = env.skateboard.data.root_link_lin_vel_w
    skate_vel_local = quat_apply_inverse(env.robot.data.root_link_quat_w, skate_vel)
    return skate_vel_local

def skate_ang_vel_local(env: G1SkaterManagerBasedRlEnv) -> torch.Tensor:
    skate_ang_vel = env.skateboard.data.root_link_ang_vel_w
    skate_ang_vel_local = quat_apply_inverse(env.robot.data.root_link_quat_w, skate_ang_vel)
    return skate_ang_vel_local

def skateboard_roll(env: G1SkaterManagerBasedRlEnv,) -> torch.Tensor:
    return env.skateboard.data.joint_pos[:, [0, 1, 4]].view(env.num_envs, -1)

def trans_target_pos_b(env: G1SkaterManagerBasedRlEnv,) -> torch.Tensor:
    target_pos_b, _, _ = env._get_transition_target_b()
    return target_pos_b[:, env.beizer_ids, :].view(env.num_envs, -1)

def trans_target_quat_b(env: G1SkaterManagerBasedRlEnv,) -> torch.Tensor:
    _, target_quat_b, _ = env._get_transition_target_b()
    return target_quat_b[:, env.slerp_ids, :].view(env.num_envs, -1)

def foot_contact_forces(env: G1SkaterManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.force is not None
    forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
    return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))

@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


