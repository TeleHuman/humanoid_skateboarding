from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  wrap_to_pi,
  quat_apply_inverse,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

@dataclass(kw_only=True)
class SkateUniformVelocityCommandCfg(CommandTermCfg):
  heading_command: bool = False
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: G1SkaterManagerBasedRlEnv) -> SkateUniformVelocityCommand:
      return SkateUniformVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
  
class SkateUniformVelocityCommand(CommandTerm):
    cfg: SkateUniformVelocityCommandCfg

    def __init__(self, cfg: SkateUniformVelocityCommandCfg, env: G1SkaterManagerBasedRlEnv):
        super().__init__(cfg, env)
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
          raise ValueError("heading_command=True but ranges.heading is set to None.")
        if self.cfg.ranges.heading and not self.cfg.heading_command:
          raise ValueError("ranges.heading is set but heading_command=False.")

        self.robot = env.robot
        self.skateboard = env.skateboard
        self.env = env

        self.command_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.heading_ref_b = torch.zeros(self.num_envs, device=self.device)
        self.heading_error = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(
          self.num_envs, dtype=torch.bool, device=self.device
        )
        self.is_standing_env = torch.zeros_like(self.is_heading_env)

        self.metrics["error_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.command_b

    @property
    def target_heading_w(self) -> torch.Tensor:
        return wrap_to_pi(self.heading_ref_b + self.command_b[:, 1])

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        vel_yaw = quat_apply_inverse(
        yaw_quat(self.skateboard.data.root_link_quat_w), 
        self.skateboard.data.root_link_lin_vel_w)
        self.metrics["error_vel"] += (
            torch.abs(self.command_b[:, 0] - vel_yaw[:, 0]) / max_command_step
        ) * self.env.contact_phase[:, 0]
        self.metrics["error_yaw"] += (
            torch.abs(wrap_to_pi(self.target_heading_w - self.skateboard.data.heading_w)) / max_command_step
        ) * self.env.contact_phase[:, 1]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
      r = torch.empty(len(env_ids), device=self.device)
      self.command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
      if self.cfg.heading_command:
        assert self.cfg.ranges.heading is not None
        self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        heading_resample_ids = env_ids[self.is_heading_env[env_ids]]
        if len(heading_resample_ids) > 0:
          self._resample_heading_command(heading_resample_ids)
      self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        if self.cfg.heading_command:
          env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
          if len(env_ids) > 0:
            resample_mask = self.env.just_exited_push2steer
            resample_env_ids = env_ids[resample_mask]
            if len(resample_env_ids) > 0:
              self._resample_heading_command(resample_env_ids)
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.command_b[standing_env_ids] = 0.0
    
    def _resample_heading_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        if not self.cfg.heading_command or self.cfg.ranges.heading is None:
            return
        r = torch.empty(len(env_ids), device=self.device)
        current_heading = self.skateboard.data.heading_w[env_ids]
        relative_heading = r.uniform_(*self.cfg.ranges.heading)
        self.command_b[env_ids, 1] = relative_heading
        self.heading_ref_b[env_ids] = current_heading
    
    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        super()._debug_vis_impl(visualizer)

        batch = visualizer.env_idx
        if batch >= self.num_envs:
            return
        if not self.cfg.heading_command or not self.is_heading_env[batch]:
            return

        skate_pos = self.skateboard.data.root_link_pos_w[batch].cpu().numpy()
        yaw = self.target_heading_w[batch].item()
        heading_goal_distance = 5.0
        direction = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        goal_pos = skate_pos + heading_goal_distance * direction
        if self.env.contact_phase[batch, 1] > 0.5:
          visualizer.add_sphere(
              center=goal_pos,
              radius=0.1,
              color=(1.0, 0.0, 0.0, 1.0),
              label="heading_target",
          )