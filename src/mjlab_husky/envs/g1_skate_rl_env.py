from dataclasses import dataclass, field
import mujoco
import numpy as np
import torch
import warp as wp
from prettytable import PrettyTable
from mjlab.envs import types
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation
from mjlab.utils.logging import print_info
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.utils.lab_api.math import (
  subtract_frame_transforms, 
  quat_apply,
  quat_mul,
  matrix_from_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer
_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))
  
@dataclass(kw_only=True)
class G1SkaterManagerBasedRlEnvCfg(ManagerBasedRlEnvCfg):

  push_rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
  steer_rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
  transition_rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
  regularization_rewards: dict[str, RewardTermCfg] = field(default_factory=dict)

  cycle_time: float = 6.0
  phase_ratios: list[float] = field(default_factory=list)
  beizer_names: list[str] = field(default_factory=list)
  slerp_names: list[str] = field(default_factory=list)
  steer_init_pos: list[float] = field(default_factory=list)
  rake_angle: float = 60.0
  eval_mode: bool = False
  """Whether in evaluation mode. If True, will save metrics to JSON and exit after all episodes complete."""
  eval_output_dir: str | None = None
  """Directory to save eval metrics JSON files. If None, saves to current directory."""

class G1SkaterManagerBasedRlEnv(ManagerBasedRlEnv):
  """Manager-based RL environment."""

  is_vector_env = True
  metadata = {
    "render_modes": [None, "rgb_array"],
    "mujoco_version": mujoco.__version__,
    "warp_version": wp.config.version,
  }
  cfg: G1SkaterManagerBasedRlEnvCfg  # type: ignore[assignment]

  def __init__(
    self,
    cfg: G1SkaterManagerBasedRlEnvCfg,
    device: str,
    render_mode: str | None = None,
    **kwargs,
  ) -> None:
    # Initialize base environment state.
    self.cfg = cfg  # type: ignore[assignment]
    if self.cfg.seed is not None:
      self.cfg.seed = self.seed(self.cfg.seed)
    self._sim_step_counter = 0
    self.extras = {}
    self.obs_buf = {}

    # Initialize scene and simulation.
    self.scene = Scene(self.cfg.scene, device=device)
    self.sim = Simulation(
      num_envs=self.scene.num_envs,
      cfg=self.cfg.sim,
      model=self.scene.compile(),
      device=device,
    )

    self.scene.initialize(
      mj_model=self.sim.mj_model,
      model=self.sim.model,
      data=self.sim.data,
    )

    # Print environment info.
    print_info("")
    table = PrettyTable()
    table.title = "Base Environment"
    table.field_names = ["Property", "Value"]
    table.align["Property"] = "l"
    table.align["Value"] = "l"
    table.add_row(["Number of environments", self.num_envs])
    table.add_row(["Environment device", self.device])
    table.add_row(["Environment seed", self.cfg.seed])
    table.add_row(["Physics step-size", self.physics_dt])
    table.add_row(["Environment step-size", self.step_dt])
    print_info(table.get_string())
    print_info("")

    self.cycle_time = self.cfg.cycle_time
    self.robot = self.scene["robot"]
    self.skateboard = self.scene["skateboard"]
    self._init_buffers()
    
    # Initialize RL-specific state.
    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(
      cfg.scene.num_envs, device=device, dtype=torch.long
    )
    self.render_mode = render_mode
    self._offline_renderer: OffscreenRenderer | None = None
    if self.render_mode == "rgb_array":
      renderer = OffscreenRenderer(
        model=self.sim.mj_model, cfg=self.cfg.viewer, scene=self.scene
      )
      renderer.initialize()
      self._offline_renderer = renderer
    self.metadata["render_fps"] = 1.0 / self.step_dt  # type: ignore
    
    # Load all managers.
    self.load_managers()
    self.setup_manager_visualizers()
  
  def _init_buffers(self):
    self._init_ids_buffers()
    self.phase_ratios = torch.tensor(self.cfg.phase_ratios, device=self.device).repeat(self.num_envs, 1)
    self.steer_init_pos = torch.tensor(self.cfg.steer_init_pos, device=self.device).repeat(self.num_envs, 1)
    self.last_contacts = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device, requires_grad=False)
    self.last_wheel_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device, requires_grad=False)
    self.last_contacts_b = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device, requires_grad=False)
    self.last_contacts_g = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device, requires_grad=False)
    self.contact_phase = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

    self.phase_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)
    self.still = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
    
    push_init_body_pose = torch.from_numpy(np.load("dataset/ref_pose/push_start_pose_b.npy")).to(self.device).repeat(self.num_envs, 1 , 1)
    steer_init_body_pose = torch.from_numpy(np.load("dataset/ref_pose/steer_start_pose_b.npy")).to(self.device).repeat(self.num_envs, 1 , 1)
    self.push_init_body_pos_b = push_init_body_pose[..., :3]
    self.steer_init_body_pos_b = steer_init_body_pose[..., :3]
    self.push_init_body_quat_b = push_init_body_pose[..., 3:]
    self.steer_init_body_quat_b = steer_init_body_pose[..., 3:]
    self.body_bezier_buffers = {
        "push2steer_start_pos_b": torch.zeros(self.num_envs,self.robot.num_bodies, 3, device=self.device, requires_grad=False),
        "steer2push_start_pos_b": torch.zeros(self.num_envs, self.robot.num_bodies, 3, device=self.device, requires_grad=False),
        "push2steer_start_quat_b": torch.zeros(self.num_envs, self.robot.num_bodies, 4, device=self.device, requires_grad=False),
        "steer2push_start_quat_b": torch.zeros(self.num_envs, self.robot.num_bodies, 4, device=self.device, requires_grad=False),
    }

  def _init_ids_buffers(self):
    self.wheel_body_ids, _ = self.skateboard.find_bodies(name_keys=[".*_wheel"], preserve_order=True)
    self.feet_body_ids, _ = self.robot.find_bodies(name_keys=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)
    self.marker_body_ids, _ = self.skateboard.find_sites(name_keys=[".*_marker"], preserve_order=True)
    self.left_foot_site_ids, _ = self.robot.find_sites(name_keys=["left_foot_1", "left_foot_2", "left_foot_3", "left_foot_4"], preserve_order=True)
    self.right_foot_site_ids, _ = self.robot.find_sites(name_keys=["right_foot_1", "right_foot_2", "right_foot_3", "right_foot_4"], preserve_order=True)
    self.beizer_ids, _ = self.robot.find_bodies(name_keys=self.cfg.beizer_names, preserve_order=True)
    self.slerp_ids, _ = self.robot.find_bodies(name_keys=self.cfg.slerp_names, preserve_order=True)
    self.truck_roll_joint_ids, _ = self.skateboard.find_joints(name_keys=["front_truck_roll_joint", "rear_truck_roll_joint"], preserve_order=True)

  def load_managers(self) -> None:
    super().load_managers()

    self.push_reward_manager = RewardManager(
      self.cfg.push_rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
    )
    print_info(f"[INFO] {self.push_reward_manager}")
    self.steer_reward_manager = RewardManager(
      self.cfg.steer_rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
    )
    print_info(f"[INFO] {self.steer_reward_manager}")
    self.transition_reward_manager = RewardManager(
      self.cfg.transition_rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
    )
    print_info(f"[INFO] {self.transition_reward_manager}")
    self.reg_reward_manager = RewardManager(
      self.cfg.regularization_rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
    )
    print_info(f"[INFO] {self.reg_reward_manager}")

  def get_heading_target_w(self, command_name: str) -> torch.Tensor | None:
    terms = getattr(self.command_manager, "_terms", None) or getattr(self.command_manager, "terms", None)
    if terms is None:
      return None
    term = terms.get(command_name)
    if term is None:
      return None
    return getattr(term, "target_heading_w", None)

  def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    self.action_manager.process_action(action.to(self.device))
    self.still = self.command_manager.get_command("skate")[:, 0] < 0.1  # pyright: ignore[reportOptionalSubscript]
    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      # self._set_skatedboard_joint_pos()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)

    # Update env counters.
    self.episode_length_buf += 1
    self.phase_length_buf += 1
    self.common_step_counter += 1
    self._compute_contact()

    # Check terminations.
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    contact_coef = self.contact_phase.clone()
    push_reward_buf = self.push_reward_manager.compute(self.step_dt) * contact_coef[:,0]
    steer_reward_buf = self.steer_reward_manager.compute(self.step_dt) * contact_coef[:,1]
    reg_reward_buf = self.reg_reward_manager.compute(self.step_dt)
    transition_reward_buf = self.transition_reward_manager.compute(self.step_dt) * torch.logical_or(contact_coef[:, 2], contact_coef[:,3])

    self.reward_buf = steer_reward_buf + push_reward_buf + reg_reward_buf + transition_reward_buf
    self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(self.reset_env_ids) > 0:
      self._reset_idx(self.reset_env_ids)
      self.scene.write_data_to_sim()
      self.sim.forward()

    self.command_manager.compute(dt=self.step_dt)
   
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)
    
    self.obs_buf = self.observation_manager.compute(update_history=True)
    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )

  def _set_skatedboard_joint_pos(self):
    joint_pos = self.skateboard.data.joint_pos
    tilt_joint_pos = joint_pos[:, 0]
    rake_angle_rad = torch.deg2rad(torch.tensor(self.cfg.rake_angle, device=self.device, dtype=torch.float32))
    truck_pos = -torch.atan(torch.sin(tilt_joint_pos) * torch.tan(rake_angle_rad))
    truck_pos = torch.clip(truck_pos, -0.1, 0.1)
    self.skateboard.write_joint_position_to_sim(
      truck_pos.unsqueeze(-1),
      joint_ids=self.truck_roll_joint_ids
    )
    
  def update_visualizers(self, visualizer: DebugVisualizer) -> None:
    super().update_visualizers(visualizer)
    self._visualize_transition_target(visualizer)
    self._visualize_contact_phase(visualizer)
    
  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    super()._reset_idx(env_ids)

    info = self.push_reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    info = self.steer_reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    info = self.reg_reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    info = self.transition_reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    
    self.phase_length_buf[env_ids] = 0
    for buf in self.body_bezier_buffers.values():
        buf[env_ids] = 0
  
  def _compute_contact(self):
    self.skateboard_contact_sensor = self.scene.sensors["skateboard_collision"]
    self.left_feet_contact_ground = self.scene.sensors["left_feet_ground_contact"]
    self.right_feet_contact_ground = self.scene.sensors["right_feet_ground_contact"]
    l_contact = torch.norm(self.left_feet_contact_ground.data.force, dim=-1) > 2.
    r_contact = torch.norm(self.right_feet_contact_ground.data.force, dim=-1) > 2.
    contact = torch.cat([l_contact, r_contact], dim=-1).squeeze(1)

    wheel_contact = torch.logical_or(torch.norm(self.skateboard_contact_sensor.data.force, dim=-1) > 1., 
                                    torch.abs(self.skateboard.data.body_link_pos_w[:, self.wheel_body_ids, 2] - 0.03) < 0.005)
    self.contact_filt = torch.logical_or(contact, self.last_contacts) 
    self.wheel_contact_filt = torch.logical_or(wheel_contact, self.last_wheel_contacts)
    self.last_contacts = contact
    self.last_wheel_contacts = wheel_contact
    self._resample_contact_phases()

  def _resample_contact_phases(self):
    self.last_contact_phase = self.contact_phase.clone()
    phase = self._get_phase()

    push_phase = (phase >= self.phase_ratios[:, 0]) & (phase < self.phase_ratios[:, 1]) & ~self.still
    push2steer = (phase >= self.phase_ratios[:, 1]) & (phase < self.phase_ratios[:, 2]) & ~self.still
    steer_phase = (phase >= self.phase_ratios[:, 2]) & (phase < self.phase_ratios[:, 3]) & ~self.still
    steer2push = (phase >= self.phase_ratios[:, 3]) & (phase <= self.phase_ratios[:, 4]) & ~self.still
    
    self.contact_phase[:,0] = push_phase.float()
    self.contact_phase[:,1] = steer_phase.float()
    self.contact_phase[:,2] = push2steer.float()
    self.contact_phase[:,3] = steer2push.float()
    
    self.just_entered_push2steer = push2steer & (self.last_contact_phase[:, 2] < 0.5)
    self.just_entered_steer2push = steer2push & (self.last_contact_phase[:, 3] < 0.5)
    self.just_exited_push2steer = (self.last_contact_phase[:, 2] > 0.5) & ~push2steer
    self.just_exited_steer2push = (self.last_contact_phase[:, 3] > 0.5) & ~steer2push
    
    body_pos_w = self.robot.data.body_link_pos_w
    body_quat_w = self.robot.data.body_link_quat_w
    root_pos_w = self.skateboard.data.root_link_pos_w[:, None, :].repeat(1, self.robot.num_bodies, 1)
    root_quat_w = self.skateboard.data.root_link_quat_w[:, None, :].repeat(1, self.robot.num_bodies, 1)
    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    if self.just_entered_push2steer.any():
        self.body_bezier_buffers["push2steer_start_pos_b"][self.just_entered_push2steer] = body_pos_b[self.just_entered_push2steer]
        self.body_bezier_buffers["push2steer_start_quat_b"][self.just_entered_push2steer] = body_quat_b[self.just_entered_push2steer]

    if self.just_entered_steer2push.any():
        self.body_bezier_buffers["steer2push_start_pos_b"][self.just_entered_steer2push] = body_pos_b[self.just_entered_steer2push]
        self.body_bezier_buffers["steer2push_start_quat_b"][self.just_entered_steer2push] = body_quat_b[self.just_entered_steer2push]
    
  def _get_phase(self):
    self.phase_length_buf[self.still] = torch.where((self.phase_length_buf[self.still]-1) % int(self.cycle_time/ 2 / self.step_dt) == 0, 
                                                            0, 
                                                            self.phase_length_buf[self.still])
    phase = ((self.phase_length_buf * self.step_dt / self.cycle_time)) % 1.0
    phase = torch.clip(phase, 0.0, 1.0)

    return phase
  
  def _steer_remaining_steps(self):
    phase = self._get_phase()
    steer_end_phase = self.phase_ratios[:, 3]
    remaining_phase = torch.where(
      phase < steer_end_phase,
      steer_end_phase - phase,
      1.0 - phase + steer_end_phase
    )
    remaining_steps = remaining_phase *  self.cycle_time / self.step_dt
    
    return remaining_steps

  def _get_feet_contact_b(self):
    left_contact_sensor = self.scene.sensors["left_feet_board_contact"]
    right_contact_sensor = self.scene.sensors["right_feet_board_contact"]
    left_contact_b = (torch.norm(left_contact_sensor.data.force, dim=-1) > 5)
    right_contact_b = (torch.norm(right_contact_sensor.data.force, dim=-1) > 5)
    contact_b = torch.cat([left_contact_b, right_contact_b], dim=-1).view(self.num_envs, 2)
    contact_filt = torch.logical_or(contact_b, self.last_contacts_b) 
    self.last_contacts_b = contact_b
    return contact_filt

  def _get_feet_contact_g(self):
    left_contact_ground_sensor = self.scene.sensors["left_feet_ground_contact"]
    right_contact_ground_sensor = self.scene.sensors["right_feet_ground_contact"]
    left_contact_g = (torch.norm(left_contact_ground_sensor.data.force, dim=-1) > 5)
    right_contact_g = (torch.norm(right_contact_ground_sensor.data.force, dim=-1) > 5)
    contact_g = torch.cat([left_contact_g, right_contact_g], dim=-1).view(self.num_envs, 2)
    
    contact_filt = torch.logical_or(contact_g, self.last_contacts_g)
    self.last_contacts_g = contact_g 
    return contact_filt

  def _get_feet_marker_dis(self):
    feet_pos = self.robot.data.body_link_pos_w[:, self.feet_body_ids, :3]
    marker_pos = self.skateboard.data.site_pos_w[:, self.marker_body_ids, :3]
    dis = marker_pos - feet_pos
    return dis
  
  def _get_transition_target_b(self):
    phase = self._get_phase()
    push2steer = (phase > self.phase_ratios[:, 1]) & (phase < self.phase_ratios[:, 2]) & ~self.still
    steer2push = (phase > self.phase_ratios[:, 3]) & (phase < self.phase_ratios[:, 4]) & ~self.still
    in_transition = push2steer | steer2push
    
    body_pos_w = self.robot.data.body_link_pos_w
    body_quat_w = self.robot.data.body_link_quat_w
    skate_pos_w = self.skateboard.data.root_link_pos_w[:, None, :].repeat(1, self.robot.num_bodies, 1)
    skate_quat_w = self.skateboard.data.root_link_quat_w[:, None, :].repeat(1, self.robot.num_bodies, 1)
    
    current_body_pos_b,current_body_quat_b = subtract_frame_transforms(skate_pos_w, skate_quat_w, body_pos_w, body_quat_w)
    target_pos_b = current_body_pos_b.clone()
    target_quat_b = current_body_quat_b.clone()
    if in_transition.any():
        t = torch.zeros(self.num_envs, device=self.device)
        
        if push2steer.any():
            t[push2steer] = (phase[push2steer] - self.phase_ratios[push2steer, 1]) / 0.1
            t[push2steer] = torch.clamp(t[push2steer], 0.0, 1.0)
            
            start_pos_b = self.body_bezier_buffers["push2steer_start_pos_b"][push2steer]
            start_quat_b = self.body_bezier_buffers["push2steer_start_quat_b"][push2steer]
            
            end_pos_b = self.steer_init_body_pos_b[push2steer]
            end_quat_b = self.steer_init_body_quat_b[push2steer]

            target_pos_b[push2steer] = bezier_curve(start_pos_b, end_pos_b, t[push2steer], offset=0.2)
            target_quat_b[push2steer] = quaternion_slerp(start_quat_b, end_quat_b, t[push2steer])

        if steer2push.any():
            t[steer2push] = (phase[steer2push] - self.phase_ratios[steer2push, 3]) / 0.05
            t[steer2push] = torch.clamp(t[steer2push], 0.0, 1.0)
            
            start_pos_b = self.body_bezier_buffers["steer2push_start_pos_b"][steer2push]
            start_quat_b = self.body_bezier_buffers["steer2push_start_quat_b"][steer2push]                
            
            end_pos_b = self.push_init_body_pos_b[steer2push]
            end_quat_b = self.push_init_body_quat_b[steer2push]

            target_pos_b[steer2push] =  bezier_curve(start_pos_b, end_pos_b, t[steer2push],offset=0.2)
            target_quat_b[steer2push] = quaternion_slerp(start_quat_b, end_quat_b, t[steer2push])

    return target_pos_b, target_quat_b, in_transition

  def get_amp_observations(self):
    return self.robot.data.joint_pos

  def _visualize_transition_target(self, visualizer: DebugVisualizer):
    target_pos_b, target_quat_b, in_transition = self._get_transition_target_b()
    
    if in_transition.any():

      root_pos_w = self.skateboard.data.root_link_pos_w[:, :3][:,None,:].repeat(1,self.robot.num_bodies,1)
      root_quat_w = self.skateboard.data.root_link_quat_w[:,None,:].repeat(1,self.robot.num_bodies,1)
      target_pos_w = root_pos_w[in_transition] + quat_apply(
          root_quat_w[in_transition], 
          target_pos_b[in_transition]
      )
      target_quat_w = quat_mul(root_quat_w[in_transition], target_quat_b[in_transition])
      desired_body_rotm = matrix_from_quat(target_quat_w).cpu().numpy()
      for i, idx in enumerate(self.beizer_ids):
          visualizer.add_frame(
          position=target_pos_w[visualizer.env_idx,idx],
          rotation_matrix=desired_body_rotm[visualizer.env_idx,idx],
          scale=0.1,
          label=f"desired_{idx}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )
      for i, idx in enumerate(self.beizer_ids):
          visualizer.add_sphere(
          center=target_pos_w[visualizer.env_idx,idx],
          radius=0.03,
          color=(1.0, 1.0, 0.0, 1.0),
          label=f"desired_{idx}",
        )

  def _visualize_contact_phase(self, visualizer: DebugVisualizer):
    contact_phase = self.contact_phase.clone()
    push_phase = contact_phase[:, 0]
    steer_phase = contact_phase[:, 1]
    transition_phase = torch.logical_or(contact_phase[:, 2], contact_phase[:, 3])
    target_pos_w = self.robot.data.root_link_pos_w
    target_pos_w[...,2] += 0.75
    if push_phase.any():
      visualizer.add_sphere(
        center=target_pos_w[visualizer.env_idx],
        radius=0.05,
        color=(1.0, 0.0, 0.0, 1.0),
        label="push_phase",
      )
    if steer_phase.any():
      visualizer.add_sphere(
        center=target_pos_w[visualizer.env_idx],
        radius=0.05,
        color=(0.0, 1.0, 0.0, 1.0),
        label="steer_phase",
      )
    if transition_phase.any():
      visualizer.add_sphere(
        center=target_pos_w[visualizer.env_idx],
        radius=0.05,
        color=(1.0, 1.0, 0.0, 1.0),
        label="transition_phase",
      )
      

def bezier_curve(start_p, end_p, t, offset=0.15):

  # middle control point, shape (num_envs, bodys, 3)
  middle_p = (start_p + end_p) / 2.0
  middle_p[..., 2] += offset
  
  t = torch.clamp(t, 0.0, 1.0).view(-1, 1, 1)  

  result_pos = (1 - t) ** 2 * start_p + 2 * (1 - t) * t * middle_p + t ** 2 * end_p

  if result_pos.shape[1] == 1:
      result_pos = result_pos.squeeze(1)
  if result_pos.shape[0] == 1:
      result_pos = result_pos.squeeze(0)
  
  return result_pos

def quaternion_slerp(q0, q1, t, shortestpath=True):
    if t.dim() == 0:
        t = t.view(1, 1, 1).expand(q0.shape[0], q0.shape[1], 1)
    elif t.dim() == 1:  # (num_envs,)
        t = t.view(-1, 1, 1)
    elif t.dim() == 2:  # (num_envs, 1)
        t = t.unsqueeze(-1)
    
    if t.shape[1] == 1 and q0.shape[1] > 1:
        t = t.repeat(1, q0.shape[1], 1)
    
    EPS = 1e-6
    
    d = torch.sum(q0 * q1, dim=-1, keepdim=True)  # (num_envs, bodys, 1)
    
    zero_mask = torch.isclose(t, torch.zeros_like(t), atol=EPS)
    ones_mask = torch.isclose(t, torch.ones_like(t), atol=EPS)
    dist_mask = (torch.abs(torch.abs(d) - 1.0) < EPS)
    
    out = torch.zeros_like(q0)
    out[zero_mask.squeeze(-1)] = q0[zero_mask.squeeze(-1)]
    out[ones_mask.squeeze(-1)] = q1[ones_mask.squeeze(-1)]
    out[dist_mask.squeeze(-1)] = q0[dist_mask.squeeze(-1)]

    if shortestpath:
        q1 = torch.where(d < 0, -q1, q1)
        d = torch.abs(d)
    
    angle = torch.acos(torch.clamp(d, -1.0 + EPS, 1.0 - EPS))
    angle_mask = (torch.abs(angle) < EPS)
    out[angle_mask.squeeze(-1)] = q0[angle_mask.squeeze(-1)]
    
    final_mask = ~(zero_mask | ones_mask | dist_mask | angle_mask)
    final_mask = final_mask.squeeze(-1)
    
    if final_mask.any():
        sin_angle = torch.sin(angle)
        isin = 1.0 / (sin_angle + EPS)
        
        t_expanded = t.expand_as(q0)
        weight0 = torch.sin((1.0 - t_expanded) * angle) * isin
        weight1 = torch.sin(t_expanded * angle) * isin
        
        result = weight0 * q0 + weight1 * q1
        result_norm = torch.norm(result, dim=-1, keepdim=True)
        result = result / (result_norm + EPS)
        
        out[final_mask] = result[final_mask]
    
    return out