from mjlab_husky.tasks.registry import register_mjlab_task
from mjlab_husky.tasks.skater.rl import SkaterOnPolicyRunner

from .env_cfgs import (
  unitree_g1_skater_env_cfg,
)
from .rl_cfg import unitree_g1_skater_ppo_runner_cfg


register_mjlab_task(
  task_id="Mjlab-Skater-Flat-Unitree-G1",
  env_cfg=unitree_g1_skater_env_cfg(),
  play_env_cfg=unitree_g1_skater_env_cfg(play=True),
  rl_cfg=unitree_g1_skater_ppo_runner_cfg(),
  runner_cls=SkaterOnPolicyRunner,
)
