from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab_husky.envs import G1SkaterManagerBasedRlEnv


def illegal_contact(env: G1SkaterManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return torch.any(sensor.data.found, dim=-1)


def bad_feet_off_board(env: G1SkaterManagerBasedRlEnv) -> torch.Tensor:
    feet_contact_b = env._get_feet_contact_b()
    bad_contact = torch.sum(feet_contact_b, dim=-1) == 0
    return bad_contact
