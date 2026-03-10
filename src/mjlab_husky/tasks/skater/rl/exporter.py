import os

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.utils.lab_api.rl.exporter import _OnnxPolicyExporter


def export_skater_policy_as_onnx(
  actor_critic: object,
  path: str,
  normalizer: object | None = None,
  filename="policy.onnx",
  verbose=False,
):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  policy_exporter = _OnnxPolicyExporter(actor_critic, normalizer, verbose)
  policy_exporter.export(path, filename)


def attach_onnx_metadata(
  env: ManagerBasedRlEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
  """Attach skater-specific metadata to ONNX model.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.
    path: Directory containing the ONNX file.
    filename: Name of the ONNX file.
  """
  onnx_path = os.path.join(path, filename)
  metadata = get_base_metadata(env, run_path)  # Skater has no extra metadata.
  attach_metadata_to_onnx(onnx_path, metadata)
