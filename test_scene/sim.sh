SCRIPT_DIR=$(dirname $(realpath $0))

if [ -z "$1" ]; then
  echo "Usage: $0 path_to_policy.onnx"
  exit 1
fi

ckpt_path=$1

uv run python test_scene/sim.py \
    --xml ${SCRIPT_DIR}/mjlab_scene.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --policy_frequency 50 
