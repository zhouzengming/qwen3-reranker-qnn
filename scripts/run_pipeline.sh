#!/usr/bin/env bash
# 完整流程：Hugging Face 模型 -> ONNX 分段 -> AI Hub fp16 QNN context binary -> 板端部署包。
#
#   bash scripts/run_pipeline.sh          # SDK 放在仓库的 qairt/ 目录下（见 qairt/README.md）
#   QAIRT_SDK=/path/to/qairt/2.50.0.xxxxxx bash scripts/run_pipeline.sh   # 或者手动指定 SDK 路径
#
# 可通过环境变量调整（方括号内为默认值）：
#   MODEL_ID [Qwen/Qwen3-Reranker-0.6B]   SEQ_LEN [4096]   PARTS [4]   ATTN_CHUNK [512]
#   DEVICE ["QCS8550 (Proxy)"]   QNN_TARGET [aarch64-oe-linux-gcc11.2]   HTP_ARCH [v73]
#   WORK [work]   OUT [dist/deploy]   EVAL [0]（设为 1 时额外在 AI Hub 真机上做端到端精度验证）
set -euo pipefail
cd "$(dirname "$0")/.."

# QAIRT SDK：优先使用环境变量 QAIRT_SDK，否则在仓库的 qairt/ 目录下自动查找（见 qairt/README.md）
if [ -z "${QAIRT_SDK:-}" ]; then
  mapfile -t _sdks < <(find -L qairt -maxdepth 6 -path "*/include/QNN/QnnInterface.h" 2>/dev/null \
                         | sed 's#/include/QNN/QnnInterface.h$##' | sort)
  if [ "${#_sdks[@]}" -eq 0 ]; then
    echo "未找到 QAIRT SDK：请按 qairt/README.md 把 SDK 解压到 qairt/ 目录，或设置环境变量 QAIRT_SDK。" >&2
    exit 1
  elif [ "${#_sdks[@]}" -gt 1 ]; then
    echo "qairt/ 下找到多个 SDK，请用 QAIRT_SDK 指定其中一个：" >&2
    printf '  %s\n' "${_sdks[@]}" >&2
    exit 1
  fi
  QAIRT_SDK="$(cd "${_sdks[0]}" && pwd)"
fi
[ -f "$QAIRT_SDK/include/QNN/QnnInterface.h" ] || { echo "QAIRT_SDK=$QAIRT_SDK 不是有效的 SDK 根目录（缺少 include/QNN）" >&2; exit 1; }
_ver="$(grep -E '^version:' "$QAIRT_SDK/sdk.yaml" 2>/dev/null | awk '{print $2}')"
echo "== 使用 QAIRT SDK：$QAIRT_SDK（版本 ${_ver:-未知}，需与 AI Hub 编译时的版本一致）"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Reranker-0.6B}"
SEQ_LEN="${SEQ_LEN:-4096}"
PARTS="${PARTS:-4}"
ATTN_CHUNK="${ATTN_CHUNK:-512}"
DEVICE="${DEVICE:-QCS8550 (Proxy)}"
QNN_TARGET="${QNN_TARGET:-aarch64-oe-linux-gcc11.2}"
HTP_ARCH="${HTP_ARCH:-v73}"
WORK="${WORK:-work}"
OUT="${OUT:-dist/deploy}"
PY="${PYTHON:-python3}"

mkdir -p "$WORK"
if [ ! -f "$WORK/hf_model/config.json" ]; then
  echo "== [0/5] 下载模型 $MODEL_ID"
  hf download "$MODEL_ID" --local-dir "$WORK/hf_model"
fi

echo "== [1/5] 导出 ONNX 分段（seq_len=$SEQ_LEN parts=$PARTS attn_chunk=$ATTN_CHUNK）"
[ -f "$WORK/onnx/manifest.json" ] || \
  $PY export/export_onnx.py --model "$WORK/hf_model" --out "$WORK/onnx" \
      --seq-len "$SEQ_LEN" --parts "$PARTS" --attn-chunk "$ATTN_CHUNK"

echo "== [2/5] 用 PyTorch 验证 ONNX 分段精度（CPU）"
$PY export/verify_onnx.py --model "$WORK/hf_model" --onnx-dir "$WORK/onnx"

echo "== [3/5] 在 AI Hub 上为 $DEVICE 编译 fp16 QNN context binary"
$PY export/aihub_compile.py --onnx-dir "$WORK/onnx" --state "$WORK/aihub_state.json" --device "$DEVICE"

if [ "${EVAL:-0}" = "1" ]; then
  echo "== [3b] 在 AI Hub 真机上串联各段验证端到端精度"
  $PY export/aihub_eval.py --model "$WORK/hf_model" --onnx-dir "$WORK/onnx" --state "$WORK/aihub_state.json"
fi

echo "== [4/5] 下载 context binary"
$PY export/aihub_download.py --onnx-dir "$WORK/onnx" --state "$WORK/aihub_state.json" --out "$WORK/context_binaries"

echo "== [5/5] 组装板端部署包 -> $OUT"
$PY export/make_deploy.py --model "$WORK/hf_model" --onnx-dir "$WORK/onnx" --binaries "$WORK/context_binaries" \
    --qairt-sdk "$QAIRT_SDK" --qnn-target "$QNN_TARGET" --htp-arch "$HTP_ARCH" --out "$OUT" \
    $($PY -c "import ziglang" 2>/dev/null && echo --cross-compile)

echo "完成。把 $OUT 拷到开发板上，然后执行：source setup_env.sh && python3 run_reranker.py selftest"
