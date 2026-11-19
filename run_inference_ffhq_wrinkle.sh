#!/usr/bin/env bash
set -euo pipefail

# Directory of the submodule itself (where inference.py is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================================================================
# CONFIGURATION - Change BASE_FOLDER to the location of your artifacts located in base folder. Expected artifacts: weak_wrinkle_masks/ and masked_face_images/
# (Relative paths are resolved from SCRIPT_DIR; absolute paths also work)
# ==============================================================================
BASE_FOLDER_RAW="../../../dados/face_samples_FaceSkin_Masked"

if [[ "${BASE_FOLDER_RAW}" = /* ]]; then
  CANDIDATE_BASE_FOLDER="${BASE_FOLDER_RAW}"
else
  CANDIDATE_BASE_FOLDER="${SCRIPT_DIR}/${BASE_FOLDER_RAW}"
fi

# ==============================================================================
# CHECK IF BASE_FOLDER EXISTS AND NORMALIZE THE PATH (WITHOUT '..')
# ==============================================================================
if [[ ! -d "${CANDIDATE_BASE_FOLDER}" ]]; then
  echo "ERROR: BASE_FOLDER not found: '${CANDIDATE_BASE_FOLDER}'"
  echo "Current working directory (pwd): $(pwd)"
  echo "Script directory (SCRIPT_DIR): ${SCRIPT_DIR}"
  echo "Check the relative or absolute path configured in BASE_FOLDER_RAW before proceeding."
  exit 1
fi

BASE_FOLDER="$(cd "${CANDIDATE_BASE_FOLDER}" && pwd)"

CKPT_DIR="${BASE_FOLDER}/pretrained_ckpt"
OUTPUT_DIR="${BASE_FOLDER}/test_wrinkle_outputs"
IMAGE_PATH="${BASE_FOLDER}/masked_face_images"
TEXTURE_PATH="${BASE_FOLDER}/weak_wrinkle_masks"

if [[ ! -d "${IMAGE_PATH}" || ! -d "${TEXTURE_PATH}" ]]; then
  echo "ERROR: Expected to find '${IMAGE_PATH}' and '${TEXTURE_PATH}' inside '${BASE_FOLDER}'."
  echo "Run the preprocessing steps before starting inference (Stage 2)."
  exit 1
fi

GDRIVE_FOLDER_ID="1Z4EagkkhdIGxDjYrxM7Gkr6sfpvXUUTs"

NETWORK="UNet"  # Options: UNet or SwinUNETR
GPU_ID="0"
IMG_SIZE="1024"

FORCE_DOWNLOAD=0
for arg in "$@"; do
  case "${arg}" in
    --force-download) FORCE_DOWNLOAD=1 ;;
    *)
      echo "Warning: unknown argument ignored: ${arg}"
      ;;
  esac
done

mkdir -p "${CKPT_DIR}" "${OUTPUT_DIR}"

TMP_DOWNLOAD=""
cleanup_tmp() {
  if [[ -n "${TMP_DOWNLOAD:-}" && "${TMP_DOWNLOAD}" == /tmp/* && -d "${TMP_DOWNLOAD}" ]]; then
    rm -rf "${TMP_DOWNLOAD}"
  fi
}
trap cleanup_tmp EXIT

# ==============================================================================
# 1. DOWNLOAD AND EXTRACT ONLY IF NO .pth CHECKPOINT IS ALREADY AVAILABLE
# ==============================================================================
existing_ckpts=$(find "${CKPT_DIR}" -type f -iname "*.pth" | head -n 1 || true)

if [[ -n "${existing_ckpts}" && "${FORCE_DOWNLOAD}" -eq 0 ]]; then
  echo "--> Checkpoint already found in '${CKPT_DIR}'. Skipping download and extraction."
else
  if ! command -v gdown >/dev/null 2>&1; then
    echo "ERROR: 'gdown' not found. Install it with: uv pip install gdown (or pip install gdown)"
    echo "Alternative: download the checkpoints manually and place them in ${CKPT_DIR}/"
    exit 1
  fi

  TMP_DOWNLOAD=$(mktemp -d "/tmp/ffhq_wrinkle_drive.XXXXXX")

  echo "--> Downloading Google Drive folder (${GDRIVE_FOLDER_ID})..."
#   gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" -O "${TMP_DOWNLOAD}" --remaining-ok
  gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" -O "${TMP_DOWNLOAD}"
  echo "--> Checking for .zip files and organizing them in ${CKPT_DIR}..."
  while IFS= read -r -d '' zip_file; do
    if ! command -v unzip >/dev/null 2>&1; then
      echo "ERROR: Found '${zip_file}', but the 'unzip' command is not installed."
      exit 1
    fi
    echo "Extracting: ${zip_file}"
    unzip -q -o "${zip_file}" -d "${TMP_DOWNLOAD}/extracted"
  done < <(find "${TMP_DOWNLOAD}" -type f -iname "*.zip" -print0)

  find "${TMP_DOWNLOAD}" -type f -iname "*.pth" -exec cp -n {} "${CKPT_DIR}/" \;
  if [[ -d "${TMP_DOWNLOAD}/extracted" ]]; then
    cp -rn "${TMP_DOWNLOAD}/extracted"/* "${CKPT_DIR}/" 2>/dev/null || true
  fi
fi

# # ==============================================================================
# # 2. LOCATE THE CHECKPOINT AND RUN INFERENCE.PY INSIDE THE SUBMODULE
# # ==============================================================================
# if [[ "${NETWORK}" == "UNet" ]]; then
#   CKPT_FILE=$(find "${CKPT_DIR}" -type f -iname "*unet*.pth" | head -n 1 || true)
# elif [[ "${NETWORK}" == "SwinUNETR" ]]; then
#   CKPT_FILE=$(find "${CKPT_DIR}" -type f -iname "*swin*.pth" | head -n 1 || true)
# else
#   echo "ERROR: Invalid model architecture ('${NETWORK}'). Use 'UNet' ou 'SwinUNETR'."
#   exit 1
# fi

# if [[ -z "${CKPT_FILE}" ]]; then
#   echo "ERRO: Could not find automatically a file .pth for '${NETWORK}' inside '${CKPT_DIR}'."
#   echo "Check the files in ${CKPT_DIR} and adjust the path manually."
#   exit 1
# fi

# echo "--> Using checkpoint: ${CKPT_FILE}"
# echo "--> Running inference.py from ${SCRIPT_DIR}..."

# (
#   cd "${SCRIPT_DIR}"
#   python inference.py \
#     --image_path "${IMAGE_PATH}" \
#     --texture_path "${TEXTURE_PATH}" \
#     --network "${NETWORK}" \
#     --num_channels 4 \
#     --num_classes 2 \
#     --checkpoint "${CKPT_FILE}" \
#     --gpu_id "${GPU_ID}" \
#     --img_size "${IMG_SIZE}" \
#     --output_dir "${OUTPUT_DIR}"
# )


# ==============================================================================
# 2. LOCATE THE STAGE 2 CHECKPOINT AND RUN INFERENCE.PY
# ==============================================================================
if [[ "${NETWORK}" == "UNet" ]]; then
  CKPT_FILE=$(find "${CKPT_DIR}/stage2_wrinkle_finetune_unet" \
    -type f -iname "*.pth" | head -n 1 || true)

elif [[ "${NETWORK}" == "SwinUNETR" ]]; then
  CKPT_FILE=$(find "${CKPT_DIR}/stage2_wrinkle_finetune_swinunetr" \
    -type f -iname "*.pth" | head -n 1 || true)

else
  echo "ERROR: Invalid NETWORK ('${NETWORK}'). Use 'UNet' or 'SwinUNETR'."
  exit 1
fi

if [[ -z "${CKPT_FILE}" ]]; then
  echo "ERROR: Stage 2 checkpoint not found for '${NETWORK}'."
  echo "Expected checkpoint directory: ${CKPT_DIR}/stage2_wrinkle_finetune_${NETWORK,,}"
  exit 1
fi

echo "--> Using Stage 2 checkpoint: ${CKPT_FILE}"
echo "--> Running inference.py from ${SCRIPT_DIR}..."

(
  cd "${SCRIPT_DIR}"
  python inference.py \
    --image_path "${IMAGE_PATH}" \
    --texture_path "${TEXTURE_PATH}" \
    --network "${NETWORK}" \
    --num_channels 4 \
    --num_classes 2 \
    --checkpoint "${CKPT_FILE}" \
    --gpu_id "${GPU_ID}" \
    --img_size "${IMG_SIZE}" \
    --output_dir "${OUTPUT_DIR}"
)


echo "--> Done! Results saved to: ${OUTPUT_DIR}"