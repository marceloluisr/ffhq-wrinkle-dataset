#!/usr/bin/env bash
set -euo pipefail

# Diretório do próprio submódulo (onde fica o inference.py)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================================================================
# CONFIGURAÇÃO - Altere BASE_FOLDER para onde estão seus artefatos no PC
# (Caminhos relativos são resolvidos a partir de SCRIPT_DIR; absolutos também funcionam)
# ==============================================================================
BASE_FOLDER_RAW="../../../dados/face_samples_FaceSkin_Masked"

if [[ "${BASE_FOLDER_RAW}" = /* ]]; then
  CANDIDATE_BASE_FOLDER="${BASE_FOLDER_RAW}"
else
  CANDIDATE_BASE_FOLDER="${SCRIPT_DIR}/${BASE_FOLDER_RAW}"
fi

# ==============================================================================
# VERIFICAR SE BASE_FOLDER EXISTE E NORMALIZAR CAMINHO (SEM '..')
# ==============================================================================
if [[ ! -d "${CANDIDATE_BASE_FOLDER}" ]]; then
  echo "ERRO: BASE_FOLDER não encontrado: '${CANDIDATE_BASE_FOLDER}'"
  echo "Diretório atual (pwd): $(pwd)"
  echo "Diretório do script (SCRIPT_DIR): ${SCRIPT_DIR}"
  echo "Verifique o caminho relativo ou absoluto em BASE_FOLDER_RAW antes de prosseguir."
  exit 1
fi

BASE_FOLDER="$(cd "${CANDIDATE_BASE_FOLDER}" && pwd)"

CKPT_DIR="${BASE_FOLDER}/pretrained_ckpt"
OUTPUT_DIR="${BASE_FOLDER}/test_wrinkle_outputs"
IMAGE_PATH="${BASE_FOLDER}/masked_face_images"
TEXTURE_PATH="${BASE_FOLDER}/weak_wrinkle_masks"

if [[ ! -d "${IMAGE_PATH}" || ! -d "${TEXTURE_PATH}" ]]; then
  echo "ERRO: Esperava encontrar '${IMAGE_PATH}' e '${TEXTURE_PATH}' dentro de '${BASE_FOLDER}'."
  echo "Rode primeiro as etapas de pré-processamento antes da inferência."
  exit 1
fi

GDRIVE_FOLDER_ID="1Z4EagkkhdIGxDjYrxM7Gkr6sfpvXUUTs"

NETWORK="UNet" # Opções: UNet ou SwinUNETR
GPU_ID="0"
IMG_SIZE="1024"

FORCE_DOWNLOAD=0
for arg in "$@"; do
  case "${arg}" in
    --force-download) FORCE_DOWNLOAD=1 ;;
    *)
      echo "Aviso: argumento desconhecido ignorado: ${arg}"
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
# 1. BAIXAR E DESCOMPACTAR APENAS SE AINDA NÃO HOUVER CHECKPOINT .pth
# ==============================================================================
existing_ckpts=$(find "${CKPT_DIR}" -type f -iname "*.pth" | head -n 1 || true)

if [[ -n "${existing_ckpts}" && "${FORCE_DOWNLOAD}" -eq 0 ]]; then
  echo "--> Checkpoint já encontrado em '${CKPT_DIR}'. Pulando download e extração."
else
  if ! command -v gdown >/dev/null 2>&1; then
    echo "ERRO: 'gdown' não encontrado. Instale com: uv pip install gdown (ou pip install gdown)"
    echo "Alternativa: baixe os checkpoints manualmente e coloque em ${CKPT_DIR}/"
    exit 1
  fi

  TMP_DOWNLOAD=$(mktemp -d "/tmp/ffhq_wrinkle_drive.XXXXXX")

  echo "--> Baixando pasta do Google Drive (${GDRIVE_FOLDER_ID})..."
#   gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" -O "${TMP_DOWNLOAD}" --remaining-ok
  gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" -O "${TMP_DOWNLOAD}"
  echo "--> Verificando arquivos .zip e organizando em ${CKPT_DIR}..."
  while IFS= read -r -d '' zip_file; do
    if ! command -v unzip >/dev/null 2>&1; then
      echo "ERRO: Encontrado '${zip_file}', mas o comando 'unzip' não está instalado."
      exit 1
    fi
    echo "Descompactando: ${zip_file}"
    unzip -q -o "${zip_file}" -d "${TMP_DOWNLOAD}/extracted"
  done < <(find "${TMP_DOWNLOAD}" -type f -iname "*.zip" -print0)

  find "${TMP_DOWNLOAD}" -type f -iname "*.pth" -exec cp -n {} "${CKPT_DIR}/" \;
  if [[ -d "${TMP_DOWNLOAD}/extracted" ]]; then
    cp -rn "${TMP_DOWNLOAD}/extracted"/* "${CKPT_DIR}/" 2>/dev/null || true
  fi
fi

# ==============================================================================
# 2. LOCALIZAR CHECKPOINT E RODAR INFERENCE.PY DENTRO DO SUBMÓDULO
# ==============================================================================
if [[ "${NETWORK}" == "UNet" ]]; then
  CKPT_FILE=$(find "${CKPT_DIR}" -type f -iname "*unet*.pth" | head -n 1 || true)
elif [[ "${NETWORK}" == "SwinUNETR" ]]; then
  CKPT_FILE=$(find "${CKPT_DIR}" -type f -iname "*swin*.pth" | head -n 1 || true)
else
  echo "ERRO: NETWORK inválida ('${NETWORK}'). Use 'UNet' ou 'SwinUNETR'."
  exit 1
fi

if [[ -z "${CKPT_FILE}" ]]; then
  echo "ERRO: Não encontrei automaticamente um arquivo .pth para '${NETWORK}' dentro de '${CKPT_DIR}'."
  echo "Verifique os arquivos em ${CKPT_DIR} e ajuste o caminho manualmente."
  exit 1
fi

echo "--> Usando checkpoint: ${CKPT_FILE}"
echo "--> Executando inference.py em ${SCRIPT_DIR}..."

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

echo "--> Concluído! Resultados salvos em: ${OUTPUT_DIR}"