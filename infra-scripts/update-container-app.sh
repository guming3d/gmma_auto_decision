#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly ENV_FILE="${REPO_ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
  set +a
else
  echo "Environment file ${ENV_FILE} is missing; create it with your Azure settings." >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/Dockerfile" ]]; then
  echo "Dockerfile not found in ${REPO_ROOT}. Run the deploy script first to generate it." >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/.streamlit/secrets.toml" ]]; then
  echo ".streamlit/secrets.toml is missing; add it before building the container." >&2
  exit 1
fi

if ! command -v az >/dev/null 2>&1; then
  echo "Required command 'az' is not installed or not in PATH." >&2
  exit 1
fi

if [[ -n "${AZ_CA_BUNDLE:-}" ]]; then
  if [[ ! -f "${AZ_CA_BUNDLE}" ]]; then
    echo "AZ_CA_BUNDLE points to ${AZ_CA_BUNDLE}, but the file does not exist." >&2
    exit 1
  fi
  export REQUESTS_CA_BUNDLE="${AZ_CA_BUNDLE}"
  export CURL_CA_BUNDLE="${AZ_CA_BUNDLE}"
fi

if [[ -n "${AZURE_CLI_DISABLE_CONNECTION_VERIFICATION:-}" ]]; then
  export AZURE_CLI_DISABLE_CONNECTION_VERIFICATION
fi

usage() {
  cat <<'EOF'
Roll out a new image revision to an existing Azure Container App.
Configuration defaults are loaded from .env at the repository root.

Required environment variables:
  AZ_SUBSCRIPTION_ID             Azure subscription ID
  AZ_RESOURCE_GROUP              Resource group that owns the Container App
  AZ_CONTAINER_APP_NAME          Container App name
  AZ_ACR_NAME                    Azure Container Registry to push the image

Optional environment variables:
  AZ_IMAGE_REPOSITORY            Repository inside ACR (default: ${AZ_CONTAINER_APP_NAME})
  AZ_IMAGE_TAG                   Image tag (default: current git SHA or timestamp)
  AZ_CONTAINER_PORT              App port exposed by the container (default: 8501)
  AZ_DOCKERFILE                  Dockerfile path relative to repo root (default: Dockerfile)
  AZ_BUILD_MODE                  local | remote (default: local). remote uses 'az acr build'
  AZ_MAX_REPLICAS                Update max replicas (default: keep current setting)
  AZ_CA_BUNDLE                   Path to trusted CA bundle used for outbound TLS (optional)
  AZURE_CLI_DISABLE_CONNECTION_VERIFICATION  Set to 1 to skip TLS verification (not recommended)

Example:
  AZ_SUBSCRIPTION_ID=xxxx \
  AZ_RESOURCE_GROUP=gmma-prod \
  AZ_CONTAINER_APP_NAME=gmma-akshare \
  AZ_ACR_NAME=gmmaacr \
  AZ_IMAGE_TAG=$(date +%Y%m%d%H%M) \
  ./infra-scripts/update-container-app.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

required_envs=(
  AZ_SUBSCRIPTION_ID
  AZ_RESOURCE_GROUP
  AZ_CONTAINER_APP_NAME
  AZ_ACR_NAME
)

for var in "${required_envs[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "Environment variable '${var}' must be set." >&2
    usage
    exit 1
  fi
done

export AZ_CONTAINER_PORT="${AZ_CONTAINER_PORT:-8501}"
export AZ_DOCKERFILE="${AZ_DOCKERFILE:-Dockerfile}"
export AZ_IMAGE_REPOSITORY="${AZ_IMAGE_REPOSITORY:-${AZ_CONTAINER_APP_NAME}}"
default_tag="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)"
export AZ_IMAGE_TAG="${AZ_IMAGE_TAG:-${default_tag}}"
export AZ_MAX_REPLICAS="${AZ_MAX_REPLICAS:-}"
readonly BUILD_MODE="${AZ_BUILD_MODE:-local}"

if [[ "${BUILD_MODE}" != "local" && "${BUILD_MODE}" != "remote" ]]; then
  echo "AZ_BUILD_MODE must be 'local' or 'remote' (got '${BUILD_MODE}')." >&2
  exit 1
fi

if [[ "${BUILD_MODE}" == "local" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "AZ_BUILD_MODE=local requires Docker CLI. Install Docker or set AZ_BUILD_MODE=remote." >&2
    exit 1
  fi
fi

if ! az account show >/dev/null 2>&1; then
  echo "Azure CLI is not logged in. Run 'az login' before executing this script." >&2
  exit 1
fi

echo "Setting Azure subscription ${AZ_SUBSCRIPTION_ID}..."
az account set --subscription "${AZ_SUBSCRIPTION_ID}"

if ! az containerapp show --name "${AZ_CONTAINER_APP_NAME}" --resource-group "${AZ_RESOURCE_GROUP}" >/dev/null 2>&1; then
  echo "Container App ${AZ_CONTAINER_APP_NAME} does not exist; run the deploy script first." >&2
  exit 1
fi

ACR_LOGIN_SERVER="$(az acr show --name "${AZ_ACR_NAME}" --query loginServer --output tsv)"
ACR_USERNAME="$(az acr credential show --name "${AZ_ACR_NAME}" --query username --output tsv)"
ACR_PASSWORD="$(az acr credential show --name "${AZ_ACR_NAME}" --query 'passwords[0].value' --output tsv)"
IMAGE_URI="${ACR_LOGIN_SERVER}/${AZ_IMAGE_REPOSITORY}:${AZ_IMAGE_TAG}"

if [[ "${BUILD_MODE}" == "remote" ]]; then
  echo "Building Docker image ${IMAGE_URI} via Azure Container Registry build service..."
  az acr build \
    --registry "${AZ_ACR_NAME}" \
    --image "${AZ_IMAGE_REPOSITORY}:${AZ_IMAGE_TAG}" \
    --file "${REPO_ROOT}/${AZ_DOCKERFILE}" \
    "${REPO_ROOT}"
else
  echo "Building Docker image ${IMAGE_URI} locally..."
  az acr login --name "${AZ_ACR_NAME}" >/dev/null
  docker build \
    --file "${REPO_ROOT}/${AZ_DOCKERFILE}" \
    --tag "${IMAGE_URI}" \
    "${REPO_ROOT}"

  echo "Pushing Docker image to ${ACR_LOGIN_SERVER}..."
  docker push "${IMAGE_URI}"
fi

update_args=(
  --name "${AZ_CONTAINER_APP_NAME}"
  --resource-group "${AZ_RESOURCE_GROUP}"
  --image "${IMAGE_URI}"
  --registry-server "${ACR_LOGIN_SERVER}"
  --registry-username "${ACR_USERNAME}"
  --registry-password "${ACR_PASSWORD}"
  --target-port "${AZ_CONTAINER_PORT}"
  --ingress external
)

if [[ -n "${AZ_MAX_REPLICAS}" ]]; then
  update_args+=(--max-replicas "${AZ_MAX_REPLICAS}")
fi

echo "Rolling out new image revision..."
az containerapp update "${update_args[@]}" --output none

FQDN="$(az containerapp show --name "${AZ_CONTAINER_APP_NAME}" --resource-group "${AZ_RESOURCE_GROUP}" --query properties.configuration.ingress.fqdn --output tsv)"
echo "Update successful."
echo "Ingress URL: https://${FQDN}"
