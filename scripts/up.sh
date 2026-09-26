#!/usr/bin/env bash
# Поднимает весь стек в локальном Kubernetes:
#   colima -> сборка образов -> helm upgrade --install -> ожидание подов -> port-forward
#
#   ./scripts/up.sh               обычный запуск
#   ./scripts/up.sh --build       пересобрать все образы, даже если они уже есть
#   ./scripts/up.sh --no-forward  не держать port-forward в конце
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHART="${ROOT}/infra/helm/estate-intelligence"
NAMESPACE="${NAMESPACE:-estate}"
RELEASE="${RELEASE:-estate}"
COLIMA_CPU="${COLIMA_CPU:-6}"
COLIMA_MEMORY="${COLIMA_MEMORY:-10}"
COLIMA_DISK="${COLIMA_DISK:-60}"

FORCE_BUILD=0
FORWARD=1
for arg in "$@"; do
  case "$arg" in
    --build) FORCE_BUILD=1 ;;
    --no-forward) FORWARD=0 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

step "Проверка зависимостей"
for bin in docker kubectl helm; do
  command -v "$bin" >/dev/null || fail "не найден $bin"
done

if [ ! -f "${ROOT}/.env" ]; then
  cp "${ROOT}/.env.example" "${ROOT}/.env"
  echo "создан .env из .env.example — впишите AVITO_COOKIES_API_KEY и AVITO_PROXY_STRING, если нужен сбор с Avito"
fi

if [ ! -f "${ROOT}/services/collector/vendor/parser_avito/config.toml" ]; then
  git -C "$ROOT" submodule update --init --recursive
fi

step "Kubernetes"
if command -v colima >/dev/null; then
  if ! colima status >/dev/null 2>&1; then
    colima start --cpu "$COLIMA_CPU" --memory "$COLIMA_MEMORY" --disk "$COLIMA_DISK" --kubernetes
  fi
  kubectl config use-context colima >/dev/null
fi
kubectl cluster-info >/dev/null 2>&1 || fail "кластер недоступен (kubectl cluster-info)"
echo "контекст: $(kubectl config current-context)"

step "Образы"
build() {
  local image="$1" dockerfile="$2"
  if [ "$FORCE_BUILD" = 1 ] || ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "сборка $image"
    docker build -t "$image" -f "${ROOT}/infra/docker/${dockerfile}" "$ROOT"
  else
    echo "$image уже есть (--build для пересборки)"
  fi
}
build estate-inference:latest Dockerfile.inference
build estate-airflow:latest   Dockerfile.airflow
build estate-mlflow:latest    Dockerfile.mlflow
build estate-collector:latest Dockerfile.collector

step "Helm"
# Настройки парсера берутся из .env, чтобы секреты не хранились в чарте.
# .env не source'ится: в URL есть '&' и '?' без кавычек.
env_get() {
  local line
  line="$(grep -E "^${1}=" "${ROOT}/.env" | tail -n 1 || true)"
  line="${line#*=}"
  line="${line%\"}"; line="${line#\"}"
  printf '%s' "$line"
}
AVITO_URLS="$(env_get AVITO_URLS)"
AVITO_USE_BYPASS_API="$(env_get AVITO_USE_BYPASS_API)"
AVITO_COOKIES_API_KEY="$(env_get AVITO_COOKIES_API_KEY)"
AVITO_PROXY_STRING="$(env_get AVITO_PROXY_STRING)"
AVITO_COUNT="$(env_get AVITO_COUNT)"

yaml_quote() { printf "'%s'" "${1//\'/\'\'}"; }

GENERATED_VALUES="$(mktemp -t estate-values.XXXXXX)"
trap 'rm -f "$GENERATED_VALUES"' EXIT
cat > "$GENERATED_VALUES" <<EOF
global:
  projectHostPath: $(yaml_quote "$ROOT")
  dagsHostPath: $(yaml_quote "$ROOT/dags")
collector:
  urls: $(yaml_quote "${AVITO_URLS:-}")
  useBypassApi: ${AVITO_USE_BYPASS_API:-false}
  cookiesApiKey: $(yaml_quote "${AVITO_COOKIES_API_KEY:-}")
  proxyString: $(yaml_quote "${AVITO_PROXY_STRING:-}")
  count: ${AVITO_COUNT:-10}
EOF

HELM_ARGS=(-f "$GENERATED_VALUES")
[ -f "${CHART}/values.local.yaml" ] && HELM_ARGS+=(-f "${CHART}/values.local.yaml")

helm upgrade --install "$RELEASE" "$CHART" \
  --namespace "$NAMESPACE" --create-namespace \
  "${HELM_ARGS[@]}" \
  --wait --timeout 15m

# ConfigMap мог измениться, а поды Airflow читают его только при старте.
if [ "$FORCE_BUILD" = 1 ]; then
  kubectl rollout restart deployment -n "$NAMESPACE" >/dev/null
fi

step "Ожидание готовности"
kubectl rollout status deployment --namespace "$NAMESPACE" --timeout 10m
kubectl rollout status statefulset --namespace "$NAMESPACE" --timeout 10m
kubectl get pods -n "$NAMESPACE"

if [ "$FORWARD" = 1 ]; then
  step "Port-forward"
  exec "${ROOT}/scripts/port_forward.sh"
fi
