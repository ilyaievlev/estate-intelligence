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
# Все настройки берутся из .env, values.yaml хранит только значения по умолчанию.
# .env не source'ится: в URL есть '&' и '?' без кавычек.
env_get() {
  local line
  line="$(grep -E "^${1}=" "${ROOT}/.env" | tail -n 1 || true)"
  line="${line#*=}"
  line="${line%\"}"; line="${line#\"}"
  printf '%s' "$line"
}

yaml_quote() { printf "'%s'" "${1//\'/\'\'}"; }

# Строка "key: 'value'" с нужным отступом; пустые и незаданные переменные
# пропускаются, чтобы сработал дефолт из values.yaml.
# Секреты парсера пишутся всегда: пустое значение в .env значит "без ключа/прокси".
from_env() {
  local indent="$1" key="$2" var="$3" mode="${4:-}" value
  value="$(env_get "$var")"
  if [ -z "$value" ] && [ "$mode" != always ]; then return; fi
  printf '%s%s: %s\n' "$indent" "$key" "$(yaml_quote "$value")"
}

# Заголовок секции печатается только если в ней есть значения: пустой
# "postgres:" Helm понял бы как null и стёр бы дефолты секции.
section() {
  local header="$1" body="$2"
  [ -n "$body" ] && printf '%s\n%s\n' "$header" "$body"
  return 0
}

GENERATED_VALUES="$(mktemp -t estate-values.XXXXXX)"
trap 'rm -f "$GENERATED_VALUES"' EXIT
{
  echo "global:"
  echo "  projectHostPath: $(yaml_quote "$ROOT")"
  echo "  dagsHostPath: $(yaml_quote "$ROOT/dags")"
  section "postgres:" "$(
    from_env "  " user POSTGRES_USER
    from_env "  " password POSTGRES_PASSWORD
    from_env "  " database POSTGRES_DB)"
  section "clickhouse:" "$(
    from_env "  " user CLICKHOUSE_USER
    from_env "  " password CLICKHOUSE_PASSWORD
    from_env "  " database CLICKHOUSE_DATABASE)"
  section "minio:" "$(
    from_env "  " rootUser MINIO_ROOT_USER
    from_env "  " rootPassword MINIO_ROOT_PASSWORD
    from_env "  " defaultBucket MINIO_DEFAULT_BUCKET)"
  section "airflow:"$'\n'"  user:" "$(
    from_env "    " username AIRFLOW_USER
    from_env "    " password AIRFLOW_PASSWORD
    from_env "    " email AIRFLOW_EMAIL)"
  section "monitoring:"$'\n'"  grafana:" "$(
    from_env "    " adminUser GRAFANA_ADMIN_USER
    from_env "    " adminPassword GRAFANA_ADMIN_PASSWORD)"
  section "ml:" "$(
    from_env "  " trainMaxAgeDays TRAIN_MAX_AGE_DAYS)"
  section "collector:" "$(
    from_env "  " intervalSec COLLECTOR_INTERVAL_SEC
    from_env "  " urls AVITO_URLS
    from_env "  " useBypassApi AVITO_USE_BYPASS_API
    from_env "  " cookiesApiKey AVITO_COOKIES_API_KEY always
    from_env "  " proxyString AVITO_PROXY_STRING always
    from_env "  " count AVITO_COUNT)"
} > "$GENERATED_VALUES"

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
