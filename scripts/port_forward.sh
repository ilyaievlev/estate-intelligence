#!/usr/bin/env bash
set -u

NAMESPACE="${NAMESPACE:-estate}"

echo "=================================================================="
echo " Estate Intelligence — port-forward (namespace: ${NAMESPACE})"
echo "=================================================================="
echo " Inference API:  http://localhost:8000/docs"
echo " Grafana:        http://localhost:3000   (admin / admin)"
echo " Airflow:        http://localhost:8080   (admin / admin)"
echo " MLflow:         http://localhost:5001"
echo " Prometheus:     http://localhost:9090"
echo " ClickHouse:     http://localhost:8123   (estate / estate)"
echo " MinIO Console:  http://localhost:9011   (minioadmin / minioadmin)"
echo " Postgres:       localhost:5433          (estate / estate)"
echo "=================================================================="
echo "Ctrl+C — закрыть все туннели"
echo ""

cleanup() {
  trap - INT TERM EXIT
  for pid in $(jobs -p); do
    pkill -P "$pid" 2>/dev/null
    kill "$pid" 2>/dev/null
  done
  exit 0
}
trap cleanup INT TERM EXIT

# port-forward обрывается при рестарте пода, поэтому переподключаемся в цикле.
forward() {
  local target="$1" ports="$2"
  while true; do
    kubectl port-forward "$target" "$ports" -n "$NAMESPACE" >/dev/null 2>&1
    sleep 2
  done
}

forward svc/inference           8000:8000 &
forward svc/grafana             3000:3000 &
forward svc/airflow             8080:8080 &
forward svc/mlflow-external     5001:5001 &
forward svc/prometheus          9090:9090 &
forward svc/clickhouse-external 8123:8123 &
forward svc/minio-external      9011:9011 &
forward svc/minio-external      9010:9010 &
forward svc/postgres-external   5433:5433 &

wait
