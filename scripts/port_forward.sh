#!/usr/bin/env bash
set -e

NAMESPACE="${NAMESPACE:-estate}"

echo "=================================================================="
echo " 🌐 Estate Intelligence - Kubernetes Port Forwarding"
echo " Namespace: ${NAMESPACE}"
echo "=================================================================="
echo " - FastAPI ML Inference: http://localhost:8000 (docs: /docs, health: /health)"
echo " - Grafana Dashboards:   http://localhost:3000 (admin / admin)"
echo " - Apache Airflow UI:    http://localhost:8080 (admin / admin)"
echo " - MLflow Tracking UI:   http://localhost:5001"
echo " - Prometheus Server:    http://localhost:9090"
echo " - ClickHouse HTTP API:  http://localhost:8123 (estate / estate)"
echo " - MinIO Object Console: http://localhost:9011 (minioadmin / minioadmin)"
echo " - PostgreSQL (PostGIS): localhost:5433 (estate / estate / estate)"
echo "=================================================================="
echo "Press Ctrl+C to terminate all port forward tunnels."
echo ""

# Terminate all child processes on exit
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT SIGINT SIGTERM

kubectl port-forward svc/inference 8000:8000 -n "${NAMESPACE}" >/dev/null 2>&1 &
kubectl port-forward svc/grafana 3000:3000 -n "${NAMESPACE}" >/dev/null 2>&1 &
kubectl port-forward svc/airflow 8080:8080 -n "${NAMESPACE}" >/dev/null 2>&1 &
kubectl port-forward svc/mlflow-external 5001:5001 -n "${NAMESPACE}" >/dev/null 2>&1 &
kubectl port-forward svc/prometheus 9090:9090 -n "${NAMESPACE}" >/dev/null 2>&1 &
kubectl port-forward svc/clickhouse-external 8123:8123 -n "${NAMESPACE}" >/dev/null 2>&1 &
kubectl port-forward svc/minio-external 9011:9011 -n "${NAMESPACE}" >/dev/null 2>&1 &
kubectl port-forward svc/postgres-external 5433:5433 -n "${NAMESPACE}" >/dev/null 2>&1 &

wait
