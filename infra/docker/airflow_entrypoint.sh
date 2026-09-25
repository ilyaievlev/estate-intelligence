#!/bin/bash
set -e

# If arguments are passed (e.g. from KubernetesExecutor: airflow tasks run ...), execute them directly
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

echo "=================================================="
echo " 🚀 Initializing Apache Airflow for Estate Intelligence"
echo "=================================================="

echo "Running Airflow database migrations..."
airflow db migrate

echo "Ensuring Airflow admin user exists..."
airflow users create \
    --username "${_AIRFLOW_WWW_USER_USERNAME:-admin}" \
    --password "${_AIRFLOW_WWW_USER_PASSWORD:-admin}" \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email "${_AIRFLOW_WWW_USER_EMAIL:-admin@estate.local}" || true

echo "=================================================="
echo " 🌐 Starting Airflow Standalone..."
echo "=================================================="
exec airflow standalone
