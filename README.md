# estate-intelligence

Пет-проект про аренду квартир в Москве. Собирает объявления с Avito, складывает их в Postgres и ClickHouse, раз в сутки переобучает модель CatBoost и отдаёт предсказание месячной аренды через HTTP API. Всё это крутится в локальном Kubernetes и ставится одним Helm-чартом.

Что умеет:

- раз в час собирает свежие объявления и обновляет базу;
- хранит историю цен по каждому объявлению (каждый проход сборщика - новый снапшот в ClickHouse);
- каждую ночь обучает новую модель и сравнивает её с текущей на одной и той же тестовой выборке. Новая модель заменяет старую только если MAE стал лучше хотя бы на 0.1%;
- отдаёт предсказание по одной квартире или пачкой, перезагружает модель без рестарта;
- пишет метрики в Prometheus, в Grafana есть два готовых дашборда.

## Как это устроено

```
Avito ──> collector ──> Postgres (текущее состояние объявлений, PostGIS)
                    └─> ClickHouse (история цен, только INSERT)

Airflow (KubernetesExecutor)
  ├─ avito_hourly_collector  - каждый час, запускает collector
  └─ ml_daily_retrain        - каждый день в 03:00 МСК
        ├─ check_data_readiness     MLflow доступен, в базе >= 500 валидных объявлений
        ├─ run_challenger_retrain   обучение + сравнение с текущей моделью
        ├─ report_retrain_outcome   печать итогов из MLflow Registry
        └─ notify_inference_reload  POST /model/reload в inference

MLflow ──> метаданные в Postgres (БД mlflow), артефакты в MinIO (бакет mlflow)

inference (FastAPI) ──> тянет модель models:/estate_rent_catboost@champion
Prometheus ──> скрейпит inference и ClickHouse ──> Grafana
```

Каждая задача Airflow выполняется в отдельном поде. Scheduler создаёт под из шаблона `pod_template.yaml`, задача отрабатывает, и под удаляется. Постоянных воркеров нет, поэтому обучение CatBoost не отнимает память у остальных сервисов, пока оно не запущено.

### Зачем здесь Kubernetes

Сначала всё было на Docker Compose (файл `infra/docker/docker-compose.yml` остался, им можно пользоваться для отладки). Переехал на k8s по трём причинам:

1. **Задачи Airflow в отдельных подах.** В Compose воркер Airflow живёт постоянно и делит память со всеми. Переобучение на 2500 итераций пару раз роняло соседние контейнеры по OOM. С `KubernetesExecutor` под обучения получает свои лимиты и после работы удаляется.
2. **Автоскейлинг inference.** Стоит HPA: 1–4 реплики, порог 80% CPU. Если прогнать `scripts/mock_traffic.py`, видно, как поднимаются новые поды.
3. **Пробы и перезапуски.** У всех сервисов есть liveness- и readiness-пробы. Упавший под поднимается заново, а трафик на него не идёт, пока он не будет готов.

Базы (Postgres, ClickHouse, MinIO) запущены как `StatefulSet` с PVC, поэтому данные переживают `helm upgrade` и рестарты подов. Остальное - обычные `Deployment`.

## Структура репозитория

```
dags/                          DAG'и Airflow
  avito_hourly_collector.py
  ml_daily_retrain.py
infra/
  docker/                      Dockerfile'ы сервисов, compose, entrypoint Airflow
  helm/estate-intelligence/    Helm-чарт всего стека
  grafana/                     дашборды и provisioning
  prometheus/                  конфиг скрейпа (для compose)
  clickhouse/                  включение /metrics в ClickHouse (для compose)
scripts/
  port_forward.sh              проброс портов всех сервисов на localhost
  mock_traffic.py              генератор нагрузки для дашбордов
services/
  collector/                   сборщик объявлений
    sources/avito/             обёртка над parser_avito + маппинг в доменную модель
    storage/                   запись в Postgres (upsert) и ClickHouse (снапшоты)
    sql/                       init-скрипты баз
    vendor/parser_avito/       git submodule с парсером
  ml/                          загрузка данных, фичи, обучение, работа с MLflow
  inference/                   FastAPI-сервис предсказаний
notebooks/experiments.ipynb    черновые эксперименты с моделью
```

## Что нужно заранее

- macOS или Linux
- [Colima](https://github.com/abiosoft/colima) (или любой другой локальный k8s: k3d, kind, minikube, Docker Desktop)
- `kubectl`, `helm` 3.x
- Python 3.12 - только для локальных скриптов и тестов

Ресурсов нужно прилично: Airflow, MLflow, ClickHouse и обучение CatBoost вместе съедают около 7–8 ГБ. Я запускаю Colima так:

```bash
colima start --cpu 6 --memory 10 --disk 60 --kubernetes
```

Если памяти мало, добавьте в VM swap. Без него у меня всё падало, когда тяжёлые сервисы стартовали одновременно. Swap настраивается через `provision` в `~/.colima/default/colima.yaml`:

```yaml
provision:
  - mode: system
    script: |
      if [ ! -f /swapfile ]; then
        fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile
      fi
      swapon /swapfile 2>/dev/null || true
```

## Запуск

### Одной командой

```bash
git clone --recurse-submodules <url-репозитория>
cd estate-intelligence
make up
```

`make up` (это `scripts/up.sh`) делает по порядку:

1. создаёт `.env` из `.env.example`, если его нет, и подтягивает сабмодуль;
2. запускает Colima с Kubernetes, если она не запущена (ресурсы задаются через `COLIMA_CPU`, `COLIMA_MEMORY`, `COLIMA_DISK`, по умолчанию 6 CPU / 10 ГБ / 60 ГБ);
3. собирает образы `estate-*`, которых ещё нет;
4. ставит или обновляет Helm-релиз. Пути к репозиторию и настройки парсера берутся из `.env`, поэтому в `values.yaml` ничего править не надо;
5. ждёт, пока все поды станут готовы;
6. открывает port-forward ко всем сервисам. Скрипт работает, пока не нажать Ctrl+C.

Первый запуск со сборкой образов занимает 10–15 минут, повторный — около минуты.

Остальные команды:

| Команда | Что делает |
|---|---|
| `make build` | пересобрать все образы и передеплоить |
| `make forward` | только port-forward (если кластер уже поднят) |
| `make status` | поды, HPA, сервисы, PVC |
| `make logs app=inference` | логи сервиса (`inference`, `mlflow`, `airflow-scheduler`, ...) |
| `make test` | тесты inference API |
| `make traffic` | генератор нагрузки для дашбордов |
| `make down` | удалить релиз, данные в PVC остаются |
| `make destroy` | удалить namespace вместе с данными |

Ниже те же шаги вручную, если нужно понять, что происходит, или запускать не в Colima.

### 1. Клонирование вместе с сабмодулем

```bash
git clone --recurse-submodules <url-репозитория>
cd estate-intelligence
# если уже склонировали без сабмодуля:
git submodule update --init --recursive
```

### 2. Переменные окружения

```bash
cp .env.example .env
```

Все сервисы читают один общий `.env` из корня репозитория. Подробнее о переменных - в разделе [Конфигурация](#конфигурация).

### 3. Сборка образов

В Colima кластер использует тот же Docker-демон, поэтому образы, собранные локально, сразу видны подам (`imagePullPolicy: IfNotPresent`). Push в registry не нужен.

```bash
docker build -t estate-inference:latest -f infra/docker/Dockerfile.inference .
docker build -t estate-airflow:latest   -f infra/docker/Dockerfile.airflow .
docker build -t estate-mlflow:latest    -f infra/docker/Dockerfile.mlflow .
docker build -t estate-collector:latest -f infra/docker/Dockerfile.collector .
```

С k3d или kind образы придётся загрузить в кластер вручную (`k3d image import ...` / `kind load docker-image ...`).

### 4. Установка чарта

Airflow монтирует DAG'и и код сервисов в поды через `hostPath`, поэтому чарту нужен абсолютный путь к репозиторию. Настройки парсера, включая ключ spfa и прокси, в `values.yaml` не хранятся и тоже передаются отдельно. Colima по умолчанию монтирует домашнюю папку в VM, так что пути с хоста работают как есть.

`make up` сам собирает эти значения из `.env` во временный values-файл. Вручную это выглядит так:

```bash
cat > /tmp/estate-values.yaml <<EOF
global:
  projectHostPath: $PWD
  dagsHostPath: $PWD/dags
collector:
  cookiesApiKey: "..."
  proxyString: "login:password@host:port"
EOF

helm upgrade --install estate ./infra/helm/estate-intelligence \
  --namespace estate --create-namespace \
  -f /tmp/estate-values.yaml --wait --timeout 15m
```

При первом запуске:

- Postgres выполняет скрипты из `services/collector/sql/postgres/`: создаёт базы `estate`, `mlflow`, `airflow`, включает PostGIS, создаёт таблицу `apartments` и заливает 454 станции метро, МЦК и МЦД с координатами;
- ClickHouse создаёт таблицу `estate.apartment_snapshots`;
- job `minio-init-bucket` создаёт бакет `mlflow`;
- init-контейнер webserver'а выполняет миграции Airflow и создаёт пользователя.

Проверить статус:

```bash
kubectl get pods,hpa,svc -n estate
```

Все поды должны быть `Running 1/1`, а `minio-init-bucket` - `Completed`. Холодный старт занимает 2–4 минуты, дольше всех поднимается Airflow.

### 5. Доступ с хоста

```bash
./scripts/port_forward.sh
```

Скрипт держит `kubectl port-forward` для всех сервисов и закрывает туннели по Ctrl+C.

| Сервис | Адрес | Логин / пароль |
|---|---|---|
| Inference API | http://localhost:8000/docs | - |
| Grafana | http://localhost:3000 | admin / admin |
| Airflow | http://localhost:8080 | admin / admin |
| MLflow | http://localhost:5001 | - |
| Prometheus | http://localhost:9090 | - |
| ClickHouse HTTP | http://localhost:8123 | estate / estate |
| MinIO Console | http://localhost:9011 | minioadmin / minioadmin |
| Postgres | localhost:5433 | estate / estate, БД `estate` |

Postgres проброшен на 5433, потому что 5432 часто занят локальным Postgres из Homebrew.

### 6. Первая модель

Сразу после установки модели в реестре нет, и `/health` вернёт `model_loaded: false`. Варианты:

- в Airflow запустить `avito_hourly_collector`, дождаться, пока наберётся хотя бы 500 объявлений, затем запустить `ml_daily_retrain`;
- если данные уже есть, обучить модель локально: `cd services/ml && python train.py`. MLflow должен быть доступен на `localhost:5001` (то есть `port_forward.sh` запущен).

Когда модель с алиасом `champion` появится в MLflow, inference подхватит её при следующем `/model/reload` или рестарте пода.

## Конфигурация

Один `.env` в корне на все сервисы. `collector`, `ml` и `inference` загружают его через `python-dotenv`. Уже заданные переменные окружения имеют приоритет над файлом. В Kubernetes те же значения передаются через `values.yaml` → ConfigMap `airflow-env`, и поды задач Airflow получают их через `envFrom`.

Основные группы переменных:

| Группа | Переменные |
|---|---|
| Postgres | `POSTGRES_*`, `DATABASE_URL` |
| ClickHouse | `CLICKHOUSE_HOST`, `CLICKHOUSE_PORT`, `CLICKHOUSE_DATABASE`, `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD` |
| MinIO / S3 | `MINIO_*`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| MLflow | `MLFLOW_TRACKING_URI`, `MLFLOW_S3_ENDPOINT_URL`, `MLFLOW_MODEL_NAME`, `MLFLOW_CHAMPION_ALIAS` |
| Inference | `INFERENCE_PORT`, `MOSCOW_CENTER_LAT`, `MOSCOW_CENTER_LON` |
| Сборщик | `COLLECTOR_INTERVAL_SEC`, `AVITO_*` (см. ниже) |
| Мониторинг | `PROMETHEUS_PORT`, `GRAFANA_*` |

Значения в `.env.example` рассчитаны на запуск с хоста: `localhost` и проброшенные порты. Внутри кластера адреса другие (`postgres:5432`, `clickhouse:8123`, `mlflow:5000`, `minio:9000`), они прописаны в чарте.

### Настройки парсера

У парсера свой `services/collector/vendor/parser_avito/config.toml`. Пять полей, которые часто приходится менять, берутся из общего `.env`:

| Поле в config.toml | Переменная | Пример |
|---|---|---|
| `urls` | `AVITO_URLS` | одна ссылка, несколько через запятую или JSON-массив |
| `use_bypass_api` | `AVITO_USE_BYPASS_API` | `true` / `false` |
| `cookies_api_key` | `AVITO_COOKIES_API_KEY` | ключ spfa.pro |
| `proxy_string` | `AVITO_PROXY_STRING` | `login:password@host:port` |
| `count` | `AVITO_COUNT` | сколько страниц выдачи обходить по каждой ссылке |

В самом `config.toml` эти поля записаны как `${AVITO_COUNT:-10}`: если переменная не задана, берётся значение после `:-`. Подстановку делает `load_config.py` в сабмодуле. Остальные настройки (фильтры по цене, паузы, ретраи) по-прежнему правятся в `config.toml`.

В Kubernetes эти значения задаются в секции `collector:` файла `values.yaml`.

## Сбор данных

`services/collector/main.py` делает один проход:

1. `AvitoClient` вызывает парсер и обходит выдачу по ссылкам из `AVITO_URLS`.
2. Сырые объявления приводятся к модели `Apartment` (`sources/avito/mapper.py`): комнаты, площадь, этаж, координаты, ближайшее метро, тип продавца.
3. В Postgres выполняется upsert по ключу `(source, external_id)`. `first_seen_at` сохраняется, `last_seen_at` обновляется.
4. В ClickHouse добавляется строка `(source, external_id, monthly_rent, collected_at)`. Таблица на `MergeTree` с партициями по месяцам, поэтому по ней удобно строить динамику цен.

При `COLLECTOR_INTERVAL_SEC > 0` сборщик работает как демон и повторяет проход с этим интервалом. В проекте запуском управляет Airflow, поэтому значение по умолчанию - `0`.

## Модель

Целевая переменная - `monthly_rent`. В выборку попадают только объявления, которые парсер видел за последние `TRAIN_MAX_AGE_DAYS` дней (по умолчанию 90), чтобы снятые с публикации квартиры со старыми ценами не тянули модель назад. Перед обучением отбрасываются выбросы: аренда вне диапазона 15 тыс.–1.5 млн ₽, площадь вне 10–400 м², больше 6 комнат, координаты за пределами Москвы и ближнего Подмосковья.

Признаки (`services/ml/features/engineering.py`):

- числовые: комнаты, площадь, этаж и этажность, доля этажа, первый/последний этаж, площадь на комнату, расстояние до метро и до центра (плюс их логарифмы), координаты, наличие и длина описания;
- категориальные: ближайшая станция, её линия и тип (метро / МЦК / МЦД). CatBoost обрабатывает их сам, без one-hot.

Станция и расстояние до неё берутся не из текста объявления, а из справочника `metro_stations` по координатам, и в обучении, и в API. Тип продавца и источник в признаки не входят: сейчас это константы. Модели, обученные раньше с этими колонками, продолжают работать: `align_features_to_model` дописывает недостающие признаки тем же значением, которое они видели при обучении.

Переобучение (`services/ml/retrain.py`):

1. Загружает данные, делит их 80/20 с фиксированным `random_state`.
2. Обучает challenger (по умолчанию 2500 итераций, `depth=7`, `lr=0.04`).
3. Загружает текущий champion из MLflow и считает метрики обеих моделей на одном и том же тесте.
4. Если MAE challenger'а лучше хотя бы на `MIN_IMPROVEMENT_RATIO` (0.1%), регистрирует новую версию и переносит на неё алиас `champion`. Причину замены записывает в теги версии. Проигравший challenger остаётся только запуском в эксперименте и в реестр не попадает.
5. Логирует в MLflow метрики, важность признаков и JSON со сравнением моделей.

Флаг `--force-promote` принудительно делает challenger чемпионом, это удобно для отладки. Остальные параметры - в `python retrain.py --help`.

На текущих данных (~15 тыс. объявлений) получается MAE около 11 тыс. ₽ и R² ≈ 0.90. Для сравнения: если всегда предсказывать медиану по обучающей выборке, MAE будет около 36 тыс. ₽.

## Inference API

FastAPI, Swagger - на `/docs`.

| Метод | Путь | Что делает |
|---|---|---|
| GET | `/health` | статус сервиса, загружена ли модель и справочник станций |
| GET | `/livez` | liveness-проба: процесс отвечает |
| GET | `/readyz` | readiness-проба: 503, пока модель не загружена |
| GET | `/model` | версия, run_id, метрики и теги текущей модели |
| POST | `/predict` | предсказание для одной квартиры |
| POST | `/predict/batch` | предсказание для списка квартир |
| POST | `/model/reload` | перечитать champion из MLflow без рестарта (только в том поде, куда попал запрос) |
| GET | `/metrics` | метрики для Prometheus |

Пример:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "rooms": 2, "area": 55, "floor": 7, "floors_total": 14,
    "latitude": 55.777, "longitude": 37.583
  }'
```

По координатам сервис сам считает расстояние до центра и находит ближайшую станцию метро, МЦК или МЦД из таблицы `metro_stations` (так же, как это делает VIEW `apartment_nearest_metro` для обучающей выборки). Поля `metro`, `metro_line`, `transport_type` и `metro_distance_m` нужны только для запросов без координат; `transport_type` принимает `metro`, `mcc` или `mcd`.

Каждая реплика раз в минуту (`inference.modelPollIntervalSec` в `values.yaml`) сверяет версию алиаса `@champion` и сама подтягивает новую модель, так что после переобучения обновляются все поды, а не только тот, до которого дошёл `/model/reload`. Если MLflow недоступен при старте, сервис пробует загрузить локальную копию модели из `services/ml/artifacts/models/`.

## Мониторинг

Prometheus находит поды inference через Kubernetes API (`kubernetes_sd_configs`) и скрейпит каждую реплику отдельно, плюс `clickhouse:9363/metrics`. Grafana поднимается с уже подключёнными источниками данных (Prometheus и ClickHouse) и двумя дашбордами в папке *Estate Intelligence*:

- **ML Инференс & Мониторинг Модели** - активная версия модели и число перезагрузок, RPS по HTTP-статусам, латентность API и чистое время инференса по перцентилям, распределение предсказанных цен, доля запросов по комнатности, средний размер батча;
- **Рыночная Аналитика (ClickHouse)** - количество снапшотов и уникальных объявлений, медианная и средняя аренда, перцентили p25/p50/p75 во времени, интенсивность сбора по часам, сегменты по цене, QPS и память ClickHouse.

Чтобы на графиках было что посмотреть, запустите генератор нагрузки (нужны `requests`, `loguru` и `clickhouse-connect`, проще всего взять venv сборщика):

```bash
services/collector/.venv/bin/python scripts/mock_traffic.py
```

Около минуты скрипт шлёт волнами одиночные и батчевые запросы, иногда невалидные, делает пару `/model/reload`, пишет в ClickHouse свежие снапшоты и выполняет аналитические запросы. Параллельно в `kubectl get hpa -n estate -w` можно наблюдать работу автоскейлера.

## Полезные команды

```bash
# состояние всего
kubectl get pods,hpa,svc,pvc -n estate

# потребление ресурсов
kubectl top pods -n estate

# смотреть, как Airflow создаёт и удаляет поды задач
kubectl get pods -n estate -w

# логи
kubectl logs -n estate -l app=inference -f
kubectl logs -n estate -l app=airflow-scheduler -f
kubectl logs -n estate -l app=mlflow -f

# консоли баз
kubectl exec -it -n estate postgres-0 -- psql -U estate -d estate
kubectl exec -it -n estate clickhouse-0 -- clickhouse-client -u estate --password estate -d estate

# применить изменения чарта
make up

# пересобрать образ и перезапустить сервис
docker build -t estate-inference:latest -f infra/docker/Dockerfile.inference .
kubectl rollout restart deployment/inference -n estate

# снести релиз (PVC с данными останутся)
helm uninstall estate -n estate
# снести совсем, вместе с данными
kubectl delete namespace estate
```

Код DAG'ов и сервисов монтируется в поды Airflow через hostPath. Поэтому правки в `dags/`, `services/ml/` и `services/collector/` подхватываются без пересборки образа. Пересобирать `estate-airflow` нужно только при изменении зависимостей.

## Тесты

```bash
python services/inference/test_api.py
```

Тест поднимает приложение через `TestClient`, загружает реальную модель из MLflow (нужен `port_forward.sh`) и проверяет все эндпоинты: health, model, predict, batch, валидацию, reload и metrics. Файл можно запускать и через `pytest`.

## Если что-то не работает

- **Поды висят в `Pending`.** Кластеру не хватает ресурсов. Проверьте `kubectl describe pod <name> -n estate` и при необходимости добавьте памяти в Colima.
- **`ErrImageNeverPull` / `ImagePullBackOff` на `estate-*`.** Образ не собран или собран в другом Docker-контексте. Проверьте `docker images | grep estate` и `docker context ls`.
- **Airflow-scheduler перезапускается.** Обычно это liveness-проба во время тяжёлых запросов к метабазе. В чарте таймауты уже увеличены, но на слабой машине их можно поднять ещё (`deployment-scheduler.yaml`).
- **Под задачи Airflow падает сразу после старта.** Посмотрите `kubectl logs <pod> -n estate` и проверьте, что `global.projectHostPath` в `values.yaml` указывает на реальный путь.
- **`/health` показывает `model_loaded: false`.** В реестре нет версии с алиасом `champion`. См. раздел [Первая модель](#6-первая-модель).
- **Парсер ловит блокировки.** См. раздел про парсер ниже.

## Парсер Avito

За сбор объявлений отвечает [parser_avito](https://github.com/Duff89/parser_avito) от Duff89. Он подключён как git submodule из форка [ilyaievlev/parser_avito](https://github.com/ilyaievlev/parser_avito), ветка `estate`. Изменения в форке:

- `apartment_ml.py` и `fetch_ml_data.py` приводят объявления о квартирах к плоскому словарю с нужными модели полями;
- `load_config.py` подставляет поля конфига из переменных окружения (см. [Настройки парсера](#настройки-парсера)).

Всё остальное (обход выдачи, пагинация, фильтры, ретраи, работа с cookies и прокси) осталось как в оригинале. Документация по настройкам - в [README оригинального репозитория](https://github.com/Duff89/parser_avito#readme) и в его [Wiki](https://github.com/Duff89/parser_avito/wiki).

Avito быстро начинает блокировать запросы с одного IP, поэтому без обхода блокировок сборщик надолго не хватит. Автор парсера описывает рабочие варианты (мобильный прокси, серверный прокси вместе с сервисом cookies spfa.pro, собственный аккаунт) в отдельном документе: [docs/ANTIBLOCK.md](https://github.com/Duff89/parser_avito/blob/master/docs/ANTIBLOCK.md). В проекте используется серверный прокси + spfa (`AVITO_USE_BYPASS_API=true`, ключ в `AVITO_COOKIES_API_KEY`, прокси в `AVITO_PROXY_STRING`).

## Технологии

**Данные и хранилища**
- PostgreSQL 16 + PostGIS 3.5 - текущее состояние объявлений, геометрия станций метро
- ClickHouse 24.8 - история цен
- MinIO - S3-совместимое хранилище для артефактов моделей

**ML**
- CatBoost - модель
- scikit-learn - сплит и метрики
- pandas, NumPy - подготовка данных
- MLflow 3 - трекинг экспериментов и Model Registry

**Сервисы**
- FastAPI + Uvicorn, Pydantic v2 - API предсказаний
- psycopg 3, clickhouse-connect - клиенты баз
- loguru - логи
- python-dotenv - общий `.env`

**Сбор данных**
- [parser_avito](https://github.com/Duff89/parser_avito) - парсер Avito (curl_cffi, BeautifulSoup)

**Оркестрация и инфраструктура**
- Apache Airflow 2.10 с KubernetesExecutor
- Kubernetes (k3s в Colima), Helm 3
- Docker, Docker Compose (для локальной отладки)

**Мониторинг**
- Prometheus, prometheus-fastapi-instrumentator
- Grafana 11 (+ плагин ClickHouse)

## Лицензия

См. [LICENSE](LICENSE). У parser_avito своя лицензия, она лежит в его репозитории.
