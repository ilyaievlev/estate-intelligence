-- История: одна строка на каждое объявление в каждом collection run. Только INSERT.
-- Init-скрипты образа выполняются без --database, поэтому имя БД указано явно.
CREATE DATABASE IF NOT EXISTS estate;

CREATE TABLE IF NOT EXISTS estate.apartment_snapshots
(
    source LowCardinality(String),
    external_id String,
    monthly_rent UInt32,
    collected_at DateTime
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(collected_at)
ORDER BY (source, external_id, collected_at);
