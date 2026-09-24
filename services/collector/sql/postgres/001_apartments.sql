-- Текущее состояние объявления: одна строка на (source, external_id).
-- История цен живёт в ClickHouse (apartment_snapshots).
CREATE TABLE IF NOT EXISTS apartments (
    source           TEXT NOT NULL,
    external_id      TEXT NOT NULL,
    url              TEXT NOT NULL,
    rooms            SMALLINT,
    area             REAL,
    floor            SMALLINT,
    floors_total     SMALLINT,
    address          TEXT,
    latitude         DOUBLE PRECISION,
    longitude        DOUBLE PRECISION,
    metro            TEXT,
    metro_distance_m INTEGER,
    seller_type      TEXT,
    description      TEXT,
    published_at     TIMESTAMPTZ,
    first_seen_at    TIMESTAMPTZ NOT NULL,
    last_seen_at     TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (source, external_id)
);

CREATE INDEX IF NOT EXISTS ix_apartments_last_seen_at
    ON apartments (last_seen_at DESC);
