-- Координаты объявления храним как PostGIS-точку (SRID 4326: X = долгота, Y = широта).
-- Одноразовая миграция: переносит latitude/longitude в location и удаляет старые колонки.
BEGIN;

CREATE EXTENSION IF NOT EXISTS postgis;

ALTER TABLE apartments ADD COLUMN IF NOT EXISTS location geometry(Point, 4326);

UPDATE apartments
SET location = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
WHERE location IS NULL
  AND latitude IS NOT NULL
  AND longitude IS NOT NULL;

ALTER TABLE apartments
    DROP COLUMN IF EXISTS latitude,
    DROP COLUMN IF EXISTS longitude;

CREATE INDEX IF NOT EXISTS ix_apartments_location
    ON apartments USING GIST (location);

COMMIT;
