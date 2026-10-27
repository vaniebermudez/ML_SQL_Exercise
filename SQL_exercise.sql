-- Here's the things I checked first.

-- 1. What years are available in the AQI data?
-- This is needed to know what the 'latest available data' year is
-- Here, I got 2025 as the latest available data
SELECT EXTRACT(YEAR FROM date_local) AS year, COUNT(*) AS record_count
FROM `bigquery-public-data.epa_historical_air_quality.o3_daily_summary`
GROUP BY year
ORDER BY year DESC;

-- Resulted to
-- year	record_count
-- 2025	113767


-- Next I checked the quick statistics on AQI
-- 2. What does the AQI distribution look like?
SELECT MIN(aqi), MAX(aqi), AVG(aqi), COUNTIF(aqi IS NULL) AS null_count
FROM `bigquery-public-data.epa_historical_air_quality.o3_daily_summary`
WHERE EXTRACT(YEAR FROM date_local) = 2025;

-- Resulted to MIN: 0, MAX: 239, AVG: ~40.03, Null: 0
-- I have an idea


-- Next is to check how many stations need to be matched
-- 3. How many monitoring stations are there, and where?
SELECT
  COUNT(DISTINCT site_num) AS stations, COUNT(DISTINCT state_code) AS states
FROM `bigquery-public-data.epa_historical_air_quality.o3_daily_summary`
WHERE EXTRACT(YEAR FROM date_local) = 2025;

-- Resulted to
-- stations	states
-- 274	53

-- Lastly, 
-- 4. Check block group population table
SELECT *
FROM `bigquery-public-data.census_bureau_acs.blockgroup_2018_5yr`
LIMIT 5;

SELECT SUM(total_pop) FROM `bigquery-public-data.census_bureau_acs.blockgroup_2018_5yr`;

-- Resulted to 326M (near 330M)


-- MAIN QUERY LOGIC

-- ============================================================
-- STEP 1: Get annual average AQI per monitoring station (2025)
-- ============================================================
WITH
  station_aqi AS (
    SELECT
      site_num,
      state_code,
      county_code,
      AVG(latitude) AS station_lat,
      AVG(longitude) AS station_lon,
      AVG(aqi) AS avg_aqi
    FROM `bigquery-public-data.epa_historical_air_quality.o3_daily_summary`
    WHERE
      aqi IS NOT NULL
      AND aqi >= 0
      AND aqi <= 500
      AND EXTRACT(YEAR FROM date_local) = 2025
    GROUP BY 1, 2, 3
  ),

  -- ============================================================
  -- STEP 2: Get block group centroids + population
  -- ============================================================
  block_groups AS (
    SELECT bg.geo_id, ST_CENTROID(bg.blockgroup_geom) AS centroid, pop.total_pop
    FROM
      `bigquery-public-data.geo_census_blockgroups.us_blockgroups_national` bg
    JOIN `bigquery-public-data.census_bureau_acs.blockgroup_2018_5yr` pop
      ON bg.geo_id = pop.geo_id
    WHERE pop.total_pop > 0
  ),

  -- ============================================================
  -- STEP 3: Convert station coordinates to geography points
  -- ============================================================
  stations_geo AS (
    SELECT
      site_num, avg_aqi, ST_GEOGPOINT(station_lon, station_lat) AS station_point
    FROM station_aqi
  ),

  -- ============================================================
  -- STEP 4: CROSS JOIN every block group to every station
  --         and compute distance
  -- ============================================================
  block_station_distances AS (
    SELECT
      bg.geo_id,
      bg.total_pop,
      s.avg_aqi,
      ST_DISTANCE(bg.centroid, s.station_point) AS distance_meters,
      ROW_NUMBER()
        OVER (
          PARTITION BY bg.geo_id
          ORDER BY ST_DISTANCE(bg.centroid, s.station_point) ASC
        )
        AS distance_rank
    FROM block_groups bg
    CROSS JOIN stations_geo s
  ),

  -- ============================================================
  -- STEP 5: Keep only the nearest station per block group
  -- ============================================================
  block_group_aqi AS (
    SELECT geo_id, total_pop, avg_aqi AS assigned_aqi
    FROM block_station_distances
    WHERE distance_rank = 1
  ),

  -- ============================================================
  -- STEP 6: Bucket by AQI zone 
  -- ============================================================
  aqi_bucketed AS (
    SELECT
      total_pop,
      CASE
        WHEN assigned_aqi >= 0 AND assigned_aqi < 51 THEN 'AQI 0-50'
        WHEN assigned_aqi >= 51 AND assigned_aqi < 101 THEN 'AQI 51-100'
        WHEN assigned_aqi >= 101 AND assigned_aqi < 151 THEN 'AQI 101-150'
        WHEN assigned_aqi >= 151 THEN 'AQI > 150'
        ELSE 'Unknown'
        END
        AS aqi_zone
    FROM block_group_aqi
    WHERE assigned_aqi IS NOT NULL
  )

-- ============================================================
-- STEP 7: Final output — % of US population by AQI zone
-- ============================================================
SELECT
  aqi_zone,
  SUM(total_pop) AS population,
  ROUND(100.0 * SUM(total_pop) / SUM(SUM(total_pop)) OVER (), 2)
    AS pct_of_us_population
FROM aqi_bucketed
GROUP BY aqi_zone
ORDER BY aqi_zone;


-- THIS QUERY RESULTED TO THE FOLLOWING TABLE
-- aqi_zone	    population	pct_of_us_population
--	AQI 0-50	319542009.0	97.93
--	AQI 51-100	6747962.0	2.07