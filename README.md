# IHG Exercise

This project contains a SQL-based air quality exposure analysis for the United States, using public BigQuery datasets to estimate population exposure by AQI band.

## Overview

The analysis combines:

- EPA historical air quality data (`o3_daily_summary`)
- Census block group geometry data
- ACS population data

The goal is to assign each block group to the nearest air quality monitoring station and summarize the population living within each AQI range.

## Included Files

- `SQL_exercise.sql` — the main SQL script for the AQI analysis
- `README.md` — project overview and explanation

## Methodology

1. Identify the latest available AQI data year.
2. Calculate the annual average AQI per monitoring station.
3. Retrieve block group centroids and total population.
4. Compute the distance between each block group centroid and all stations.
5. Assign each block group to the nearest station.
6. Bucket the assigned AQI values into AQI ranges.
7. Aggregate total population by AQI zone and compute percentage of the U.S. population.

## AQI Buckets Used

- AQI 0-50
- AQI 51-100
- AQI 101-150
- AQI > 150

## Result Summary

The final query produced the following population distribution for 2025:

- AQI 0-50: 319,542,009 people (97.93%)
- AQI 51-100: 6,747,962 people (2.07%)

## Tools and Data Sources

This analysis uses BigQuery public datasets, including:

- `bigquery-public-data.epa_historical_air_quality.o3_daily_summary`
- `bigquery-public-data.geo_census_blockgroups.us_blockgroups_national`
- `bigquery-public-data.census_bureau_acs.blockgroup_2018_5yr`

It relies on geospatial functions such as:

- `ST_GEOGPOINT`
- `ST_CENTROID`
- `ST_DISTANCE`

## Notes

- The script is designed for BigQuery SQL.
- Ensure the relevant Google Cloud BigQuery datasets are accessible in your environment.
- The logic is intended for exploratory analysis and can be extended with additional air quality metrics or geographic filters.
