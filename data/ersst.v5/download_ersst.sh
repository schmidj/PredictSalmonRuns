#!/bin/bash

ersst_url="https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf"
yr_start=1984
yr_end=2025

for iy in $(seq $yr_start $yr_end); do
  for im in {01..12}; do
     echo "Downloading data for $iy-$im"
     curl -s -O ${ersst_url}/ersst.v5.${iy}${im}.nc
  done
done