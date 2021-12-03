#!/bin/bash

speed_coeffs="1 2 4"
completed_coeffs="1 2 3"

cfg_template="$(cat cfg_template.yaml)"
for speed_coeff in $speed_coeffs
do for completed_coeff in $completed_coeffs
do
  echo "$cfg_template" > cfg.yaml
  sed -i "s/speed_coeff/$speed_coeff/g" cfg.yaml
  sed -i "s/completed_coeff/$completed_coeff/g" cfg.yaml
  echo "$speed_coeff $completed_coeff" >> tuning_log.txt
  python runner.py

done
done
