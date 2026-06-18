#!/usr/bin/env bash
# Regenerate the 6-3 vs 5-5 (TANH attack / DC-GAN defense) r_repair quantile comparison.
# Both runs are forced onto one figure by symlinking their logs under minibatch-style
# names so the existing plot script (which globs *_minibatch*.log and groups by arch
# pair) picks them up. The leading legend number encodes norm scale x10 (63=6.3, 55=5.5).
set -euo pipefail

RR=/cephyr/users/garciafe/temp/CIFAR10/run_results
CONTAINER=/cephyr/users/garciafe/containers/fl_env_v5.sif
SCRIPT=/cephyr/users/garciafe/scripts/plot_gen_corr_pert_ratio_log.py
QUANTILE="${1:-0.8}"   # optional first arg: quantile (default 0.8)
HOME_COPY=/cephyr/users/garciafe/quantile_compare.pdf

TMP="$RR/_cmp_6-3_5-5"
rm -rf "$TMP"; mkdir -p "$TMP"
ln -s "$RR/console_final_N_kd_mb_6-3_tanh_dcgan.log" "$TMP/console_minibatch_63_01_tanh_dcgan.log"
ln -s "$RR/console_final_N_kd_mb_5-5_tanh_mask.log"  "$TMP/console_minibatch_55_01_tanh_dcgan.log"

# The plot script prints each produced path (it now appends a constraint token + arch pair
# and writes PDF); capture the produced file rather than hard-coding the name.
PRODUCED=$(apptainer exec "$CONTAINER" python3 "$SCRIPT" \
  --run-results-path "$TMP" \
  --output "$RR/quantile_compare_6-3_vs_5-5.pdf" \
  --quantile "$QUANTILE" \
  --no-attack-gen | tail -n 1)

rm -rf "$TMP"
cp "$PRODUCED" "$HOME_COPY"
echo "Updated: $HOME_COPY (from $PRODUCED, q=$QUANTILE)"


# bash ./scripts/regen_compare.sh
