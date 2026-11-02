#!/usr/bin/env bash
# Driver for diagnostic_validation.py: one model per process.
#
# Why not just run the script once over all models? Because the WSL2 VM has 7 GB and five
# torch models loaded back-to-back in a single interpreter never give the memory back --
# that froze the VM on 2026-08-03. A fresh process per model releases everything on exit.
# Results accumulate in results/diagnostic_validation.json via the script's resume logic,
# so an interrupted run continues where it stopped instead of starting over.
set -u
cd "$(dirname "$0")/.."

PY=${PY:-$HOME/.venvs/umlip/bin/python}
# SCRIPT=scripts/run_mno.py ./scripts/run_validation.sh   -- same driver, different stage
SCRIPT=${SCRIPT:-scripts/diagnostic_validation.py}
MODELS=("MACE-MP-0" "CHGNet" "SevenNet-0" "M3GNet-MatPES" "MatterSim")
if [ "${1:-}" = "--extra" ]; then
    PY=$HOME/.venvs/umlip2/bin/python
    MODELS=("Orb-v3" "GRACE-2L-OMAT")
    EXTRA=--extra
else
    EXTRA=
fi

for m in "${MODELS[@]}"; do
    avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    echo "### $m  (MemAvailable ${avail} MB)"
    if [ "$avail" -lt 1500 ]; then
        echo "### SKIP $m - under 1.5 GB available, refusing to risk the VM"
        continue
    fi
    # nice: keep the interactive shell responsive while this runs
    nice -n 10 "$PY" "$SCRIPT" $EXTRA --only "$m"
    echo "### $m exit=$?"
done
echo "ALL DONE"
