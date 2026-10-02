#!/bin/bash
# Submit the OMol-4M LR experiment arms from a Perlmutter login node.
# Each arm runs on 1 node (4x A100) via fairchem's SLURM mode (submitit),
# which checkpoints at the 24h limit and requeues up to 3 times.
#
# Usage:
#   bash configs/lr_omol4m/submit.sh tier1          # core comparison (5 runs)
#   bash configs/lr_omol4m/submit.sh tier2          # ablations (5 runs)
#   bash configs/lr_omol4m/submit.sh smoke          # 200-step throughput check
#   EXTRA="seed=1" bash configs/lr_omol4m/submit.sh tier1   # extra overrides
set -euo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
UMA=configs/lr_omol4m/uma_s_1p2p1.yaml
ALLSCAIP=configs/lr_omol4m/allscaip_sm.yaml
EXTRA=${EXTRA:-}

submit() {
  echo ">> fairchem -c $*"
  # shellcheck disable=SC2086
  fairchem -c "$@" $EXTRA
}

case "${1:-}" in
  tier1)
    submit $UMA uma_lr=none
    submit $UMA uma_lr=coulomb
    submit $UMA uma_lr=coulomb_6A
    submit $ALLSCAIP allscaip_lr=none
    submit $ALLSCAIP allscaip_lr=coulomb
    ;;
  tier2)
    submit $UMA uma_lr=coulomb_heis
    submit $UMA uma_lr=coulomb_equil
    submit $UMA uma_lr=coulomb_12A
    submit $ALLSCAIP allscaip_lr=coulomb_6A
    submit $ALLSCAIP allscaip_lr=coulomb_heis
    ;;
  smoke)
    for cfg in "$UMA uma_lr=coulomb" "$ALLSCAIP allscaip_lr=coulomb"; do
      # shellcheck disable=SC2086
      submit $cfg epochs=null steps=200 runner.evaluate_every_n_steps=100 \
        cluster.timeout_hr=1 ~job.logger
    done
    ;;
  *)
    echo "usage: $0 {tier1|tier2|smoke}" >&2
    exit 1
    ;;
esac
