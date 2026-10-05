#!/bin/bash
# Launch OMol-4M LR runs from a Perlmutter login node (not inside salloc).
# Each job runs on 1 node (4x A100) via fairchem's SLURM mode, which
# checkpoints at the time limit and requeues up to 3 times; continue longer
# runs with `resume`.
#
# Usage:
#   submit.sh direct   <uma|allscaip> <arm> [<arm> ...]   # phase 1, 60 epochs
#   submit.sh conserve <uma|allscaip> <arm> <phase1_run>   # phase 2, 10 epochs
#   submit.sh resume   <run_dir>                           # continue a run
#   submit.sh tier1                                        # phase 1, core arms
#   submit.sh tier2                                        # phase 1, ablations
#
# <arm> is a file name in uma_lr/ or allscaip_lr/ (e.g. coulomb, coulomb_6A).
# <phase1_run> / <run_dir> is a run's timestamp directory, e.g.
#   /pscratch/sd/s/santiago/lr_experiments/202610-0413-1532-8dcb
# Extra overrides go in EXTRA, e.g. EXTRA="seed=1" submit.sh direct uma coulomb
set -euo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
D=configs/lr_omol4m
EXTRA=${EXTRA:-}

group() { [ "$1" = uma ] && echo uma_lr || echo allscaip_lr; }

submit() {
  echo ">> fairchem -c $*"
  # shellcheck disable=SC2086
  fairchem -c "$@" $EXTRA
}

direct() {
  local arch=$1
  shift
  for arm in "$@"; do
    submit "$D/${arch}_direct.yaml" "$(group "$arch")=$arm"
  done
}

conserve() {
  local arch=$1 arm=$2 run=$3
  local ckpt="$run/checkpoints/final/inference_ckpt.pt"
  [ -f "$ckpt" ] || { echo "missing $ckpt (has phase 1 finished?)" >&2; exit 1; }
  if [ "$arch" = uma ]; then
    # LR settings come from the checkpoint; the arm only names the run
    submit "$D/uma_conserve.yaml" "starting_checkpoint=$ckpt" \
      "job.run_name=uma_omol4m_${arm}_conserve_s0"
  else
    submit "$D/allscaip_conserve.yaml" "allscaip_lr=$arm" "starting_checkpoint=$ckpt"
  fi
}

resume() {
  local run=$1 latest
  latest=$(ls -td "$run"/checkpoints/step_*/ 2>/dev/null | head -1)
  [ -n "$latest" ] || { echo "no step_* checkpoints in $run/checkpoints" >&2; exit 1; }
  submit "${latest%/}/resume.yaml"
}

case "${1:-}" in
  direct) shift; direct "$@" ;;
  conserve) shift; conserve "$@" ;;
  resume) shift; resume "$@" ;;
  tier1)
    direct uma none coulomb coulomb_6A
    direct allscaip none coulomb
    ;;
  tier2)
    direct uma coulomb_heis coulomb_equil coulomb_12A
    direct allscaip coulomb_6A coulomb_heis
    ;;
  *)
    sed -n '2,19p' "$0" >&2
    exit 1
    ;;
esac
