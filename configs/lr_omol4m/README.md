# Long-range components on OMol-4M

Do long-range (LR) components help molecular MLIPs at the 4M-structure scale,
and is any gain due to *range* rather than extra head capacity? Two backbones,
each trained from scratch on the same data:

- **UMA-S-1.2.1 architecture**: 4 layers, 128 channels, lmax 2. Uses 8 MoLE
  experts where the release uses 64; the expert count does not change step
  time, only parameter count.
- **AllScAIP-small**: 6 layers, 512 hidden, about 34M parameters.

## Procedure

| Phase | Config | Forces | Precision | Epochs | lr | Loss (E:F) |
|---|---|---|---|---|---|---|
| 1 | `{uma,allscaip}_direct.yaml` | direct | bf16 | 60 | 8e-4 | 10:10 |
| 2 | `{uma,allscaip}_conserve.yaml` | energy gradient | fp32 (TF32 matmuls) | 10 | 4e-4 | 40:20 |

- **Phase 1.** LR terms add to the energy; forces come from a short-range
  direct head (UMA `Linear_Force_Head`, AllScAIP `AllScAIP_direct_force_head`).
- **Phase 2** starts from phase 1's `checkpoints/final/inference_ckpt.pt`. The
  backbone loads unchanged and the phase-1 energy head, including the LR
  charge networks, is copied into the conservative head through
  `initialize_finetuning_model(head_init_from=...)`. Loading is strict, so a
  mismatched head fails at startup instead of silently reinitializing. For
  AllScAIP, pass the same `allscaip_lr` arm in both phases.
- Phase hyperparameters live in `phase/direct.yaml` and `phase/conserve.yaml`.
  Set `runner.train_eval_unit.tf32=false` in phase 2 for strict IEEE fp32,
  at a large speed cost on A100.

## Arms

| Arm | UMA (`uma_lr=`) | AllScAIP (`allscaip_lr=`) | Question |
|---|---|---|---|
| Baseline | `none` | `none` | reference |
| Coulomb, all pairs | `coulomb` | `coulomb` | does LR electrostatics help? |
| Coulomb, 6 A (placebo) | `coulomb_6A` | `coulomb_6A` | is the gain from range, or from a charge-aware head? |
| Coulomb, 12 A | `coulomb_12A` | | how far does it need to reach? |
| + Heisenberg spin | `coulomb_heis` | `coulomb_heis` | spin coupling (open-shell, metal complexes) |
| + charge equilibration | `coulomb_equil` | | electronegativity/hardness prior |

Tier 1 (core claim) is the first three UMA arms plus the AllScAIP baseline and
Coulomb arms. Tier 2 is the rest. Add `seed=1` replicates for any arm whose
effect is close to run-to-run noise.

## Launch (Perlmutter login node, not inside salloc)

```bash
S=configs/lr_omol4m/submit.sh
bash $S tier1                                  # phase 1 for the tier-1 arms
bash $S direct uma coulomb_heis                # phase 1 for chosen arms
bash $S resume <run_dir>                       # continue after the requeue limit
bash $S conserve uma coulomb <phase1_run_dir>  # phase 2 once phase 1 finishes
```

`<run_dir>` is a run's timestamp directory under
`/pscratch/sd/s/santiago/lr_experiments`. Each job uses one node (4x A100)
through fairchem's SLURM mode, which checkpoints at the 24 h limit and
requeues itself up to 3 times (96 h). After that, `resume` resubmits from the
newest `checkpoints/step_*/resume.yaml`; the run keeps its directory, LR
schedule and step count. Runs log to wandb project `lr-omol4m`, grouped by
architecture and phase.

## Budget (one 4x A100 node per job)

One OMol-4M epoch is 218.7M atoms. Measured on one A100: UMA conservative
about 1,520 atoms/s in fp32 + TF32; AllScAIP conservative about 850 atoms/s in
bf16. Direct-force throughput is estimated from an A5000 at 2.3x conservative.

| | Phase 1 (60 ep, direct, bf16) | Phase 2 (10 ep, conservative, fp32) | Total |
|---|---|---|---|
| UMA | ~18.5k atoms/s, ~200 h, ~9 jobs | ~5.5k atoms/s, ~110 h, ~5 jobs | ~310 node-hours |
| AllScAIP | measure in the first hour | slower than UMA | measure |

Check each run's 4-GPU atoms/s in its first hour; the cosine schedule assumes
the run finishes all its epochs.

## Data and memory

- Train reads `mlip_data/train_4M`, val reads `mlip_data/val` (aselmdb).
  Energy and forces map to `omol_energy` / `omol_forces`; charge and spin
  come from `atoms.info`. The per-domain val splits (biomolecules,
  electrolytes, metal complexes, neutral organics) filter on `data_ids` in
  `val/metadata.npz`. Validation runs once per epoch.
- `max_atoms` is per GPU: 600 for UMA and 350 for AllScAIP (padded to 350 in
  phase 1). Keep it the same across arms of one architecture.

## Fixed choices (same for every arm)

- UMA uses `max_neighbors: 300`, effectively uncapped at 6 A. A tight
  short-range cap would let the 6 A placebo arm recover truncated neighbors
  and inflate the LR effect.
- Charges are constrained to each system's total charge (and spin, when
  Heisenberg is on) and computed in fp32 even under bf16 autocast.
- Molecules sit in 120 A vacuum boxes with `pbc=True`. The LR code detects
  that no periodic image is in reach and sums over exact intra-molecule
  pairs.

## Evaluation

Overall val MAE will understate LR effects, because most of each system's
energy is local. Compare arms on:

- the per-domain val splits logged during training
- OMol benchmarks that probe range, in `configs/uma/benchmark/`: distance
  scaling (`omol-scale.yaml`), ligand–pocket interaction (`omol-pocket.yaml`)
  and IE/EA (`omol-ieea.yaml`); AllScAIP equivalents are in
  `configs/allscaip/benchmarks/`
- force error vs system size, and for atoms more than 6 A from any charged group

## Known caveats

- In phase 1, LR physics reaches forces only through the shared backbone; the
  direct force head is short-range. Compare LR arms on forces after phase 2.
- The Heisenberg coupling J(r) is an unconstrained MLP of distance applied to
  every intra-molecule pair. It has no decay envelope, so watch the
  `coulomb_heis` arms for instability at large separations.
- AllScAIP runs uncompiled in both phases: compile is untested with the LR
  heads, and the LR gradient head does not support it.
