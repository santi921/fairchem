# Long-range components on OMol-4M

Do long-range (LR) components help molecular MLIPs at the 4M-structure scale,
and is any gain due to *range* rather than extra head capacity? Two backbones,
each trained from scratch on the same data with conservative (energy-gradient)
forces:

- **UMA-S-1.2.1 architecture**: 4 layers, 128 channels, lmax 2, 64 MoLE
  experts (291M total parameters). Matches the `uma-s-1p2p1` checkpoint
  config, whose weights load into the LR backbone with no missing keys.
- **AllScAIP-small**: 6 layers, 512 hidden, about 34M parameters.

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

## Launch (Perlmutter login node)

```bash
bash configs/lr_omol4m/submit.sh smoke   # 200 steps each: check data + throughput
bash configs/lr_omol4m/submit.sh tier1
bash configs/lr_omol4m/submit.sh tier2
```

Runs go to `/pscratch/sd/s/santiago/lr_experiments` and log to wandb project
`lr-omol4m`, grouped by architecture. Each arm uses one node (4x A100) through
fairchem's SLURM mode, which checkpoints at the 24h limit and requeues up to
3 times (72h total). `sbatch_local.sh` wraps one arm in a plain sbatch
allocation instead, with no automatic requeue.

## Before tier 1

1. **Data.** Train reads `mlip_data/train_4M` and val reads `mlip_data/val`
   as `ase_db` (aselmdb). Energy and forces are mapped to `omol_energy` and
   `omol_forces`, and charge/spin are read from `atoms.info`. The per-domain
   val splits (biomolecules, electrolytes, metal complexes, neutral organics)
   filter on `data_ids` in the val `metadata.npz`. If that key is missing,
   delete those splits from `dataset/omol_4M.yaml`.
2. **Budget.** One OMol-4M epoch is 218.7M atoms. UMA-S-1.2.1 measured
   about 1,520 atoms/s per A100 (fp32 + TF32, `max_atoms` 600), so an epoch
   takes about 11-12 h on one node including evals and checkpoints.
   `epochs: 5` (about 58 h) fits the 72 h requeue budget with margin. A run
   that times out never finishes its cosine schedule, so check the first
   hour's 4-GPU atoms/s before committing. All arms in a comparison must use
   the same `epochs`.
3. **Memory.** `max_atoms` is per GPU: 600 for UMA (about 15 GB activations
   with `max_neighbors: 300`) and 350 for AllScAIP. Lower it if a run hits
   OOM on 40 GB A100s, and keep it the same across arms of one architecture.

## Fixed choices (same for every arm)

- Conservative forces, energy:force loss ratio 2:1 (40/20, per-atom energy
  MAE plus L2 force), AdamW lr 4e-4 with cosine decay, fp32 with TF32 matmuls.
- UMA uses `max_neighbors: 300`, effectively uncapped at 6 A. A tight
  short-range cap would let the 6 A placebo arm recover truncated neighbors
  and inflate the LR effect.
- Charges are constrained to each system's total charge (and spin, when
  Heisenberg is on).
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

- The Heisenberg coupling J(r) is an unconstrained MLP of distance applied to
  every intra-molecule pair. It has no decay envelope, so watch the
  `coulomb_heis` arms for instability at large separations.
- The AllScAIP LR gradient head requires `wrap_property: False` and does not
  support `torch.compile`, so every AllScAIP arm runs uncompiled.
