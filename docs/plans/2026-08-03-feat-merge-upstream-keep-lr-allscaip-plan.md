---
title: "feat: Merge upstream main keeping LR/LES/AllScAIP-LR work"
type: feat
date: 2026-08-03
---

# Merge upstream/main into v2_esen (keeping LR / LES / AllScAIP-LR work)

## Context

`v2_esen` carries the long-range electrostatics research stack (LES, Ewald, charge
equilibration, BEC, `LRChargePredictor`, AllScAIP-LR, EScAIP additions) and is
**55 commits ahead / 104 commits behind** `upstream/main` (FAIR-Chem/fairchem).
Last sync point: merge-base `eb817d65c` (2026-03-26). Goal: bring in ~4 months of
upstream work (torch 2.13, GP all-to-all, mixed PBC, perf work, upstream AllScAIP
tests, `TrainCheckpointCallback` move) without losing any branch-only LR components.

A dry-run `git merge-tree` plus a targeted API audit produced the findings below.
User decisions already made: **repair** `escn_md_les.py` (don't delete), and
**remove** all three global `torch.set_float32_matmul_precision("high")` calls.

## Textual conflicts (only 4 files, all trivial — resolve by taking upstream)

| File | Conflict | Resolution |
|---|---|---|
| `src/fairchem/core/components/train/train_runner.py` | Upstream moved `TrainCheckpointCallback` → `components/callbacks.py` (#2057); our only change there was ruff formatting | Take upstream (class now lives in `callbacks.py`) |
| `src/fairchem/core/models/allscaip/AllScAIP.py` | Upstream added license header + removed constructor-level `set_float32_matmul_precision` (#2108) | Take upstream both hunks |
| `src/fairchem/core/models/allscaip/README.md` | Rename `radius_graph_v2.py` → `allscaip_radius_graph.py` + pretrained-model table | Take upstream |
| `src/fairchem/core/models/allscaip/utils/data_preprocess.py` | Import path of the (byte-identical) radius-graph module | Take upstream import |

`CLAUDE.md` also merges — keep upstream's new matmul-precision section AND our
local "Shared Skills" section; update the `torch~=2.8.0` mention to 2.13.

Verified non-issues:
- Branch's `radius_graph_v2.py` is **byte-identical** to upstream's `allscaip_radius_graph.py`.
- `InferenceSettings.max_atoms` added identically on both sides → merges to a single field (verify once post-merge).
- Registry names for AllScAIP are identical on both sides, same file path → no collision.
- `models/__init__.py`: upstream unchanged (empty) → our version survives; its `escn_md` imports all still exist upstream.
- #1960 "spin/charge no longer core fields" is **torch-sim only**; `AtomicData.charge/spin` are unchanged → LR charge code unaffected.
- `mlip_unit.py`: our changes purely cosmetic; upstream added orthogonal `tf32` plumbing → clean.
- `modules/loss.py`, `graph/compute.py`, `evaluator.py`, `_metrics.py` (r2 metric): ours-only changes, upstream untouched.
- All branch-only file sets (`escn_md_lr.py`, `models/les/**`, `models/utils/lr.py`, `lr_charges.py`, `AllScAIP_lr.py`, `allscaip/utils/lr_utils.py`, escaip module additions) survive untouched.

## Semantic breakage to fix post-merge (auto-merges, then breaks at runtime)

### CRITICAL 1 — `node_offset` → `scatter_target` API rewrite (GP all-to-all work)
Upstream changed `EdgeDegreeEmbedding.forward` (5→4 args; arg 3 is now `scatter_target: [E]`
instead of `edge_index: [2,E]`) and `eSCNMD_Block.forward` (`node_offset: int` →
`scatter_target: Tensor|None, gp_ctx: GPContext|None`). Branch call sites that break:
- `src/fairchem/core/models/uma/escn_md_lr.py:493-498` and `:502-513` (`node_offset=` at :512)
- `src/fairchem/core/models/uma/escn_md_les.py:517-524` and `:531-541` (also has pre-existing stale args — repair fully per user decision)
- Transitively `eSCNMDMoeBackboneLR` in `escn_moe.py` (used by `configs/uma/lr/backbone/small_lr.yaml`)

Fix pattern: mirror upstream `escn_md.py:729-755, 856-859, 885` — compute
`scatter_target` (defaults to `graph_dict["edge_index"][1]` when GP off) and pass it through.
**No existing test covers these forwards** — add a smoke test that runs
`eSCNMDBackboneLR` (and ideally `eSCNMDMoeBackboneLR`) forward.

### CRITICAL 2 — `regress_config` now hard-required (#1928)
`MLIPPredictUnit` (`units/mlip_unit/predict.py:204, :488`) unconditionally reads
`backbone.regress_config.direct_forces`. Branch backbones lacking it:
- `eSCNMDBackboneLR` (`escn_md_lr.py:115-117`)
- `eSCNMDBackboneLES` (`escn_md_les.py:131-133`)
- `EScAIPBackbone` (`escaip/EScAIP.py:63-65`)

Every LR inference path (calculator, predict runners) dies at init. Fix: give each a
`GradRegressConfig` (copy the pattern from `allscaip/AllScAIP.py:77-81`), keeping the
legacy plain attributes for the LR heads that read them. This also restores
`HydraInterfaceMixin.skip_property` behavior (`base.py:187-192` now only checks
`regress_config`, so direct-force LR models would otherwise silently stop skipping
inference-only derivative tasks → missing/NaN outputs).

### CRITICAL 3 — global TF32 mutations (user chose: remove all)
Remove `torch.set_float32_matmul_precision("high")` from:
- `src/fairchem/core/models/__init__.py:25` (fires on any models import, process-wide)
- `src/fairchem/core/models/allscaip/AllScAIP.py:121-122` (conflict resolution handles it)
- `src/fairchem/core/models/escaip/EScAIP.py:125-126`
- test copies: `tests/core/models/allscaip/test_forward.py:35`, `tests/core/models/escaip/test_forward.py:32` (if these files survive dedup)

TF32 for training runs is now opt-in via the `tf32` flag upstream added to
`MLIPTrainEvalUnit` — note in the summary which dev configs may want `tf32: true`.

### Cleanup / dedup (do in the merge commit or immediately after)
- Delete `src/fairchem/core/models/allscaip/utils/radius_graph_v2.py` (dup of `allscaip_radius_graph.py`).
- Delete branch's `tests/core/models/allscaip/test_forward.py` (upstream's `test_allscaip_forward.py` supersedes it).
- Delete `src/fairchem/core/calculate/ase_calculatorOG.py` (stale copy; reads removed `backbone.direct_forces` → AttributeError for UMA checkpoints).
- Fix duplicate registration: `escn_md_les.py:619` and `:755` both register `"esen_efs_head_les"` (second silently wins) — rename one (pre-existing bug, fix while repairing LES).
- `git checkout upstream/main -- configs/uma/training_release/` — our branch accidentally corrupted a head key (`energyandforcehead:` → `:`) in `uma_sm_conserve_finetune.yaml`; upstream's copies are intact and already include our `pass_through_head_outputs` addition.
- Optional: add `escn_moe` / `escn_md_les` imports to `models/__init__.py` so short registry names (`escnmd_moe_backbone_lr`, `escnmd_backbone_les`) resolve regardless of import order (pre-existing order-dependent KeyError).

### Watch items (no action unless they bite)
- torch 2.8→2.13 pin comes with the merge; branch LR torch APIs audited — all stable.
  Two things to exercise, not statically provable: `torch.vmap` under `torch.compile` in
  `allscaip_radius_graph.py:301,401` (run the allscaip tests), and confirm
  `models/utils/lr.py:145` `torch.meshgrid` passes `indexing="ij"`.
- Don't re-pin `nvalchemi-toolkit-ops==0.2.0`; take upstream pins (`torch~=2.13.0`,
  `torch-sim-atomistic>=0.6.0`, `requires-python <3.15`).
- Branch `# noqa: TC001` renames in `mlip_unit.py` may upset the pinned ruff 0.5.1 — revert to `TCH001` if pre-commit complains.

## Execution plan

### Phase 0 — Secure working tree
0. Persist this plan into the repo as
   `docs/plans/2026-08-03-feat-merge-upstream-keep-lr-allscaip-plan.md`
   (same convention as the 2026-03-30 merge plan) so it can be referenced during and after the merge; commit it with the working-tree cleanup below.
1. Commit uncommitted work (6 config files under `configs/escaip/` + `dev/configs/OMol25/`,
   updated plan doc, untracked `docs/plans/2026-03-31-research-lr-components-at-scale.md`).
   None of these paths conflict with upstream.
2. Push `v2_esen` to origin as a safety point.

### Phase 1 — Merge
3. `git merge upstream/main` (no rebase — history has merge knots).
4. Resolve the 4 conflicts per table (take upstream); merge `CLAUDE.md` keeping both sides' additions.
5. Apply the dedup/cleanup deletions and `configs/uma/training_release/` reset inside the merge commit.
6. Commit the merge.

### Phase 2 — Fix semantic breakage (separate commits after the merge)
7. `regress_config` on `eSCNMDBackboneLR` / `eSCNMDBackboneLES` / `EScAIPBackbone`.
8. `scatter_target` migration in `escn_md_lr.py` forward; full repair of `escn_md_les.py`
   forward (stale arg lists) + fix the duplicate `esen_efs_head_les` registration.
9. Remove the three global TF32 calls (+ test copies).
10. Add a forward smoke test for `eSCNMDBackboneLR` (currently zero coverage of its forward).

### Phase 2.5 — Environment update to torch 2.13 (explicit step, user-requested)
11. After the merge brings in the new pins, upgrade the environment:
    `pip install -e packages/fairchem-core[dev]` (pulls `torch~=2.13.0`,
    `torch-sim-atomistic>=0.6.0`, `nvalchemi-toolkit-ops>=0.3.0`).
    Verify with `python -c "import torch; print(torch.__version__)"` and confirm
    CUDA availability before running the GPU-relevant smoke tests / training run.

### Phase 3 — Verification
- `python -c "import fairchem.core"`; import each branch-only module explicitly
  (`escn_md_lr`, `escn_md_les`, `models.les`, `allscaip.AllScAIP_lr`, `escn_moe`).
- `pytest tests/core/models/test_lr.py -c packages/fairchem-core/pyproject.toml`
- `pytest tests/core/models/allscaip -c packages/fairchem-core/pyproject.toml` (includes upstream's new tests + branch's `test_allscaip_lr.py`, exercises the vmap/compile path)
- New `eSCNMDBackboneLR` forward test passes.
- Broader smoke: `pytest tests/core/models tests/core/units -c packages/fairchem-core/pyproject.toml -m "not gpu"` (as time allows).
- 50-step LR training smoke run with `dev/configs/OMol25/fair_direct_4M_local_lr.yml`
  (mirrors the 2026-03-30 merge verification).
- `pre-commit run --files <all touched files>`.

Note: the environment likely needs `torch 2.13` installed to run verification
(`pip install -e packages/fairchem-core[dev]` after the merge).
