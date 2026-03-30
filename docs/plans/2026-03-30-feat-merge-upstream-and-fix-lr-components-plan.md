---
title: "feat: Merge upstream main and fix LR components"
type: feat
date: 2026-03-30
---

# Merge Upstream main and Fix LR Components

## Overview

Merge 146 upstream commits from `facebookresearch/fairchem/main` into `santi921/fairchem/v2_esen`, then ensure the long-range (LR) electrostatic/spin components work correctly with the updated codebase. The LR feature adds Coulomb electrostatics, Ewald summation, Heisenberg spin coupling, and charge/spin renormalization to the eSCNMD model backbone.

## Problem Statement / Motivation

The `v2_esen` branch adds LR interaction components (charges, Ewald summation, Heisenberg coupling) to eSCNMD, but has fallen 146 commits behind upstream. Upstream has significantly refactored `escn_md.py` (3584 changed lines across 15 commits), adding Triton kernels, quaternion-based Wigner D, hessian support, refactored MLP heads, and a new `outputs.py` module. The branch also has broken tests, debug print statements, and several critical bugs in the LR head classes that must be fixed.

## Proposed Solution

A phased approach: (1) clean merge, (2) fix broken imports and basic training, (3) audit and fix LR logic, (4) enable LR training, (5) write proper tests.

## Broad Overview of Upstream Changes (146 commits)

**Total: 409 files changed, +38,971 / -22,740 lines**

### Model Architecture & Performance (High Impact)

| Commit | Change | Impact on LR Branch |
|--------|--------|---------------------|
| `4e7f5228` Refactor MLP Energy/EFS Heads (#1734) | Extracted shared helper functions from head classes into `outputs.py` (`compute_energy`, `compute_forces`, `compute_forces_and_stress`, `reduce_node_to_system`) | **Critical** — LR heads must adopt these helpers or they'll diverge from upstream head patterns |
| `8f431ae6` Add Hessian matrix calculation (#1735) | Added hessian support to heads | Low — LR heads don't need hessian initially |
| `de1df4f8` Umas fast GPU backend (#1826) | New Triton kernels for wigner transforms, `execution_backends.py` | Medium — backbone forward pass now dispatches to Triton; LR backbone must support this |
| `e468b645` Quaternion-based Wigner D (#1771) | New `quaternion/` module for handling y-aligned edge edge case | Medium — changes backbone graph processing |
| `5f9aa799` Consolidate turbo and turbo_umas (#1898) | Merged turbo variants into main backbone | Medium — backbone class may have different constructor params |
| `c45dbcde` Optimize md torch compile (#1892) | Compile optimizations | Low — may affect graph breaks in LR code |
| `08f7b0b8` UnifiedRadialMLP (#1831) | Batched radial computation module | Low — new module, no direct conflict |
| `ab247c59` UMA-S 1.2 (#1861) | New model release with updated defaults | Low — new configs/checkpoints |

### Embeddings & Features

| Commit | Change | Impact on LR Branch |
|--------|--------|---------------------|
| `9d8a66ff` Embeddings update (#1765) | **Renamed `embedding_dev.py` -> `embedding.py`** | **Critical** — all branch files importing `embedding_dev` will break |
| `2a50de36` Channel charge and spin (#1768) | Changes to charge/spin handling | **High** — directly relevant to LR charge prediction |
| `7ec5e934` Fix graph break in ChgSpinEmbedding (#1907) | Compile fix for charge/spin embeddings | Medium — relevant if using `rand_emb` |
| `3eae94d0` Composition dropout (#1767) | New training regularization | Low |

### Infrastructure & Testing

| Commit | Change | Impact |
|--------|--------|--------|
| `36b6ff03` Claude init (#1783) | **Added upstream `CLAUDE.md`** | Should adopt their CLAUDE.md and merge with local additions |
| `62087c4f` Early Partition and Parallel Graph Gen (#1630) | New graph generation pipeline | Medium — affects backbone's graph construction |
| `8f74b9ed` NVIDIA graph gen support (#1737) | Alternative graph gen backend | Low |
| `d76d689c` Refactor FAIRChemCalculator (#1731) | Model-agnostic calculator | Low — inference path |
| `62e8827a` Refactor MLIPPredictUnit (#1727) | Model-agnostic prediction | Low |
| `56677783` MD runner abstractions (#1835) | New MD runner patterns | Low |
| `eb77d0de` Generic MD Runner (#1782) | Generic MD runner | Low |

### Dependencies & Compatibility

| Commit | Change |
|--------|--------|
| `cd0636e2` numpy 2.4 (#1862) | numpy 2.4 compatibility |
| `ac340073` e3nn 0.6.0 (#1798) | e3nn version bump |
| `d582b843` ase 3.28.0 (#1912) | ASE version bump |
| `82fdc5a5` numpy/numba upgrade (#1537) | numba compatibility |

### Documentation & Tutorials

| Commit | Change |
|--------|--------|
| `932ca6a3` UMA catalysis tutorial (#1667) | New tutorial |
| `9c398155` Jupyterbook>=2 docs refactor (#1627) | Docs infrastructure change |
| `e646dafa` Changelogs for UMA (#1866) | Release notes |
| `5a70a280` Update docs (#1865) | General docs |

### Bug Fixes (relevant to us)

| Commit | Change |
|--------|--------|
| `67fff240` Default untrained predictions (#1883) | Heads return defaults when untrained |
| `d0298207` Enable untrained property predictions (#1811) | Related to above |
| `1d8a39d3` Single atom bug fix (#1882) | Edge case for single-atom systems |
| `088a9b11` Refactor single atoms (#1732) | Related |
| `d5c6cc5d` Reduce at higher precision (#1889) | Numerical precision in reductions |
| `c4244586` Fix per atom MAE bug (#1825) | Metrics fix |

### Key Takeaway

The most impactful upstream changes for this branch are:
1. **Head refactoring** — `outputs.py` with shared helpers means LR heads should use `compute_energy()`, `compute_forces()` etc.
2. **Embeddings rename** — `embedding_dev` -> `embedding` is a hard break
3. **Channel charge and spin** (#1768) — upstream now has charge/spin channel support that may overlap or complement the LR charge prediction approach
4. **Upstream CLAUDE.md** — should be adopted (it has stricter pre-commit requirements)

## Upstream Model Architecture Changes (UMA/eSCNMD)

This section documents what changed structurally in the eSCNMD backbone and head classes. The upstream `escn_md.py` went from **2356 lines** (this branch) to **1407 lines** — a significant simplification.

### Backbone: `eSCNMDBackbone`

**New constructor parameters (not in this branch):**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `direct_stress` | `bool` | `False` | Direct stress prediction (not autograd) |
| `regress_hessian` | `bool` | `False` | Enable Hessian matrix computation |
| `hessian_vmap` | `bool` | `True` | Use vmap for Hessian (faster) |
| `dataset_mapping` | `dict[str,str]\|None` | `None` | Maps config dataset names to embedding names (replaces `dataset_list`) |
| `use_cuda_graph_wigner` | `bool` | `False` | CUDA graph optimization for Wigner D |
| `use_quaternion_wigner` | `bool` | `True` | New quaternion-based Wigner D method (fixes y-aligned edge edge case) |
| `charge_balanced_channels` | `list[int]\|None` | `None` | Which embedding channels to charge-balance |
| `spin_balanced_channels` | `list[int]\|None` | `None` | Which embedding channels to spin-balance |
| `execution_mode` | `str` | `"general"` | Dispatch to Triton/fast backends |

**Removed parameters:**
- `edge_chunk_size: int | None = None` → now `edge_chunk_size: int = 1` (always set, auto-computed)

**New `GradRegressConfig` dataclass:** Replaces the old `direct_forces`/`regress_forces`/`regress_stress` boolean soup with a structured config:
```python
@dataclass
class GradRegressConfig:
    direct_forces: bool = False
    direct_stress: bool = False
    forces: bool = False
    stress: bool = False
    hessian: bool = False
    hessian_vmap: bool = True
```
The backbone stores `self.regress_config = GradRegressConfig(...)` and heads read from it.

**New `balance_channels()` method:** Called after every message-passing layer. Enforces that certain embedding channels sum to target charge/spin per batch. This is upstream's approach to charge/spin consistency — operates at the embedding level, not at the output level like this branch's `batch_spin_charge_renormalization()`. Key difference: upstream balances embeddings during message passing; this branch renormalizes predicted scalar charges after the head MLP.

**New `balance_channels_batched()` standalone function:** The actual math — subtracts the per-system mean from selected channels, adds the target divided by natoms. Operates on the full `[N, (lmax+1)^2, C]` embedding tensor.

**New Wigner D path:** `use_quaternion_wigner=True` (default) uses `axis_angle_wigner_hybrid()` from new `common/quaternion/` module. Fixes a numerical issue with y-aligned edges that the Euler angle method had. Falls back to the old Euler path when `use_quaternion_wigner=False`.

**New execution backend dispatch:** `self.backend = get_execution_backend(execution_mode)` returns an `ExecutionBackend` that dispatches rotation/scatter operations. Three modes:
- `"general"` — standard PyTorch (default)
- `"umas_fast_pytorch"` — fused operations
- `"umas_fast_gpu"` — Triton custom kernels (new `triton/` directory)

**Simplified forward output:** Now returns only `{"node_embedding": x_message, "batch": batch}`. No longer returns `displacement` or `orig_cell` — those are handled by heads via data dict directly.

**New methods:**
- `build_inference_settings(cls, settings)` — builds config overrides from inference settings
- `get_default_untrained_tasks(self, checkpoint_tasks, inference_settings)` — returns default tasks for untrained properties (e.g., auto-add stress for energy-only checkpoints)

### Heads: Major Refactoring

**New `outputs.py` module with shared helpers:**

| Function | Purpose |
|----------|---------|
| `get_l_component_range(x, l_min, l_max)` | Extract spherical harmonic components by L range |
| `reduce_node_to_system(node_values, batch, num_systems)` | Sum node values to system level, in **float64** precision |
| `compute_energy(emb, energy_block, batch, num_systems, ...)` | Full energy computation pipeline |
| `compute_forces(energy_part, pos, training)` | Autograd force computation |
| `compute_forces_and_stress(energy_part, pos, cell, ...)` | Autograd force + stress computation |
| `compute_hessian(forces, pos, vmap, training)` | Hessian matrix computation |

**`MLP_EFS_Head` — now the primary head class:**
- Constructor takes `reduce`, `prefix`, `wrap_property` params
- Uses `compute_energy()`, `compute_forces()`, `compute_forces_and_stress()`, `compute_hessian()` from `outputs.py`
- Supports hessian computation
- Uses `self.regress_config` (from backbone) instead of separate booleans

**`MLP_Energy_Head` — now inherits from `MLP_EFS_Head`:**
```python
class MLP_Energy_Head(MLP_EFS_Head):
    """Deprecated: use MLP_EFS_Head with regress_forces=False."""
```
Just validates that forces/stress are disabled, delegates everything to parent.

**`Linear_Energy_Head` — simplified:**
- No LR code at all (clean 20-line class)
- Uses `compute_energy()` from `outputs.py`

**`Linear_Force_Head` — uses `get_l_component_range()`:**
- Extracts L=0,1 components, applies `SO3_Linear`, extracts L=1 as forces

**`MLP_Stress_Head` — cleaner decomposition:**
- Uses `get_l_component_range()` and `reduce_node_to_system()` from outputs.py
- Separate scalar (L=0) and anisotropic (L=2) paths

### Key Structural Differences: Branch vs Upstream

| Aspect | This Branch | Upstream |
|--------|-------------|----------|
| Backbone count | 2 (`eSCNMDBackbone` + copied `eSCNMDBackboneLR`) | 1 (`eSCNMDBackbone`) |
| Head count | 9 (including 3 LR variants) | 5 (no LR variants) |
| Force/stress config | Separate booleans | `GradRegressConfig` dataclass |
| Charge balancing | `batch_spin_charge_renormalization()` on head output | `balance_channels()` on embeddings during message passing |
| Energy computation | Inline `index_add_` in each head | Shared `compute_energy()` in outputs.py |
| Wigner D method | Euler angles only | Quaternion (default) + Euler fallback |
| Execution backends | None | 3 modes (general, fast pytorch, triton) |
| Embedding import | `embedding_dev` | `embedding` |
| `torch_scatter` | Used in heads and lr.py | Removed — uses PyTorch builtins |
| Precision | Input dtype | `float64` for energy reduction |
| Hessian support | None | Full support |
| `escn_md.py` LOC | 2356 | 1407 |

### Implications for LR Re-integration

When re-applying LR code on the upstream backbone:

1. **`eSCNMDBackboneLR` should inherit from `eSCNMDBackbone`** rather than copying it. The LR backbone only needs ~10 extra `__init__` params and a small `forward()` override to add `edge_index_lr`.

2. **LR heads should use `outputs.py` helpers** — call `compute_energy()` for SR energy, then add LR energy on top. This gets float64 precision for free and reduces code.

3. **`regress_config` must be used** instead of `self.regress_forces` / `self.regress_stress` booleans.

4. **Consider whether `balance_channels` overlaps with `batch_spin_charge_renormalization`** — upstream balances charge/spin at the embedding level; this branch balances at the output level. These may be complementary (embedding-level for SR, output-level for LR) or redundant.

5. **Replace `torch_scatter` with PyTorch builtins** — upstream has removed this dependency.

6. **Use `get_l_component_range()` instead of `emb["node_embedding"].narrow(1, 0, 1).squeeze()`**.

## Technical Considerations

### Pre-Merge: Adopt Upstream CLAUDE.md

Upstream added a `CLAUDE.md` at commit `36b6ff03`. It has stricter requirements than our local version:
- Requires `pre-commit run --files` on every modified file before committing
- Requires Meta copyright header on all files
- Has specific docstring formatting rules (no text on opening/closing quote lines)
- Uses `pytest tests -c packages/fairchem-core/pyproject.toml` syntax

**Recommendation:** Before merging, either stash or commit our local `CLAUDE.md` (currently untracked). During merge, adopt upstream's CLAUDE.md since it reflects the project's official conventions, and append any local additions (LR-specific guidance) as a separate section.

### Merge Conflict Hotspot: `escn_md.py`

This is the single biggest risk. The file has:
- **Upstream:** 15 commits, 3584 changed lines (new imports, refactored heads, Triton backends, hessian support, consolidated turbo models)
- **Branch:** Added `eSCNMDBackboneLR` backbone, 3 LR head classes (`MLP_EFS_Head_LR`, `MLP_Energy_Head_LR`, `Linear_Energy_Head`), and LR utility imports

**Recommended merge strategy:** Accept upstream's `escn_md.py` entirely, then re-apply the LR additions (backbone class + head classes + imports) on top of the new upstream code. This is safer than resolving a 3-way merge of 3584 lines.

### Files That Only Exist on This Branch (No Conflict Risk)

| File | Purpose |
|------|---------|
| `src/fairchem/core/models/utils/lr.py` | Core LR functions (Coulomb, Ewald, Heisenberg, renormalization) |
| `src/fairchem/core/models/les/` | LES model with production Ewald module |
| `src/fairchem/core/models/uma/escn_md_les.py` | LES backbone/heads |
| `src/fairchem/core/models/esen/` | ESEN model directory |
| `src/fairchem/core/models/uma/nn/embedding_dev.py` | Dev embedding (must be migrated to `embedding.py`) |
| `tests/core/models/test_lr.py` | Broken LR tests (must be rewritten) |

### Critical Upstream Renames

- `embedding_dev.py` -> `embedding.py` (upstream uses `embedding.py`; this branch imports `embedding_dev`)
- `torch_scatter.scatter_add` replaced with other patterns in upstream

### Known Bugs to Fix (Discovered During Research)

1. **`Linear_Energy_Head` typed to wrong backbone:** Declares `backbone: eSCNMDBackbone` but accesses LR attributes (`conv_function_tf`, `hidden_channels_lr`, etc.). Must be `eSCNMDBackboneLR`.
2. **`Linear_Energy_Head` missing `use_ewald_tf`:** Never set, but referenced in `get_lr_energies()`.
3. **`det_cells` used before assignment:** All 3 LR heads have `if data["cell"] is not None: det_cells = ...` followed by `if torch.any(det_cells < 1e-6)` — crashes when `cell` is None.
4. **Ewald per-atom vs per-batch shape mismatch:** `potential_full_ewald_batched()` returns a flat vector, but the energy distribution logic (`pot_b = factors / volume - q_b**2 / ...`) broadcasts incorrectly — `factors` is per-k-vector while `q_b**2` is per-atom.
5. **Debug `print()` statements:** 7+ active print statements across `lr.py` and head forward passes that will flood training logs.
6. **Broken test file:** `test_lr.py` has wrong import path (`fairchem.core.models.models.utils.lr`) and references `self.batch` outside a class.
7. **HACK block in `get_lr_energies()`:** 40-line commented-out numpy dump block (lines 1568-1609).

### Architecture: LR Component Integration Points

```
eSCNMDBackboneLR (registered: "escnmd_backbone_lr")
  ├── Extra params: hidden_channels_lr, heisenberg_tf, latent_charge_tf,
  │   conv_function_tf, normalize_charges_tf, equil_charges_tf, use_ewald_tf,
  │   cutoff_lr, return_bec, lr_output_scaling_factor
  └── forward() -> emb dict (same as base, no LR logic in backbone)

MLP_EFS_Head_LR (registered: "esen_efs_head_lr")
  ├── get_charges(node_features, data) -> {charges, charges_raw, spin, hardness, electroneg}
  ├── get_lr_energies(emb, data) -> {energy, energy_spin, charges}
  └── forward(data, emb) -> {energy, forces, stress} (autograd-based)

MLP_Energy_Head_LR (registered: "esen_mlp_energy_head_lr")
  ├── get_charges(), get_lr_energies() (duplicated from above)
  └── forward(data, emb) -> {energy} (direct, no autograd)

Linear_Energy_Head (registered: "esen_linear_energy_head")
  ├── get_charges(), get_lr_energies() (duplicated, with bugs)
  └── forward(data, emb) -> {energy}
```

### Key LR Functions in `lr.py`

| Function | Purpose | Status |
|----------|---------|--------|
| `potential_full_from_edge_inds()` | Direct Coulomb sum for non-periodic | Mostly correct, needs print cleanup |
| `potential_full_ewald_batched()` | Ewald summation for periodic | Has shape bugs, debug prints |
| `heisenberg_potential_full_from_edge_inds()` | Spin-spin coupling via learned NN | Appears correct |
| `batch_spin_charge_renormalization()` | Constrain total charge/spin per batch | Appears correct |

### Open Design Questions

- **Q1:** Should `cutoff_lr` actually generate a separate long-range graph? Currently defined but unused — LR uses the same edge set as short-range.
- **Q2:** Should the two Ewald implementations (`lr.py` vs `les/module/ewald.py`) be consolidated? The LES version is more robust.
- **Q3:** Should the three duplicated `get_charges()`/`get_lr_energies()` methods be refactored into a shared mixin?

## Acceptance Criteria

### Phase 1: Merge
- [ ] Upstream `facebookresearch/fairchem/main` merged into branch
- [ ] All imports resolve (`python -c "import fairchem.core"` succeeds)
- [ ] No merge conflict markers in any file
- [ ] `ruff check src/` passes

### Phase 2: Base Training
- [ ] Training runs with standard (non-LR) backbone + head configs
- [ ] Config: `escnmd_backbone` + `esen_efs_head` (or upstream equivalent after refactoring)

### Phase 3: LR Logic Review
- [ ] All debug `print()` statements removed from `lr.py` and head classes
- [ ] HACK block removed from `get_lr_energies()`
- [ ] `Linear_Energy_Head` type annotation fixed to `eSCNMDBackboneLR`
- [ ] `Linear_Energy_Head` `use_ewald_tf` attribute added
- [ ] `det_cells` unbound variable bug fixed in all 3 heads
- [ ] Ewald `potential_full_ewald_batched()` shape bug fixed
- [ ] `embedding_dev` imports updated to `embedding`
- [ ] `from __future__ import annotations` added to LES module files

### Phase 4: LR Training
- [ ] Non-periodic LR training runs: `fair_direct_4M_local_lr_experiment_non_periodic_small.yml`
- [ ] Periodic LR training runs: `fair_direct_4M_local_lr_experiment_periodic_small.yml`
- [ ] Heisenberg training compiles (if applicable config exists)
- [ ] `torch.compile` does not break with LR components

### Phase 5: Tests
- [ ] Fix import path in `test_lr.py` (`fairchem.core.models.utils.lr`)
- [ ] Unit test: `potential_full_from_edge_inds()` with known 2-atom system
- [ ] Unit test: `potential_full_ewald_batched()` with simple periodic cell
- [ ] Unit test: `heisenberg_potential_full_from_edge_inds()` with known coupling
- [ ] Unit test: `batch_spin_charge_renormalization()` verifies charge/spin conservation
- [ ] Tests pass with `pytest tests/core/models/test_lr.py`

## Success Metrics

1. Clean merge with no regressions to existing tests (`pytest tests/` passes)
2. Base training completes 1+ epoch without errors
3. LR training (both periodic and non-periodic) completes 1+ epoch
4. All new LR tests pass
5. No debug print statements in production code paths

## Dependencies & Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| `escn_md.py` merge conflicts are massive | High — could take significant manual effort | Accept upstream version, re-apply LR code on top |
| Upstream refactored head patterns (new `outputs.py`) | Medium — LR heads may need to use new patterns | Examine upstream's refactored heads as templates |
| Ewald shape bug causes silent incorrect energies | High — training would converge to wrong values | Unit test against analytical Madelung constant |
| `torch_scatter` removed upstream | Medium — `lr.py` uses `from torch_scatter import scatter` | Check if upstream provides a replacement |
| Config YAML keys may have changed upstream | Low — configs may need updating | Compare upstream backbone YAML schema |

## References & Research

### Internal References
- LR backbone: `src/fairchem/core/models/uma/escn_md.py:688` (`eSCNMDBackboneLR`)
- LR functions: `src/fairchem/core/models/utils/lr.py`
- LR head (EFS): `src/fairchem/core/models/uma/escn_md.py:1371` (`MLP_EFS_Head_LR`)
- LR head (Energy): `src/fairchem/core/models/uma/escn_md.py:1770` (`MLP_Energy_Head_LR`)
- LR head (Linear): `src/fairchem/core/models/uma/escn_md.py:2029` (`Linear_Energy_Head`)
- LES Ewald module: `src/fairchem/core/models/les/module/ewald.py`
- LES backbone: `src/fairchem/core/models/uma/escn_md_les.py`
- Existing broken tests: `tests/core/models/test_lr.py`
- Training configs: `dev/configs/OMol25/fair_direct_4M_local_lr_experiment_*.yml`
- Job script: `job_test.sh`

### Key Upstream Commits Affecting `escn_md.py`
- `fef6eccc` Regress model attrs patch (#1925)
- `5f9aa799` Consolidate turbo and turbo_umas (#1898)
- `c45dbcde` Optimize md torch compile (#1892)
- `4e05b968` Refactor MLP Energy/EFS Heads with Shared Helper Functions (#1734)
- `8f431ae6` Add Hessian matrix calculation support (#1735)
- `e468b645` Quaternion-based Wigner D method for fixing y-aligned edges (#1771)
- `de1df4f8` Umas fast gpu backend (#1826)

### Registry Names
- Backbones: `escnmd_backbone`, `escnmd_backbone_lr`, `escnmd_backbone_les`
- LR Heads: `esen_efs_head_lr`, `esen_mlp_energy_head_lr`, `esen_linear_energy_head`
- Standard Heads: `esen_efs_head`, `esen_mlp_energy_head`, `esen_linear_force_head`, `esen_mlp_stress_head`
