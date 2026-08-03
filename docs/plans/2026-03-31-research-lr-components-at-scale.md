# Research Plan: Long-Range Components for Foundation-Scale MLIPs

## Context

We want to systematically study which bolt-on long-range (LR) components improve machine-learned interatomic potentials (MLIPs) when training at foundation-model scale. The core question: **as backbone models grow in capacity and data, which explicit physics modules (Coulomb, Ewald, spin coupling, charge constraints, etc.) remain beneficial vs. become redundant?** This is timely because foundation MLIPs like UMA explicitly lack long-range interactions (6Å cutoff), while recent work (LES, MACE-Polar, Bamboo) shows LR components are critical for specific chemical domains.

Target venue: NeurIPS/ICML (AI4Science track).

### Decisions Made
- **Dataset**: OMol-4M (fixed random subset from OMol team) for sweeps; full OMol for final models
- **Backbones**: UMA + AllScAIP (in fairchem, full sweep), NequIP (own codebase, selective runs)
- **Strategy**: Full LR component sweep on UMA + AllScAIP → pick best combo → train at full scale
- **Compute budget**: ~60 GPU-days
- **Dispersion (D3/D4)**: Out of scope
- **Collaborator training**: Not available; we run all models ourselves

---

## 1. Experimental Design

### 1.1 Backbone Locality Spectrum

| Model | Locality | Key Mechanism | Effective Range | Codebase | Sweep Role |
|-------|----------|--------------|-----------------|----------|-----------|
| **NequIP** | Strictly local | Body-ordered equivariant message passing | ~5-6Å cutoff | Own codebase (ChengUCB/NequIP-LES) | Selective (best combo only) |
| **UMA (eSCNMD)** | Local + global routing | SO(3)-equivariant + MoLE (global composition features) | ~6Å cutoff but global info via MoLE | fairchem `src/fairchem/core/models/uma/escn_md_lr.py` | **Full sweep** |
| **AllScAIP** | Local + all-to-all attention | Neighborhood attention + optional node-level global attention | Unbounded (with node attention) | fairchem `src/fairchem/core/models/allscaip/` | **Full sweep** |

### 1.2 LR Component Matrix

| Component | Code Status | What It Does | Physics It Captures |
|-----------|------------|-------------|-------------------|
| **Latent Coulomb** | Implemented (fairchem `lr.py`) | NN predicts charges → direct pairwise 1/r | Electrostatics (non-periodic) |
| **Charge/spin constraints** | Implemented (fairchem `lr.py`) | Renormalize predicted charges to match Q_total, S_total | Conservation laws |
| **Electronegativity/hardness** | Implemented (fairchem `lr.py`) | Learned χ, η → charge equilibration energy | Chemical potential equalization |
| **Heisenberg coupling** | Implemented (fairchem `lr.py`) | NN-predicted J-coupling × spin dot product | Magnetic exchange interactions |
| **BEC (Born Effective Charges)** | Implemented (fairchem `les/module/bec.py`) | ∂P/∂r via autograd on predicted polarization | IR spectra, dielectric response |
| **Euclidean Fast Attention** | Not implemented | O(n) attention via random Fourier features on 3D coordinates | Global geometric context (stretch goal) |

Note: Ewald summation is implemented but scoped down since we are OMol-only (non-periodic). It remains available if periodic evaluation is added later.

### 1.3 Training Protocol

- **Sweep phase**: OMol-4M (fixed random subset), UMA × all LR combos + AllScAIP × all LR combos
- **Selection**: Pick best LR component combination from sweep
- **NequIP validation**: Train NequIP ± best LR combo via its own codebase on OMol-4M
- **Scale phase**: Top 1-2 configurations on full OMol (release checkpoints)

---

## 2. Critique & Gaps

### 2.1 Confounded Variables (Critical)

**Problem**: Comparing UMA vs AllScAIP vs NequIP conflates locality with architecture differences. NequIP in a separate codebase adds optimizer/scheduler confounds.

**Mitigations** (must do):
- **Within-backbone comparisons are the primary story**: For UMA and AllScAIP (same fairchem codebase, same training loop), the Δ from adding LR components is clean and unconfounded. This is the paper's strongest evidence.
- Match parameter count across UMA and AllScAIP as closely as possible
- Run each backbone WITH and WITHOUT its native "global" feature (UMA ±MoLE, AllScAIP ±node attention) → 2×2: {native-global ON/OFF} × {LR ON/OFF}
- **Parameter-matched larger-backbone control**: Take LR component's param budget and add to backbone instead. The #1 reviewer objection: "just make the backbone bigger."
- NequIP: match optimizer/LR schedule/batch size where possible. Present NequIP results as a **supporting data point** (does the same LR combo help a strictly local model?), not a head-to-head comparison. Acknowledge codebase confounds.

### 2.2 OMol-Only Risk (High — Acknowledged Trade-off)

**Problem**: OMol is non-periodic molecules. Long-range electrostatics matter most in condensed-phase / periodic / large charge-transfer systems. Training only on OMol biases toward finding LR components "useless."

**Mitigations**:
- **Evaluation on long-range-sensitive molecular benchmarks** (no retraining needed):
  - Charged molecular fragment dissociation curves (test 1/r Coulomb tail recovery)
  - Large donor-acceptor complexes from SPICE or similar (>50 atoms, charge transfer across >6Å)
  - Zwitterionic amino acids / salt bridges in peptides
- **System-size stratification**: Evaluate separately on small (≤20 atoms), medium (20-50), and large (50+) molecules. LR benefit should emerge at larger sizes.
- **Acknowledge this explicitly** in paper limitations. Frame as: "we study the molecular regime; periodic systems are future work."
- **Interesting angle**: If LR components help even for molecules (where you'd expect them to matter less), that's actually a stronger result than showing they help for crystals (where it's obvious).

### 2.3 Missing Baselines (High)

**Must include** (within ~60 GPU-day budget):

| Baseline | Why | Cost |
|----------|-----|------|
| **Increased cutoff (10Å, 12Å)** | Simplest "long-range" approach — just see more atoms. If this matches Coulomb, the physics module is unnecessary. | 4 runs |
| **Parameter-matched larger backbone** | Spend LR component's param budget on backbone capacity instead. | 2 runs |

**Acknowledge but don't run** (cite from literature):
- MACE-Polar multipoles: cite their results, note that monopole-only Coulomb is a simpler starting point
- Bamboo D3 decomposition: cite, note D3 is out of scope for molecular-only study

### 2.4 Evaluation Framework (High)

| Level | Metric | What It Tests |
|-------|--------|--------------|
| **Accuracy** | Energy MAE (meV/atom), Force MAE (meV/Å) | Raw predictive quality |
| **LR sensitivity** | ΔE on dissociation curves at r > 6Å (tail behavior) | Whether the model captures 1/r asymptotics |
| **Physical consistency** | Charge conservation error per system | Whether constraints actually work |
| **Force-energy consistency** | ‖F_predicted + ∇E‖ for direct-force models | Thermodynamic consistency |
| **Scaling curves** | Accuracy vs {data size, model size, molecule size} | The paper's central contribution |
| **Computational overhead** | Wall-clock ms/step, peak memory, % overhead from LR | Practicality at scale |

### 2.5 Scaling Analysis (High — Central Claim)

This is the paper's differentiator. Three axes:

1. **Data scaling**: Train each config at {500K, 1M, 2M, 4M} subsets. Plot accuracy vs data for LR-on vs LR-off. Key question: does the LR benefit persist, shrink, or grow with more data?
2. **Model scaling**: 2 backbone sizes for UMA and AllScAIP (where you control code). Does LR help more for small or large models?
3. **System-size scaling**: Stratify OMol test set by molecule size. At what atom count does LR benefit emerge?

### 2.6 Heisenberg Spin Coupling (Medium)

**Problem**: OMol has spin states but limited magnetic diversity. Heisenberg coupling may not have enough signal.

**Recommendation**: Present as a **case study**, not part of the main ablation. Select a subset of OMol with diverse spin states (transition metal complexes, radicals). If it helps on that subset, great. If not, you've still contributed the implementation.

### 2.7 NequIP Separate Codebase (Medium — Acknowledged)

**Problem**: NequIP in a separate repo means different training loop. Any difference in results could be optimizer, data loading, or backbone.

**Mitigations**:
- Match hyperparams where possible (optimizer, schedule, batch size, total steps)
- Report full training configs in appendix
- Frame NequIP as a **supporting experiment**: "the best LR combo also helps a strictly local equivariant model trained in a different framework"
- The paper's core claims rest on the clean UMA/AllScAIP within-fairchem comparisons

### 2.8 Euclidean Fast Attention (Low Priority / Stretch)

If included, frame as the **learned attention baseline**: "can a learned global attention mechanism replace explicit physics?" This is a good foil for the physics components. Budget: 2 runs on the best-performing backbone only.

---

## 3. Suggested Paper Structure

### Title Candidates
- "Do Foundation MLIPs Need Long-Range Physics? A Systematic Study Across Scales"
- "Scaling Long-Range Interactions in Machine-Learned Interatomic Potentials"

### Outline

1. **Introduction** (~1.5 pages): Foundation MLIPs are local by design. When does this fail? How do LR corrections interact with model/data scale? Position relative to LES, MACE-Polar, Bamboo.
2. **Background** (~1 page): Taxonomy of LR approaches (Coulomb, Ewald, multipole, learned attention). Locality spectrum of backbone architectures. What's been shown and what's missing.
3. **Method** (~2 pages):
   - Modular LR framework (backbone-agnostic, plug-and-play components)
   - Backbone implementations across locality spectrum
   - LR modules: latent Coulomb, charge constraints, electronegativity/hardness, Heisenberg, BEC
4. **Experiments** (~3 pages):
   - 4a: Ablation matrix on OMol-4M (backbone × LR component)
   - 4b: Controls (increased cutoff, parameter-matched larger backbone)
   - 4c: Scaling analysis (data scaling, model scaling, system-size scaling)
   - 4d: Long-range sensitivity benchmarks (dissociation curves, large molecules)
   - 4e: Computational overhead analysis
5. **Results & Discussion** (~2 pages): What scales, what doesn't, and why. Per-backbone analysis (within-backbone Δ is the clean comparison).
6. **Conclusion** (~0.5 pages): Practical guidelines for practitioners. Released checkpoints.

### Key Figures (planned)
- **Fig 1**: Backbone locality spectrum diagram + LR component taxonomy
- **Fig 2**: Ablation heatmap (backbone × component → accuracy)
- **Fig 3**: Scaling curves (accuracy vs data/model/system size, LR-on vs LR-off)
- **Fig 4**: Dissociation curve tail behavior (1/r recovery)
- **Fig 5**: Computational overhead bar chart

---

## 4. Experiment Schedule (~60 GPU-days)

### Phase 1: Full LR Sweep on UMA + AllScAIP (~30 GPU-days)

Each backbone × every LR component combination on OMol-4M:

| # | Backbone | LR Configuration | Runs | Est. GPU-days |
|---|----------|-----------------|------|---------------|
| 1 | UMA | no-LR (baseline) | 1 | 1.5 |
| 2 | UMA | Coulomb only | 1 | 1.5 |
| 3 | UMA | Coulomb + charge constraints | 1 | 1.5 |
| 4 | UMA | Coulomb + constraints + χ/η (equilibration) | 1 | 1.5 |
| 5 | UMA | Coulomb + constraints + Heisenberg | 1 | 1.5 |
| 6 | UMA | Coulomb + constraints + BEC | 1 | 1.5 |
| 7 | AllScAIP | no-LR (baseline) | 1 | 1.5 |
| 8 | AllScAIP | Coulomb only | 1 | 1.5 |
| 9 | AllScAIP | Coulomb + charge constraints | 1 | 1.5 |
| 10 | AllScAIP | Coulomb + constraints + χ/η | 1 | 1.5 |
| 11 | AllScAIP | Coulomb + constraints + Heisenberg | 1 | 1.5 |
| 12 | AllScAIP | Coulomb + constraints + BEC | 1 | 1.5 |
| 13 | UMA | increased cutoff (10Å, 12Å) | 2 | 3 |
| 14 | AllScAIP | increased cutoff (10Å, 12Å) | 2 | 3 |
| 15 | UMA | parameter-matched larger backbone (no LR) | 1 | 1.5 |
| 16 | AllScAIP | parameter-matched larger backbone (no LR) | 1 | 1.5 |
| 17 | UMA w/o MoLE | no-LR vs best LR (isolate native global) | 2 | 3 |
| 18 | AllScAIP w/o node attn | no-LR vs best LR (isolate native global) | 2 | 3 |
| **Subtotal** | | | **~22** | **~30** |

### Phase 2: Scaling Analysis + NequIP (~18 GPU-days)

| # | Experiment | Runs | Est. GPU-days |
|---|-----------|------|---------------|
| 19 | Data scaling: best UMA config × {500K, 1M, 2M, 4M} (LR on + off) | 8 | 8 |
| 20 | Model size: UMA small vs medium × {best LR, no-LR} | 4 | 4 |
| 21 | NequIP baseline (no LR) via own codebase | 1 | 2 |
| 22 | NequIP + best LR combo via NequIP-LES codebase | 1 | 2 |
| 23 | NequIP + increased cutoff | 1 | 2 |
| **Subtotal** | | **15** | **~18** |

### Phase 3: Full-Scale Training + Stretch (~12 GPU-days)

| # | Experiment | Runs | Est. GPU-days |
|---|-----------|------|---------------|
| 24 | Full OMol: best overall config | 1 | 5 |
| 25 | Full OMol: best config without LR (scaling comparison) | 1 | 5 |
| 26 | EFA baseline on best backbone (stretch goal) | 1 | 2 |
| **Subtotal** | | **3** | **~12** |

**Total: ~40 runs, ~60 GPU-days**

### Evaluation (inference only, negligible cost)
- Long-range sensitivity: dissociation curves at r > 6Å, large-molecule stratification
- Physical consistency: charge conservation, force-energy consistency
- Computational overhead: ms/step, peak memory, % overhead per LR component
- System-size analysis: stratify OMol test by molecule size (≤20, 20-50, 50+ atoms)

---

## 5. Remaining Questions

1. **MACE-Polar finding**: They found training on partial charge labels *hurts* — total Q/S constraints alone worked better. Is this consistent with your observations? Informs whether charge constraint component is sufficient.
2. **AllScAIP node attention + Coulomb**: The most scientifically interesting cell. If global attention already captures LR effects, adding Coulomb may add noise. If they're complementary, that's a strong finding for physics-informed components.
3. **OMol-4M composition**: Since it's a fixed random subset, what's the distribution of charged species, large molecules (>50 atoms), and radicals? If underrepresented, the sweep may underestimate LR benefit. Worth profiling the subset before starting.
4. **NequIP-LES integration**: Which LR components does ChengUCB/NequIP-LES already support? Do they match the fairchem LR implementations (same Coulomb formula, same charge normalization)?

---

## 6. Strengths of Current Plan

1. **Timeliness**: UMA explicitly acknowledges the long-range gap. LES and MACE-Polar just published. This is the right moment for a systematic comparison.
2. **Unique position**: Having LR components integrated into fairchem (UMA, AllScAIP) AND access to external implementations (NequIP-LES, MACE) is a rare vantage point.
3. **Modular design**: The LES/LR code is already backbone-agnostic in principle — this makes the ablation credible.
4. **Scale**: Training on OMol-4M/full-OMol is genuinely at foundation scale, not toy-scale. This is the paper's differentiator vs. prior work that tested LR on small datasets.
5. **Locality spectrum**: The NequIP → MACE → UMA → AllScAIP ladder is a clean experimental axis that maps directly to the research question.
6. **Scaling analysis as central contribution**: "Here's how LR benefit interacts with data/model/system scale" is novel and actionable.

---

## 7. Key References

- **Bamboo** (Gong et al., Nature Machine Intelligence 2025): SR + Coulomb + D3 decomposition; ensemble distillation; showed explicit charge prediction critical for condensed-phase accuracy
- **AllScAIP** (arxiv:2603.06567): All-to-all attention backbone; neighborhood + optional global node attention
- **MACE** (Batatia et al., NeurIPS 2022): Higher body-order equivariant message passing; ACE-based
- **MACE-Polar** (arxiv:2602.19411): Multipole (l≤2) + Ewald + Fukui charge equilibration; 89% error reduction on protein-ligand; training on partial charges hurts
- **LES** (Kim et al., JCTC 2025): Latent Ewald summation; backbone-agnostic; <5% overhead; Au₂/MgO long-range benchmark
- **UMA** (Wood et al., arxiv:2506.23971): eSCNMD + MoLE; 500M structures; explicit long-range limitation acknowledged
- **NequIP** (Batzner et al., Nature Communications 2022): E(3)-equivariant message passing
- **Euclidean Fast Attention** (arxiv:2412.08541): O(n) attention via random Fourier features on coordinates
