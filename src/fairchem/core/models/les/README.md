# LES — Long-range Electrostatics/Spin

`models/les/` is the shared math library for long-range (non-local) interactions
that supplement the short-range message passing in eSCNMD and AllScAIP. It
predicts per-atom latent charges and/or spins from backbone features, then
computes their contribution to the energy via Ewald summation (periodic) or a
direct Coulomb sum (non-periodic), optionally with a learned Heisenberg
spin-spin coupling term. Charge/spin totals are renormalized per-system so the
long-range term does not break charge/spin conservation.

## File Structure

```
les/
├── les.py                   # Les module: charge prediction (Atomwise MLP) + Ewald + BEC
├── module/
│   ├── ewald.py              # Ewald k-space summation (periodic electrostatics)
│   ├── bec.py                 # Born effective charge computation
│   ├── atomwise.py            # Per-atom MLP for latent charge prediction
│   └── blocks.py              # Shared MLP building blocks
└── util/
    ├── grad.py                # Autograd helpers (forces/BEC from energy)
    └── scatter.py              # Segment-sum utilities
```

Two other locations implement the same family of long-range physics against
different backbones and are the actual entry points used by configs/heads:

| Component | Backbone | Consumes |
|---|---|---|
| `models/uma/escn_md_lr.py` (`eSCNMDBackboneLR`, `MLP_EFS_Head_LR`, `MLP_Energy_Head_LR`, `Linear_Energy_Head_LR`) | UMA eSCNMD | `models/utils/lr.py` (direct Coulomb, Ewald, Heisenberg, charge/spin renormalization) + `models/utils/lr_charges.py` (`LRChargePredictor`) |
| `models/uma/escn_md_les.py` (`eSCNMDBackboneLES`, `MLP_EFS_Head_LES`, `MLP_Energy_Head_LES`) | UMA eSCNMD | this package (`Les`) directly |
| `models/allscaip/AllScAIP_lr.py` | AllScAIP | `models/les/module.Ewald` + `models/allscaip/utils/lr_utils.py` |

## Quick Start

Runnable training configs live under `configs/uma/lr/` (`uma_lr.yml`,
`uma_lr_heis.yml`, `uma_lr_constr.yml`, `uma_lr_heis_constr.yml`, with matching
backbones under `configs/uma/lr/backbone/`):

```bash
fairchem -c configs/uma/lr/uma_lr.yml
```

## Terminology

- **LR** (`escn_md_lr.py`, `models/utils/lr.py`): the electrostatics/Heisenberg
  implementation integrated with UMA's eSCNMD backbone and heads.
- **LES**: the standalone `Les` module in this directory, providing a
  production-quality Ewald/BEC implementation; wrapped for eSCNMD via
  `escn_md_les.py` and reused by AllScAIP's LR head.
- **BEC** (Born effective charge): the derivative of predicted per-atom charge
  with respect to atomic displacement, exposed via `models/les/module/bec.py`
  for consumers that need it (e.g. dielectric/IR-active properties).
