# Deep Learning the metric functions from QCD lattice data

Machine learning project to use network setup to learn the deformation parameters in our deformed AdS-BH model metric. 

## Problem setup

We have lattice data for QCD potential for different temperatures. We want to find the metric functions just from the potential data, without any information about the background itself. One solution to this inverse problem is to set up a network that learns the metric functions, such that the potential computed using the metric functions matches the data. This project has additional complexity due to the data being partially complex: The potential has real branch and complex branch, so that when we allow the holographic coordinate to have complex values, we have linear tail for the potential as afunction of separation, instead of it breaking and creating swallowtail shape. This allows us to 1) probe the physics at larger separations, and 2) Construct metric just from lattice data, and compare it to existing models such as IHQCD.

The model we use is deformed AdS-BH, so that we learn deformation parameters `a(z)` and `b(z)` that are polynomials. For the non-deformed, baseline case of pure AdS-BH, we have simply $a=b=0$. The metric is 
$$ ds^2 = \frac{R^2}{z^2}\left( -f(z)dt^2 + g(z) dz^2 + d\vec{x}^2\right) \ , $$

and the metric functions are (parametrization in the learning uses $z_h = 1$)

$$ f(z) = e^{a(z)}\left(1-z^4/z_h^4\right) $$
$$ g(z) = \frac{e^{b(z)}}{1-z^4/z_h^4} .$$

The learned parameters are the coefficients $a_i$ and $b_i$, in the polynomials:
$$ a(z) = \sum_{i=1}^N a_i(z)^i \ , \quad b(z) = \sum_{i=1}^N b_i(z)^i.$$

## Setup

1. Create virtual environment: 
```bash
python3 -m venv .venv
source .venv/bin/activate
````

2. Install dependencies:
```bash
pip install -r requirements.txt
```


## Data

Data files should be placed in the `data/`directory (our data is not public so it will not display here in the repository)

## Modules 

- `dataset_HR.py`- Module for dataset handling
- `constants.py`- Module for the dtype constants
- `model_HR_new.py`- Module for the model


## Network Architecture

**Model: `AdSBHNet`** — A physics-informed parametric model (not a traditional deep neural network)

**Learnable Parameters:**
- `a[0..N-1]`: Polynomial coefficients for metric function `a(z) = Σ aᵢ·zⁱ`
- `b[0..N-1]`: Polynomial coefficients for metric function `b(z) = Σ bᵢ·zⁱ`
- `logcoef`: Log-scale vertical coefficient (represents `R²/(2πα')`)
- `shift`: Additive offset for UV normalization

**Metric Functions:**
```
f(z) = (1 - z⁴) · exp(a(z))
g(z) = exp(b(z)) / (1 - z⁴)
```

## Training Pipeline

1. **Data Loading**: Lattice QCD data (L, V, σ) filtered to stable branch (L < 1.4)
2. **Train/Val Split**: 80/20 random split
3. **Optimizer**: Adam with parameter-specific learning rates (5e-3 for a,b,shift; 3e-3 for logcoef)
4. **Scheduler**: ReduceLROnPlateau (factor=0.7, patience=50)
5. **Epochs**: 1000 with staged constraint scheduling

**Staged Training Schedule:**

| Epoch | Parameter Bounds | Magnitude λ | L_max λ | Monotonicity λ |
|-------|------------------|-------------|---------|----------------|
| 1-200 | Tight | 3.0 | 100 | 0.0 |
| 201-400 | Medium | 1.5 | 200 | 5.0 |
| 401-700 | Relaxed | 0.5 | 200 | 10.0 |
| 701-1000 | Wide | 0.5 | 300 | 15.0 |

## Loss Function

Multi-component physics-informed loss:

```
L_total = L_data + L_Lmax + L_NEC + L_monotonicity + L_magnitude + L_reg
```

| Component | Weight | Description |
|-----------|--------|-------------|
| **Data Loss** | λ=10 | Weighted MSE: `Σ (1/σ²) · (V_pred - V_true)²` on stable branch |
| **L_max Penalty** | λ=100-300 | Penalizes if L_max deviates from [0.35, 0.7] |
| **NEC Penalty** | λ=0 | Null Energy Condition: `d(a+b)/dz ≤ 0` |
| **Monotonicity** | λ=0-15 | Ensures `f(z)/z⁴` is decreasing |
| **Magnitude** | λ=20→2 | Encourages non-trivial parameter values |
| **L2 Regularization** | λ=1e-3 | Prevents parameter explosion |

## Core Algorithms

**Forward Pass:**
1. Sample z_* on power-law grid: `zs = zmin + (zmax - zmin) · u^q`
2. Compute parametric curve (L(z_*), V(z_*)) via numerical integration
3. Interpolate V(L) using local quadratic fitting
4. Apply shift: `V_out = V_interp + shift`

**Key Integrals (Complex-Safe):**
```
L(z_*) = (2/π) ∫₀^z_* dz · √g(z) / √[(z_*⁴·f(z_*))/(z⁴·f(z)) - 1]

V(z_*) = 2π [∫_ε^z_* dz · (1/z²) · (√(f·g)/√[1 - (z⁴·f(z_*))/(z_*⁴·f(z))] - 1) - 1/z_*]
```

**Hard Constraints (Post-Step):**
- `a[0] < b[0] < 0`
- `a[1:], b[1:] > 0`
- `|a|, |b| ≤ 2.0`

## Usage

Main training script: `training.py`
