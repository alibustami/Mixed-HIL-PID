# Mixed HIL Pseudocode for Academic Publication

This directory contains formal algorithmic specifications of the **Mixed Human-in-the-Loop (Mixed HIL) Optimization Framework** for PID controller tuning.

## Contents

### 1. `mixed_hil_pseudocode.md` - Core Framework
**Main algorithms for the complete Mixed HIL system:**
- Algorithm 1: Main Mixed HIL optimization framework
- Algorithm 2-3: Preference learning mechanism
- Algorithm 4-5: Differential Evolution with feasibility handling
- Algorithm 6-7: Constrained Bayesian Optimization
- Algorithm 8-9: Adaptive search space control
- Algorithm 10: Candidate injection
- Algorithm 11-12: PID evaluation and metrics

**Key Features:**
- Complete, executable specifications
- Formal notation with complexity analysis
- Ready for inclusion in research papers
- 12 algorithms covering all system components

### 2. `human_feedback_taxonomy.md` - Feedback Mechanism
**Detailed specification of the four-way feedback system:**
- Formal definitions of PREFER_DE, PREFER_BO, TIE_REFINE, REJECT_BOTH
- State transition diagrams
- Mathematical properties and proofs
- Comparative analysis with alternative schemes

**Key Features:**
- Action-by-action algorithmic responses
- Information flow diagrams
- Asymmetry rationale (why PREFER_BO gives 2 injections vs. 1 for PREFER_DE)
- Auto-termination criteria

## Usage Guidelines for Academic Papers

### Recommended Citation Structure

When referencing these algorithms in your paper:

#### For the Main Framework:
```latex
The Mixed HIL optimization framework (Algorithm 1) coordinates
Differential Evolution (Algorithm 4) and Bayesian Optimization 
(Algorithm 6) through adaptive preference learning (Algorithm 2).
```

#### For Feedback Mechanism:
```latex
Human operators provide feedback from a four-way action space
ℱ = {PREFER_DE, PREFER_BO, TIE_REFINE, REJECT_BOTH}, each
triggering distinct algorithmic responses (see Supplementary Materials).
```

### Inclusion Strategies

#### **Strategy 1: Main Paper + Supplementary**
- **Main Paper**: Include Algorithms 1, 2, 4, 6 (core framework)
- **Supplementary Materials**: Full set of 12 algorithms + feedback taxonomy

#### **Strategy 2: Focused Technical Paper**
- **Main Paper**: All 12 algorithms
- **Supplementary**: Detailed implementation notes, code repository link

#### **Strategy 3: Systems Paper**
- **Main Paper**: Algorithms 1, 11-12 (system overview + evaluation)
- **Supplementary**: Optimizer details (Algorithms 4-9)

## Notation Consistency

All algorithms use consistent notation:

| Symbol | Meaning | Type |
|--------|---------|------|
| θ | Parameter vector [K_p, K_i, K_d] | ℝⁿ |
| J | Objective function (cost) | ℝ |
| g | Constraint violation | ℝ |
| w | Preference weights | [0,1]ⁿ |
| F | DE mutation factor | ℝ₊ |
| α | Preference learning rate | (0,1) |
| ρ | Probability of feasibility | [0,1] |
| P | DE population matrix | ℝ^(N_pop×n) |
| GP | Gaussian Process | Model |

## Mathematical Rigor

### Verified Properties:
1. **Bounds Safety**: All algorithms respect `bounds_adaptive ⊆ global_bounds`
2. **Feasibility Preservation**: Best feasible solution never discarded
3. **Preference Convergence**: Weight vector converges under consistent feedback
4. **Constraint Handling**: Implements Deb's feasibility rules correctly

### Computational Complexity:
- Per-iteration: O(N_pop · T_sim / Δt)
- Dominated by physics simulation costs
- BO: O(N_obs³) for GP training
- DE: O(N_pop · n) for evolution

## Implementation Validation

These algorithms have been validated against the reference implementation:
- **Reference Code**: `main_macos.py`, `differential_evolution.py`, `bayesian_optimization.py`
- **Test System**: Husky robot yaw control in PyBullet
- **Parameter Space**: 3D (K_p, K_i, K_d)
- **Constraint**: Actuator saturation at 255 rad/s

## Reproducibility Checklist

To reproduce results using these algorithms:

- [ ] Implement Algorithms 1-12 in your target language
- [ ] Use global bounds: K_p ∈ [0.1, 10.0], K_i ∈ [0.01, 10.0], K_d ∈ [0.01, 10.0]
- [ ] Set hyperparameters:
  - [ ] DE population: N_pop = 6
  - [ ] Initial mutation: F₀ = 0.5
  - [ ] Preference learning rate: α = 0.3
  - [ ] BO min PoF: ρ_min = 0.95
  - [ ] Shrink factor: λ = 0.5
  - [ ] Expand factor: γ = 1.5
- [ ] Implement PID evaluation with:
  - [ ] Derivative-on-measurement (Algorithm 11, line 20)
  - [ ] Anti-windup (lines 33-35)
  - [ ] Actuator saturation at u_max = 255
- [ ] Use feasibility-aware selection (Algorithm 5)
- [ ] Implement all four feedback actions correctly

## Extending the Framework

### Possible Extensions:

1. **Multi-Objective HIL**
   - Modify Algorithm 1 to handle Pareto fronts
   - Feedback becomes "Select from 2 solutions on Pareto front"

2. **Confidence-Weighted Preferences**
   - Modify Algorithm 2 to scale α by human confidence
   - Quick decisions → higher α, slow → lower α

3. **Transfer Learning**
   - Initialize w from previous similar tuning sessions
   - Warm-start both DE and BO with historical data

4. **Active Learning**
   - System suggests when human feedback would be most valuable
   - Based on GP uncertainty or population diversity

## Common Pitfalls (Implementation)

⚠️ **Watch out for these when implementing:**

1. **Global vs. Adaptive Bounds Confusion**
   - Always clamp adaptive bounds to global bounds
   - See Algorithm 8, lines 10-12 and Algorithm 9, lines 13-15

2. **Preference Normalization**
   - Must normalize parameters to [0,1] before updating weights
   - Unnormalized updates will fail (Algorithm 2, lines 3-4)

3. **Best Protection in Injection**
   - Never replace the best individual when `protect_best=true`
   - See Algorithm 10, lines 9-14

4. **Feasibility Rules**
   - Must implement Deb's rules exactly (Algorithm 5)
   - Wrong: `minimize J + λ·g`, Right: hierarchical comparison

5. **BO Nudging Condition**
   - Only nudge if preferred candidate is feasible (g ≤ 0)
   - Algorithm 7, lines 2-4

## Visualization Recommendations

For publications, consider these visualizations:

1. **Optimization Trajectory**: Parameter space showing θ_DE, θ_BO over iterations
2. **Bounds Evolution**: How adaptive bounds shrink/expand with feedback
3. **Preference Weight Convergence**: w(t) trajectory
4. **PID Response Curves**: Side-by-side DE vs. BO responses at each iteration
5. **Feedback Distribution**: Histogram of feedback types used

## Contact and Questions

These algorithms are derived from the Mixed HIL implementation described in:

**Repository**: `/home/waleed/Desktop/DE&BO/Mixed-HIL-PID`

**Core Files**:
- `main_macos.py` - Main framework (Algorithm 1)
- `differential_evolution.py` - DE implementation (Algorithms 4-5, 8-10)
- `bayesian_optimization.py` - BO implementation (Algorithms 6-7)

For clarifications on specific algorithmic details, refer to the inline comments in these files.

---

## Version History

- **v1.0** (2026-01-12): Initial pseudocode formalization
  - 12 core algorithms
  - Feedback taxonomy
  - Comprehensive documentation

---

*These algorithms provide a complete, reproducible specification of the Mixed HIL framework for inclusion in peer-reviewed publications.*
