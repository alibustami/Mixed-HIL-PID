# Mixed HIL Algorithm - Condensed for Main Paper

**This is a condensed, publication-ready version suitable for inclusion in the main body of a research paper (typically 2-3 columns).**

---

## Algorithm: Mixed Human-in-the-Loop Optimization

```
──────────────────────────────────────────────────────────────────────────
Input:  bounds ∈ ℝⁿˣ² - Parameter bounds
        max_iter - Maximum iterations    
Output: θ* - Optimal parameters
──────────────────────────────────────────────────────────────────────────
Initialize:
  w ← RANDOM([0,1]ⁿ)                          // Preference weights
  DE ← DifferentialEvolution(bounds)
  BO ← BayesianOptimizer(bounds)
  Warm-start BO with DE population

Main Loop (i = 1 to max_iter):
  
  // 1. GENERATE CANDIDATES
  θ_DE, J_DE, g_DE ← DE.Evolve()              // DE candidate
  θ_BO ← BO.Propose()
  J_BO, g_BO ← Evaluate(θ_BO)                 // BO candidate
  
  // 2. KNOWLEDGE SHARING
  BO.Update(θ_DE, J_DE, g_DE)                 // BO learns from DE
  BO.Update(θ_BO, J_BO, g_BO)
  
  // 3. HUMAN COMPARISON
  response_DE ← Simulate(θ_DE)                // Full response
  response_BO ← Simulate(θ_BO)
  
  choice ← Human.Compare(response_DE, response_BO)
  
  // 4. ADAPTIVE RESPONSE
  switch choice:
    
    case PREFER_DE:                           // Validate DE
      w ← w + α(θ̂_DE - w)                     // Update preferences
      θ_anchor ← bounds_min + w⊙span
      DE.InjectCandidate(θ_anchor)            // Guide DE
      BO.Nudge(θ_DE, J_DE, J_BO, g_DE)        // Guide BO
    
    case PREFER_BO:                           // Validate BO  
      w ← w + α(θ̂_BO - w)
      θ_anchor ← bounds_min + w⊙span
      DE.InjectCandidate(θ_BO)                // Share BO solution
      DE.InjectCandidate(θ_anchor)            // + anchor (2×)
      BO.Nudge(θ_BO, J_BO, J_DE, g_BO)
    
    case TIE_REFINE:                          // Converge
      θ_mid ← (θ_DE + θ_BO)/2                 // Midpoint
      DE.ShrinkBounds(θ_mid, 0.5)             // 50% reduction
      BO.ShrinkBounds(θ_mid, 0.5)
      DE.mutation ← 0.8 · DE.mutation         // Reduce exploration
    
    case REJECT_BOTH:                         // Explore
      DE.ExpandBounds(1.5)                    // 50% expansion
      BO.ExpandBounds(1.5)
      DE.mutation ← min(1.2 · DE.mutation, 1.0) // More exploration
  
  // 5. AUTO-TERMINATE
  if PerformanceTargetsMet(response_DE) ∧ g_DE ≤ 0:
    return θ_DE
  if PerformanceTargetsMet(response_BO) ∧ g_BO ≤ 0:
    return θ_BO

return Best feasible solution
──────────────────────────────────────────────────────────────────────────
```

---

## Key Subroutines

### Preference Update
```
w ← w + α · ((θ_pref - bounds_min)/span - w)   // EMA toward preferred
θ_anchor ← bounds_min + w ⊙ (bounds_max - bounds_min)
```

### BO Preference Nudge
```
if g_pref ≤ 0:                                  // Only if feasible
  Δ ← |J_other - J_pref|
  J_pseudo ← J_pref - 0.2·Δ                     // Synthetic improvement
  GP.AddObservation(θ_pref, J_pseudo, g_pref)   // Bias future search
```

### Feasibility-Aware Comparison (DE)
```
// Deb's constraint handling rules:
if feas_a ∧ ¬feas_b: return a                   // Feasible beats infeasible
if ¬feas_a ∧ feas_b: return b
if feas_a ∧ feas_b: return argmin(J_a, J_b)    // Minimize cost
return argmin(g_a, g_b)                         // Minimize violation
```

### Constrained BO Acquisition
```
// Expected Improvement with Constraints (EIC):
EI(θ) ← σ_Y(θ) · [Z·Φ(Z) + φ(Z)]              // Z = (J_best-μ_Y-ξ)/σ_Y
PoF(θ) ← Φ(-μ_G(θ)/σ_G(θ))                    // P(g(θ) ≤ 0)
EIC(θ) ← EI(θ) · PoF(θ)

θ_next ← argmax{EIC(θ) | PoF(θ) ≥ 0.95}        // Constrained acquisition
```

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| α | 0.3 | Preference learning rate |
| N_pop | 6 | DE population size |
| F₀ | 0.5 | Initial DE mutation |
| ρ_min | 0.95 | Min BO probability of feasibility |
| λ | 0.5 | Bounds shrink factor (TIE) |
| γ | 1.5 | Bounds expand factor (REJECT) |

---

## Complexity

**Per iteration:** O(N_pop · T_sim/Δt)  
Dominated by physics simulations; optimization overhead is negligible.

---

## Notation

- **θ**: Parameter vector [K_p, K_i, K_d]
- **J**: Objective (cost)
- **g**: Constraint violation (g ≤ 0 is feasible)
- **w**: Preference weights ∈ [0,1]ⁿ
- **α**: Learning rate
- **θ̂**: Normalized parameters
- **GP**: Gaussian Process
- **Φ, φ**: Standard normal CDF and PDF

---

*See supplementary materials for complete algorithmic details.*
