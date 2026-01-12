# Mixed Human-in-the-Loop Optimization: Formal Pseudocode

## Abstract

This document presents the formal algorithmic specification of the **Mixed Human-in-the-Loop (Mixed HIL)** optimization framework for automated PID controller tuning. The approach synergistically combines Differential Evolution (DE) and Bayesian Optimization (BO) with adaptive preference learning to incorporate human expertise into the optimization process.

---

## Algorithm 1: Main Mixed HIL Framework

```
Algorithm 1: Mixed HIL Optimization Framework
─────────────────────────────────────────────────────────────────────
Input:  bounds ∈ ℝⁿˣ² - Parameter bounds [min, max] for n parameters
        max_iter - Maximum optimization iterations
        α_pref - Preference learning rate
        F₀ - Initial DE mutation factor
        N_pop - DE population size
        ρ_min - Minimum BO probability of feasibility
        
Output: θ* - Optimal PID parameters
        H - Complete optimization history
─────────────────────────────────────────────────────────────────────

1:  // === INITIALIZATION ===
2:  w ← RANDOM([0,1]ⁿ)                    ▷ Initialize preference weights
3:  θ_anchor ← bounds_min + w ⊙ (bounds_max - bounds_min)
4:  
5:  DE ← InitializeDifferentialEvolution(bounds, N_pop, F₀)
6:  DE.population[0] ← θ_anchor            ▷ Seed with preference anchor
7:  
8:  BO ← InitializeBayesianOptimizer(bounds, ρ_min)
9:  
10: // Warm-start BO with DE's initial population
11: for each θ ∈ DE.population do
12:     J, g ← EvaluatePID(θ)              ▷ J: cost, g: constraint violation
13:     BO.Update(θ, J, g)
14: end for
15: 
16: θ_best ← argmin{J(θ) | g(θ) ≤ 0}       ▷ Best feasible solution
17: H ← ∅                                   ▷ Optimization history
18: 
19: // === MAIN OPTIMIZATION LOOP ===
20: for i ← 1 to max_iter do
21:     
22:     // === CANDIDATE GENERATION ===
23:     θ_DE, J_DE, g_DE ← DE.Evolve()     ▷ DE proposes candidate A
24:     θ_BO ← BO.Propose()                 ▷ BO proposes candidate B
25:     
26:     J_BO, g_BO ← EvaluatePID(θ_BO)
27:     
28:     // === KNOWLEDGE SHARING ===
29:     BO.Update(θ_DE, J_DE, g_DE)         ▷ BO learns from DE
30:     BO.Update(θ_BO, J_BO, g_BO)         ▷ BO learns from itself
31:     
32:     // === DETAILED EVALUATION ===
33:     response_DE ← SimulatePID(θ_DE)     ▷ Full system response
34:     response_BO ← SimulatePID(θ_BO)
35:     
36:     metrics_DE ← ComputeMetrics(response_DE)  ▷ Overshoot, rise time, etc.
37:     metrics_BO ← ComputeMetrics(response_BO)
38:     
39:     // === TERMINATION CHECK ===
40:     if PerformanceTargetsMet(metrics_DE) ∧ g_DE ≤ 0 then
41:         return θ_DE, H ∪ {(i, θ_DE, "auto_terminate_de")}
42:     end if
43:     
44:     if PerformanceTargetsMet(metrics_BO) ∧ g_BO ≤ 0 then
45:         return θ_BO, H ∪ {(i, θ_BO, "auto_terminate_bo")}
46:     end if
47:     
48:     // === HUMAN FEEDBACK ===
49:     choice ← HumanComparison(response_DE, response_BO, 
50:                               θ_DE, θ_BO, metrics_DE, metrics_BO)
51:     
52:     // === ADAPTIVE RESPONSE TO FEEDBACK ===
53:     switch choice do
54:         case PREFER_DE:
55:             gap ← PreferenceUpdate(w, θ_DE, θ_BO, α_pref)
56:             θ_anchor ← ComputeAnchor(w, bounds)
57:             
58:             DE.mutation ← F₀                    ▷ Reset exploration
59:             DE.InjectCandidate(θ_anchor)        ▷ Inject learned anchor
60:             
61:             BO.PreferenceNudge(θ_DE, J_DE, J_BO, g_DE)
62:             H ← H ∪ {(i, "prefer_de", θ_DE, θ_BO, gap)}
63:             
64:         case PREFER_BO:
65:             gap ← PreferenceUpdate(w, θ_BO, θ_DE, α_pref)
66:             θ_anchor ← ComputeAnchor(w, bounds)
67:             
68:             DE.mutation ← F₀
69:             DE.InjectCandidate(θ_BO)            ▷ Share BO's solution
70:             DE.InjectCandidate(θ_anchor)        ▷ Also inject anchor
71:             
72:             BO.PreferenceNudge(θ_BO, J_BO, J_DE, g_BO)
73:             H ← H ∪ {(i, "prefer_bo", θ_BO, θ_DE, gap)}
74:             
75:         case TIE_REFINE:
76:             θ_mid ← (θ_DE + θ_BO) / 2           ▷ Midpoint
77:             
78:             DE.RefineSearchSpace(θ_mid, 0.5)    ▷ Shrink 50%
79:             BO.RefineBounds(θ_mid, 0.5)
80:             
81:             H ← H ∪ {(i, "tie_refine", θ_mid)}
82:             
83:         case REJECT_BOTH:
84:             DE.ExpandSearchSpace(1.5)           ▷ Expand 50%
85:             BO.ExpandBounds(1.5)
86:             
87:             H ← H ∪ {(i, "reject_expand")}
88:             
89:         case EXIT:
90:             return θ_best, H
91:     end switch
92:     
93:     // === UPDATE BEST ===
94:     θ_best ← UpdateBest(θ_best, θ_DE, θ_BO)   ▷ Feasibility-aware
95:     
96: end for
97: 
98: return θ_best, H
```

---

## Algorithm 2: Preference Learning Mechanism

```
Algorithm 2: Preference Weight Update
─────────────────────────────────────────────────────────────────────
Input:  w ∈ [0,1]ⁿ - Current preference weights
        θ_pref ∈ ℝⁿ - Preferred candidate
        θ_other ∈ ℝⁿ - Other candidate
        α ∈ (0,1) - Learning rate
        bounds ∈ ℝⁿˣ² - Parameter bounds
        
Output: gap ∈ ℝⁿ - Normalized distance vector
        w_new ∈ [0,1]ⁿ - Updated weights
─────────────────────────────────────────────────────────────────────

1:  // === NORMALIZATION ===
2:  span ← bounds_max - bounds_min
3:  θ̂_pref ← (θ_pref - bounds_min) / (span + ε)   ▷ ε = 10⁻⁹
4:  θ̂_other ← (θ_other - bounds_min) / (span + ε)
5:  
6:  // === GAP COMPUTATION ===
7:  gap ← θ̂_other - θ̂_pref                         ▷ Direction vector
8:  
9:  // === EXPONENTIAL MOVING AVERAGE UPDATE ===
10: w_new ← w + α · (θ̂_pref - w)                   ▷ Shift toward preference
11: w_new ← CLIP(w_new, 0, 1)                       ▷ Ensure [0,1] bounds
12: 
13: return gap, w_new


Algorithm 3: Anchor Point Generation
─────────────────────────────────────────────────────────────────────
Input:  w ∈ [0,1]ⁿ - Preference weights
        bounds ∈ ℝⁿˣ² - Parameter bounds
        
Output: θ_anchor ∈ ℝⁿ - Anchor point in parameter space
─────────────────────────────────────────────────────────────────────

1:  θ_anchor ← bounds_min + w ⊙ (bounds_max - bounds_min)
2:  return θ_anchor
```

---

## Algorithm 3: Differential Evolution with HIL Adaptations

```
Algorithm 4: Feasibility-Aware Differential Evolution
─────────────────────────────────────────────────────────────────────
Input:  bounds ∈ ℝⁿˣ² - Adaptive search bounds
        global_bounds ∈ ℝⁿˣ² - Hard domain limits
        N_pop - Population size
        F - Mutation factor
        CR - Crossover probability
        
Output: θ_best - Best candidate from current generation
        J_best - Best objective value
        g_best - Best constraint violation
─────────────────────────────────────────────────────────────────────

1:  // === INITIALIZATION (if needed) ===
2:  if population = ∅ then
3:      for i ← 1 to N_pop do
4:          P[i] ← bounds_min + RANDOM([0,1]ⁿ) ⊙ (bounds_max - bounds_min)
5:          P[i] ← CLIP(P[i], global_bounds_min, global_bounds_max)
6:          J[i] ← ∞, g[i] ← ∞
7:      end for
8:  end if
9:  
10: // === EVOLUTION STEP ===
11: for i ← 1 to N_pop do
12:     
13:     // === MUTATION (DE/rand/1) ===
14:     r₁, r₂, r₃ ← RANDOM_DISTINCT({1,...,N_pop} \ {i})
15:     v ← P[r₁] + F · (P[r₂] - P[r₃])
16:     v ← CLIP(v, bounds_min, bounds_max)
17:     v ← CLIP(v, global_bounds_min, global_bounds_max)
18:     
19:     // === CROSSOVER (binomial) ===
20:     u ← P[i]
21:     j_rand ← RANDOM({1,...,n})
22:     for j ← 1 to n do
23:         if RANDOM() < CR ∨ j = j_rand then
24:             u[j] ← v[j]
25:         end if
26:     end for
27:     
28:     // === EVALUATION ===
29:     J_trial, g_trial ← EvaluatePID(u)
30:     
31:     // === FEASIBILITY-AWARE SELECTION ===
32:     if IsBetter(J_trial, g_trial, J[i], g[i]) then
33:         P[i] ← u
34:         J[i] ← J_trial
35:         g[i] ← g_trial
36:     end if
37:     
38: end for
39: 
40: // === FIND BEST (feasibility-aware) ===
41: idx_best ← argmin_i {J[i] | using feasibility rules}
42: return P[idx_best], J[idx_best], g[idx_best]


Algorithm 5: Feasibility-Aware Comparison
─────────────────────────────────────────────────────────────────────
Input:  J_a, g_a - Objective and constraint for candidate A
        J_b, g_b - Objective and constraint for candidate B
        
Output: true if A is better than B, false otherwise
─────────────────────────────────────────────────────────────────────

1:  feas_a ← (g_a ≤ 0)
2:  feas_b ← (g_b ≤ 0)
3:  
4:  // === FEASIBILITY RULES (Deb et al.) ===
5:  if feas_a ∧ ¬feas_b then
6:      return true                          ▷ Feasible beats infeasible
7:  end if
8:  
9:  if ¬feas_a ∧ feas_b then
10:     return false
11: end if
12: 
13: if feas_a ∧ feas_b then
14:     return J_a < J_b                     ▷ Both feasible: compare cost
15: end if
16: 
17: return g_a < g_b                         ▷ Both infeasible: less violation
```

---

## Algorithm 4: Bayesian Optimization with Constraints

```
Algorithm 6: Constrained Bayesian Optimization
─────────────────────────────────────────────────────────────────────
Input:  bounds ∈ ℝⁿˣ² - Adaptive search bounds
        ρ_min - Minimum probability of feasibility
        GP_Y - Gaussian Process for objective
        GP_G - Gaussian Process for constraints
        D - Historical observations {(θᵢ, Jᵢ, gᵢ)}
        
Output: θ_next - Next candidate to evaluate
─────────────────────────────────────────────────────────────────────

1:  // === UPDATE GAUSSIAN PROCESSES ===
2:  X ← [θ₁, θ₂, ..., θ_|D|]ᵀ
3:  Y ← [J₁, J₂, ..., J_|D|]ᵀ
4:  G ← [g₁, g₂, ..., g_|D|]ᵀ
5:  
6:  GP_Y.Fit(X, Y)                          ▷ Train objective surrogate
7:  GP_G.Fit(X, G)                          ▷ Train constraint surrogate
8:  
9:  // === IDENTIFY BEST FEASIBLE ===
10: D_feas ← {(θ, J) ∈ D | g ≤ 0}
11: if D_feas ≠ ∅ then
12:     J_best ← min{J | (θ, J) ∈ D_feas}
13: else
14:     J_best ← min{J | (θ, J) ∈ D}        ▷ Fallback if none feasible
15: end if
16: 
17: // === GENERATE CANDIDATES ===
18: U ← SampleLatinHypercube(bounds, N_candidates)
19: 
20: // === COMPUTE ACQUISITION (EIC) ===
21: for each u ∈ U do
22:     
23:     // Expected Improvement
24:     μ_Y(u), σ_Y(u) ← GP_Y.Predict(u)
25:     if σ_Y(u) > 0 then
26:         Z ← (J_best - μ_Y(u) - ξ) / σ_Y(u)
27:         EI(u) ← σ_Y(u) · [Z · Φ(Z) + φ(Z)]  ▷ Φ: CDF, φ: PDF
28:     else
29:         EI(u) ← 0
30:     end if
31:     
32:     // Probability of Feasibility
33:     μ_G(u), σ_G(u) ← GP_G.Predict(u)
34:     if σ_G(u) > 0 then
35:         PoF(u) ← Φ(-μ_G(u) / σ_G(u))      ▷ P(g(u) ≤ 0)
36:     else
37:         PoF(u) ← 1 if μ_G(u) ≤ 0 else 0
38:     end if
39:     
40:     // Expected Improvement with Constraints
41:     EIC(u) ← EI(u) · PoF(u)
42:     
43: end for
44: 
45: // === SELECT BEST ACQUISITION ===
46: θ_next ← argmax_u {EIC(u) | PoF(u) ≥ ρ_min}
47: 
48: // === FALLBACK IF NO FEASIBLE CANDIDATES ===
49: if ∄ u : PoF(u) ≥ ρ_min then
50:     θ_next ← argmax_u {PoF(u)}           ▷ Most likely feasible
51: end if
52: 
53: return θ_next


Algorithm 7: BO Preference Nudging
─────────────────────────────────────────────────────────────────────
Input:  θ_pref - Preferred candidate
        J_pref - Objective value of preferred
        J_other - Objective value of other candidate
        g_pref - Constraint violation of preferred
        strength β ∈ (0,1) - Nudge strength
        
Output: Updated GP with synthetic observation
─────────────────────────────────────────────────────────────────────

1:  // === ONLY NUDGE IF PREFERRED IS FEASIBLE ===
2:  if g_pref > 0 then
3:      return                               ▷ Skip if infeasible
4:  end if
5:  
6:  // === COMPUTE SYNTHETIC COST ===
7:  Δ ← |J_other - J_pref|                   ▷ Performance gap
8:  
9:  if Δ > 0 then
10:     J_pseudo ← J_pref - β · Δ            ▷ Artificially improve
11: else
12:     J_pseudo ← J_pref - β · 0.01         ▷ Small improvement
13: end if
14: 
15: // === ADD SYNTHETIC OBSERVATION TO GP ===
16: GP_Y.AddObservation(θ_pref, J_pseudo)    ▷ "Better" than reality
17: 
18: // Note: This biases BO to explore near θ_pref in future iterations
```

---

## Algorithm 5: Search Space Adaptation

```
Algorithm 8: Refine Search Space (Exploitation)
─────────────────────────────────────────────────────────────────────
Input:  θ_center - Center point for refinement
        shrink_factor λ ∈ (0,1) - Shrinkage ratio
        bounds_current - Current adaptive bounds
        global_bounds - Hard domain limits
        
Output: bounds_new - Refined (smaller) bounds
─────────────────────────────────────────────────────────────────────

1:  // === COMPUTE NEW RANGE ===
2:  range_current ← bounds_current_max - bounds_current_min
3:  range_new ← λ · range_current
4:  
5:  // === CENTER NEW BOUNDS ===
6:  bounds_new_min ← θ_center - range_new / 2
7:  bounds_new_max ← θ_center + range_new / 2
8:  
9:  // === CLAMP TO GLOBAL BOUNDS ===
10: bounds_new_min ← MAX(bounds_new_min, global_bounds_min)
11: bounds_new_max ← MIN(bounds_new_max, global_bounds_max)
12: bounds_new_max ← MAX(bounds_new_max, bounds_new_min + ε)
13: 
14: // === FOR DE: RESTART POPULATION ===
15: if optimizer = DE then
16:     mutation ← 0.8 · mutation            ▷ Reduce exploration
17:     population ← InitializePopulation(bounds_new)
18:     population[0] ← θ_center             ▷ Keep center point
19: end if
20: 
21: return bounds_new


Algorithm 9: Expand Search Space (Exploration)
─────────────────────────────────────────────────────────────────────
Input:  expand_factor γ > 1 - Expansion ratio
        bounds_current - Current adaptive bounds
        global_bounds - Hard domain limits
        
Output: bounds_new - Expanded (larger) bounds
─────────────────────────────────────────────────────────────────────

1:  // === COMPUTE CENTER ===
2:  θ_center ← (bounds_current_min + bounds_current_max) / 2
3:  
4:  // === COMPUTE NEW RANGE ===
5:  range_current ← bounds_current_max - bounds_current_min
6:  range_new ← γ · range_current
7:  
8:  // === EXPAND SYMMETRICALLY ===
9:  bounds_new_min ← θ_center - range_new / 2
10: bounds_new_max ← θ_center + range_new / 2
11: 
12: // === CLAMP TO GLOBAL BOUNDS ===
13: bounds_new_min ← MAX(bounds_new_min, global_bounds_min)
14: bounds_new_max ← MIN(bounds_new_max, global_bounds_max)
15: bounds_new_max ← MAX(bounds_new_max, bounds_new_min + ε)
16: 
17: // === FOR DE: RESTART WITH MORE EXPLORATION ===
18: if optimizer = DE then
19:     mutation ← MIN(1.2 · mutation, 1.0)  ▷ Increase up to max 1.0
20:     population ← InitializePopulation(bounds_new)
21: end if
22: 
23: return bounds_new
```

---

## Algorithm 6: Candidate Injection

```
Algorithm 10: Inject Candidate into DE Population
─────────────────────────────────────────────────────────────────────
Input:  θ_inject - Candidate to inject
        P - Current DE population
        J, g - Current fitness and violations
        protect_best - Whether to protect best individual
        
Output: Updated population with injected candidate
─────────────────────────────────────────────────────────────────────

1:  // === CLIP TO VALID BOUNDS ===
2:  θ_inject ← CLIP(θ_inject, bounds_min, bounds_max)
3:  θ_inject ← CLIP(θ_inject, global_bounds_min, global_bounds_max)
4:  
5:  // === EVALUATE CANDIDATE ===
6:  J_inject, g_inject ← EvaluatePID(θ_inject)
7:  
8:  // === SELECT REPLACEMENT INDEX ===
9:  if protect_best then
10:     idx_best ← argmin_i {J[i] | using feasibility rules}
11:     candidates ← {1,...,N_pop} \ {idx_best}
12: else
13:     candidates ← {1,...,N_pop}
14: end if
15: 
16: idx_replace ← RANDOM_CHOICE(candidates)
17: 
18: // === REPLACE POPULATION MEMBER ===
19: P[idx_replace] ← θ_inject
20: J[idx_replace] ← J_inject
21: g[idx_replace] ← g_inject
22: 
23: return P, J, g
```

---

## Algorithm 7: PID Evaluation and Metrics

```
Algorithm 11: PID Controller Evaluation
─────────────────────────────────────────────────────────────────────
Input:  θ = [K_p, K_i, K_d] - PID parameters
        r(t) - Reference trajectory (target)
        T_sim - Simulation duration
        Δt - Time step
        u_max - Actuator saturation limit
        
Output: J - Total cost (objective)
        g - Constraint violation
        metrics - Performance metrics
        response - Time-series response
─────────────────────────────────────────────────────────────────────

1:  // === INITIALIZATION ===
2:  e_integral ← 0
3:  y_prev ← 0
4:  J_total ← 0
5:  u_max_observed ← 0
6:  sat_steps ← 0
7:  response ← ∅
8:  
9:  // === SIMULATION LOOP ===
10: for t ← 0 to T_sim step Δt do
11:     
12:     // === GET CURRENT STATE ===
13:     y(t) ← MeasureSystemOutput()         ▷ From physics simulator
14:     
15:     // === COMPUTE ERROR ===
16:     e(t) ← r(t) - y(t)
17:     e_integral ← e_integral + e(t) · Δt
18:     
19:     // === DERIVATIVE-ON-MEASUREMENT ===
20:     ė_measured ← (y(t) - y_prev) / Δt    ▷ Reduces derivative kick
21:     
22:     // === PID CONTROL LAW ===
23:     u_raw(t) ← K_p · e(t) + K_i · e_integral - K_d · ė_measured
24:     
25:     // === SATURATION HANDLING ===
26:     u_max_observed ← MAX(u_max_observed, |u_raw(t)|)
27:     u(t) ← CLIP(u_raw(t), -u_max, u_max)
28:     
29:     if |u_raw(t)| > u_max then
30:         sat_steps ← sat_steps + 1
31:         
32:         // === ANTI-WINDUP ===
33:         if SIGN(u_raw(t)) = SIGN(e(t)) then
34:             e_integral ← e_integral - e(t) · Δt
35:         end if
36:     end if
37:     
38:     // === APPLY CONTROL ===
39:     ApplyControlInput(u(t))              ▷ To physics simulator
40:     StepSimulation(Δt)
41:     
42:     // === ACCUMULATE COST ===
43:     sat_excess ← MAX(0, |u_raw(t)| - u_max)
44:     J_total ← J_total + e(t)² + 0.001·u(t)² + λ_sat·sat_excess²
45:     
46:     // === RECORD RESPONSE ===
47:     response ← response ∪ {(t, r(t), y(t), u(t))}
48:     y_prev ← y(t)
49:     
50: end for
51: 
52: // === STRICT SATURATION PENALTY ===
53: if u_max_observed > u_max then
54:     excess_ratio ← (u_max_observed - u_max) / u_max
55:     J_total ← J_total + λ_hard · (1 + excess_ratio)
56: end if
57: 
58: // === NORMALIZE COST ===
59: J ← J_total / (T_sim / Δt)
60: 
61: // === CONSTRAINT VIOLATION ===
62: g ← u_max_observed - u_max               ▷ g ≤ 0 is feasible
63: 
64: // === PERFORMANCE METRICS ===
65: metrics ← ComputePerformanceMetrics(response)
66: 
67: return J, g, metrics, response


Algorithm 12: Performance Metrics Computation
─────────────────────────────────────────────────────────────────────
Input:  response - Time series {(t, r, y, u)}
        r_target - Target value
        
Output: metrics - {overshoot, rise_time, settling_time}
─────────────────────────────────────────────────────────────────────

1:  y_max ← MAX{y | (t, r, y, u) ∈ response}
2:  
3:  // === OVERSHOOT ===
4:  if y_max > r_target then
5:      overshoot ← 100 · (y_max - r_target) / r_target
6:  else
7:      overshoot ← 0
8:  end if
9:  
10: // === RISE TIME (10% to 90%) ===
11: t_10 ← MIN{t | y(t) ≥ 0.1 · r_target}
12: t_90 ← MIN{t | y(t) ≥ 0.9 · r_target}
13: rise_time ← t_90 - t_10
14: 
15: // === SETTLING TIME (±5% of target) ===
16: tolerance ← 0.05 · r_target
17: last_violation ← MAX{t | |y(t) - r_target| > tolerance}
18: settling_time ← last_violation
19: 
20: return {overshoot, rise_time, settling_time}
```

---

## Key Algorithmic Features

### 1. **Dual-Algorithm Synergy**
- DE provides robust global exploration via population-based evolution
- BO leverages probabilistic surrogate models for sample-efficient local optimization
- Bidirectional knowledge sharing: BO learns from all DE evaluations

### 2. **Adaptive Preference Learning**
- Exponential moving average (α = 0.3) balances responsiveness and stability
- Anchor injection guides both optimizers toward human-preferred regions
- Synthetic observation nudging biases BO's Gaussian Process toward preferences

### 3. **Feasibility-Aware Selection**
- Deb's constraint handling: feasible solutions always dominate infeasible ones
- Among feasible: minimize objective; among infeasible: minimize violation
- Ensures physically realizable controllers (actuator saturation constraints)

### 4. **Multi-Modal Feedback Mechanism**
- **Preference** (PREFER_DE/BO): Update weights, inject anchors, cross-pollinate
- **Refinement** (TIE): Shrink bounds by 50%, reduce mutation, intensify local search
- **Expansion** (REJECT): Expand bounds by 50%, increase mutation, explore new regions
- **Termination**: Auto-stop when performance targets and constraints are satisfied

### 5. **Hierarchical Bounds Management**
- **Global bounds**: Hard parameter limits, never violated
- **Adaptive bounds**: Dynamically adjusted per human feedback
- Ensures mathematical validity while enabling focused search

---

## Computational Complexity

| Component | Complexity | Notes |
|-----------|------------|-------|
| DE Evolution | 𝒪(N_pop · n) | n = parameter dimensions |
| BO GP Training | 𝒪(N_obs³) | N_obs = number of observations |
| BO Acquisition | 𝒪(N_cand · n) | N_cand = candidate samples |
| PID Simulation | 𝒪(T_sim/Δt) | Physics simulation steps |
| Overall Iteration | 𝒪(N_pop · T_sim/Δt) | Dominated by simulations |

---

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| θ | PID parameter vector [K_p, K_i, K_d] |
| J | Objective function (cost) |
| g | Constraint violation (g ≤ 0 is feasible) |
| w | Preference weight vector |
| α | Preference learning rate |
| F | DE mutation factor |
| ρ_min | Minimum probability of feasibility (BO) |
| λ | Shrink/expand factor for bounds |
| GP | Gaussian Process |
| EI | Expected Improvement |
| PoF | Probability of Feasibility |
| EIC | Expected Improvement with Constraints |

---

## References

**Algorithmic Foundations:**
1. Storn & Price (1997) - Differential Evolution
2. Mockus (1975) - Bayesian Optimization
3. Deb (2000) - Constraint handling in evolutionary algorithms
4. Schonlau et al. (1998) - Expected Improvement acquisition
5. Gardner et al. (2014) - Constrained Bayesian Optimization

**PID Control:**
6. Åström & Hägglund (1995) - PID Controllers: Theory, Design, and Tuning
7. Anti-windup techniques for saturating controllers

---

*This pseudocode provides a complete, unambiguous specification of the Mixed HIL optimization framework suitable for reproduction in academic publications.*
