# Complete Guide: How DE and BO React to Human Feedback

This guide explains in **simple terms** with **complete code details** how Differential Evolution (DE) and Bayesian Optimization (BO) respond to different types of human feedback.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Mixed HIL: 4 Feedback Options](#mixed-hil-4-feedback-options)
3. [Single DE HIL: 2 Feedback Options](#single-de-hil-2-feedback-options)
4. [Single BO HIL: 2 Feedback Options](#single-bo-hil-2-feedback-options)
5. [Deep Dive: Core Mechanisms](#deep-dive-core-mechanisms)

---

## Introduction

### What You Need to Know

**Human-in-the-Loop (HIL)** means you guide the optimization by giving feedback. Think of it like training a dog - you tell it "good" or "bad" and it learns.

**Two Systems:**
1. **Mixed HIL**: DE and BO compete together. You compare both and pick the winner.
2. **Single HIL**: Only one algorithm runs. You say "good" or "bad" to single candidates.

**The Goal:** Find the best PID controller parameters (Kp, Ki, Kd) that make a robot turn correctly.

---

## Mixed HIL: 4 Feedback Options

**File:** `main_macos.py`

In each iteration, DE and BO each propose one candidate. You see both and choose from 4 options.

---

### Option 1: "Prefer A (DE)" - I like the DE candidate better

#### What This Means
You're telling the system: "DE found something good, BO should learn from it."

#### What Happens to DE

##### 1. Reset Mutation Factor
**Location:** `main_macos.py`, Line 586

```python
de.mutation_factor = BASE_MUTATION
```

**What it does:** Resets mutation to 0.5 (default value). This makes DE more exploratory again after it might have become too conservative.

**Why:** When you prefer DE, it means DE is on the right track. Resetting mutation keeps it exploring effectively.

---

##### 2. Update Preference Model
**Location:** `main_macos.py`, Line 583

```python
gap = pref_model.update_towards(cand_a, cand_b)
```

**What happens inside `update_towards()`:**
**Location:** `main_macos.py`, Lines 237-248

```python
def update_towards(self, preferred, other):
    preferred = np.array(preferred, dtype=float)
    other = np.array(other, dtype=float)
    min_b = self.bounds[:, 0]
    span = self.bounds[:, 1] - min_b

    # Normalize parameters to [0,1] range
    pref_norm = (preferred - min_b) / (span + 1e-9)
    other_norm = (other - min_b) / (span + 1e-9)
    gap = other_norm - pref_norm

    # Move weights toward preferred parameters
    self.weights = self.weights + self.lr * (pref_norm - self.weights)
    self._normalize()
    return gap
```

**What it does:** 
- Normalizes both candidates to [0,1] scale
- Calculates the "gap" (difference) between them
- Moves internal "preference weights" closer to the DE candidate
- Learning rate (lr) is 0.3, so weights shift 30% toward DE

**Example:**
- DE candidate: `[5.0, 2.0, 1.0]` (Kp, Ki, Kd)
- BO candidate: `[2.0, 1.0, 3.0]`
- Bounds: `[(0.1, 10.0), (0.01, 10.0), (0.01, 10.0)]`
- Normalized DE: `[0.495, 0.199, 0.099]`
- Weights shift 30% toward these values

---

##### 3. Calculate Anchor Point
**Location:** `main_macos.py`, Line 584

```python
anchor = pref_model.anchor_params()
```

**What happens inside `anchor_params()`:**
**Location:** `main_macos.py`, Lines 232-235

```python
def anchor_params(self):
    min_b = self.bounds[:, 0]
    max_b = self.bounds[:, 1]
    return min_b + self.weights * (max_b - min_b)
```

**What it does:** 
- Uses the updated preference weights to calculate a "preference anchor"
- This anchor represents the parameter region you seem to prefer
- Formula: `anchor = min + weights × (max - min)`

**Example:**
- Bounds: `[(0.1, 10.0), (0.01, 10.0), (0.01, 10.0)]`
- Weights after update: `[0.5, 0.3, 0.2]`
- Anchor = `[0.1 + 0.5×9.9, 0.01 + 0.3×9.99, 0.01 + 0.2×9.99]` = `[5.05, 3.01, 2.01]`

---

##### 4. Inject Anchor into DE Population
**Location:** `main_macos.py`, Line 587

```python
de.inject_candidate(anchor, eval_func=fitness_wrapper, protect_best=True)
```

**What happens inside `inject_candidate()`:**
**Location:** `differential_evolution.py`, Lines 132-168

```python
def inject_candidate(self, candidate, eval_func=None, fitness=None, violation=None, protect_best=True):
    # Clip candidate to bounds
    cand = np.array(candidate, dtype=float).reshape(-1)
    cand = np.clip(cand, self.bounds[:, 0], self.bounds[:, 1])
    cand = np.clip(cand, self.global_bounds[:, 0], self.global_bounds[:, 1])

    # Evaluate the candidate
    if fitness is None or violation is None:
        if eval_func is None:
            raise ValueError("inject_candidate requires either (fitness, violation) or eval_func.")
        fitness, violation = self._as_fit_viol(eval_func(cand))
    else:
        fitness = float(fitness)
        violation = float(violation)

    # Choose which population member to replace
    if self.pop_size <= 1:
        idx = 0
    else:
        # Protect the best individual from being replaced
        protected = self.best_idx if (protect_best and self.best_idx >= 0) else None
        choices = [i for i in range(self.pop_size) if i != protected]
        idx = int(np.random.choice(choices))

    # Replace chosen member with the anchor
    self.population[idx] = cand
    self.fitness_scores[idx] = fitness
    self.violations[idx] = violation
    self._update_best_idx()

    return idx, float(fitness), float(violation)
```

**What it does:**
1. Takes the anchor point
2. Clips it to valid bounds (ensures valid parameters)
3. Evaluates it by running the simulation
4. Chooses a random population member to replace (but NEVER replaces the best one)
5. Puts the anchor into the population
6. Updates the best individual tracker

**Why protect_best=True:** We don't want to lose the best solution we've found so far.

---

#### What Happens to BO

##### BO Receives Preference Nudge
**Location:** `main_macos.py`, Line 589

```python
bo.nudge_with_preference(cand_a, fit_a, fit_b, viol_a)
```

**What happens inside `nudge_with_preference()`:**
**Location:** `bayesian_optimization.py`, Lines 232-242

```python
def nudge_with_preference(self, preferred, preferred_cost, other_cost, preferred_violation, strength=0.2):
    # Only nudge if the preferred candidate is feasible
    preferred_violation = float(preferred_violation)
    if preferred_violation > 0.0:
        return

    preferred = np.array(preferred, dtype=float).reshape(-1)
    gap = abs(float(other_cost) - float(preferred_cost))
    
    # Create artificial lower cost
    pseudo_cost = float(preferred_cost) - (strength * gap if gap > 0 else strength * 0.01)

    # Add this virtual data point to BO's knowledge
    self.update(preferred, pseudo_cost, preferred_violation)
    print(f"[BO] Preference nudge applied. Gap={gap:.4f}, pseudo_cost={pseudo_cost:.4f}")
```

**What it does:**
1. Checks if DE candidate is feasible (violation ≤ 0)
2. Calculates the gap (difference) between DE and BO costs
3. Creates a "pseudo cost" that's lower than the actual cost
4. Adds this fake lower cost to BO's Gaussian Process

**Example:**
- DE cost: 10.5
- BO cost: 15.2
- Gap: 4.7
- Strength: 0.2 (20%)
- Pseudo cost: 10.5 - (0.2 × 4.7) = 10.5 - 0.94 = 9.56

**Why do this:** This "tricks" BO into thinking the DE region is even better than it really is, so BO will search there more in the future.

---

##### BO Also Updates with Both Candidates
**Location:** `main_macos.py`, Lines 473-474

```python
bo.update(cand_b, fit_b_fast, viol_b_fast)
bo.update(cand_a, fit_a_fast, viol_a_fast)
```

**What happens inside `update()`:**
**Location:** `bayesian_optimization.py`, Lines 73-85

```python
def update(self, X, y, g):
    x = np.array(X, dtype=float).reshape(-1)
    y = float(y)
    g = float(g)

    # Clamp to global domain
    x = np.clip(x, self.global_bounds[:, 0], self.global_bounds[:, 1])

    # Add to historical data
    self.X_raw.append(x)
    self.Y.append(y)
    self.G.append(g)

    # Retrain Gaussian Processes
    self._fit_models()
```

**What it does:**
- Adds the candidate to BO's history
- `X_raw`: all parameter combinations tried
- `Y`: all costs/fitness values
- `G`: all constraint violations
- Retrains both Gaussian Processes (one for cost, one for constraints)

**Why:** BO learns from BOTH candidates, not just its own. This is knowledge sharing.

---

#### Summary of "Prefer A (DE)"

```
YOU: "DE is better"
     ↓
DE Actions:
  1. ✓ Reset mutation to 0.5
  2. ✓ Update preference weights toward DE parameters
  3. ✓ Calculate anchor from weights
  4. ✓ Inject anchor into population (protect best)
     ↓
BO Actions:
  1. ✓ Update with both DE and BO candidates
  2. ✓ Add virtual "better" version of DE candidate
     ↓
RESULT: DE gets reinforced, BO learns to search near DE's region
```

---

### Option 2: "Prefer B (BO)" - I like the BO candidate better

#### What This Means
You're telling the system: "BO found something good, DE should learn from it."

#### What Happens to BO

##### Same as Prefer A, but for BO
**Location:** `main_macos.py`, Line 601

```python
bo.nudge_with_preference(cand_b, fit_b, fit_a, viol_b)
```

This is identical to the nudge in Prefer A, but now:
- Preferred = BO candidate
- Other = DE candidate
- BO gets the preference boost

---

#### What Happens to DE

##### 1. Reset Mutation Factor
**Location:** `main_macos.py`, Line 597

```python
de.mutation_factor = BASE_MUTATION
```

Same as Prefer A - reset to 0.5.

---

##### 2. Update Preference Model Toward BO
**Location:** `main_macos.py`, Line 594

```python
gap = pref_model.update_towards(cand_b, cand_a)
```

Same logic as Prefer A, but now weights move toward BO's parameters.

---

##### 3. Calculate New Anchor
**Location:** `main_macos.py`, Line 595

```python
anchor = pref_model.anchor_params()
```

Anchor is now based on weights that favor BO's region.

---

##### 4. Inject BOTH BO Candidate AND Anchor
**Location:** `main_macos.py`, Lines 598-599

```python
de.inject_candidate(cand_b, eval_func=fitness_wrapper, protect_best=True)
de.inject_candidate(anchor, eval_func=fitness_wrapper, protect_best=True)
```

**Why TWO injections?**
1. **BO candidate**: Gives DE the exact good solution BO found
2. **Anchor**: Gives DE a point based on your historical preferences

This is a stronger signal than Prefer A. When you prefer BO, DE might be searching in the wrong area, so we give it TWO new directions to explore.

**How this works:**
- First injection: Replaces a random population member (except best) with BO's candidate
- Second injection: Replaces another random member (except best) with the anchor
- Net result: 2 out of ~6-8 population members are now influenced by BO

---

#### Summary of "Prefer B (BO)"

```
YOU: "BO is better"
     ↓
BO Actions:
  1. ✓ Update with both candidates
  2. ✓ Add virtual "better" version of BO candidate
     ↓
DE Actions:
  1. ✓ Reset mutation to 0.5
  2. ✓ Update preference weights toward BO parameters
  3. ✓ Calculate anchor from new weights
  4. ✓ Inject BO candidate into population (protect best)
  5. ✓ Inject anchor into population (protect best)
     ↓
RESULT: BO gets reinforced, DE gets STRONG push toward BO's region
```

---

### Option 3: "TIE (Refine)" - Both are similarly good, zoom in

#### What This Means
You're telling the system: "You're both in the right area, now search more carefully in this region."

#### What Happens to BOTH DE and BO

##### 1. Calculate Midpoint
**Location:** `main_macos.py`, Line 606

```python
avg_c = (np.array(cand_a) + np.array(cand_b)) / 2.0
```

**What it does:**
- Takes the average of DE and BO candidates
- Example: DE = `[5.0, 2.0, 1.0]`, BO = `[3.0, 4.0, 3.0]`
- Midpoint = `[4.0, 3.0, 2.0]`

**Why:** This midpoint becomes the center of the new, smaller search region.

---

##### 2. DE Refines Search Space
**Location:** `main_macos.py`, Line 607

```python
de.refine_search_space(avg_c)
```

**What happens inside `refine_search_space()`:**
**Location:** `differential_evolution.py`, Lines 174-205

```python
def refine_search_space(self, center_candidate, shrink_factor=0.5):
    center = np.array(center_candidate, dtype=float).reshape(-1)
    shrink_factor = float(shrink_factor)

    # Calculate new range (50% of current)
    current_range = self.bounds[:, 1] - self.bounds[:, 0]
    new_range = current_range * shrink_factor

    # Calculate new bounds centered on the midpoint
    min_b = center - (new_range / 2.0)
    max_b = center + (new_range / 2.0)

    # CRITICAL: Clamp to global bounds (never go outside allowed range)
    min_b = np.maximum(min_b, self.global_bounds[:, 0])
    max_b = np.minimum(max_b, self.global_bounds[:, 1])
    max_b = np.maximum(max_b, min_b + 1e-9)

    self.bounds = np.column_stack((min_b, max_b))

    # Reduce mutation for exploitation
    self.mutation_factor *= 0.8

    # Reinitialize population in new bounds
    self.population = self._initialize_population()
    self.population[0] = np.clip(center, self.bounds[:, 0], self.bounds[:, 1])

    # Reset scores (force re-evaluation)
    self.fitness_scores[:] = np.inf
    self.violations[:] = np.inf
    self.best_idx = -1

    print(f"[DE] Refine Mode Active. New Bounds (clipped to global): {self.bounds}")
```

**Step-by-step example:**
```
Current bounds: Kp=[0.1, 10.0], Ki=[0.01, 10.0], Kd=[0.01, 10.0]
Current range:  Kp=9.9,         Ki=9.99,         Kd=9.99
Shrink by 50%:  Kp=4.95,        Ki=4.995,        Kd=4.995
Center (midpoint): [4.0, 3.0, 2.0]

New bounds:
  Kp: [4.0 - 4.95/2, 4.0 + 4.95/2] = [1.525, 6.475]
  Ki: [3.0 - 4.995/2, 3.0 + 4.995/2] = [0.5025, 5.4975]
  Kd: [2.0 - 4.995/2, 2.0 + 4.995/2] = [-0.4975, 4.4975]
  
After clamping to global [0.1, 10.0], [0.01, 10.0], [0.01, 10.0]:
  Kp: [1.525, 6.475]  ✓ valid
  Ki: [0.5025, 5.4975] ✓ valid
  Kd: [0.01, 4.4975]  ✓ clamped min to 0.01

Mutation: 0.5 × 0.8 = 0.4 (smaller, more exploitative)
```

**What else happens:**
- Entire population is randomized within new smaller bounds
- First population member is set to the midpoint
- All fitness scores reset to infinity (will re-evaluate)

**Why reinitialize population:**
Old population members might be outside the new bounds. Starting fresh ensures all individuals are in the refined region.

---

##### 3. BO Refines Bounds
**Location:** `main_macos.py`, Line 608

```python
bo.refine_bounds(avg_c)
```

**What happens inside `refine_bounds()`:**
**Location:** `bayesian_optimization.py`, Lines 198-213

```python
def refine_bounds(self, center_candidate, shrink_factor=0.5):
    center = np.array(center_candidate, dtype=float).reshape(-1)
    shrink_factor = float(shrink_factor)

    # Calculate new range (50% of current)
    current_range = self.bounds[:, 1] - self.bounds[:, 0]
    new_range = current_range * shrink_factor

    # Calculate new bounds centered on the midpoint
    min_b = center - (new_range / 2.0)
    max_b = center + (new_range / 2.0)

    # Clamp to global bounds
    min_b = np.maximum(min_b, self.global_bounds[:, 0])
    max_b = np.minimum(max_b, self.global_bounds[:, 1])
    max_b = np.maximum(max_b, min_b + 1e-9)

    self.bounds = np.column_stack((min_b, max_b))
    print(f"[BO] Refine Mode Active. New Bounds: {self.bounds}")
```

**What it does:**
- Same math as DE (shrink bounds by 50%, center on midpoint)
- BUT: BO keeps its Gaussian Process (doesn't reset)
- All historical data is preserved
- Only future proposals are limited to new bounds

**Key difference from DE:**
- DE: Restarts population, loses individuals
- BO: Keeps all knowledge, just narrows search

---

#### Summary of "TIE (Refine)"

```
YOU: "Both are good, zoom in"
     ↓
Calculate midpoint: (DE + BO) / 2
     ↓
DE Actions:
  1. ✓ Shrink bounds by 50% around midpoint
  2. ✓ Reduce mutation by 20% (0.5 → 0.4)
  3. ✓ Reinitialize population in new bounds
  4. ✓ Put midpoint as first population member
  5. ✓ Reset all scores
     ↓
BO Actions:
  1. ✓ Shrink bounds by 50% around midpoint
  2. ✓ Keep all Gaussian Process knowledge
  3. ✓ Limit future proposals to new bounds
     ↓
RESULT: Both focus on smaller region around the midpoint, intensive local search
```

---

### Option 4: "REJECT Both" - Neither is good, explore elsewhere

#### What This Means
You're telling the system: "You're both looking in the wrong place, expand your search!"

#### What Happens to DE

##### DE Expands Search Space
**Location:** `main_macos.py`, Line 613

```python
de.expand_search_space()
```

**What happens inside `expand_search_space()`:**
**Location:** `differential_evolution.py`, Lines 207-245

```python
def expand_search_space(self, expand_factor=1.5):
    expand_factor = float(expand_factor)

    # Calculate center of current bounds
    center = np.mean(self.bounds, axis=1)
    current_range = self.bounds[:, 1] - self.bounds[:, 0]
    
    # Expand range by 50%
    new_range = current_range * expand_factor

    # Calculate new bounds
    min_b = center - (new_range / 2.0)
    max_b = center + (new_range / 2.0)

    # CRITICAL: Clamp to global bounds
    min_b = np.maximum(min_b, self.global_bounds[:, 0])
    max_b = np.minimum(max_b, self.global_bounds[:, 1])
    max_b = np.maximum(max_b, min_b + 1e-9)

    prev_bounds = self.bounds.copy()
    self.bounds = np.column_stack((min_b, max_b))

    # Increase mutation for exploration (cap at 1.0)
    self.mutation_factor = min(self.mutation_factor * 1.2, 1.0)

    # Restart population
    self.population = self._initialize_population()
    self.fitness_scores[:] = np.inf
    self.violations[:] = np.inf
    self.best_idx = -1

    # Log message
    changed = not np.allclose(prev_bounds, self.bounds)
    if changed:
        print(f"[DE] Search Space Expanded (clipped to global). New Bounds: {self.bounds}")
    else:
        print(f"[DE] Search Space at GLOBAL bounds; diversification restart. Bounds: {self.bounds}")
```

**Step-by-step example:**
```
Current bounds: Kp=[2.0, 6.0], Ki=[1.0, 5.0], Kd=[0.5, 3.0]
Current range:  Kp=4.0,        Ki=4.0,       Kd=2.5
Expand by 50%:  Kp=6.0,        Ki=6.0,       Kd=3.75
Center:         Kp=4.0,        Ki=3.0,       Kd=1.75

New bounds:
  Kp: [4.0 - 6.0/2, 4.0 + 6.0/2] = [1.0, 7.0]
  Ki: [3.0 - 6.0/2, 3.0 + 6.0/2] = [0.0, 6.0]
  Kd: [1.75 - 3.75/2, 1.75 + 3.75/2] = [-0.125, 3.625]

After clamping to global [0.1, 10.0], [0.01, 10.0], [0.01, 10.0]:
  Kp: [1.0, 7.0] ✓ valid
  Ki: [0.01, 6.0] ✓ clamped min
  Kd: [0.01, 3.625] ✓ clamped min

Mutation: 0.4 × 1.2 = 0.48 (larger, more exploratory)
If already 0.8: 0.8 × 1.2 = 0.96 (capped at 1.0)
```

**What else happens:**
- Entire population is randomized
- All scores reset
- If bounds were already at global limits, this becomes a "restart" with higher mutation

---

#### What Happens to BO

##### BO Expands Bounds
**Location:** `main_macos.py`, Line 614

```python
bo.expand_bounds()
```

**What happens inside `expand_bounds()`:**
**Location:** `bayesian_optimization.py`, Lines 215-230

```python
def expand_bounds(self, expand_factor=1.5):
    expand_factor = float(expand_factor)

    # Calculate center of current bounds
    center = np.mean(self.bounds, axis=1)
    current_range = self.bounds[:, 1] - self.bounds[:, 0]
    
    # Expand range by 50%
    new_range = current_range * expand_factor

    # Calculate new bounds
    min_b = center - (new_range / 2.0)
    max_b = center + (new_range / 2.0)

    # Clamp to global bounds
    min_b = np.maximum(min_b, self.global_bounds[:, 0])
    max_b = np.minimum(max_b, self.global_bounds[:, 1])
    max_b = np.maximum(max_b, min_b + 1e-9)

    self.bounds = np.column_stack((min_b, max_b))
    print(f"[BO] Search Space Expanded. New Bounds: {self.bounds}")
```

**What it does:**
- Same expansion math as DE (expand by 50%)
- BUT: BO keeps its Gaussian Process intact
- All historical "bad" data is preserved - BO learns what NOT to do
- Future proposals will explore the wider region

**Key difference from DE:**
- DE: Loses population, gets fresh start
- BO: Keeps all knowledge, including the rejected regions (helps avoid them)

---

#### Summary of "REJECT Both"

```
YOU: "Both are bad, look elsewhere"
     ↓
DE Actions:
  1. ✓ Expand bounds by 50% (within global limits)
  2. ✓ Increase mutation by 20% (0.4 → 0.48)
  3. ✓ Completely restart population
  4. ✓ Reset all scores
     ↓
BO Actions:
  1. ✓ Expand bounds by 50% (within global limits)
  2. ✓ Keep all Gaussian Process knowledge
  3. ✓ Future proposals explore wider region
     ↓
RESULT: Both explore wider area, DE with fresh start, BO learning from mistakes
```

---

## Single DE HIL: 2 Feedback Options

**File:** `de_hil_updated.py`

Only DE runs. Each iteration, you see one DE candidate and choose: ACCEPT or REJECT.

---

### Option 1: ACCEPT - This candidate is good

#### What Happens

**Location:** `de_hil_updated.py`, Lines 555-562

```python
if choice == 1:  # ACCEPT -> REFINE
    print("User ACCEPTED. Refining search space around best candidate.")
    de.refine_search_space(cand)
    clamp_de_bounds_to_global(de)
    # keep a conservative seed in-pop (optional)
    if getattr(de, "population", None) is not None and de.population.shape[0] >= 2:
        de.population[1] = np.clip(safe_seed, de.bounds[:, 0], de.bounds[:, 1])
    log_label = "accept_refine"
```

**What this does:**

##### 1. Refine Search Space
Calls the same `refine_search_space()` function from Mixed HIL:
- Shrinks bounds by 50% around accepted candidate
- Reduces mutation by 20%
- Reinitializes population
- Resets scores

##### 2. Clamp to Global Bounds
**Location:** `de_hil_updated.py`, Lines 408-424

```python
def clamp_de_bounds_to_global(de_obj):
    """
    If your DE expands/refines bounds, clamp them back into the declared PID_BOUNDS.
    This prevents the 'DE escapes bounds after reject' effect when PID_BOUNDS are meant
    to be hard limits for the experiment.
    """
    if not hasattr(de_obj, "bounds"):
        return

    b = np.array(de_obj.bounds, dtype=float)
    b[:, 0] = np.maximum(b[:, 0], GLOBAL_BOUNDS[:, 0])
    b[:, 1] = np.minimum(b[:, 1], GLOBAL_BOUNDS[:, 1])
    b[:, 1] = np.maximum(b[:, 1], b[:, 0] + 1e-9)
    de_obj.bounds = b

    if hasattr(de_obj, "population") and de_obj.population is not None:
        de_obj.population = np.clip(de_obj.population, de_obj.bounds[:, 0], de_obj.bounds[:, 1])
```

**What this does:**
- Double-checks that bounds don't go outside global limits
- Also clips the entire population to bounds
- This is a safety measure to prevent numerical errors

##### 3. Inject Safe Seed
```python
de.population[1] = np.clip(safe_seed, de.bounds[:, 0], de.bounds[:, 1])
```

**What is safe_seed?**
**Location:** `de_hil_updated.py`, Lines 441-443

```python
safe_seed = np.array([b[0] for b in PID_BOUNDS], dtype=float)
```

This is simply the minimum values: `[0.1, 0.01, 0.01]`

**Why inject it?**
Provides a conservative fallback. Even if all other population members are aggressive, this one is very safe.

---

#### Summary of ACCEPT in Single DE HIL

```
YOU: "This DE candidate is good"
     ↓
DE Actions:
  1. ✓ Refine bounds by 50% around candidate
  2. ✓ Reduce mutation by 20%
  3. ✓ Reinitialize population
  4. ✓ Clamp everything to global bounds
  5. ✓ Inject safe seed as backup
  6. ✓ Reset scores
     ↓
RESULT: DE zooms in on the good region, with a safety net
```

---

### Option 2: REJECT - This candidate is bad

#### What Happens

**Location:** `de_hil_updated.py`, Lines 564-570

```python
elif choice == 2:  # REJECT -> EXPAND
    print("User REJECTED. Expanding search space.")
    de.expand_search_space()
    clamp_de_bounds_to_global(de)
    if getattr(de, "population", None) is not None and de.population.shape[0] >= 2:
        de.population[1] = np.clip(safe_seed, de.bounds[:, 0], de.bounds[:, 1])
    log_label = "reject_expand"
```

**What this does:**

##### 1. Expand Search Space
Calls the same `expand_search_space()` function from Mixed HIL:
- Expands bounds by 50% (within global limits)
- Increases mutation by 20%
- Restarts population
- Resets scores

##### 2. Clamp to Global Bounds
Same safety check as ACCEPT.

##### 3. Inject Safe Seed
Same conservative backup as ACCEPT.

---

#### Summary of REJECT in Single DE HIL

```
YOU: "This DE candidate is bad"
     ↓
DE Actions:
  1. ✓ Expand bounds by 50% (within global)
  2. ✓ Increase mutation by 20%
  3. ✓ Restart population
  4. ✓ Clamp everything to global bounds
  5. ✓ Inject safe seed as backup
  6. ✓ Reset scores
     ↓
RESULT: DE explores wider area with fresh start, with a safety net
```

---

## Single BO HIL: 2 Feedback Options

**File:** `bo_hil_updated.py`

Only BO runs. Each iteration, you see one BO candidate and choose: ACCEPT or REJECT.

---

### Option 1: ACCEPT - This candidate is good

#### What Happens

**Location:** `bo_hil_updated.py`, Lines 553-570

```python
if choice == 1:  # ACCEPT
    print(f"User ACCEPTED. Refining search space around {cand}")
    bo.refine_bounds(cand)

    # Preference nudge (only if feasible), consistent with main_macos approach
    try:
        bo.nudge_with_preference(
            preferred=cand,
            preferred_cost=float(fit),
            other_cost=float(best_overall_fit),
            preferred_violation=float(violation),
            strength=float(HIL_NUDGE_STRENGTH),
        )
    except TypeError:
        # If your optimizer signature differs, you can remove this block.
        pass

    log_label = "accept_refine"
```

**What this does:**

##### 1. Refine Bounds
Calls `refine_bounds()` - same as Mixed HIL:
- Shrinks bounds by 50% around accepted candidate
- Keeps all Gaussian Process knowledge
- Future proposals limited to smaller region

##### 2. Preference Nudge
Calls `nudge_with_preference()` - same as Mixed HIL:
- Creates virtual data point with lower cost
- Strength is 20% (configurable via `HIL_NUDGE_STRENGTH`)
- Only if candidate is feasible (violation ≤ 0)

**The nudge parameters:**
- `preferred`: The accepted candidate
- `preferred_cost`: Its actual fitness
- `other_cost`: Best fitness seen so far overall
- `strength`: 0.20 (20%)

**Example:**
```
Accepted candidate cost: 12.5
Best overall cost: 10.0
Gap: 2.5
Pseudo cost: 12.5 - (0.2 × 2.5) = 12.5 - 0.5 = 12.0
```

BO adds a virtual data point at the accepted location with cost 12.0 instead of 12.5.

---

#### Summary of ACCEPT in Single BO HIL

```
YOU: "This BO candidate is good"
     ↓
BO Actions:
  1. ✓ Refine bounds by 50% around candidate
  2. ✓ Keep all Gaussian Process knowledge
  3. ✓ Add virtual "better" data point (20% boost)
  4. ✓ Limit future proposals to smaller region
     ↓
RESULT: BO zooms in with reinforcement from preference nudge
```

---

### Option 2: REJECT - This candidate is bad

#### What Happens

**Location:** `bo_hil_updated.py`, Lines 572-575

```python
elif choice == 2:  # REJECT
    print("User REJECTED. Expanding search space.")
    bo.expand_bounds()
    log_label = "reject_expand"
```

**What this does:**

##### 1. Expand Bounds
Calls `expand_bounds()` - same as Mixed HIL:
- Expands bounds by 50% (within global limits)
- Keeps all Gaussian Process knowledge
- Future proposals explore wider region

##### 2. NO Preference Nudge
Unlike ACCEPT, there's no virtual data point added.
The rejected candidate is already in BO's history with its actual (bad) cost.

---

#### Summary of REJECT in Single BO HIL

```
YOU: "This BO candidate is bad"
     ↓
BO Actions:
  1. ✓ Expand bounds by 50% (within global)
  2. ✓ Keep all Gaussian Process knowledge (learns from bad result)
  3. ✓ NO preference nudge
  4. ✓ Future proposals explore wider region
     ↓
RESULT: BO explores wider area, learning to avoid bad regions
```

---

## Deep Dive: Core Mechanisms

### 1. Feasibility Rules

**Used in:** Both DE and BO, all approaches

**Location:** `differential_evolution.py`, Lines 54-70

```python
@staticmethod
def _is_better(fit_a, viol_a, fit_b, viol_b):
    """
    Feasibility rules (Deb-style):
      1) feasible beats infeasible
      2) among feasible: lower fitness wins
      3) among infeasible: lower violation wins
    """
    a_feas = viol_a <= 0.0
    b_feas = viol_b <= 0.0

    if a_feas and not b_feas:
        return True  # A is feasible, B is not -> A wins
    if b_feas and not a_feas:
        return False  # B is feasible, A is not -> B wins
    if a_feas and b_feas:
        return fit_a < fit_b  # Both feasible -> lower cost wins
    return viol_a < viol_b  # Both infeasible -> smaller violation wins
```

**What this means:**

A solution is "feasible" if it satisfies constraints (violation ≤ 0).

**Comparison rules:**
1. Feasible ALWAYS beats infeasible (even if infeasible has better cost)
2. Two feasible → pick the one with lower cost
3. Two infeasible → pick the one with smaller violation

**Example:**
```
Solution A: cost=10, violation=-5 (feasible)
Solution B: cost=5, violation=2 (infeasible)
Winner: A (feasible beats infeasible, even though B has lower cost)

Solution C: cost=10, violation=-5 (feasible)
Solution D: cost=8, violation=-2 (feasible)
Winner: D (both feasible, D has lower cost)

Solution E: cost=10, violation=5 (infeasible)
Solution F: cost=8, violation=10 (infeasible)
Winner: E (both infeasible, E has smaller violation)
```

---

### 2. DE Evolution (How DE Generates New Candidates)

**Location:** `differential_evolution.py`, Lines 72-116

**The Algorithm:**
```python
def evolve(self, eval_func):
    # For each member of the population
    for i in range(self.pop_size):
        # Pick 3 random different members (not including current)
        idxs = [idx for idx in range(self.pop_size) if idx != i]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]

        # Create mutant: a + F × (b - c)
        mutant = a + self.mutation_factor * (b - c)
        mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

        # Crossover: mix mutant with current
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, self.population[i])

        # Evaluate trial
        trial_fit, trial_viol = self._as_fit_viol(eval_func(trial))

        # Replace if better (using feasibility rules)
        if self._is_better(trial_fit, trial_viol, self.fitness_scores[i], self.violations[i]):
            self.population[i] = trial
            self.fitness_scores[i] = trial_fit
            self.violations[i] = trial_viol
```

**Step-by-step example:**

**Population (6 members):**
```
0: [1.0, 2.0, 1.5]  fit=50
1: [3.0, 1.0, 2.0]  fit=45
2: [2.0, 3.0, 1.0]  fit=55
3: [4.0, 2.5, 3.0]  fit=40  ← best
4: [2.5, 1.5, 2.5]  fit=48
5: [3.5, 2.0, 1.5]  fit=52
```

**Evolving member #0:**

1. **Pick 3 random others:** Say we pick #1, #3, #5
   - a = [3.0, 1.0, 2.0]
   - b = [4.0, 2.5, 3.0]
   - c = [3.5, 2.0, 1.5]

2. **Create mutant:** F=0.5
   ```
   b - c = [4.0-3.5, 2.5-2.0, 3.0-1.5] = [0.5, 0.5, 1.5]
   F × (b - c) = 0.5 × [0.5, 0.5, 1.5] = [0.25, 0.25, 0.75]
   mutant = a + [0.25, 0.25, 0.75] = [3.25, 1.25, 2.75]
   ```

3. **Crossover:** CR=0.7, random check for each dimension
   ```
   Random checks: [0.3, 0.8, 0.2] < 0.7 → [True, False, True]
   Trial = mix(mutant, current)
   Trial[0] = mutant[0] = 3.25  (crossover)
   Trial[1] = current[1] = 2.0   (keep current)
   Trial[2] = mutant[2] = 2.75  (crossover)
   Trial = [3.25, 2.0, 2.75]
   ```

4. **Evaluate trial:** Simulate and get fit=42, viol=-1

5. **Compare:** trial_fit=42 vs current_fit=50, both feasible → 42 < 50 → Replace!
   ```
   New population[0] = [3.25, 2.0, 2.75]
   New fitness[0] = 42
   ```

Repeat for all 6 members → One generation complete.

---

### 3. BO Acquisition Function (How BO Picks Next Candidate)

**The Goal:** BO needs to balance:
- **Exploration:** Try uncertain areas (high variance in GP prediction)
- **Exploitation:** Try areas where we predict good values (low mean in GP prediction)
- **Feasibility:** Only try areas likely to satisfy constraints

**Location:** `bayesian_optimization.py`, Lines 136-177

**The Process:**

##### Step 1: Generate Random Candidates
```python
u_min, u_max = self._search_bounds_unit()
U = np.random.uniform(u_min, u_max, size=(2048, self.dim))
```

Generates 2048 random candidates within current bounds.

##### Step 2: Calculate Best Feasible Cost So Far
```python
Y = np.array(self.Y, dtype=float)
G = np.array(self.G, dtype=float)
feas = G <= 0.0
have_feasible = bool(np.any(feas))
best_y = float(np.min(Y[feas])) if have_feasible else None
```

##### Step 3: For Each Candidate, Calculate EIC
**Location:** `bayesian_optimization.py`, Lines 130-134

```python
def _eic(self, u, best_y, xi=0.01):
    pof = self._probability_feasible(u)
    if pof <= 0.0:
        return 0.0
    return self._expected_improvement(u, best_y, xi=xi) * pof
```

**EIC = Expected Improvement × Probability of Feasibility**

**Expected Improvement:**
**Location:** `bayesian_optimization.py`, Lines 108-118

```python
def _expected_improvement(self, u, best_y, xi=0.01):
    if not self._f_is_fit:
        return 0.0
    u = np.array(u, dtype=float).reshape(1, -1)
    mu, sigma = self.gp_f.predict(u, return_std=True)
    mu = float(mu[0])
    sigma = float(max(sigma[0], 1e-12))
    imp = best_y - mu - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    return float(max(0.0, ei))
```

**What this means:**
- `mu`: GP's predicted cost at this location
- `sigma`: GP's uncertainty (high = unexplored)
- `imp`: How much improvement we expect (best_y - mu - small_bonus)
- `Z`: Standardized improvement
- `ei`: Combines improvement and uncertainty

**High EI when:**
- GP predicts low cost (low `mu`) → exploitation
- GP is very uncertain (high `sigma`) → exploration
- Both → best!

**Probability of Feasibility:**
**Location:** `bayesian_optimization.py`, Lines 120-128

```python
def _probability_feasible(self, u):
    if not self._g_is_fit:
        return 0.0
    u = np.array(u, dtype=float).reshape(1, -1)
    mu, sigma = self.gp_g.predict(u, return_std=True)
    mu = float(mu[0])
    sigma = float(max(sigma[0], 1e-12))
    z = (0.0 - mu) / sigma
    return float(norm.cdf(z))
```

**What this means:**
- GP_g predicts violation at this location
- `mu`: predicted violation
- `sigma`: uncertainty
- `z`: How many standard deviations away from feasible (0)
- `norm.cdf(z)`: Probability that actual violation ≤ 0

**Example:**
```
mu = -2.0, sigma = 1.0
z = (0 - (-2)) / 1 = 2.0
PoF = norm.cdf(2.0) = 0.977 (97.7% chance of being feasible)

mu = 2.0, sigma = 1.0
z = (0 - 2) / 1 = -2.0
PoF = norm.cdf(-2.0) = 0.023 (2.3% chance of being feasible)

mu = 0.5, sigma = 2.0
z = (0 - 0.5) / 2 = -0.25
PoF = norm.cdf(-0.25) = 0.401 (40.1% chance of being feasible)
```

##### Step 4: Filter and Select Best
```python
for u in U:
    pof = self._probability_feasible(u)
    if pof < self.pof_min:  # pof_min = 0.95
        continue  # Skip if less than 95% chance of feasibility
    acq = self._eic(u, best_y, xi=xi)
    if acq > best_acq:
        best_acq = acq
        best_u = u
```

Only considers candidates with ≥95% probability of being feasible, then picks the one with highest EIC.

---

### 4. Gaussian Process Training

**How BO Learns:**

**Location:** `bayesian_optimization.py`, Lines 87-106

```python
def _fit_models(self):
    X_unit = self._to_unit(np.array(self.X_raw))
    Y = np.array(self.Y, dtype=float)
    G = np.array(self.G, dtype=float)

    # Train constraint GP on all data
    try:
        self.gp_g.fit(X_unit, G)
        self._g_is_fit = True
    except Exception:
        self._g_is_fit = False

    # Train objective GP only on feasible data (if enough)
    feas = G <= 0.0
    try:
        if np.sum(feas) >= 2:
            self.gp_f.fit(X_unit[feas], Y[feas])
        else:
            self.gp_f.fit(X_unit, Y)
        self._f_is_fit = True
    except Exception:
        self._f_is_fit = False
```

**Key points:**

1. **Two GPs:**
   - `gp_f`: Models objective (cost/fitness)
   - `gp_g`: Models constraint (violation)

2. **Unit normalization:**
   All inputs normalized to [0,1] for numerical stability

3. **Feasible-only objective:**
   If we have ≥2 feasible points, `gp_f` trains only on those
   Otherwise, trains on everything

4. **Always-on constraint:**
   `gp_g` always trains on all data (feasible and infeasible)

**Example:**
```
Historical data:
X: [[1.0, 2.0, 1.5], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0], [4.0, 2.5, 3.0]]
Y: [50, 45, 55, 40]
G: [5.0, -2.0, 3.0, -1.0]  (violations)

Feasible: indices 1 and 3 (G ≤ 0)

gp_g trains on all 4 points
gp_f trains on points 1 and 3 only (the feasible ones)
```

---

### 5. The Preference Model (Mixed HIL Only)

**Purpose:** Learn which parameter regions you prefer over time.

**Structure:**
**Location:** `main_macos.py`, Lines 223-249

```python
class PreferenceModel:
    def __init__(self, bounds, lr=PREFERENCE_LR):
        self.bounds = np.array(bounds, dtype=float)
        self.lr = float(lr)  # learning rate = 0.3
        self.weights = np.random.rand(len(self.bounds))  # random start
```

**State:** Maintains weights `w = [w_Kp, w_Ki, w_Kd]`

**Initial weights:** Random, e.g., `[0.42, 0.67, 0.31]`

**Update rule:**
```python
def update_towards(self, preferred, other):
    # Normalize to [0,1]
    min_b = self.bounds[:, 0]
    span = self.bounds[:, 1] - min_b
    pref_norm = (preferred - min_b) / (span + 1e-9)
    
    # Move weights toward preferred
    self.weights = self.weights + self.lr * (pref_norm - self.weights)
    self._normalize()
```

**What this does:**
- Treats weights as "preferences" in [0,1] normalized space
- Each update moves weights 30% closer to the preferred candidate's normalized values
- Over time, weights converge to your preference patterns

**Example evolution:**
```
Iteration 1: Prefer DE [5.0, 2.0, 1.0]
  Normalized: [0.495, 0.199, 0.099]
  Weights: [0.42, 0.67, 0.31] + 0.3×([0.495, 0.199, 0.099] - [0.42, 0.67, 0.31])
         = [0.42, 0.67, 0.31] + 0.3×[0.075, -0.471, -0.211]
         = [0.4425, 0.5287, 0.2467]

Iteration 2: Prefer BO [3.0, 4.0, 3.0]
  Normalized: [0.293, 0.399, 0.299]
  Weights: [0.4425, 0.5287, 0.2467] + 0.3×([0.293, 0.399, 0.299] - [0.4425, 0.5287, 0.2467])
         = [0.3977, 0.4898, 0.2624]

Over 10 iterations of preferring higher Kp:
  Weights converge toward [0.7, 0.3, 0.2] → You like higher Kp, lower Ki and Kd
```

**Anchor calculation:**
```python
def anchor_params(self):
    min_b = self.bounds[:, 0]
    max_b = self.bounds[:, 1]
    return min_b + self.weights * (max_b - min_b)
```

**Example:**
```
Weights: [0.7, 0.3, 0.2]
Bounds: [(0.1, 10), (0.01, 10), (0.01, 10)]
Anchor = [0.1 + 0.7×9.9, 0.01 + 0.3×9.99, 0.01 + 0.2×9.99]
       = [7.03, 3.01, 2.01]
```

This anchor reflects your learned preferences and gets injected into DE's population.

---

## Complete Flow Diagrams

### Mixed HIL - Complete Flow

```
START ITERATION
    ↓
DE generates candidate A
BO generates candidate B
    ↓
Simulate both with history
    ↓
Calculate metrics (overshoot, rise time, settling time)
Calculate violations (actuator saturation)
    ↓
Show both to human
    ↓
┌─────────┬──────────┬──────────┬──────────┐
│ Prefer A│ Prefer B │   TIE    │  REJECT  │
│   (DE)  │   (BO)   │ (Refine) │ (Expand) │
└─────────┴──────────┴──────────┴──────────┘
    │         │           │          │
    ↓         ↓           ↓          ↓
```

**Prefer A Path:**
```
Update preference model → toward A
Calculate anchor from weights
    ↓
DE: ← reset mutation
    ← inject anchor (protect best)
    ↓
BO: ← update with both A and B
    ← nudge toward A (virtual better cost)
    ↓
Log and continue
```

**Prefer B Path:**
```
Update preference model → toward B
Calculate anchor from weights
    ↓
BO: ← update with both A and B
    ← nudge toward B (virtual better cost)
    ↓
DE: ← reset mutation
    ← inject B candidate (protect best)
    ← inject anchor (protect best)
    ↓
Log and continue
```

**TIE Path:**
```
Calculate midpoint = (A + B) / 2
    ↓
DE: ← shrink bounds 50% around midpoint
    ← reduce mutation 20%
    ← reinitialize population
    ← reset scores
    ↓
BO: ← shrink bounds 50% around midpoint
    ← keep GP knowledge
    ↓
Log and continue
```

**REJECT Path:**
```
DE: ← expand bounds 50%
    ← increase mutation 20%
    ← restart population
    ← reset scores
    ↓
BO: ← expand bounds 50%
    ← keep GP knowledge
    ↓
Log and continue
```

---

### Single DE HIL - Complete Flow

```
START ITERATION
    ↓
DE evolves one generation
    ↓
Get best candidate from DE
    ↓
Simulate with history
    ↓
Calculate metrics and violations
    ↓
Check auto-termination:
  If (target_ok AND safe_ok) → DONE
    ↓
Show candidate to human
    ↓
┌──────────┬──────────┐
│  ACCEPT  │  REJECT  │
└──────────┴──────────┘
    │          │
    ↓          ↓
```

**ACCEPT Path:**
```
DE: ← refine bounds 50% around candidate
    ← reduce mutation 20%
    ← reinitialize population
    ← inject safe seed
    ← clamp to global bounds
    ← reset scores
    ↓
Log and continue
```

**REJECT Path:**
```
DE: ← expand bounds 50%
    ← increase mutation 20%
    ← restart population
    ← inject safe seed
    ← clamp to global bounds
    ← reset scores
    ↓
Log and continue
```

---

### Single BO HIL - Complete Flow

```
START ITERATION
    ↓
BO proposes candidate (with retry for feasibility)
    ↓
Simulate with history
    ↓
Calculate metrics and violations
    ↓
Check auto-termination:
  If (target_ok AND safe_ok) → DONE
    ↓
Show candidate to human
    ↓
┌──────────┬──────────┐
│  ACCEPT  │  REJECT  │
└──────────┴──────────┘
    │          │
    ↓          ↓
```

**ACCEPT Path:**
```
BO: ← refine bounds 50% around candidate
    ← keep GP knowledge
    ← nudge toward candidate (virtual better cost, 20%)
    ↓
Log and continue
```

**REJECT Path:**
```
BO: ← expand bounds 50%
    ← keep GP knowledge (learns from bad area)
    ↓
Log and continue
```

---

## Summary Table

| Action | Mixed HIL DE | Mixed HIL BO | Single DE | Single BO |
|--------|--------------|--------------|-----------|-----------|
| **Prefer A** | Reset mutation, inject anchor, update preference | Update with both, nudge toward A | N/A | N/A |
| **Prefer B** | Reset mutation, inject B + anchor, update preference | Update with both, strong nudge toward B | N/A | N/A |
| **TIE/Refine** | Shrink bounds 50%, reduce mutation 20%, restart pop | Shrink bounds 50%, keep GP | N/A | N/A |
| **REJECT/Expand** | Expand bounds 50%, increase mutation 20%, restart pop | Expand bounds 50%, keep GP | N/A | N/A |
| **ACCEPT** | N/A | N/A | Refine 50%, reduce mutation, restart, inject safe seed | Refine 50%, keep GP, nudge 20% |
| **REJECT** | N/A | N/A | Expand 50%, increase mutation, restart, inject safe seed | Expand 50%, keep GP |

---

## Key Takeaways

1. **Preference Learning (Mixed HIL):** 
   - Weights learn your patterns over time
   - Anchors guide DE population
   - Nudges guide BO's Gaussian Process

2. **Bounds Adaptation:**
   - Refine = Shrink 50% = Focus on good region
   - Expand = Grow 50% = Explore wider area
   - Always clamped to global safety limits

3. **Knowledge Retention:**
   - DE: Loses population on refine/expand, gets fresh start
   - BO: Never loses Gaussian Process, learns from everything

4. **Safety Mechanisms:**
   - Feasibility-first selection (feasible always beats infeasible)
   - Protected best individual in DE
   - 95% probability threshold in BO
   - Global bounds hard limits
   - Safe seed injection in Single DE HIL

5. **Mutation Control:**
   - Higher mutation = More exploration
   - Lower mutation = More exploitation
   - Adjusted automatically based on feedback

This system creates a powerful collaboration between human expertise and algorithmic optimization!
