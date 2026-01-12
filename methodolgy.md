# Weight Adjustment Methodology in Mixed HIL-PID Optimization

## Overview

This document explains the **weight adjustment mechanism** (also called "preference learning") used in our Mixed Hardware-in-the-Loop (HIL) PID tuning system. The weight adjustment is a key component that helps the optimization algorithms learn from human feedback and progressively improve their search strategy.

---

## What is Weight Adjustment?

Weight adjustment is a **learning mechanism** that captures and encodes human preferences about PID controller parameters. Think of it as teaching the system what kinds of solutions you prefer, so it can search more intelligently in those directions.

### The Core Idea

Instead of blindly searching the entire parameter space, the system maintains **internal weights** (one for each PID parameter: Kp, Ki, Kd) that represent where in the search space the user tends to prefer solutions. These weights are continuously updated based on your choices during the optimization process.

---

## How Does It Work?

### 1. **Weight Initialization**

At the start of optimization, the `PreferenceModel` class (lines 223-249 in `main_macos.py`) initializes with:

```python
self.weights = np.random.rand(len(self.bounds))
```

This creates random weights between 0 and 1 for each parameter dimension (Kp, Ki, Kd).

### 2. **Converting Weights to Anchor Points**

The weights represent normalized positions within the parameter bounds. They are converted to actual parameter values using the `anchor_params()` method:

```python
def anchor_params(self):
    min_b = self.bounds[:, 0]
    max_b = self.bounds[:, 1]
    return min_b + self.weights * (max_b - min_b)
```

**Example:**
- If bounds for Kp are [0.1, 10.0] and weight is 0.5
- Anchor = 0.1 + 0.5 × (10.0 - 0.1) = 5.05

This anchor point represents the system's current "best guess" of where good solutions might be.

### 3. **Learning from User Feedback**

When you choose between two candidates (A from DE, B from BO), the system updates its weights toward your preferred choice using the `update_towards()` method:

```python
def update_towards(self, preferred, other):
    # Normalize parameters to [0,1] space
    pref_norm = (preferred - min_b) / (span + 1e-9)
    other_norm = (other - min_b) / (span + 1e-9)
    
    # Calculate the direction from other to preferred
    gap = other_norm - pref_norm
    
    # Update weights: move toward preferred choice
    self.weights = self.weights + self.lr * (pref_norm - self.weights)
    self._normalize()  # Keep weights in [0,1]
    
    return gap
```

**What happens:**
1. **Normalization**: Both candidates are normalized to [0,1] based on parameter bounds
2. **Learning Rate**: The `PREFERENCE_LR` (default 0.3) controls how fast the weights move
3. **Update Rule**: Weights shift toward the preferred candidate's normalized position
4. **Gap Calculation**: The "gap" shows how different the two candidates were

### 4. **Applying the Learned Preferences**

The updated weights influence the search in two ways:

#### A. **Injecting Anchor Points into DE**

After you prefer a candidate, a new anchor point is calculated and injected into the Differential Evolution population:

```python
if choice == 1:  # User preferred DE
    gap = pref_model.update_towards(cand_a, cand_b)
    anchor = pref_model.anchor_params()
    de.inject_candidate(anchor, eval_func=fitness_wrapper, protect_best=True)
```

This ensures DE explores around regions similar to your preferences.

#### B. **Nudging Bayesian Optimization**

The BO surrogate model is also updated with a "preference nudge":

```python
bo.nudge_with_preference(cand_a, fit_a, fit_b, viol_a)
```

This method (in `bayesian_optimization.py`, lines 232-242) creates a **pseudo-observation** with artificially improved cost, encouraging BO to search near the preferred region.

---

## Mathematical Details

### Learning Rate (α = 0.3)

The learning rate controls the balance between:
- **Exploration**: Low α keeps weights stable, maintaining broader search
- **Exploitation**: High α makes weights change quickly, focusing on recent preferences

Our choice of α = 0.3 provides a good middle ground.

### Weight Update Formula

For each dimension *i*:

```
w_i^(new) = w_i^(old) + α × (w_i^(preferred) - w_i^(old))
```

This is an **exponential moving average** that gradually shifts weights toward preferred regions.

### Normalization to [0,1]

After each update, weights are clipped:

```python
self.weights = np.clip(self.weights, 0.0, 1.0)
```

This ensures they always represent valid normalized positions within bounds.

---

## Example Scenario

**Initial State:**
- Bounds: Kp=[0.1, 10.0], Ki=[0.01, 10.0], Kd=[0.01, 10.0]
- Initial weights: [0.45, 0.61, 0.33] (random)
- Initial anchor: [4.56, 6.08, 3.30]

**Iteration 1:**
- DE proposes: [3.2, 4.5, 2.1]
- BO proposes: [7.8, 8.9, 6.5]
- User prefers DE (faster settling, less overshoot)

**What happens:**
1. Normalize candidates to [0,1] space
   - DE normalized: [0.31, 0.45, 0.21]
   - BO normalized: [0.78, 0.89, 0.65]
2. Calculate gap: [0.47, 0.44, 0.44] (showing BO was far from DE)
3. Update weights toward DE:
   - New weights: [0.45 + 0.3×(0.31-0.45), ...] = [0.41, 0.56, 0.29]
4. New anchor: [4.10, 5.61, 2.91] (shifted toward DE's region)
5. Inject anchor into DE population
6. Add preference nudge to BO model

**Result:** Both optimizers now bias their search toward the lower-parameter region where you found success.

---

## Why Is This Important?

### 1. **Accelerated Convergence**

Without weight adjustment, optimizers might waste iterations exploring regions you consistently reject. With it, they quickly learn to focus on promising areas.

### 2. **Implicit Constraint Encoding**

Sometimes you have preferences that are hard to formalize (e.g., "I prefer conservative controllers"). Weight adjustment captures these implicitly through your choices.

### 3. **Synergy Between Algorithms**

The shared preference model creates a communication channel:
- DE finds good exploitative solutions
- BO explores promising regions
- Weight adjustment ensures both learn from ALL feedback

### 4. **Robustness to Subjectivity**

The exponential moving average (via learning rate) prevents one-off choices from drastically changing the search strategy, while still adapting to consistent patterns.

---

## Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `PREFERENCE_LR` | 0.3 | How fast weights shift toward preferences |
| `BASE_MUTATION` | 0.5 | Reset value for DE mutation after learning |
| Anchor injection | Every preference | Frequency of injecting learned anchors |

---

## Code Flow Summary

1. **Initialize** `PreferenceModel` with random weights
2. **Generate anchor** from weights
3. **Seed DE** population with anchor
4. **Each iteration:**
   - Generate candidates from DE and BO
   - Simulate and visualize both
   - **User chooses** preferred candidate
   - **Update weights** toward preferred candidate
   - **Calculate new anchor** from updated weights
   - **Inject anchor** into DE (and possibly BO via nudge)
   - **Both optimizers** now search near learned preferences

---

## Limitations and Trade-offs

### Advantages ✓
- Learns quickly from sparse feedback
- Simple and interpretable (weights are just normalized positions)
- Computationally cheap (no complex learning algorithm)

### Limitations ✗
- Assumes user preferences are consistent in parameter space
- May over-commit to early preferences if learning rate is too high
- Single anchor may not capture multi-modal preference landscapes

---

## Conclusion

The weight adjustment mechanism is an elegant solution for **human-in-the-loop optimization**. It transforms subjective preferences into actionable search bias, enabling both Differential Evolution and Bayesian Optimization to:

1. Learn from every piece of human feedback
2. Focus exploration on promising regions
3. Avoid repeatedly suggesting solutions the user dislikes
4. Converge faster to satisfactory PID parameters

By maintaining a simple weighted anchor point and updating it with each choice, the system achieves effective **preference learning** without complex machinery, making it both practical and interpretable for real-world controller tuning applications.
