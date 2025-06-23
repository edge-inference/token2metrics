# Prefill-Only Latency Predictor

## Data Available

For each prompt *i*:
- **Prefill token count**: \( L_i \)
- **Server prefill time**: \( T_{i,\text{srv}} \)

> **Dataset:**

**Server**

All 3,000 questions at 100 questions by subjet in MMLU dataset

**Jetson**
150 questions at 5 questions per subject in MMLU dataset

- **Jetson prefill time**: \( T_{j,\text{jet}} \) for some \( j \)

---

## Goal

Predict Jetson prefill time for any prompt, given only server data and a small Jetson subset.

---

## Approach

### 1. Fit a Linear Model on Server Data

\[
T_{\text{srv}}(L) \approx \alpha L + \beta
\]

---

### 2. Map to Jetson

#### **A. Simple Scaling**

- Compute scaling ratio:

    \[
    r = \frac{\text{mean}\{T_{j,\text{jet}}\}}{\text{mean}\{T_{j,\text{srv}}\}}
    \]

- Predict Jetson time:

    \[
    \hat{T}_{\text{jet}}(L) = r \cdot (\alpha L + \beta)
    \]

#### **B. Edge-Specific Regression**

- Fit directly on Jetson samples:

    \[
    T_{\text{jet}}(L) \approx \alpha_{\text{jet}} L + \beta_{\text{jet}}
    \]

- Extrapolate to all prompts:

    \[
    \hat{T}_{\text{jet}}(L_i) \text{ for all } i
    \]

---

## Summary Table

| Step                | Formula/Method                                        |
|---------------------|------------------------------------------------------|
| Server Fit          | \( T_{\text{srv}}(L) \approx \alpha L + \beta \)     |
| Simple Scaling      | \( \hat{T}_{\text{jet}}(L) = r(\alpha L + \beta) \)  |
| Edge Regression     | \( T_{\text{jet}}(L) \approx \alpha_{\text{jet}} L + \beta_{\text{jet}} \) |

