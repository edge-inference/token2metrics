# Decode-Only Latency Predictor

This guide describes how to build and extrapolate a **decode-only** inference time model—from output token count to decode latency—using full server measurements and a small Jetson sample.

---

## 🎯 Data Available

- **Full server dataset**: For each question *i*, we have
  - Output token count: `(O_i)`
  - Measured server decode time: `(T_srv_dec(O_i))` (all 3,000 questions at 100 questions by subjet in MMLU dataset)

- **Jetson sample**: A question *j* with
  - Output token count: `(O_j)`
  - Measured Jetson decode time: `(T_jet_dec(O_j))` (150 questions at 5 questions per subject in MMLU dataset)

---

## 🔢 Server-Side Model Fit

Fit a **linear regression** on the server data:

T_srv_dec(O) ≈ γ * O + δ


Use all `(O_i, T_srv_dec(O_i))` pairs to solve for `γ` and `δ`.

---

## 🧪 Jetson Calibration

Two methods:

1. **Scaling factor**:
Ť_srv_j = γ * O_j + δ
r = mean(T_jet_dec(O_j)) / mean(Ť_srv_j)
T_jet_dec_pred(O) = r * (γ * O + δ)


2. **Direct regression**:


---

## 🚀 Extrapolation & Usage

1. Usage will be like having `(O_i)` for every prompt offline.
2. Apply chosen Jetson model (`scaled server` or `direct regression`).
3. Aggregate results (sum, percentiles, histograms).

---
