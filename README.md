# Token2Metrics
Translating tokens to latency, energy metrics for edge inference

## Overview
Token2Metrics is a module part of the edgereasoning project. This module handles data postprocessing and plotting figures for latency, power and energy. The evaluations supported are:

prefill
decode
scaling

## Phase-Specific Modeling
- **Prefill latency**: Model predicts prefill time from input tokens only.
- **Decode latency**: Model predicts decode time from output tokens only.
- Each phase is trained and calibrated separately for each model size.

## Quick Start

look into the prefilltokens, prefillenergy and decodetokens on how run those for datasets available. 
