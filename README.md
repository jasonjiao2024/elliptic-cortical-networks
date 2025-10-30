# Elliptic Cortical Networks (ECN) — Reference Implementation

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This repository contains a reference implementation of Elliptic Cortical Networks (ECN) aligned with the Neurocomputing manuscript. It includes the six validation dimensions described in the paper: memory compression, non-linear learning (XOR), mathematical constraint satisfaction, computational throughput, architecture scalability, and predictive learning.

The core implementation lives in `ecn_implementation.py` and can be executed directly to reproduce the validation suite and save results to `ecn_validation_results.json`.

## Requirements

- Python 3.8+
- NumPy
- Optional: `coincurve` (if installed, secp256k1 acceleration is used)

Install dependencies:
```bash
python3 -m pip install numpy coincurve
```

## Quick Start

Run the full validation suite:
```bash
python3 ecn_implementation.py
```

This will print the outcomes of all six validations and write a summary to `ecn_validation_results.json`.

## What’s Implemented

- Six-layer cortical column with per-layer curve counts and finite fields
- Inter-curve projections via three-stage pipeline: ψ (extract) → τ (transform) → φ (remap)
- XOR learner for non-linear capability
- Throughput benchmark and scalability presets (`simple`, `medium`, `complex`)
- Predictive learner demonstrating temporal sequence learning

## Notes on Results

- The code computes and displays actual results from your environment; it does not hard-code paper figures. Targets have been removed from console output for clean, factual reporting.
- If `coincurve` is present, operations use secp256k1-backed acceleration; otherwise, a pure-Python fallback is used.

## File Guide

- `ecn_implementation.py`: Main ECN implementation and validation entry point (run this)

## Citation

If this code aids your research, please cite the ECN manuscript:
Jiao, D. (2025). Elliptic cortical networks: A mathematically constrained architecture for biologically-inspired intelligence. Neurocomputing, 658, 131802. https://doi.org/10.1016/j.neucom.2025.131802

