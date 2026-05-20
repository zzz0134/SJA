# Structured Multi-step Jailbreaking under a Hamiltonian Generative Formulation

## Abstract
Recent work shows that even safety aligned large language models (LLM) can be pushed into unsafe behavior by carefully crafted jailbreak prompts. Existing jailbreaking attack methods often rely on disfluent or incoherent prompts, which limit their success and make them easy to detect. We introduce SJA, a structured jailbreak attack built around two ideas. First, inspired by the logic of Spilsbury puzzle, SJA decomposes a harmful query into a sequence of harmless sub-questions and reconstructs the original answer by combining the sub-question responses. Second, by leveraging the theory of Hamiltonian dynamics on hyperbolic space, we propose a hyperbolic Hamiltonian dynamics-based sub-question generation framework that effectively captures the structural and temporal dependencies. We provide a theoretical analysis of how each sub-question evolves along the trajectory and show that the hyperbolic Hamiltonian system effectively captures the underlying semantic structure. Finally, we propose a hyperbolic narrative fusion mechanism built on fractional embedding and Möbius fusion. This mechanism integrates coherent narratives into sub-questions while preserving geometric consistency and improving stealth performance. We theoretically validate that the combination of the generated harmless sub-questions, guided by the stealthy narrative, can effectively preserve the contextual semantics of the original harmful question.

## 1. Method Overview
Given an input goal $g$, SJA performs:

1. **Direction Decomposition**: Generate $K$ semantically distinct intermediate directions.
2. **Narrative Construction**: Generate a short refusal-free story context tied to $g$.
3. **Hyperbolic Fusion**:
   - Embed goal/directions/story with a causal LM hidden-state encoder.
   - Evolve direction-conditioned state with Hamiltonian-like dynamics in hyperbolic space.
   - Fuse dynamic state with narrative embedding via Mobius addition.
4. **Sub-question Decoding**: Decode one distinct question per direction under guardrails.

## 2. Safety and Generation Constraints
The implementation enforces:

- refusal/policy-text filtering,
- semantic de-duplication for directions and sub-questions,
- question-shape validation,
- fallback templates when constrained decoding fails.

These checks are implemented in [`sja_guardrails.py`](./sja_guardrails.py) and consumed by [`question2.py`](./question2.py).

## 3. Repository Structure
- [`question2.py`](./question2.py): main pipeline (decomposition, hyperbolic dynamics, decoding, tracing).
- [`sja_guardrails.py`](./sja_guardrails.py): filtering, validation, and semantic distinctness utilities.

## 4. Environment
Recommended dependencies:

- Python >= 3.10
- `torch`
- `transformers`
- `pandas`
- `geoopt`

Install example:

```bash
pip install torch transformers pandas geoopt
```

## 5. Usage
Default run:

```bash
python question2.py \
  --csv-path /path/to/input.csv \
  --output-csv generated_subquestions.csv \
  --trace-jsonl generated_subquestions_trace.jsonl
```

Important arguments:

- `--goal-column`: explicit goal column name (auto-detected if omitted)
- `--max-dirs`: number of decomposed directions
- `--dir-sim-threshold`: direction distinctness threshold
- `--subq-sim-threshold`: sub-question distinctness threshold
- `--max-retries`: retry budget before fallback

Input goal column is auto-detected from one of:

- `goal`
- `Behavior`
- `plain_query`
- `question`

## 6. Output Format
- `output-csv`: rows of `(goal, sub_question)`.
- `trace-jsonl`: per-goal audit record including directions, story, generated sub-questions, and fallback flags.

## 7. Reproducibility Notes
- Fix CUDA device mapping and model path in `question2.py` before running.
- Determinism may vary with sampling retries and backend kernels.
- For controlled evaluation, pin package versions and random seeds.

## 8. Citation
If you use this codebase in academic work, please cite as:

```bibtex
@inproceedings{
anonymous2026structured,
title={Structured Multi-step Jailbreaking under a Hamiltonian Generative Formulation},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=lpa6hHaukP}
}
```
