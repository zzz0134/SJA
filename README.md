# SJA: Story-Jacobian Alignment for Narrative Sub-question Generation

## Abstract
We present **Story-Jacobian Alignment (SJA)**, a geometric generation framework for decomposing a complex harmful-goal query into diverse, non-refusal, and question-shaped narrative sub-questions. SJA combines (i) semantic direction decomposition, (ii) hyperbolic dynamics on the Poincare ball, and (iii) narrative fusion to produce auditable sub-question sets with controllable diversity constraints.

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
@misc{sja2026,
  title        = {SJA: Story-Jacobian Alignment for Narrative Sub-question Generation},
  author       = {Anonymous},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {\url{https://github.com/zzz0134/SJA}}
}
```
