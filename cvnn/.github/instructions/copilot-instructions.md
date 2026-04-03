Purpose
=======
Provide concise, repository-specific guidance for AI coding agents working on this codebase. Keep answers short, actionable, and focused on changes that are small, verifiable, and respectful of existing style and tests.

Key project facts
-----------------
- Language: Python (PEP 8 conventions expected). Project uses Poetry: `pyproject.toml`, `poetry.lock`.
- Source code: `src/cvnn/` is the primary package.
- Configs: `configs/` contains run configs; `docs/` and `README.md` contain usage notes and examples.
- Tests: `tests/` (when present) and `pyproject.toml` define test dependencies. Prefer `poetry run pytest` locally.

Operational rules (merge of repo guidance + agent constraints)
------------------------------------------------------------
- Start with a one-line task receipt and a short plan when starting work.
- Read the user's request fully, extract explicit requirements into a checklist, and keep them visible.
- Prefer the smallest, lowest-risk code changes that achieve the requirement.
- Preserve existing public APIs and coding style; follow PEP 8 for new code.
- For non-trivial edits, add a small unit test (pytest) covering the change and run tests locally.
- Always run quick verification after edits: import the modified module and run a minimal smoke test where feasible.
- Do not change unrelated files or reformat the entire repository.

Editing and commit rules
------------------------
- Use the repository's apply_patch/api to edit files programmatically when possible.
- Complete edits for a single file in one patch when practical. If multiple files are changed, batch them and provide a brief checkpoint after every 3–5 file edits.
- If edits introduce errors, attempt up to 3 small fixes; if still failing, report the failing outputs and root cause.

Project-specific patterns and examples
-------------------------------------
- Complex-valued models: code uses `torchcvnn` for complex ops (e.g., complex pooling). Avoid using real-only PyTorch ops where complex tensors are expected. Example files: `src/cvnn/models/blocks.py`, `src/cvnn/models/linear.py`.

Developer workflows
-------------------
- Setup: repository uses Poetry; prefer creating a venv with `poetry install` then `poetry shell` or use `poetry run` to run commands.
- Run tests: `poetry run pytest` (or the specific pytest invocation in CI). If CI uses additional env vars, mirror them locally when possible.
- Lint / typecheck: run project's linters if present; otherwise ensure code is syntactically correct and follows PEP8.

When unsure
----------
- Ask me questions
- If a change affects model shapes, write a small reproducible snippet in `tests/` that constructs a minimal model and runs a forward pass to confirm shapes.
- If a required runtime (e.g., `torch`) is missing in the agent environment, report that tests couldn't be executed and include the exact failure output.

What not to do
--------------
- Do not leave some obsolete code to maintain compatibility.
- Do not add unnecessary fallback options.
- Do not modify the code or experiment configs in `configs/` unless without my explicit approval.
- Don't call external network services or exfiltrate secrets.

Feedback loop
-------------
After applying changes, report: files edited, tests run (PASS/FAIL), and a short summary mapping requirements to status. Ask for any clarifications or missing behaviors.

---
Be concise: keep instructions short and actionable. When in doubt, add a focused test and a minimal docstring.