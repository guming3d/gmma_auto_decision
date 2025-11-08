# Repository Guidelines

## Project Structure & Module Organization
Core Streamlit entry points live at the repo root (`gmma_akshare.py`, `gmma_etf_akshare.py`, `gmma_hk_akshare.py`) and feed shared helpers in `ui/` (layout, sidebar controls) and `utils/` (data munging, chart helpers). Trading logic and backtests sit in `models/`, while reusable parameters live in `config.py`. Cached API payloads drop into `cache/`, and visual assets belong in `images/`. All unit tests reside in `tests/` and target the production modules via lightweight mocks.

## Build, Test, and Development Commands
Install dependencies in a fresh virtualenv with `pip install -r requirements.txt`. Launch the ETF-focused UI via `streamlit run main.py`, or start market dashboards with `streamlit run gmma_akshare.py`, `gmma_etf_akshare.py`, or `gmma_hk_akshare.py`. Static validation should always include `python run_tests.py`, which runs the unittest discovery suite under `tests/`. When iterating on a single module, `python -m unittest tests.test_gmma_etf_akshare` keeps the feedback loop tight.

## Coding Style & Naming Conventions
Follow Python 3.10+ practices: 4-space indentation, `snake_case` functions, and short docstrings for public helpers. Keep Streamlit callbacks lean and gate side effects under `if __name__ == "__main__":`. Reuse EMA period constants from `config.py` rather than hardcoding values, and name cached artifacts with the existing `industry_data_YYYY-MM-DD.json` pattern.

## Testing Guidelines
The suite is built on `unittest` with synthetic data mocks; mirror that structure when adding scenarios. New tests belong in files named `tests/test_<feature>.py`, with methods prefixed by `test_` and docstrings outlining intent. Aim to cover crossover detection branches and failure paths (empty frames, API errors). Before submitting a PR, run `python run_tests.py` plus any Streamlit flows you touched.

## Commit & Pull Request Guidelines
History shows concise, imperative titles (`adding stock total market value change`, `roll back akshare`); continue with ≤72-character summaries and include context or issue IDs in the body when helpful. PRs should explain the user-facing change, list key commands reviewers can run (`streamlit run …`, `python run_tests.py`), and attach screenshots or GIFs whenever UI output changes. Cross-link any tracking tickets and call out configuration or cache updates that reviewers need to reproduce the scenario locally.

## Security & Configuration Tips
No secrets should live in `config.py`; consume API keys via environment variables and document them in README if needed. Cached industry JSON files may contain dated vendor data—clear `cache/` before sharing archives outside the team. When working with third-party APIs (Akshare, Tushare), respect rate limits by reusing the provided LRU caching and Streamlit `st.cache_data` wrappers rather than adding ad-hoc sleeps.
