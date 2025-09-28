# Repository Guidelines

## Project Structure & Module Organization
- `offer_curation.ipynb` — primary workflow for importing Klook offers, transforming content, and grading via OpenAI Responses API.
- `offers/` — source JSON payloads pulled from Klook; each file holds one `activity` object used as grading input.
- `.env.example` — template for required environment variables; copy to `.env` or export manually before running the notebook.
- Ancillary artifacts such as `graded_offers.csv` should be regenerated locally and excluded from commits unless explicitly required.

## Environment & Configuration
- Python 3.11+ with `openai`, `pandas`, and `concurrent.futures` (standard library) are required; create a virtual environment (`python3 -m venv .venv && source .venv/bin/activate`).
- Configure OpenAI access with `export OPENAI_API_KEY="sk-..."` **before** launching Jupyter, or store the value in `.env` and load it via your notebook setup.
- Keep API keys out of version control; `.env` is gitignored.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` *(create if needed)* — install notebook dependencies; pin versions whenever the stack changes.
- `jupyter notebook offer_curation.ipynb` — launch the interactive workflow; execute cells sequentially to reproduce grading outputs.
- `python3 scripts/validate_offers.py` *(optional future hook)* — use for automated checks when added; ensure new scripts log to stdout and exit non-zero on failure.

## Coding Style & Naming Conventions
- Prefer Python typing annotations and descriptive function names (`load_offers`, `grade_offer`).
- Follow PEP 8 (4-space indentation, snake_case functions/variables). Keep notebook cells concise; refactor shared logic into helper modules if they exceed ~100 lines.
- When adding files, keep ASCII encoding and document non-obvious logic with concise comments.

## Testing Guidelines
- As this is a simple project we do not need automated tests

## Commit & Pull Request Guidelines
- Use conventional, action-oriented commit messages (`feat: add schema-enforced grading`, `fix: ensure image payload uses input_image`).
- Each PR should summarize notebook or script changes, list verification steps (e.g., grading run results), and reference related issues or task IDs.
- Attach screenshots or notebook previews when UI/output formatting changes, and confirm secrets were not accidentally committed.
