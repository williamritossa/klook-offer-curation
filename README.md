# Klook Offer Curation

This repository supports the process of reviewing and curating Klook activities before they go live on our production storefront. The Jupyter notebook ingests raw offer payloads, normalises their content, and applies grading rules through the OpenAI Responses API so the merchandising team can focus on high-confidence offers.

## Getting started
- Ensure Python 3.11+ is available.
- (Optional) Create a virtual environment with `python3 -m venv .venv && source .venv/bin/activate`.
- Install dependencies via `pip install -r requirements.txt` (add `python-dotenv` if you plan to read from `.env`).
- Provide OpenAI access with `export OPENAI_API_KEY=...` or by adding the key to a local `.env` file.
- Launch `jupyter notebook offer_curation.ipynb` and step through the cells in order.

## Working in the notebook
- Early cells load helper functions and parse JSON files from `offers/` into structured records.
- Preview cells (8–10) surface a quick overview of the current dataset.
- Grading uses the OpenAI Responses API; keep an eye on rate limits and quota usage.
- When `results_df` appears, the final cell exports `graded_offers.csv` beside the notebook for downstream review.

## Repository layout
- `offer_curation.ipynb`: main workflow for loading, structuring, and grading offers.
- `offers/`: source JSON payloads; place new activities here before running the notebook.
- `graded_offers.csv`: latest grading export; regenerate locally as needed.
- `requirements.txt`: minimal runtime dependencies for the notebook.
- `.env`: gitignored file for sensitive configuration such as `OPENAI_API_KEY`.

## Regenerating outputs
- Remove or archive any existing `graded_offers.csv` when starting a fresh curation pass.
- Confirm `OPENAI_API_KEY` is available to the kernel before running the grading cell.
- Rerun the grading section to produce updated scores, Klook links, and OpenAI log references.

## Operational notes
- `grade_offers_parallel` uses `ThreadPoolExecutor` to process multiple offers; adjust worker settings if you encounter rate limiting.
- The grading response is parsed as JSON—if parsing fails, inspect the `reason` column and refine the prompt or offending offers.