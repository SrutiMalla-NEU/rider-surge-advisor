# Surge Predictor (working name)

A rider-side tool that tells people stranded by rideshare surge pricing their cheapest realistic move: hail a cab, walk a few blocks, wait, or book now.

Version 1 is trained on New York City public trip records (TLC), weather, holidays, and subway alerts.

**Status:** in progress. Data pipeline, exploration, and first models complete.

## Setup

1. `python -m venv .venv` and activate it
2. `pip install -r requirements.txt`
3. `python -m src.download` to download the TLC data (see `data/SOURCES.md`)

© 2026 Sruti Smitha Malla. All rights reserved.