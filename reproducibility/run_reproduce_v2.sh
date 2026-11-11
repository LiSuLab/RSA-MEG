#!/usr/bin/env bash
set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPRO_DIR"

echo "=== [1/3] Running V2 Numerical & Zero-Tie Audit ==="
python3 scripts/v2_07_supervisor_robustness_and_zero_hallucination_audit.py

echo "=== [2/3] Rendering Main Figures 1-8 ==="
python3 scripts/v2_09_render_v5_style_publication_figures.py

echo "=== [3/3] Rendering Supplementary Figures S1-S10 ==="
python3 scripts/v2_10_render_v5_style_supplementary_figures.py

echo "=== [COMPLETED] All V2 figures verified and rendered ==="
