#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_all.sh — Build all PMUT analysis outputs from scratch
#
# Usage:
#   chmod +x run_all.sh   (first time only)
#   ./run_all.sh
#
# Scripts run (in order):
#   00  00_export_data.py       Excel → data/pmuts_all.csv
#
#   01  01_train_cache.py       Train all models ONCE and cache predictions
#                                 data/progressive_cache.csv
#                                 data/regime_cache.csv
#
#   04  04_metrics_table.py     MAE/RMSE/R² tables from cache (no retraining)
#                                 outputs/04_metrics/progressive_metrics.{csv,html}
#                                 outputs/04_metrics/regime_metrics.{csv,html}
#
#   06  06_progressive_training.py  Interactive HTML from cache
#                                 outputs/06_progressive/all_combined.html
#                                 outputs/06_progressive/progressive_metrics.csv
#
#   07  07_regime_aware_training.py  Interactive HTML from cache
#                                 outputs/07_regime/all_combined.html
#                                 outputs/07_regime/regime_metrics.csv
#
#   08  08_progressive_images.py  Static PNG images from cache
#                                 outputs/08_images/pmut_NN_vector_full.png  (×15)
#                                 outputs/08_images/pmut_NN_vector_full_mlp.png (×1)
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=".venv/bin/python3"

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'

step() { echo -e "\n${BOLD}${CYAN}▶  $*${RESET}"; }
ok()   { echo -e "${GREEN}✓  $* complete${RESET}"; }
fail() { echo -e "${RED}✗  $* failed — see error above${RESET}"; exit 1; }

run() {
    local script="$1"
    step "$script"
    "$PYTHON" "$script" && ok "$script" || fail "$script"
}

# ── Sanity check ──────────────────────────────────────────────────────────────
if [[ ! -x "$PYTHON" ]]; then
    echo -e "${RED}ERROR: $PYTHON not found. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt${RESET}"
    exit 1
fi

echo -e "${BOLD}PMUT Frequency Response Analysis — full rebuild${RESET}"
echo "Python : $("$PYTHON" --version)"
echo "Dir    : $SCRIPT_DIR"
echo "Started: $(date)"

# ── 00  Export raw Excel data → CSV ───────────────────────────────────────────
run 00_export_data.py

# ── 01  Train all models, write prediction caches ─────────────────────────────
run 01_train_cache.py

# ── 04  Metrics tables (Vector · Full Spectrum · both training schemes) ────────
run 04_metrics_table.py

# ── 06  Progressive training — interactive HTML (reads cache) ─────────────────
run 06_progressive_training.py

# ── 07  Regime-aware training — interactive HTML (reads cache) ────────────────
run 07_regime_aware_training.py

# ── 08  Static matplotlib images (reads cache) ────────────────────────────────
run 08_progressive_images.py

# ── Summary ────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}${GREEN}══════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  All outputs built successfully.${RESET}"
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════${RESET}"
echo ""
echo "Data / cache:"
echo "  data/pmuts_all.csv"
echo "  data/progressive_cache.csv"
echo "  data/regime_cache.csv"
echo ""
echo "  outputs/04_metrics/"
echo "    progressive_metrics.csv  progressive_metrics.html"
echo "    regime_metrics.csv       regime_metrics.html"
echo ""
echo "  outputs/06_progressive/"
echo "    all_combined.html        progressive_metrics.csv"
echo ""
echo "  outputs/07_regime/"
echo "    all_combined.html        regime_metrics.csv"
echo ""
echo "  outputs/08_images/"
echo "    pmut_NN_vector_full.png      (15 images — RF + GB)"
echo "    pmut_NN_vector_full_mlp.png  ( 1 image  — RF + GB + MLP)"
echo ""
echo "Finished: $(date)"
