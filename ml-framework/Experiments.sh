#!/bin/bash
# Run Multiple Pipeline Configurations
# This script runs the streaming growth pipeline with different settings
# and saves results to descriptive folders for easy comparison
set -e  # Exit on error
# Color output for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
# Base settings
DATA_DIR="data"
BASE_OUTPUT="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Starting Pipeline Test Suite${NC}"
echo -e "${BLUE}Timestamp: ${TIMESTAMP}${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
# Test 1: Baseline - Minimal Features with Single Model
echo -e "${GREEN}[1/4] Running: Baseline (Linear Regression, Minimal Features)${NC}"
OUTPUT_DIR="${BASE_OUTPUT}/test1_baseline_linear_minimal_${TIMESTAMP}"
poetry run python scripts/streaming_growth_pipeline.py \
    --data-dir ${DATA_DIR} \
    --output ${OUTPUT_DIR} \
    --model linear_regression \
    --lags 1 \
    --rolling-windows 4 \
    --use-cumulative \
    --seed 42
echo -e "${YELLOW}✓ Test 1 complete. Results: ${OUTPUT_DIR}${NC}"
echo "Generating comprehensive HTML report..."
poetry run python scripts/generate_report.py --results-dir ${OUTPUT_DIR} --mode full
echo ""
# Test 2: Extended Features - All Models with More Lags
echo -e "${GREEN}[2/4] Running: Extended Features (All Models, Extended Lags)${NC}"
OUTPUT_DIR="${BASE_OUTPUT}/test2_extended_all_models_lag1248_${TIMESTAMP}"
poetry run python scripts/streaming_growth_pipeline.py \
    --data-dir ${DATA_DIR} \
    --output ${OUTPUT_DIR} \
    --model all \
    --lags 1 2 4 8 \
    --rolling-windows 4 8 12 \
    --use-cumulative \
    --seed 42
echo -e "${YELLOW}✓ Test 2 complete. Results: ${OUTPUT_DIR}${NC}"
echo "Generating comprehensive HTML report..."
poetry run python scripts/generate_report.py --results-dir ${OUTPUT_DIR} --mode full
echo ""
# Test 3: Random Forest Focus - Optimized for tree-based model
echo -e "${GREEN}[3/4] Running: Random Forest Optimized${NC}"
OUTPUT_DIR="${BASE_OUTPUT}/test3_random_forest_optimized_${TIMESTAMP}"
poetry run python scripts/streaming_growth_pipeline.py \
    --data-dir ${DATA_DIR} \
    --output ${OUTPUT_DIR} \
    --model random_forest \
    --lags 1 2 4 \
    --rolling-windows 4 8 \
    --use-cumulative \
    --cv-folds 10 \
    --seed 42
echo -e "${YELLOW}✓ Test 3 complete. Results: ${OUTPUT_DIR}${NC}"
echo "Generating comprehensive HTML report..."
poetry run python scripts/generate_report.py --results-dir ${OUTPUT_DIR} --mode full
echo ""
# Test 4: Data Leakage Comparison - With cumulative features
echo -e "${GREEN}[4/4] Running: Data Leakage Comparison (WARNING: For comparison only!)${NC}"
OUTPUT_DIR="${BASE_OUTPUT}/test4_with_leakage_comparison_${TIMESTAMP}"
poetry run python scripts/streaming_growth_pipeline.py \
    --data-dir ${DATA_DIR} \
    --output ${OUTPUT_DIR} \
    --model all \
    --lags 1 2 4 \
    --rolling-windows 4 8 \
    --use-cumulative \
    --include-cumulative-features \
    --seed 42
echo -e "${YELLOW}✓ Test 4 complete. Results: ${OUTPUT_DIR}${NC}"
echo "Generating comprehensive HTML report..."
poetry run python scripts/generate_report.py --results-dir ${OUTPUT_DIR} --mode full
echo ""
# Generate comparison summary
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}All Tests Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "Results saved in:"
echo "  1. ${BASE_OUTPUT}/test1_baseline_linear_minimal_${TIMESTAMP}/"
echo "  2. ${BASE_OUTPUT}/test2_extended_all_models_lag1248_${TIMESTAMP}/"
echo "  3. ${BASE_OUTPUT}/test3_random_forest_optimized_${TIMESTAMP}/"
echo "  4. ${BASE_OUTPUT}/test4_with_leakage_comparison_${TIMESTAMP}/"
echo ""
echo "To compare results, open the HTML reports:"
echo "  open ${BASE_OUTPUT}/test*_${TIMESTAMP}/plots/analysis_report.html"
echo ""
echo -e "${GREEN}Done! Check the HTML reports for detailed comparison.${NC}"