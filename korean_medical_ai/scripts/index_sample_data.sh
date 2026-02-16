#!/bin/bash
# ============================================================
# KorMedAI - 샘플 데이터 인덱싱 스크립트
# ============================================================
# 초기 의료 데이터를 벡터 DB에 인덱싱합니다.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " KorMedAI - 데이터 인덱싱"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Python 가상환경 활성화
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

cd "$PROJECT_DIR"

# 1. 로컬 가이드라인 데이터 인덱싱
echo ""
echo "[1/2] 로컬 의료 가이드라인 인덱싱..."
python -m korean_medical_ai.main --index-data korean_medical_ai/data/guidelines

# 2. HuggingFace 데이터셋 인덱싱 (KorMedLawQA)
echo ""
echo "[2/2] SNUH KorMedLawQA 데이터셋 인덱싱..."
python -m korean_medical_ai.main --index-hf snuh/KorMedLawQA

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " 인덱싱 완료"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
