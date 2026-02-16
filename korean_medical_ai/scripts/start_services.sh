#!/bin/bash
# ============================================================
# KorMedAI - 서비스 시작 스크립트
# ============================================================
# Mac Mini에서 모든 서비스를 순차적으로 시작합니다.
#
# 서비스 구성:
#   1. Ollama (Qwen3-8B) - 포트 11434
#   2. KorMedAI API 서버  - 포트 8000
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_DIR/korean_medical_ai/logs"

mkdir -p "$LOG_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " KorMedAI - 서비스 시작"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Ollama 서비스 시작 ──
echo ""
echo "[1/2] Ollama 서비스 확인..."
if ! pgrep -x "ollama" > /dev/null 2>&1; then
    echo "  Ollama 서비스를 시작합니다..."
    ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    sleep 3
    echo "  Ollama 시작됨 (PID: $(pgrep -x ollama || echo 'N/A'))"
else
    echo "  Ollama 이미 실행 중"
fi

# 모델 상태 확인
echo "  로드된 모델:"
ollama list 2>/dev/null || echo "  (모델 목록 조회 실패)"

# ── 2. KorMedAI API 서버 시작 ──
echo ""
echo "[2/2] KorMedAI API 서버 시작..."

# Python 가상환경 활성화
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# 서버 시작
cd "$PROJECT_DIR"
python -m korean_medical_ai.main \
    --host 0.0.0.0 \
    --port 8000 \
    > "$LOG_DIR/kormedai.log" 2>&1 &

KORMEDAI_PID=$!
echo "  KorMedAI API 서버 시작됨 (PID: $KORMEDAI_PID)"
sleep 2

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " 서비스 시작 완료"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo " 접속 정보:"
echo "  API 서버:  http://localhost:8000"
echo "  API 문서:  http://localhost:8000/docs"
echo "  Ollama:    http://localhost:11434"
echo ""
echo " 로그 확인:"
echo "  tail -f $LOG_DIR/kormedai.log"
echo "  tail -f $LOG_DIR/ollama.log"
echo ""
echo " 서비스 중지:"
echo "  kill $KORMEDAI_PID"
echo "  pkill ollama"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
