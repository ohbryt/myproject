#!/bin/bash
# ============================================================
# KorMedAI - Mac Mini 서버 설정 스크립트
# ============================================================
# Mac Mini (Apple Silicon M2/M4) 에서 한국형 의료 AI 시스템을
# 구동하기 위한 환경을 설정합니다.
#
# 필요 사양:
#   - Mac Mini M2 Pro 이상 (M4 권장)
#   - RAM: 32GB 이상 (64GB 권장)
#   - SSD: 500GB 이상
#
# 설치 대상:
#   1. Ollama (Qwen3-8B 서빙)
#   2. vLLM (HARI-Q3-14B 서빙)
#   3. Python 환경 및 의존성
#   4. ChromaDB (벡터 DB)
# ============================================================

set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " KorMedAI - Mac Mini 서버 설정"
echo " 한국형 의료 AI 진단/치료 보조 시스템"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── 1. Homebrew 확인 ──
echo ""
echo "[1/6] Homebrew 확인..."
if ! command -v brew &> /dev/null; then
    echo "  Homebrew를 설치합니다..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "  Homebrew 이미 설치됨: $(brew --version | head -1)"
fi

# ── 2. Ollama 설치 ──
echo ""
echo "[2/6] Ollama 설치 및 모델 다운로드..."
if ! command -v ollama &> /dev/null; then
    echo "  Ollama를 설치합니다..."
    brew install ollama
else
    echo "  Ollama 이미 설치됨: $(ollama --version 2>/dev/null || echo 'installed')"
fi

# Ollama 서비스 시작
echo "  Ollama 서비스 시작..."
ollama serve &> /dev/null &
sleep 3

# Qwen3-8B 모델 다운로드 (Apache 2.0 - 상용 가능)
echo "  Qwen3-8B 모델 다운로드 (문헌 검토/규정 준수 에이전트용)..."
echo "  ※ Apache 2.0 라이선스 - 상업적 사용 가능"
ollama pull qwen3:8b || echo "  ⚠️ 모델 다운로드 실패 - 수동 설치 필요"

# ── 3. Python 환경 설정 ──
echo ""
echo "[3/6] Python 환경 설정..."
if ! command -v python3 &> /dev/null; then
    brew install python@3.11
fi

# venv 생성
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "  가상환경 생성: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ── 4. Python 의존성 설치 ──
echo ""
echo "[4/6] Python 의존성 설치..."
pip install --upgrade pip

# requirements.txt에서 설치
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -r "$PROJECT_DIR/requirements.txt"
else
    echo "  requirements.txt를 생성하고 설치합니다..."
    pip install \
        langchain>=0.3.0 \
        langchain-community>=0.3.0 \
        langchain-core>=0.3.0 \
        chromadb>=0.5.0 \
        faiss-cpu>=1.8.0 \
        sentence-transformers>=3.0.0 \
        transformers>=4.45.0 \
        torch>=2.4.0 \
        accelerate>=0.34.0 \
        fastapi>=0.115.0 \
        uvicorn>=0.30.0 \
        python-multipart>=0.0.9 \
        pypdf>=4.0.0 \
        beautifulsoup4>=4.12.0 \
        datasets>=3.0.0 \
        Pillow>=10.0.0 \
        pydantic>=2.0.0 \
        httpx>=0.27.0
fi

# ── 5. vLLM 설치 (HARI-Q3-14B 서빙용) ──
echo ""
echo "[5/6] vLLM 설치 (HARI-Q3-14B 서빙용)..."
echo "  ※ Apple Silicon에서는 vLLM 대신 llama.cpp 또는 MLX 사용 권장"

# Apple Silicon에서는 MLX가 더 적합할 수 있음
pip install mlx-lm>=0.18.0 || echo "  ⚠️ MLX 설치 실패 - 선택적 구성요소"

# vLLM은 Linux/CUDA 환경에서만 완전 지원
# Mac에서는 Ollama를 통해 모든 모델을 서빙하는 것이 더 실용적
echo "  Mac Mini에서는 Ollama로 통합 서빙을 권장합니다."
echo "  HARI-Q3-14B 모델은 Ollama 커스텀 모델로 등록하거나"
echo "  llama.cpp GGUF 변환 후 사용합니다."

# ── 6. 디렉토리 구조 확인 ──
echo ""
echo "[6/6] 데이터 디렉토리 확인..."
mkdir -p "$PROJECT_DIR/data/guidelines"
mkdir -p "$PROJECT_DIR/data/embeddings"
mkdir -p "$PROJECT_DIR/data/papers"
mkdir -p "$PROJECT_DIR/data/laws"
mkdir -p "$PROJECT_DIR/data/drug_info"
mkdir -p "$PROJECT_DIR/logs"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " 설정 완료!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo " 다음 단계:"
echo "  1. HARI-Q3-14B 모델 설정:"
echo "     - HuggingFace에서 snuh/hari-q3-14b 다운로드"
echo "     - GGUF 변환 후 Ollama에 등록 또는 MLX로 변환"
echo ""
echo "  2. MedGemma-4B 모델 설정:"
echo "     - HuggingFace에서 google/medgemma-4b-it 다운로드"
echo "     - Transformers로 직접 로드됩니다"
echo ""
echo "  3. 의료 데이터 준비:"
echo "     - data/guidelines/ : 진료지침 PDF 파일"
echo "     - data/papers/     : 의학 논문"
echo "     - data/laws/       : 의료법 자료"
echo "     - data/drug_info/  : 약품 정보"
echo ""
echo "  4. 서버 시작:"
echo "     source .venv/bin/activate"
echo "     python -m korean_medical_ai.main"
echo ""
echo "  5. API 문서 확인:"
echo "     http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
