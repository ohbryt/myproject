"""
KorMedAI 시스템 설정
====================
Mac Mini 서버 기반 한국형 의료 AI 설정
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "korean_medical_ai" / "data"
GUIDELINES_DIR = DATA_DIR / "guidelines"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"


@dataclass
class ModelConfig:
    """개별 LLM 모델 설정"""

    name: str
    model_id: str
    backend: str  # "ollama", "vllm", "transformers"
    context_length: int = 8192
    temperature: float = 0.1  # 의료용이므로 낮은 temperature 기본값
    max_tokens: int = 2048
    quantization: Optional[str] = None  # "Q4_K_M", "Q5_K_M", "AWQ" 등
    gpu_memory_fraction: float = 0.0  # 0이면 자동 할당
    role: str = ""  # "diagnostic", "literature", "compliance", "vision"


@dataclass
class SystemConfig:
    """전체 시스템 설정"""

    # ── 서버 설정 (Mac Mini) ──
    host: str = "0.0.0.0"
    port: int = 8000
    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str = "http://localhost:8080"

    # ── 3개 핵심 LLM 모델 설정 ──
    models: dict = field(default_factory=lambda: {
        # 1) 한국어 의료 전문 LLM (서울대병원 HARI)
        "diagnostic": ModelConfig(
            name="SNUH HARI-Q3-14B",
            model_id="snuh/hari-q3-14b",
            backend="vllm",
            context_length=8192,
            temperature=0.1,
            quantization="AWQ",  # Mac Mini 메모리 절약
            role="diagnostic",
        ),
        # 2) 한국어 범용 LLM (Apache 2.0 상용 가능)
        "literature": ModelConfig(
            name="Qwen3-8B",
            model_id="Qwen/Qwen3-8B",
            backend="ollama",
            context_length=32768,
            temperature=0.2,
            quantization="Q4_K_M",
            role="literature",
        ),
        # 3) 의료 비전-언어 모델 (영상 분석)
        "vision": ModelConfig(
            name="MedGemma-4B",
            model_id="google/medgemma-4b-it",
            backend="transformers",
            context_length=8192,
            temperature=0.1,
            role="vision",
        ),
    })

    # ── RAG 설정 ──
    vector_db_type: str = "chromadb"  # "chromadb" or "faiss"
    embedding_model: str = "intfloat/multilingual-e5-large"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 5

    # ── Multi-Agent 설정 ──
    agent_discussion_rounds: int = 3  # 에이전트 간 토론 라운드
    consensus_threshold: float = 0.7  # 합의 임계값
    enable_compliance_check: bool = True  # 의료법 준수 검사

    # ── 안전 설정 ──
    disclaimer_required: bool = True
    max_confidence_display: float = 0.95  # 최대 표시 신뢰도
    require_physician_confirmation: bool = True


# ──────────────────────────────────────
# 한국 의료 데이터 소스 설정
# ──────────────────────────────────────
@dataclass
class KoreanMedicalDataSources:
    """한국 의료 데이터 소스 목록"""

    sources: dict = field(default_factory=lambda: {
        "guidelines": {
            "보건복지부_진료지침": "https://www.mohw.go.kr",
            "대한의학회_임상진료지침": "https://www.kma.org",
            "건강보험심사평가원": "https://www.hira.or.kr",
            "질병관리청_가이드라인": "https://www.kdca.go.kr",
        },
        "databases": {
            "PubMed_Korean": "https://pubmed.ncbi.nlm.nih.gov",
            "KoreaMed": "https://koreamed.org",
            "KMLE_기출문제": "snuh/KorMedLawQA",
            "약학정보원": "https://www.health.kr",
        },
        "insurance": {
            "건강보험_수가체계": "https://www.hira.or.kr/rd/insuadtcrtr/",
            "비급여_항목": "https://www.hira.or.kr/rd/dissdtcrtr/",
        },
    })


# ──────────────────────────────────────
# 의료 면책 조항
# ──────────────────────────────────────
MEDICAL_DISCLAIMER_KO = """
⚠️ 의료 면책 조항 (Medical Disclaimer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
본 시스템은 의료진의 진단 및 치료 결정을 보조하기 위한 참고 도구입니다.
- 본 시스템의 출력은 최종 의학적 판단을 대체하지 않습니다.
- 모든 진단 및 치료 결정은 반드시 의사의 임상적 판단에 따라야 합니다.
- 본 시스템은 의료기기가 아니며, 의료법에 따른 의료행위에 해당하지 않습니다.
- 출력 결과는 참고용이며, 환자 안전을 위해 반드시 검증이 필요합니다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

MEDICAL_DISCLAIMER_EN = """
⚠️ Medical Disclaimer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This system is a reference tool to assist clinical decision-making.
- Outputs do not replace professional medical judgment.
- All diagnostic and treatment decisions must be made by licensed physicians.
- This system is not a certified medical device.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
