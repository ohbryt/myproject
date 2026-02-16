"""
기본 테스트 (Basic Tests)
=========================
모듈 임포트 및 기본 설정 테스트
"""

import pytest


def test_import_package():
    """패키지 임포트 테스트"""
    import korean_medical_ai
    assert korean_medical_ai.__version__ == "0.1.0"
    assert korean_medical_ai.__app_name__ == "KorMedAI"


def test_import_settings():
    """설정 모듈 임포트 테스트"""
    from korean_medical_ai.configs.settings import (
        SystemConfig,
        ModelConfig,
        KoreanMedicalDataSources,
        MEDICAL_DISCLAIMER_KO,
    )
    config = SystemConfig()
    assert "diagnostic" in config.models
    assert "literature" in config.models
    assert "vision" in config.models
    assert config.vector_db_type in ("chromadb", "faiss")
    assert len(MEDICAL_DISCLAIMER_KO) > 0


def test_model_config():
    """모델 설정 테스트"""
    from korean_medical_ai.configs.settings import SystemConfig

    config = SystemConfig()

    # HARI-Q3-14B (진단)
    diag = config.models["diagnostic"]
    assert "hari" in diag.model_id.lower()
    assert diag.backend == "vllm"
    assert diag.temperature <= 0.2  # 의료용 낮은 temperature

    # Qwen3-8B (문헌 검토)
    lit = config.models["literature"]
    assert "qwen" in lit.model_id.lower()
    assert lit.backend == "ollama"

    # MedGemma (영상)
    vis = config.models["vision"]
    assert "medgemma" in vis.model_id.lower()
    assert vis.backend == "transformers"


def test_data_sources():
    """한국 의료 데이터 소스 설정 테스트"""
    from korean_medical_ai.configs.settings import KoreanMedicalDataSources

    sources = KoreanMedicalDataSources()
    assert "guidelines" in sources.sources
    assert "databases" in sources.sources
    assert "insurance" in sources.sources


def test_text_splitter():
    """한국어 의료 텍스트 분할기 테스트"""
    from korean_medical_ai.rag.vector_store import KoreanMedicalTextSplitter

    splitter = KoreanMedicalTextSplitter(chunk_size=100, chunk_overlap=20)
    text = "고혈압 진단 기준은 수축기혈압 140mmHg 이상입니다. " * 20
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 120  # chunk_size + margin


def test_document_loader_init():
    """문서 로더 초기화 테스트"""
    from korean_medical_ai.rag.document_loader import KoreanMedicalDocumentLoader

    loader = KoreanMedicalDocumentLoader()
    assert ".pdf" in loader.SUPPORTED_EXTENSIONS
    assert ".json" in loader.SUPPORTED_EXTENSIONS


def test_model_manager_init():
    """모델 매니저 초기화 테스트"""
    from korean_medical_ai.models.model_manager import ModelManager

    mm = ModelManager()
    assert "diagnostic" in mm.available_roles
    assert "literature" in mm.available_roles
    assert "vision" in mm.available_roles
    assert len(mm.loaded_models) == 0  # 아직 로드하지 않음


def test_image_modality():
    """영상 모달리티 enum 테스트"""
    from korean_medical_ai.image_analysis.medical_vision import ImageModality

    assert ImageModality.CHEST_XRAY == "chest_xray"
    assert ImageModality.DERMATOLOGY == "dermatology"


def test_fine_tuning_config():
    """파인튜닝 설정 테스트"""
    from korean_medical_ai.models.fine_tuning import FineTuningConfig

    config = FineTuningConfig()
    assert "Qwen" in config.base_model_id  # Apache 2.0 모델
    assert config.use_qlora is True
    assert config.lora_r > 0


def test_dataset_builder():
    """데이터셋 빌더 테스트"""
    from korean_medical_ai.models.fine_tuning import KoreanMedicalDatasetBuilder

    builder = KoreanMedicalDatasetBuilder()
    builder.add_qa_pair(
        question="고혈압의 진단 기준은?",
        answer="수축기혈압 140mmHg 이상 또는 이완기혈압 90mmHg 이상",
    )

    alpaca = builder.to_alpaca_format()
    assert len(alpaca) == 1
    assert alpaca[0]["instruction"] == "고혈압의 진단 기준은?"

    chat = builder.to_chat_format()
    assert len(chat) == 1
    assert len(chat[0]["messages"]) == 3  # system + user + assistant
