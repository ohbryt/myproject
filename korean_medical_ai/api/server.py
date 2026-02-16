"""
FastAPI 서버 (API Server)
=========================
KorMedAI REST API 서버

엔드포인트:
  POST /api/v1/consult        → 의료 상담 (멀티 에이전트)
  POST /api/v1/quick-consult  → 빠른 상담 (단일 에이전트)
  POST /api/v1/soap-note      → SOAP 노트 생성
  POST /api/v1/image-analyze  → 의료 영상 분석
  POST /api/v1/literature     → 문헌 검색
  GET  /api/v1/health         → 헬스 체크
  GET  /api/v1/models         → 로드된 모델 상태
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from korean_medical_ai import __app_name__, __version__
from korean_medical_ai.agents.multi_agent_orchestrator import MultiAgentOrchestrator
from korean_medical_ai.configs.settings import MEDICAL_DISCLAIMER_KO, SystemConfig
from korean_medical_ai.image_analysis.medical_vision import (
    ImageModality,
    MedicalImageAnalyzer,
)
from korean_medical_ai.models.model_manager import ModelManager
from korean_medical_ai.rag.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ──────────────────────────────────────
# Pydantic 요청/응답 모델
# ──────────────────────────────────────


class ConsultRequest(BaseModel):
    """의료 상담 요청"""

    patient_info: str = Field(..., description="환자 정보 (증상, 병력, 검사결과 등)")
    discussion_rounds: int = Field(default=3, ge=1, le=5, description="에이전트 토론 라운드 수")


class QuickConsultRequest(BaseModel):
    """빠른 상담 요청"""

    question: str = Field(..., description="의료 질문")
    source_type: Optional[str] = Field(None, description="검색 소스 필터")


class SOAPNoteRequest(BaseModel):
    """SOAP 노트 생성 요청"""

    clinical_info: str = Field(..., description="진료 정보")


class LiteratureRequest(BaseModel):
    """문헌 검색 요청"""

    topic: str = Field(..., description="검색 주제")


class ConsultResponse(BaseModel):
    """상담 응답"""

    answer: str
    confidence: float
    agents_used: list[str] = []
    sources: list[str] = []
    disclaimer: str = ""


class ImageAnalysisResponse(BaseModel):
    """영상 분석 응답"""

    findings: str
    modality: str
    model_used: str
    disclaimer: str = ""


class HealthResponse(BaseModel):
    """헬스 체크 응답"""

    status: str
    version: str
    loaded_models: list[str]


# ──────────────────────────────────────
# FastAPI 앱 생성
# ──────────────────────────────────────


def create_app(config: Optional[SystemConfig] = None) -> FastAPI:
    """FastAPI 앱 생성 및 설정"""

    config = config or SystemConfig()

    app = FastAPI(
        title=__app_name__,
        version=__version__,
        description="한국형 의료 AI 진단/치료 보조 시스템 - 개원가 의사를 위한 AI 어시스턴트",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 컴포넌트 초기화 (lazy loading)
    state = {
        "model_manager": None,
        "orchestrator": None,
        "rag_pipeline": None,
        "image_analyzer": None,
        "config": config,
    }

    def get_model_manager() -> ModelManager:
        if state["model_manager"] is None:
            state["model_manager"] = ModelManager(state["config"])
        return state["model_manager"]

    def get_orchestrator() -> MultiAgentOrchestrator:
        if state["orchestrator"] is None:
            state["orchestrator"] = MultiAgentOrchestrator(
                model_manager=get_model_manager(), config=state["config"],
            )
        return state["orchestrator"]

    def get_rag_pipeline() -> RAGPipeline:
        if state["rag_pipeline"] is None:
            state["rag_pipeline"] = RAGPipeline(
                model_manager=get_model_manager(), config=state["config"],
            )
        return state["rag_pipeline"]

    def get_image_analyzer() -> MedicalImageAnalyzer:
        if state["image_analyzer"] is None:
            state["image_analyzer"] = MedicalImageAnalyzer(
                model_manager=get_model_manager(), config=state["config"],
            )
        return state["image_analyzer"]

    # ── 엔드포인트 ──

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health_check():
        """시스템 상태 확인"""
        mm = get_model_manager()
        return HealthResponse(
            status="ok",
            version=__version__,
            loaded_models=mm.loaded_models,
        )

    @app.get("/api/v1/models")
    async def list_models():
        """모델 상태 조회"""
        mm = get_model_manager()
        return {
            "available_roles": mm.available_roles,
            "loaded_models": mm.loaded_models,
            "model_details": {
                role: {
                    "name": cfg.name,
                    "model_id": cfg.model_id,
                    "backend": cfg.backend,
                    "role": cfg.role,
                }
                for role, cfg in state["config"].models.items()
            },
        }

    @app.post("/api/v1/consult", response_model=ConsultResponse)
    async def consult(request: ConsultRequest):
        """
        멀티 에이전트 의료 상담

        3개 에이전트(진단/문헌/검수)가 토론하여 합의된 답변을 생성합니다.
        """
        try:
            orchestrator = get_orchestrator()
            result = orchestrator.discuss(
                patient_info=request.patient_info,
                num_rounds=request.discussion_rounds,
            )
            return ConsultResponse(
                answer=result.final_answer,
                confidence=result.confidence,
                agents_used=result.agents_used,
                disclaimer=result.disclaimer,
            )
        except Exception as e:
            logger.error(f"상담 실패: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/quick-consult", response_model=ConsultResponse)
    async def quick_consult(request: QuickConsultRequest):
        """빠른 상담 (RAG + 단일 모델)"""
        try:
            pipeline = get_rag_pipeline()
            result = pipeline.query(
                question=request.question,
                source_type=request.source_type,
            )
            return ConsultResponse(
                answer=result.answer,
                confidence=result.confidence,
                sources=[
                    doc.metadata.get("file_name", "")
                    for doc in result.source_documents
                ],
                disclaimer=result.disclaimer,
            )
        except Exception as e:
            logger.error(f"빠른 상담 실패: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/soap-note", response_model=ConsultResponse)
    async def generate_soap_note(request: SOAPNoteRequest):
        """SOAP 노트 생성"""
        try:
            pipeline = get_rag_pipeline()
            result = pipeline.generate_soap_note(clinical_info=request.clinical_info)
            return ConsultResponse(
                answer=result.answer,
                confidence=result.confidence,
                disclaimer=result.disclaimer,
            )
        except Exception as e:
            logger.error(f"SOAP 노트 생성 실패: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/image-analyze", response_model=ImageAnalysisResponse)
    async def analyze_image(
        image: UploadFile = File(...),
        modality: Optional[str] = Form(None),
        clinical_context: str = Form(""),
    ):
        """의료 영상 분석"""
        try:
            analyzer = get_image_analyzer()

            # 임시 파일로 저장
            suffix = Path(image.filename or "image.jpg").suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                content = await image.read()
                tmp.write(content)
                tmp_path = tmp.name

            # 모달리티 변환
            img_modality = None
            if modality:
                try:
                    img_modality = ImageModality(modality)
                except ValueError:
                    pass

            result = analyzer.analyze(
                image_path=tmp_path,
                modality=img_modality,
                clinical_context=clinical_context,
            )

            # 임시 파일 정리
            Path(tmp_path).unlink(missing_ok=True)

            return ImageAnalysisResponse(
                findings=result.raw_response,
                modality=result.modality,
                model_used=result.model_used,
                disclaimer=MEDICAL_DISCLAIMER_KO,
            )
        except Exception as e:
            logger.error(f"영상 분석 실패: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/literature")
    async def search_literature(request: LiteratureRequest):
        """문헌 검색"""
        try:
            from korean_medical_ai.agents.literature_agent import LiteratureAgent

            agent = LiteratureAgent(
                model_manager=get_model_manager(), config=state["config"],
            )
            result = agent.review(request.topic)
            return {
                "topic": result.query_topic,
                "review": result.raw_response,
                "sources": result.sources,
            }
        except Exception as e:
            logger.error(f"문헌 검색 실패: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app
