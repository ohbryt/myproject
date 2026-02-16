"""
진단 보조 에이전트 (Diagnostic Agent)
=====================================
증상을 분석하고 의심 질환 리스트를 생성합니다.

모델: SNUH HARI-Q3-14B (한국어 의료 전문)
역할: 환자 증상 → 감별진단 → 추천 검사 → 치료 계획
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from korean_medical_ai.configs.settings import SystemConfig
from korean_medical_ai.models.model_manager import ModelManager
from korean_medical_ai.rag.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticResult:
    """진단 보조 결과"""

    chief_complaint: str = ""
    differential_diagnoses: list[dict] = field(default_factory=list)
    recommended_tests: list[dict] = field(default_factory=list)
    treatment_plan: str = ""
    red_flags: list[str] = field(default_factory=list)
    kcd_codes: list[str] = field(default_factory=list)
    insurance_notes: str = ""
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)
    raw_response: str = ""


DIAGNOSTIC_PROMPT = ChatPromptTemplate.from_template("""
당신은 한국 개원가의 숙련된 내과 전문의를 보조하는 AI입니다.
SNUH(서울대병원) 임상 가이드라인에 기반하여 진단을 보조합니다.

## 역할
- 환자의 증상과 소견을 분석하여 감별진단 목록을 제시합니다.
- 각 감별진단에 대한 확률과 근거를 설명합니다.
- 추가로 필요한 검사를 건강보험 수가코드와 함께 제시합니다.
- Red flag (응급 상황 징후)을 반드시 확인합니다.

## 참고 가이드라인
{context}

## 환자 정보
{patient_info}

## 분석 결과

### 1. 주소 (Chief Complaint) 분석


### 2. 감별진단 (Differential Diagnoses)
각 진단에 대해 다음을 포함:
- 진단명 (한국어/영어)
- KCD 코드
- 확률 (높음/중간/낮음)
- 주요 근거
- 감별 포인트

### 3. Red Flags (위험 징후)
즉시 조치가 필요한 징후를 나열

### 4. 추천 검사
- 검사명 / 보험코드 / 목적 / 우선순위

### 5. 치료 계획 (초기)
- 경험적 치료 (약물명/용량/용법)
- 비약물적 치료
- 환자 교육 사항

### 6. 건강보험 관련 참고사항
- 급여/비급여 구분
- 본인부담금 관련 사항
""")


class DiagnosticAgent:
    """
    진단 보조 에이전트

    HARI-Q3-14B 모델을 사용하여 한국 의료 환경에 최적화된
    감별진단과 치료 계획을 생성합니다.
    """

    AGENT_NAME = "진단 보조 에이전트"
    AGENT_ROLE = "diagnostic"

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
        config: Optional[SystemConfig] = None,
    ):
        self.config = config or SystemConfig()
        self.model_manager = model_manager or ModelManager(self.config)
        self.rag_pipeline = rag_pipeline or RAGPipeline(
            model_manager=self.model_manager, config=self.config,
        )

    def analyze(self, patient_info: str) -> DiagnosticResult:
        """
        환자 정보를 분석하여 진단 보조 결과를 생성

        Args:
            patient_info: 환자 정보 (증상, 병력, 검사결과 등)

        Returns:
            DiagnosticResult 객체
        """
        logger.info(f"[{self.AGENT_NAME}] 진단 분석 시작")

        # RAG를 통해 관련 가이드라인 검색 + LLM 답변 생성
        rag_response = self.rag_pipeline.query(
            question=patient_info,
            model_role=self.AGENT_ROLE,
            prompt_template=DIAGNOSTIC_PROMPT,
        )

        result = DiagnosticResult(
            chief_complaint=patient_info[:200],
            raw_response=rag_response.answer,
            confidence=rag_response.confidence,
            sources=[
                doc.metadata.get("file_name", "Unknown")
                for doc in rag_response.source_documents
            ],
        )

        logger.info(f"[{self.AGENT_NAME}] 진단 분석 완료 (신뢰도: {result.confidence:.2f})")
        return result

    def get_agent_opinion(self, patient_info: str, context: str = "") -> str:
        """
        Multi-Agent 토론을 위한 에이전트 의견 생성

        Args:
            patient_info: 환자 정보
            context: 다른 에이전트의 이전 의견

        Returns:
            에이전트의 의견 텍스트
        """
        prompt = f"""
[진단 보조 에이전트 의견]

## 환자 정보
{patient_info}

## 다른 에이전트의 이전 의견
{context if context else "첫 번째 라운드입니다."}

## 진단 분석 의견
한국 임상 가이드라인에 기반하여 감별진단과 근거를 제시해 주세요.
다른 에이전트의 의견이 있다면, 동의하거나 반론을 제시해 주세요.
"""
        return self.model_manager.generate(self.AGENT_ROLE, prompt)
