"""
문헌 검토 에이전트 (Literature Review Agent)
============================================
최신 의학 논문에서 관련 근거를 검색하고 검토합니다.

모델: Qwen3-8B (Apache 2.0, 한국어 범용)
역할: PubMed/KoreaMed 문헌 검색 → 근거 수준 평가 → 요약
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
class LiteratureReviewResult:
    """문헌 검토 결과"""

    query_topic: str = ""
    relevant_studies: list[dict] = field(default_factory=list)
    evidence_summary: str = ""
    evidence_level: str = ""  # "Ia", "Ib", "IIa", "IIb", "III", "IV"
    recommendation_grade: str = ""  # "A", "B", "C", "D"
    korean_guidelines_match: bool = False
    raw_response: str = ""
    sources: list[str] = field(default_factory=list)


LITERATURE_REVIEW_PROMPT = ChatPromptTemplate.from_template("""
당신은 의학 연구자이자 근거중심의학(EBM) 전문가입니다.
아래 의료 질문에 대해 최신 문헌을 검토하고 근거를 정리해 주세요.

## 역할
- 관련 의학 논문과 가이드라인의 근거를 검토합니다.
- 근거 수준(Level of Evidence)과 권고 등급(Grade of Recommendation)을 평가합니다.
- 한국 의료 환경에서의 적용 가능성을 분석합니다.
- 최신 연구 동향과 메타분석 결과를 우선 참조합니다.

## 참고 문헌
{context}

## 검토 주제
{question}

## 문헌 검토 결과

### 1. 관련 연구 요약
각 연구에 대해:
- 저자/연도/저널
- 연구 디자인 (RCT, 코호트, 메타분석 등)
- 주요 결과
- 한계점

### 2. 근거 수준 평가
- Level of Evidence: (Ia/Ib/IIa/IIb/III/IV)
- Grade of Recommendation: (A/B/C/D)
- 근거 평가 이유

### 3. 한국 임상 적용
- 한국 진료지침과의 일치 여부
- 한국 환자 집단에서의 적용 시 고려사항
- 한국 건강보험 급여 현황

### 4. 최신 연구 동향
- 진행 중인 주요 임상시험
- 최근 발표된 주목할 연구
""")


class LiteratureAgent:
    """
    문헌 검토 에이전트

    Qwen3-8B 모델을 사용하여 의학 문헌을 검토하고
    근거중심의학(EBM) 기반 분석을 수행합니다.
    """

    AGENT_NAME = "문헌 검토 에이전트"
    AGENT_ROLE = "literature"

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

    def review(self, topic: str) -> LiteratureReviewResult:
        """
        의학 주제에 대한 문헌 검토 수행

        Args:
            topic: 검토할 의학 주제

        Returns:
            LiteratureReviewResult 객체
        """
        logger.info(f"[{self.AGENT_NAME}] 문헌 검토 시작: {topic[:100]}")

        rag_response = self.rag_pipeline.query(
            question=topic,
            model_role=self.AGENT_ROLE,
            source_type="paper",
            prompt_template=LITERATURE_REVIEW_PROMPT,
        )

        result = LiteratureReviewResult(
            query_topic=topic,
            raw_response=rag_response.answer,
            sources=[
                doc.metadata.get("file_name", "Unknown")
                for doc in rag_response.source_documents
            ],
        )

        logger.info(f"[{self.AGENT_NAME}] 문헌 검토 완료")
        return result

    def get_agent_opinion(self, patient_info: str, context: str = "") -> str:
        """
        Multi-Agent 토론을 위한 에이전트 의견 생성

        Args:
            patient_info: 환자 정보 / 질문
            context: 다른 에이전트의 이전 의견

        Returns:
            에이전트의 의견 텍스트
        """
        prompt = f"""
[문헌 검토 에이전트 의견]

## 검토 주제
{patient_info}

## 다른 에이전트의 이전 의견
{context if context else "첫 번째 라운드입니다."}

## 문헌 기반 의견
최신 의학 문헌과 근거중심의학(EBM) 원칙에 기반하여 의견을 제시해 주세요.
다른 에이전트의 의견에 대해 문헌적 근거를 바탕으로 검증해 주세요.
"""
        return self.model_manager.generate(self.AGENT_ROLE, prompt)
