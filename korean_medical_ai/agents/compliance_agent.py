"""
검수/규정준수 에이전트 (Compliance & Verification Agent)
========================================================
최종 답변이 한국 의료법, 가이드라인, 보험 규정에 부합하는지 검증합니다.

모델: Qwen3-8B (규정 준수 검증에 사용)
역할: 의료법 검증 → 보험 수가 확인 → 안전성 체크 → 최종 검수
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
class ComplianceResult:
    """규정 준수 검증 결과"""

    is_compliant: bool = True
    medical_law_issues: list[str] = field(default_factory=list)
    insurance_issues: list[str] = field(default_factory=list)
    safety_concerns: list[str] = field(default_factory=list)
    guideline_deviations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    risk_level: str = "low"  # "low", "medium", "high", "critical"
    raw_response: str = ""


COMPLIANCE_CHECK_PROMPT = ChatPromptTemplate.from_template("""
당신은 한국 의료법 및 건강보험 전문가입니다.
아래 의료 답변이 한국 의료 법규와 가이드라인에 부합하는지 검증하세요.

## 검증 항목
1. **의료법 준수**: 의료법, 약사법, 의료기기법 등 관련 법규 위반 여부
2. **진료지침 준수**: 대한의학회/학회별 임상진료지침 부합 여부
3. **보험 규정**: 건강보험 급여 기준, 비급여 항목, 본인부담금 관련 정확성
4. **안전성**: 약물 상호작용, 금기 사항, 부작용 경고 누락 여부
5. **윤리적 사항**: 환자 권리, 사전 동의, 개인정보 보호 관련 사항

## 참고 법규/가이드라인
{context}

## 검증 대상 답변
{question}

## 검증 결과

### 1. 의료법 준수 여부
- 위반 사항:
- 근거 조항:

### 2. 진료지침 부합 여부
- 편차 사항:
- 참조 가이드라인:

### 3. 건강보험 관련 정확성
- 급여/비급여 구분 정확성:
- 수가 관련 오류:

### 4. 안전성 검토
- 약물 상호작용 위험:
- 금기 사항 누락:
- 부작용 경고 누락:

### 5. 종합 판정
- 위험 수준: (low/medium/high/critical)
- 수정 필요 사항:
- 추가 권고사항:
""")


class ComplianceAgent:
    """
    규정 준수 검증 에이전트

    다른 에이전트의 답변을 한국 의료법, 보험규정,
    임상진료지침에 비추어 검증합니다.
    """

    AGENT_NAME = "검수 에이전트"
    AGENT_ROLE = "literature"  # Qwen3-8B 사용 (규정 검증에 적합)

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

    def verify(self, answer_to_check: str) -> ComplianceResult:
        """
        의료 답변의 규정 준수 여부를 검증

        Args:
            answer_to_check: 검증할 의료 답변 텍스트

        Returns:
            ComplianceResult 객체
        """
        logger.info(f"[{self.AGENT_NAME}] 규정 준수 검증 시작")

        rag_response = self.rag_pipeline.query(
            question=answer_to_check,
            model_role=self.AGENT_ROLE,
            source_type="law",
            prompt_template=COMPLIANCE_CHECK_PROMPT,
        )

        result = ComplianceResult(
            raw_response=rag_response.answer,
        )

        # 위험 키워드 기반 간단한 안전성 체크
        high_risk_keywords = [
            "금기", "contraindicated", "위험", "즉시 중단",
            "응급", "생명 위험", "과량 투여", "치명적",
        ]
        for keyword in high_risk_keywords:
            if keyword in rag_response.answer:
                result.safety_concerns.append(f"'{keyword}' 관련 주의 사항 감지됨")
                result.risk_level = "high"

        logger.info(
            f"[{self.AGENT_NAME}] 검증 완료 (위험 수준: {result.risk_level})"
        )
        return result

    def get_agent_opinion(self, patient_info: str, context: str = "") -> str:
        """
        Multi-Agent 토론을 위한 에이전트 의견 (규정 준수 관점)

        Args:
            patient_info: 환자 정보 / 의료 답변
            context: 다른 에이전트의 이전 의견

        Returns:
            규정 준수 관점의 의견
        """
        prompt = f"""
[검수 에이전트 의견 - 규정 준수 관점]

## 검토 대상
{patient_info}

## 다른 에이전트의 이전 의견
{context if context else "첫 번째 라운드입니다."}

## 규정 준수 검증 의견
다른 에이전트들의 의견이 한국 의료법, 보험 규정, 임상진료지침에
부합하는지 검증하고, 수정이 필요한 사항을 지적해 주세요.

특히 다음 사항을 확인:
1. 의료법 위반 소지가 없는가?
2. 건강보험 급여 기준에 맞는가?
3. 약물 안전성 (상호작용, 금기) 확인했는가?
4. 환자 안전을 위한 주의사항이 포함되었는가?
"""
        return self.model_manager.generate(self.AGENT_ROLE, prompt)
