"""
멀티 에이전트 오케스트레이터 (Multi-Agent Orchestrator)
=======================================================
3개의 전문 에이전트가 LangChain을 통해 협업하며 토론합니다.

워크플로우:
  Round 1: 각 에이전트가 독립적으로 분석
  Round 2: 다른 에이전트의 의견을 참고하여 보완/반론
  Round 3: 합의 도출 및 최종 답변 생성

에이전트:
  1. 진단 보조 에이전트 (HARI-Q3-14B) → 감별진단/치료계획
  2. 문헌 검토 에이전트 (Qwen3-8B)    → 근거 확인/최신 문헌
  3. 검수 에이전트 (Qwen3-8B)          → 의료법/보험/안전성 검증
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from korean_medical_ai.agents.diagnostic_agent import DiagnosticAgent
from korean_medical_ai.agents.literature_agent import LiteratureAgent
from korean_medical_ai.agents.compliance_agent import ComplianceAgent
from korean_medical_ai.configs.settings import SystemConfig, MEDICAL_DISCLAIMER_KO
from korean_medical_ai.models.model_manager import ModelManager
from korean_medical_ai.rag.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class DiscussionRound:
    """토론 라운드 기록"""

    round_number: int
    diagnostic_opinion: str = ""
    literature_opinion: str = ""
    compliance_opinion: str = ""


@dataclass
class MultiAgentResult:
    """멀티 에이전트 최종 결과"""

    final_answer: str = ""
    discussion_rounds: list[DiscussionRound] = field(default_factory=list)
    consensus_reached: bool = False
    confidence: float = 0.0
    risk_level: str = "low"
    disclaimer: str = ""
    agents_used: list[str] = field(default_factory=list)


class MultiAgentOrchestrator:
    """
    멀티 에이전트 오케스트레이터

    3개의 LLM 모델이 각각 전문 에이전트 역할을 맡아
    LangChain 기반으로 토론하고 합의된 답변을 생성합니다.

    Mac Mini에서 실행 시:
    - HARI-Q3-14B: vLLM 서빙 (진단)
    - Qwen3-8B: Ollama 서빙 (문헌 검토 + 규정 준수)
    - MedGemma-4B: Transformers 직접 로드 (영상 분석, 필요 시)
    """

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        config: Optional[SystemConfig] = None,
    ):
        self.config = config or SystemConfig()
        self.model_manager = model_manager or ModelManager(self.config)
        self.rag_pipeline = RAGPipeline(
            model_manager=self.model_manager, config=self.config,
        )

        # 3개 전문 에이전트 초기화
        self.diagnostic_agent = DiagnosticAgent(
            model_manager=self.model_manager,
            rag_pipeline=self.rag_pipeline,
            config=self.config,
        )
        self.literature_agent = LiteratureAgent(
            model_manager=self.model_manager,
            rag_pipeline=self.rag_pipeline,
            config=self.config,
        )
        self.compliance_agent = ComplianceAgent(
            model_manager=self.model_manager,
            rag_pipeline=self.rag_pipeline,
            config=self.config,
        )

    def discuss(
        self,
        patient_info: str,
        num_rounds: Optional[int] = None,
    ) -> MultiAgentResult:
        """
        멀티 에이전트 토론 실행

        3개 에이전트가 여러 라운드에 걸쳐 토론하고
        최종 합의된 답변을 생성합니다.

        Args:
            patient_info: 환자 정보 / 의료 질문
            num_rounds: 토론 라운드 수 (기본: config 설정값)

        Returns:
            MultiAgentResult 객체
        """
        rounds = num_rounds or self.config.agent_discussion_rounds
        discussion_history: list[DiscussionRound] = []
        accumulated_context = ""

        logger.info(f"멀티 에이전트 토론 시작 ({rounds} 라운드)")

        for round_num in range(1, rounds + 1):
            logger.info(f"━━━ 라운드 {round_num}/{rounds} ━━━")
            round_record = DiscussionRound(round_number=round_num)

            # ── 1. 진단 보조 에이전트 ──
            logger.info(f"  [Round {round_num}] 진단 보조 에이전트 분석 중...")
            round_record.diagnostic_opinion = self.diagnostic_agent.get_agent_opinion(
                patient_info=patient_info,
                context=accumulated_context,
            )

            # ── 2. 문헌 검토 에이전트 ──
            logger.info(f"  [Round {round_num}] 문헌 검토 에이전트 분석 중...")
            round_record.literature_opinion = self.literature_agent.get_agent_opinion(
                patient_info=patient_info,
                context=accumulated_context + f"\n\n진단 에이전트 의견:\n{round_record.diagnostic_opinion}",
            )

            # ── 3. 검수 에이전트 ──
            logger.info(f"  [Round {round_num}] 검수 에이전트 검증 중...")
            round_record.compliance_opinion = self.compliance_agent.get_agent_opinion(
                patient_info=patient_info,
                context=(
                    accumulated_context
                    + f"\n\n진단 에이전트 의견:\n{round_record.diagnostic_opinion}"
                    + f"\n\n문헌 검토 에이전트 의견:\n{round_record.literature_opinion}"
                ),
            )

            discussion_history.append(round_record)

            # 누적 컨텍스트 업데이트
            accumulated_context = self._compile_round_context(discussion_history)

            logger.info(f"  라운드 {round_num} 완료")

        # ── 최종 합의 답변 생성 ──
        final_answer = self._generate_consensus(patient_info, discussion_history)

        result = MultiAgentResult(
            final_answer=final_answer,
            discussion_rounds=discussion_history,
            consensus_reached=True,
            confidence=self._calculate_confidence(discussion_history),
            disclaimer=MEDICAL_DISCLAIMER_KO if self.config.disclaimer_required else "",
            agents_used=[
                self.diagnostic_agent.AGENT_NAME,
                self.literature_agent.AGENT_NAME,
                self.compliance_agent.AGENT_NAME,
            ],
        )

        logger.info(f"멀티 에이전트 토론 완료 (신뢰도: {result.confidence:.2f})")
        return result

    def _compile_round_context(self, rounds: list[DiscussionRound]) -> str:
        """토론 라운드 기록을 컨텍스트 문자열로 컴파일"""
        parts = []
        for r in rounds:
            parts.append(f"""
=== 라운드 {r.round_number} ===

[진단 보조 에이전트]
{r.diagnostic_opinion[:1000]}

[문헌 검토 에이전트]
{r.literature_opinion[:1000]}

[검수 에이전트]
{r.compliance_opinion[:1000]}
""")
        return "\n".join(parts)

    def _generate_consensus(
        self,
        patient_info: str,
        rounds: list[DiscussionRound],
    ) -> str:
        """토론 결과를 종합하여 최종 합의 답변 생성"""
        discussion_summary = self._compile_round_context(rounds)

        consensus_prompt = f"""
당신은 한국 개원가 의사를 위한 의료 AI 시스템의 최종 답변 생성기입니다.
3명의 전문 에이전트(진단 보조, 문헌 검토, 규정 준수 검수)가 {len(rounds)} 라운드에 걸쳐
토론한 내용을 종합하여 최종 합의 답변을 생성해 주세요.

## 원칙
1. 세 에이전트 모두 동의한 사항을 최우선으로 반영
2. 이견이 있는 사항은 근거 수준이 높은 의견을 우선
3. 검수 에이전트가 지적한 안전/규정 문제는 반드시 반영
4. 불확실한 사항은 명확히 불확실하다고 표시
5. 최종 답변에는 출처와 근거 수준을 명시

## 환자 정보 / 질문
{patient_info}

## 에이전트 토론 내용
{discussion_summary}

## 최종 합의 답변

### 진단 분석

### 근거 및 문헌

### 치료 계획

### 주의사항 및 안전 정보

### 건강보험 관련 참고

### 불확실한 사항 / 추가 확인 필요 사항
"""
        # 진단 모델(HARI-Q3-14B)로 최종 답변 생성
        return self.model_manager.generate("diagnostic", consensus_prompt)

    def _calculate_confidence(self, rounds: list[DiscussionRound]) -> float:
        """토론 결과 기반 신뢰도 계산 (간단한 휴리스틱)"""
        if not rounds:
            return 0.0

        # 마지막 라운드의 의견 길이 기반 간단한 휴리스틱
        last_round = rounds[-1]
        opinions = [
            last_round.diagnostic_opinion,
            last_round.literature_opinion,
            last_round.compliance_opinion,
        ]
        # 모든 에이전트가 응답했으면 기본 신뢰도 0.7
        non_empty = sum(1 for o in opinions if len(o) > 50)
        base_confidence = non_empty / len(opinions) * 0.7

        # 라운드 수 보너스 (최대 0.3)
        round_bonus = min(len(rounds) / 5, 0.3)

        return min(base_confidence + round_bonus, self.config.max_confidence_display)

    def quick_consult(self, question: str) -> str:
        """
        빠른 상담 (단일 라운드, 진단 에이전트만 사용)

        Args:
            question: 간단한 의료 질문

        Returns:
            답변 텍스트
        """
        result = self.diagnostic_agent.analyze(question)
        return result.raw_response
