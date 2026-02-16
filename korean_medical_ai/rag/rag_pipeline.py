"""
RAG 파이프라인 (Retrieval-Augmented Generation Pipeline)
========================================================
한국 의료 지식 기반 RAG 시스템의 핵심 파이프라인입니다.

워크플로우:
  1. 사용자 질문 수신 (한국어/영어)
  2. 벡터 DB에서 관련 의료 문서 검색
  3. 검색된 문서를 컨텍스트로 LLM에 전달
  4. 출처가 명시된 답변 생성
  5. 의료법/가이드라인 준수 검증
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from korean_medical_ai.configs.settings import SystemConfig, MEDICAL_DISCLAIMER_KO
from korean_medical_ai.models.model_manager import ModelManager
from korean_medical_ai.rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# ──────────────────────────────────────
# 한국 의료 RAG 프롬프트 템플릿
# ──────────────────────────────────────

KOREAN_MEDICAL_RAG_PROMPT = ChatPromptTemplate.from_template("""
당신은 한국의 개원가 의사들을 보조하는 의료 AI 어시스턴트입니다.
아래 제공된 의료 문헌과 가이드라인을 참고하여 답변해 주세요.

## 답변 규칙
1. 반드시 제공된 근거 자료에 기반하여 답변하세요.
2. 근거가 없는 내용은 "근거 자료에서 확인되지 않았습니다"라고 명시하세요.
3. 답변에는 반드시 출처를 [출처: 문서명] 형태로 표기하세요.
4. 한국 의료법과 건강보험 수가 체계를 고려하세요.
5. 약물명은 성분명(영문)과 상품명(한국어)을 병기하세요.
6. 진단코드(KCD)와 처치코드를 가능한 경우 포함하세요.
7. 불확실한 경우 반드시 전문의 상담을 권고하세요.

## 참고 자료
{context}

## 환자 정보 / 질문
{question}

## 답변
""")

KOREAN_MEDICAL_SOAP_PROMPT = ChatPromptTemplate.from_template("""
당신은 한국의 개원가 의사를 위한 SOAP 노트 작성 보조 AI입니다.
아래 진료 정보를 바탕으로 SOAP 노트를 작성해 주세요.

## 참고 가이드라인
{context}

## 진료 정보
{question}

## SOAP 노트

### S (Subjective, 주관적 소견)
환자 주소, 현병력, 과거력, 가족력 등

### O (Objective, 객관적 소견)
활력징후, 이학적 검사, 검사 결과 등

### A (Assessment, 평가)
진단명 (KCD 코드 포함), 감별진단

### P (Plan, 계획)
치료 계획, 처방, 추적 검사, 환자 교육 등
- 처방: 약품명(성분명) / 용량 / 용법 / 일수
- 검사: 검사명 / 보험코드
- 다음 진료: 예약일
""")


@dataclass
class RAGResponse:
    """RAG 응답 데이터 클래스"""

    answer: str
    source_documents: list[Document] = field(default_factory=list)
    confidence: float = 0.0
    model_used: str = ""
    disclaimer: str = ""


class RAGPipeline:
    """
    한국 의료 RAG 파이프라인

    LangChain을 사용하여 벡터 DB 검색 → LLM 생성을 체인합니다.
    3개 LLM 중 적절한 모델을 자동 선택하여 사용합니다.
    """

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        vector_store_manager: Optional[VectorStoreManager] = None,
        config: Optional[SystemConfig] = None,
    ):
        self.config = config or SystemConfig()
        self.model_manager = model_manager or ModelManager(self.config)
        self.vector_store = vector_store_manager or VectorStoreManager(self.config)

    def _format_documents(self, docs: list[Document]) -> str:
        """검색된 문서를 컨텍스트 문자열로 포맷"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", doc.metadata.get("source", "Unknown"))
            source_type = doc.metadata.get("source_type", "")
            formatted.append(
                f"[자료 {i}] (출처: {source}, 유형: {source_type})\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)

    def query(
        self,
        question: str,
        model_role: str = "diagnostic",
        source_type: Optional[str] = None,
        top_k: Optional[int] = None,
        prompt_template: Optional[ChatPromptTemplate] = None,
    ) -> RAGResponse:
        """
        RAG 기반 의료 질의응답

        Args:
            question: 의료 질문 (한국어/영어)
            model_role: 사용할 모델 역할 ("diagnostic", "literature")
            source_type: 검색할 문서 종류 필터
            top_k: 검색할 문서 수
            prompt_template: 커스텀 프롬프트 템플릿

        Returns:
            RAGResponse 객체 (답변, 출처 문서, 신뢰도)
        """
        # 1. 관련 문서 검색
        retrieved_docs = self.vector_store.search(
            query=question,
            top_k=top_k or self.config.top_k_retrieval,
            source_type=source_type,
        )

        # 2. 컨텍스트 구성
        context = self._format_documents(retrieved_docs) if retrieved_docs else (
            "관련 의료 문헌을 찾지 못했습니다. 일반 의학 지식을 바탕으로 답변합니다."
        )

        # 3. LangChain 체인 구성
        template = prompt_template or KOREAN_MEDICAL_RAG_PROMPT
        llm = self.model_manager.get_langchain_llm(model_role)

        chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | template
            | llm
            | StrOutputParser()
        )

        # 4. 답변 생성
        answer = chain.invoke(question)

        # 5. 신뢰도 계산 (검색된 문서 수 기반 간단한 휴리스틱)
        confidence = min(len(retrieved_docs) / self.config.top_k_retrieval, 1.0)

        # 6. 면책 조항 추가
        disclaimer = MEDICAL_DISCLAIMER_KO if self.config.disclaimer_required else ""

        return RAGResponse(
            answer=answer,
            source_documents=retrieved_docs,
            confidence=confidence,
            model_used=self.config.models[model_role].name,
            disclaimer=disclaimer,
        )

    def generate_soap_note(
        self,
        clinical_info: str,
        model_role: str = "diagnostic",
    ) -> RAGResponse:
        """
        SOAP 노트 생성

        Args:
            clinical_info: 진료 정보 (주소, 증상, 검사결과 등)
            model_role: 사용할 모델 역할
        """
        return self.query(
            question=clinical_info,
            model_role=model_role,
            prompt_template=KOREAN_MEDICAL_SOAP_PROMPT,
        )

    def multi_source_query(self, question: str) -> dict[str, RAGResponse]:
        """
        다중 소스 검색 (가이드라인 + 논문 + 의료법을 각각 검색)

        Returns:
            소스 타입별 RAGResponse 딕셔너리
        """
        source_types = ["guideline", "paper", "law", "drug_info"]
        results = {}

        for source_type in source_types:
            try:
                results[source_type] = self.query(
                    question=question,
                    source_type=source_type,
                    top_k=3,
                )
            except Exception as e:
                logger.warning(f"{source_type} 검색 실패: {e}")

        return results
