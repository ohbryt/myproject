"""
벡터 스토어 관리 (Vector Store Manager)
=======================================
한국 의료 가이드라인, 논문, 의료법 등을 벡터 DB에 저장하고 검색합니다.

지원 DB:
  - ChromaDB (기본값, 로컬 persistent)
  - FAISS (고속 검색)
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from korean_medical_ai.configs.settings import SystemConfig, EMBEDDINGS_DIR

logger = logging.getLogger(__name__)


class KoreanMedicalTextSplitter:
    """
    한국어 의료 문서 전용 텍스트 분할기

    한국어 의료 문서의 특성을 고려합니다:
    - 한국어/영어 혼용 (의학 용어)
    - 가이드라인 구조 (1. 2. 3. / 가. 나. 다.)
    - 처방 정보, 용법/용량 등의 구조적 데이터
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n\n",  # 대제목 구분
                "\n\n",    # 문단 구분
                "\n",      # 줄바꿈
                "。",      # 한국어 마침표 (일부 문서)
                ".",       # 영어 마침표
                " ",       # 공백
                "",        # 문자 단위
            ],
            length_function=len,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return self._splitter.split_documents(documents)

    def split_text(self, text: str) -> list[str]:
        return self._splitter.split_text(text)


class VectorStoreManager:
    """
    벡터 스토어 통합 관리자

    한국 의료 데이터를 벡터화하여 저장하고,
    RAG 파이프라인에서 관련 문서를 검색합니다.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vector_store: Optional[VectorStore] = None
        self._text_splitter = KoreanMedicalTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        """임베딩 모델 로드 (multilingual-e5-large)"""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": "mps"},  # Mac Mini Apple Silicon
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 32,
                },
            )
            logger.info(f"임베딩 모델 로드: {self.config.embedding_model}")
        return self._embeddings

    def initialize_store(self, persist_directory: Optional[str] = None) -> VectorStore:
        """벡터 스토어 초기화"""
        persist_dir = persist_directory or str(EMBEDDINGS_DIR / "medical_kb")
        embeddings = self._get_embeddings()

        if self.config.vector_db_type == "chromadb":
            self._vector_store = Chroma(
                collection_name="korean_medical_knowledge",
                embedding_function=embeddings,
                persist_directory=persist_dir,
            )
            logger.info(f"ChromaDB 초기화: {persist_dir}")
        elif self.config.vector_db_type == "faiss":
            faiss_path = Path(persist_dir) / "faiss_index"
            if faiss_path.exists():
                self._vector_store = FAISS.load_local(
                    str(faiss_path), embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info(f"FAISS 인덱스 로드: {faiss_path}")
            else:
                # 빈 인덱스 생성은 문서 추가 시 자동 처리
                self._vector_store = None
                logger.info("FAISS 인덱스가 없습니다. 문서 추가 시 생성됩니다.")
        else:
            raise ValueError(f"지원하지 않는 벡터 DB: {self.config.vector_db_type}")

        return self._vector_store

    def add_documents(
        self,
        documents: list[Document],
        source_type: str = "guideline",
    ) -> int:
        """
        문서를 벡터 스토어에 추가

        Args:
            documents: 추가할 문서 목록
            source_type: 문서 종류 (guideline, paper, law, drug_info, insurance)

        Returns:
            추가된 청크 수
        """
        # 메타데이터에 소스 타입 추가
        for doc in documents:
            doc.metadata["source_type"] = source_type
            doc.metadata["language"] = "ko"

        # 텍스트 분할
        chunks = self._text_splitter.split_documents(documents)
        logger.info(f"문서 {len(documents)}개 → 청크 {len(chunks)}개 (source: {source_type})")

        if self._vector_store is None:
            embeddings = self._get_embeddings()
            if self.config.vector_db_type == "faiss":
                self._vector_store = FAISS.from_documents(chunks, embeddings)
            else:
                self.initialize_store()
                self._vector_store.add_documents(chunks)
        else:
            self._vector_store.add_documents(chunks)

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_type: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> list[Document]:
        """
        관련 문서 검색

        Args:
            query: 검색 쿼리 (한국어/영어)
            top_k: 반환할 최대 문서 수
            source_type: 특정 소스 타입으로 필터링
            score_threshold: 최소 유사도 점수

        Returns:
            관련 문서 목록
        """
        if self._vector_store is None:
            logger.warning("벡터 스토어가 초기화되지 않았습니다")
            return []

        k = top_k or self.config.top_k_retrieval

        # 필터 설정
        filter_dict = {}
        if source_type:
            filter_dict["source_type"] = source_type

        if score_threshold > 0 and hasattr(self._vector_store, "similarity_search_with_score"):
            results_with_scores = self._vector_store.similarity_search_with_score(
                query, k=k, filter=filter_dict if filter_dict else None,
            )
            return [doc for doc, score in results_with_scores if score >= score_threshold]
        else:
            return self._vector_store.similarity_search(
                query, k=k, filter=filter_dict if filter_dict else None,
            )

    def save(self, path: Optional[str] = None) -> None:
        """벡터 스토어 저장"""
        if self._vector_store is None:
            logger.warning("저장할 벡터 스토어가 없습니다")
            return

        if isinstance(self._vector_store, FAISS):
            save_path = path or str(EMBEDDINGS_DIR / "medical_kb" / "faiss_index")
            self._vector_store.save_local(save_path)
            logger.info(f"FAISS 인덱스 저장: {save_path}")
        elif isinstance(self._vector_store, Chroma):
            # ChromaDB는 자동으로 persist됨
            logger.info("ChromaDB 자동 저장됨")

    @property
    def document_count(self) -> int:
        """저장된 문서 수"""
        if self._vector_store is None:
            return 0
        if isinstance(self._vector_store, Chroma):
            return self._vector_store._collection.count()
        return -1  # FAISS는 직접 카운트 불가
