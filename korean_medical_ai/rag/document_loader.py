"""
한국 의료 문서 로더 (Document Loader)
=====================================
다양한 한국 의료 데이터 소스에서 문서를 로드합니다.

지원 포맷:
  - PDF (진료지침, 논문)
  - HTML (웹 가이드라인)
  - JSON (구조화된 의료 데이터)
  - TXT/MD (일반 텍스트)
  - KMLE 기출문제 (HuggingFace 데이터셋)
"""

import json
import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class KoreanMedicalDocumentLoader:
    """한국 의료 문서 통합 로더"""

    SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".json", ".txt", ".md", ".csv"}

    def load_directory(self, directory: str, recursive: bool = True) -> list[Document]:
        """
        디렉토리에서 의료 문서 일괄 로드

        Args:
            directory: 문서 디렉토리 경로
            recursive: 하위 디렉토리 포함 여부
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"디렉토리가 존재하지 않습니다: {directory}")
            return []

        documents = []
        pattern = "**/*" if recursive else "*"

        for file_path in dir_path.glob(pattern):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    docs = self.load_file(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"파일 로드 실패: {file_path} - {e}")

        logger.info(f"총 {len(documents)}개 문서 로드됨 (경로: {directory})")
        return documents

    def load_file(self, file_path: str) -> list[Document]:
        """단일 파일 로드"""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._load_pdf(path)
        elif ext in (".html", ".htm"):
            return self._load_html(path)
        elif ext == ".json":
            return self._load_json(path)
        elif ext in (".txt", ".md"):
            return self._load_text(path)
        elif ext == ".csv":
            return self._load_csv(path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {ext}")

    def _load_pdf(self, path: Path) -> list[Document]:
        """PDF 문서 로드"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(path))
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "file_name": path.name,
                    "file_type": "pdf",
                    "source_path": str(path),
                })
            return docs
        except ImportError:
            logger.error("pypdf 미설치. pip install pypdf 실행 필요")
            return []

    def _load_html(self, path: Path) -> list[Document]:
        """HTML 문서 로드"""
        try:
            from langchain_community.document_loaders import BSHTMLLoader
            loader = BSHTMLLoader(str(path), open_encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "file_name": path.name,
                    "file_type": "html",
                    "source_path": str(path),
                })
            return docs
        except ImportError:
            logger.error("beautifulsoup4 미설치. pip install beautifulsoup4 실행 필요")
            return []

    def _load_json(self, path: Path) -> list[Document]:
        """JSON 의료 데이터 로드"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []

        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    content = item.get("content") or item.get("text") or json.dumps(
                        item, ensure_ascii=False
                    )
                    metadata = {
                        k: v for k, v in item.items()
                        if k not in ("content", "text") and isinstance(v, (str, int, float, bool))
                    }
                    metadata.update({
                        "file_name": path.name,
                        "file_type": "json",
                        "index": i,
                    })
                    documents.append(Document(page_content=str(content), metadata=metadata))
        elif isinstance(data, dict):
            content = data.get("content") or data.get("text") or json.dumps(
                data, ensure_ascii=False
            )
            documents.append(Document(
                page_content=str(content),
                metadata={"file_name": path.name, "file_type": "json"},
            ))

        return documents

    def _load_text(self, path: Path) -> list[Document]:
        """텍스트 파일 로드"""
        text = path.read_text(encoding="utf-8")
        return [Document(
            page_content=text,
            metadata={
                "file_name": path.name,
                "file_type": path.suffix.lstrip("."),
                "source_path": str(path),
            },
        )]

    def _load_csv(self, path: Path) -> list[Document]:
        """CSV 데이터 로드 (약품 정보, 수가 정보 등)"""
        try:
            from langchain_community.document_loaders.csv_loader import CSVLoader
            loader = CSVLoader(str(path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "file_name": path.name,
                    "file_type": "csv",
                })
            return docs
        except ImportError:
            logger.error("csv loader 미사용. 기본 처리로 전환")
            return self._load_text(path)

    def load_huggingface_dataset(
        self,
        dataset_id: str,
        split: str = "train",
        text_column: str = "question",
        max_samples: Optional[int] = None,
    ) -> list[Document]:
        """
        HuggingFace 데이터셋 로드 (예: snuh/KorMedLawQA)

        Args:
            dataset_id: HuggingFace 데이터셋 ID
            split: 데이터셋 분할
            text_column: 텍스트 컬럼명
            max_samples: 최대 샘플 수
        """
        try:
            from datasets import load_dataset

            dataset = load_dataset(dataset_id, split=split)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

            documents = []
            for i, row in enumerate(dataset):
                content_parts = []
                for col in dataset.column_names:
                    val = row.get(col)
                    if val is not None:
                        content_parts.append(f"{col}: {val}")

                documents.append(Document(
                    page_content="\n".join(content_parts),
                    metadata={
                        "source": dataset_id,
                        "split": split,
                        "index": i,
                        "file_type": "huggingface",
                    },
                ))

            logger.info(f"HuggingFace 데이터셋 로드: {dataset_id} ({len(documents)}건)")
            return documents

        except ImportError:
            logger.error("datasets 미설치. pip install datasets 실행 필요")
            return []
