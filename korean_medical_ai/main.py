"""
KorMedAI 메인 엔트리포인트
===========================
한국형 의료 AI 진단/치료 보조 시스템

사용법:
  # API 서버 시작
  python -m korean_medical_ai.main

  # 특정 포트로 시작
  python -m korean_medical_ai.main --port 9000

  # 개발 모드 (자동 리로드)
  python -m korean_medical_ai.main --reload

  # RAG 데이터 인덱싱
  python -m korean_medical_ai.main --index-data /path/to/medical/docs
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def run_server(host: str, port: int, reload: bool = False) -> None:
    """FastAPI 서버 실행"""
    import uvicorn

    from korean_medical_ai.api.server import create_app
    from korean_medical_ai.configs.settings import SystemConfig

    config = SystemConfig(host=host, port=port)
    app = create_app(config)

    logger.info(f"KorMedAI 서버 시작: http://{host}:{port}")
    logger.info(f"API 문서: http://{host}:{port}/docs")

    uvicorn.run(
        app if not reload else "korean_medical_ai.api.server:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=reload,
    )


def index_data(data_dir: str) -> None:
    """의료 데이터를 벡터 DB에 인덱싱"""
    from korean_medical_ai.configs.settings import SystemConfig
    from korean_medical_ai.rag.document_loader import KoreanMedicalDocumentLoader
    from korean_medical_ai.rag.vector_store import VectorStoreManager

    config = SystemConfig()
    loader = KoreanMedicalDocumentLoader()
    store = VectorStoreManager(config)
    store.initialize_store()

    logger.info(f"데이터 인덱싱 시작: {data_dir}")

    # 디렉토리별로 소스 타입 매핑
    source_type_map = {
        "guidelines": "guideline",
        "papers": "paper",
        "laws": "law",
        "drug_info": "drug_info",
    }

    from pathlib import Path

    data_path = Path(data_dir)

    if data_path.is_dir():
        # 하위 디렉토리가 있으면 소스 타입별로 처리
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]
        if subdirs:
            for subdir in subdirs:
                source_type = source_type_map.get(subdir.name, "guideline")
                docs = loader.load_directory(str(subdir))
                if docs:
                    count = store.add_documents(docs, source_type=source_type)
                    logger.info(f"  {subdir.name}: {count}개 청크 인덱싱됨")
        else:
            # 단일 디렉토리
            docs = loader.load_directory(data_dir)
            if docs:
                count = store.add_documents(docs)
                logger.info(f"  {count}개 청크 인덱싱됨")
    else:
        # 단일 파일
        docs = loader.load_file(data_dir)
        if docs:
            count = store.add_documents(docs)
            logger.info(f"  {count}개 청크 인덱싱됨")

    store.save()
    logger.info(f"인덱싱 완료. 총 문서 수: {store.document_count}")


def index_huggingface_dataset(dataset_id: str, split: str = "train") -> None:
    """HuggingFace 데이터셋을 벡터 DB에 인덱싱"""
    from korean_medical_ai.configs.settings import SystemConfig
    from korean_medical_ai.rag.document_loader import KoreanMedicalDocumentLoader
    from korean_medical_ai.rag.vector_store import VectorStoreManager

    config = SystemConfig()
    loader = KoreanMedicalDocumentLoader()
    store = VectorStoreManager(config)
    store.initialize_store()

    logger.info(f"HuggingFace 데이터셋 인덱싱: {dataset_id}")
    docs = loader.load_huggingface_dataset(dataset_id, split=split)

    if docs:
        count = store.add_documents(docs, source_type="dataset")
        logger.info(f"  {count}개 청크 인덱싱됨")

    store.save()
    logger.info(f"인덱싱 완료. 총 문서 수: {store.document_count}")


def main() -> None:
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="KorMedAI - 한국형 의료 AI 진단/치료 보조 시스템",
    )
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트 (기본: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트 (기본: 8000)")
    parser.add_argument("--reload", action="store_true", help="개발 모드 (자동 리로드)")
    parser.add_argument("--index-data", type=str, help="의료 데이터 인덱싱 경로")
    parser.add_argument("--index-hf", type=str, help="HuggingFace 데이터셋 인덱싱 (예: snuh/KorMedLawQA)")
    parser.add_argument("--hf-split", type=str, default="train", help="HuggingFace 데이터셋 스플릿")

    args = parser.parse_args()

    if args.index_data:
        index_data(args.index_data)
    elif args.index_hf:
        index_huggingface_dataset(args.index_hf, args.hf_split)
    else:
        run_server(args.host, args.port, args.reload)


if __name__ == "__main__":
    main()
