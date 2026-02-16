"""
의료 영상 분석 모듈 (Medical Image Analysis)
=============================================
MedGemma-4B 및 한국형 의료 비전 모델을 사용하여
의료 영상(X-ray, CT 등)을 분석합니다.

지원 모델:
  - Google MedGemma-4B-IT: 범용 의료 영상 (CXR, Derm, Ophtho, Histo)
  - SNUH HARI Vision (향후): 한국 환자 데이터 특화
  - CXR-LLaVA (연구용): 흉부 X-ray 전문

지원 영상:
  - 흉부 X-ray (Chest X-ray)
  - 피부 병변 (Dermatology)
  - 안저 사진 (Ophthalmology)
  - 병리 슬라이드 (Histopathology)
  - CT/MRI (향후 지원)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from korean_medical_ai.configs.settings import SystemConfig
from korean_medical_ai.models.model_manager import ModelManager

logger = logging.getLogger(__name__)


class ImageModality(str, Enum):
    """의료 영상 모달리티"""

    CHEST_XRAY = "chest_xray"
    DERMATOLOGY = "dermatology"
    OPHTHALMOLOGY = "ophthalmology"
    HISTOPATHOLOGY = "histopathology"
    CT = "ct"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    OTHER = "other"


@dataclass
class ImageAnalysisResult:
    """영상 분석 결과"""

    image_path: str = ""
    modality: str = ""
    findings: str = ""
    impression: str = ""
    recommendations: list[str] = field(default_factory=list)
    abnormalities_detected: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    model_used: str = ""
    raw_response: str = ""


# ──────────────────────────────────────
# 모달리티별 분석 프롬프트 (한국어)
# ──────────────────────────────────────

ANALYSIS_PROMPTS = {
    ImageModality.CHEST_XRAY: """
당신은 영상의학과 전문의입니다. 이 흉부 X-ray 영상을 분석해 주세요.

## 분석 항목
1. **기술적 평가**: 촬영 자세(AP/PA), 노출, 회전
2. **폐야(Lung fields)**: 투과도, 침윤, 결절, 종괴, 무기폐
3. **심장/종격동**: 심비대(CTR), 종격동 확장, 림프절
4. **횡격막**: 위치, 각도, CP angle 둔화
5. **골격**: 늑골 골절, 척추 이상
6. **연부조직**: 피하기종, 기타 이상

## 보고서 형식
Findings: (소견)
Impression: (판독 소견)
Recommendation: (추천 사항)
""",

    ImageModality.DERMATOLOGY: """
당신은 피부과 전문의입니다. 이 피부 병변 사진을 분석해 주세요.

## 분석 항목
1. **병변 기술**: 위치, 크기, 형태, 색상, 경계
2. **ABCDE 기준**: 비대칭(A), 경계(B), 색상(C), 직경(D), 변화(E)
3. **감별진단**: 가능한 진단 목록 (확률순)
4. **권장 검사**: 조직검사 필요 여부

## 보고서 형식
Description: (병변 기술)
Differential: (감별진단)
Recommendation: (추천 사항)
""",

    ImageModality.OPHTHALMOLOGY: """
당신은 안과 전문의입니다. 이 안저 사진을 분석해 주세요.

## 분석 항목
1. **시신경유두**: 크기, CDR(cup-to-disc ratio), 색상, 부종
2. **혈관**: 동정맥비, 교차 징후, 출혈, 신생혈관
3. **황반부**: 부종, 삼출물, 드루젠, 색소 변화
4. **망막**: 출혈, 삼출물, 박리, 열공
5. **기타**: 유리체 혼탁 등

## 보고서 형식
Findings: (소견)
Impression: (판독 소견)
Recommendation: (추천 사항)
""",

    ImageModality.HISTOPATHOLOGY: """
당신은 병리과 전문의입니다. 이 조직 슬라이드를 분석해 주세요.

## 분석 항목
1. **저배율**: 전체 구조, 배열 패턴
2. **고배율**: 세포 형태, 핵/세포질 비, 유사분열
3. **특수 소견**: 침윤 패턴, 괴사, 혈관 침범
4. **면역조직화학**: 해당 시 표지자 결과

## 보고서 형식
Microscopic Description: (현미경 소견)
Diagnosis: (병리진단)
Comment: (참고사항)
""",
}

# 기본 프롬프트 (모달리티 미지정 시)
DEFAULT_ANALYSIS_PROMPT = """
당신은 의료 영상 분석 전문가입니다. 이 의료 영상을 분석해 주세요.

## 분석 요청
1. 영상의 종류와 촬영 부위를 식별하세요.
2. 정상/비정상 소견을 구분하세요.
3. 이상 소견이 있다면 상세히 기술하세요.
4. 추가 검사나 조치가 필요한 경우 권고해 주세요.

## 보고서 형식
Modality: (영상 종류)
Findings: (소견)
Impression: (판독 소견)
Recommendation: (추천 사항)
"""


class MedicalImageAnalyzer:
    """
    의료 영상 분석기

    MedGemma-4B를 사용하여 다양한 의료 영상을 분석합니다.
    한국 의료 환경에 맞는 보고서 형식으로 결과를 출력합니다.
    """

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        config: Optional[SystemConfig] = None,
    ):
        self.config = config or SystemConfig()
        self.model_manager = model_manager or ModelManager(self.config)

    def analyze(
        self,
        image_path: str,
        modality: Optional[ImageModality] = None,
        clinical_context: str = "",
        custom_prompt: str = "",
    ) -> ImageAnalysisResult:
        """
        의료 영상 분석

        Args:
            image_path: 영상 파일 경로
            modality: 영상 모달리티 (None이면 자동 감지)
            clinical_context: 임상 정보 (증상, 병력 등)
            custom_prompt: 커스텀 분석 프롬프트

        Returns:
            ImageAnalysisResult 객체
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {image_path}")

        # 프롬프트 구성
        if custom_prompt:
            prompt = custom_prompt
        elif modality and modality in ANALYSIS_PROMPTS:
            prompt = ANALYSIS_PROMPTS[modality]
        else:
            prompt = DEFAULT_ANALYSIS_PROMPT

        # 임상 정보 추가
        if clinical_context:
            prompt += f"\n\n## 임상 정보\n{clinical_context}"

        prompt += "\n\n위 영상을 분석하고 한국어로 보고서를 작성해 주세요."

        logger.info(f"영상 분석 시작: {path.name} (모달리티: {modality or 'auto'})")

        # MedGemma 비전 모델로 분석
        try:
            raw_response = self.model_manager.analyze_image(image_path, prompt)
        except Exception as e:
            logger.error(f"영상 분석 실패: {e}")
            raw_response = f"영상 분석 중 오류 발생: {e}"

        result = ImageAnalysisResult(
            image_path=image_path,
            modality=modality.value if modality else "auto",
            raw_response=raw_response,
            model_used=self.config.models["vision"].name,
        )

        logger.info(f"영상 분석 완료: {path.name}")
        return result

    def batch_analyze(
        self,
        image_paths: list[str],
        modality: Optional[ImageModality] = None,
        clinical_context: str = "",
    ) -> list[ImageAnalysisResult]:
        """
        다수 영상 일괄 분석

        Args:
            image_paths: 영상 파일 경로 목록
            modality: 영상 모달리티
            clinical_context: 임상 정보

        Returns:
            ImageAnalysisResult 목록
        """
        results = []
        for path in image_paths:
            try:
                result = self.analyze(path, modality, clinical_context)
                results.append(result)
            except Exception as e:
                logger.error(f"영상 분석 실패 ({path}): {e}")
                results.append(ImageAnalysisResult(
                    image_path=path,
                    raw_response=f"분석 실패: {e}",
                ))
        return results

    def compare_images(
        self,
        current_image: str,
        previous_image: str,
        modality: Optional[ImageModality] = None,
        clinical_context: str = "",
    ) -> ImageAnalysisResult:
        """
        영상 비교 분석 (경과 관찰용)

        Args:
            current_image: 현재 영상 경로
            previous_image: 이전 영상 경로
            modality: 영상 모달리티
            clinical_context: 임상 정보

        Returns:
            비교 분석 결과
        """
        prompt = f"""
당신은 영상의학과 전문의입니다.
이전 영상과 현재 영상을 비교 분석해 주세요.

## 비교 분석 항목
1. 이전 영상 대비 변화 사항
2. 호전/악화/불변 판단
3. 새로 발생한 소견
4. 소실된 소견
5. 추적 관찰 권고 사항

## 임상 정보
{clinical_context if clinical_context else "정보 없음"}
"""
        # 현재는 단일 이미지만 분석 (향후 멀티이미지 지원 시 확장)
        return self.analyze(
            image_path=current_image,
            modality=modality,
            clinical_context=f"이전 영상 경로: {previous_image}\n{clinical_context}",
            custom_prompt=prompt,
        )
