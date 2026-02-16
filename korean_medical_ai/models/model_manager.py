"""
모델 관리자 (Model Manager)
===========================
3개 LLM 모델의 로딩, 관리, 인퍼런스를 담당합니다.

지원 모델:
  1. SNUH HARI-Q3-14B  → 한국어 의료 진단 (vLLM)
  2. Qwen3-8B          → 문헌 검토 / 범용 한국어 (Ollama)
  3. MedGemma-4B       → 의료 영상 분석 (Transformers)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_community.llms import Ollama, VLLMOpenAI
from langchain_core.language_models import BaseLLM
from langchain_core.messages import HumanMessage

from korean_medical_ai.configs.settings import ModelConfig, SystemConfig

logger = logging.getLogger(__name__)


class BaseModelWrapper(ABC):
    """모델 래퍼 기본 클래스"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._model: Optional[Any] = None

    @abstractmethod
    def load(self) -> None:
        """모델 로드"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        """메모리에서 모델 해제"""
        self._model = None
        logger.info(f"모델 해제됨: {self.config.name}")


class OllamaModelWrapper(BaseModelWrapper):
    """Ollama 백엔드 모델 래퍼 (Qwen3-8B 등)"""

    def __init__(self, config: ModelConfig, base_url: str = "http://localhost:11434"):
        super().__init__(config)
        self.base_url = base_url

    def load(self) -> None:
        self._model = Ollama(
            model=self.config.model_id,
            base_url=self.base_url,
            temperature=self.config.temperature,
            num_predict=self.config.max_tokens,
            num_ctx=self.config.context_length,
        )
        logger.info(f"Ollama 모델 로드됨: {self.config.name}")

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_loaded:
            self.load()
        return self._model.invoke(prompt)

    def as_langchain_llm(self) -> BaseLLM:
        if not self.is_loaded:
            self.load()
        return self._model


class VLLMModelWrapper(BaseModelWrapper):
    """vLLM 백엔드 모델 래퍼 (HARI-Q3-14B 등)"""

    def __init__(self, config: ModelConfig, base_url: str = "http://localhost:8080"):
        super().__init__(config)
        self.base_url = base_url

    def load(self) -> None:
        self._model = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=f"{self.base_url}/v1",
            model_name=self.config.model_id,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        logger.info(f"vLLM 모델 로드됨: {self.config.name}")

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_loaded:
            self.load()
        return self._model.invoke(prompt)

    def as_langchain_llm(self) -> BaseLLM:
        if not self.is_loaded:
            self.load()
        return self._model


class TransformersVisionWrapper(BaseModelWrapper):
    """Transformers 백엔드 비전 모델 래퍼 (MedGemma 등)"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._processor = None

    def load(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info(f"비전 모델 디바이스: {device}")

            self._processor = AutoProcessor.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
                device_map=device,
                trust_remote_code=True,
            )
            logger.info(f"비전 모델 로드됨: {self.config.name} on {device}")
        except ImportError:
            logger.error("transformers/torch 미설치. pip install transformers torch 실행 필요")
            raise

    def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 전용 생성"""
        if not self.is_loaded:
            self.load()
        inputs = self._processor(text=prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        outputs = self._model.generate(**inputs, max_new_tokens=self.config.max_tokens)
        return self._processor.decode(outputs[0], skip_special_tokens=True)

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """의료 영상 분석"""
        if not self.is_loaded:
            self.load()

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return self._processor.decode(outputs[0], skip_special_tokens=True)


class ModelManager:
    """
    전체 모델 관리자

    3개 LLM을 통합 관리하며, LangChain 호환 인터페이스를 제공합니다.
    Mac Mini (Apple Silicon) 환경에서 최적화되어 동작합니다.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self._wrappers: dict[str, BaseModelWrapper] = {}
        self._initialize_wrappers()

    def _initialize_wrappers(self) -> None:
        """모델 래퍼 초기화 (아직 모델 로드는 하지 않음)"""
        for role, model_config in self.config.models.items():
            if model_config.backend == "ollama":
                self._wrappers[role] = OllamaModelWrapper(
                    model_config, self.config.ollama_base_url
                )
            elif model_config.backend == "vllm":
                self._wrappers[role] = VLLMModelWrapper(
                    model_config, self.config.vllm_base_url
                )
            elif model_config.backend == "transformers":
                self._wrappers[role] = TransformersVisionWrapper(model_config)
            else:
                logger.warning(f"알 수 없는 백엔드: {model_config.backend}")

    def load_model(self, role: str) -> None:
        """특정 역할의 모델 로드"""
        if role in self._wrappers:
            self._wrappers[role].load()
        else:
            raise ValueError(f"알 수 없는 역할: {role}. 사용 가능: {list(self._wrappers.keys())}")

    def load_all(self) -> None:
        """모든 모델 로드"""
        for role in self._wrappers:
            try:
                self.load_model(role)
            except Exception as e:
                logger.error(f"모델 로드 실패 ({role}): {e}")

    def generate(self, role: str, prompt: str, **kwargs) -> str:
        """특정 역할의 모델로 텍스트 생성"""
        if role not in self._wrappers:
            raise ValueError(f"알 수 없는 역할: {role}")
        return self._wrappers[role].generate(prompt, **kwargs)

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """의료 영상 분석 (비전 모델 사용)"""
        vision_wrapper = self._wrappers.get("vision")
        if not isinstance(vision_wrapper, TransformersVisionWrapper):
            raise RuntimeError("비전 모델이 설정되지 않았습니다")
        return vision_wrapper.analyze_image(image_path, prompt)

    def get_langchain_llm(self, role: str) -> BaseLLM:
        """LangChain 호환 LLM 객체 반환"""
        wrapper = self._wrappers.get(role)
        if wrapper is None:
            raise ValueError(f"알 수 없는 역할: {role}")
        if isinstance(wrapper, (OllamaModelWrapper, VLLMModelWrapper)):
            return wrapper.as_langchain_llm()
        raise TypeError(f"{role} 모델은 LangChain LLM 인터페이스를 지원하지 않습니다")

    @property
    def loaded_models(self) -> list[str]:
        """로드된 모델 역할 목록"""
        return [role for role, w in self._wrappers.items() if w.is_loaded]

    @property
    def available_roles(self) -> list[str]:
        """사용 가능한 역할 목록"""
        return list(self._wrappers.keys())
