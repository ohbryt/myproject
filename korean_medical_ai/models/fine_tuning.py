"""
파인튜닝 파이프라인 (Fine-Tuning Pipeline)
==========================================
한국 의료 데이터를 사용하여 오픈소스 LLM을 파인튜닝합니다.

지원 기법:
  - QLoRA (4-bit 양자화 + LoRA): 메모리 효율적
  - LoRA: 표준 어댑터 기반 튜닝
  - Full Fine-tuning: 전체 파라미터 업데이트 (GPU 클러스터 필요)

파인튜닝 대상:
  - 한국어 의학 용어 / 약어 이해
  - 한국 EMR 스타일 (한영 혼용 패턴)
  - 한국 보험 수가 체계
  - 한국 진료지침 기반 추론

라이선스 호환 베이스 모델:
  - Qwen3-8B (Apache 2.0) ← 권장
  - Kakao Kanana 1.5 8B (Apache 2.0)
  - SOLAR 10.7B (Apache 2.0)
  - Llama 3.1 8B (Llama License)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from korean_medical_ai.configs.settings import BASE_DIR

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """파인튜닝 설정"""

    # 베이스 모델
    base_model_id: str = "Qwen/Qwen3-8B"  # Apache 2.0, 상용 가능
    output_dir: str = str(BASE_DIR / "korean_medical_ai" / "models" / "fine_tuned")

    # LoRA 설정
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # 학습 설정
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    fp16: bool = True

    # 데이터 설정
    train_data_path: str = ""
    eval_data_path: str = ""
    eval_ratio: float = 0.1  # 학습 데이터의 10%를 평가용으로 분리


@dataclass
class TrainingExample:
    """학습 데이터 예시 형식"""

    instruction: str  # 지시문 (의료 질문)
    input: str = ""   # 추가 입력 (환자 정보 등)
    output: str = ""  # 기대 답변


class KoreanMedicalDatasetBuilder:
    """
    한국 의료 파인튜닝 데이터셋 생성기

    다양한 소스에서 학습 데이터를 수집하고
    파인튜닝용 instruction-output 형식으로 변환합니다.
    """

    def __init__(self):
        self.examples: list[TrainingExample] = []

    def add_qa_pair(self, question: str, answer: str, context: str = "") -> None:
        """의료 Q&A 쌍 추가"""
        self.examples.append(TrainingExample(
            instruction=question,
            input=context,
            output=answer,
        ))

    def add_kmle_questions(self, dataset_path: str = "snuh/KorMedLawQA") -> int:
        """KMLE 기출문제 데이터 추가"""
        try:
            from datasets import load_dataset
            dataset = load_dataset(dataset_path, split="train")

            count = 0
            for row in dataset:
                question = row.get("question", "")
                answer = row.get("answer", "")
                if question and answer:
                    self.examples.append(TrainingExample(
                        instruction=f"다음 의학 문제에 답하세요:\n{question}",
                        output=answer,
                    ))
                    count += 1

            logger.info(f"KMLE 데이터 {count}건 추가됨")
            return count
        except Exception as e:
            logger.error(f"KMLE 데이터 로드 실패: {e}")
            return 0

    def add_clinical_scenarios(self, json_path: str) -> int:
        """임상 시나리오 데이터 추가"""
        path = Path(json_path)
        if not path.exists():
            logger.warning(f"파일 없음: {json_path}")
            return 0

        with open(path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)

        count = 0
        for scenario in scenarios:
            self.examples.append(TrainingExample(
                instruction=scenario.get("scenario", ""),
                input=scenario.get("patient_info", ""),
                output=scenario.get("expected_response", ""),
            ))
            count += 1

        logger.info(f"임상 시나리오 {count}건 추가됨")
        return count

    def to_alpaca_format(self) -> list[dict]:
        """Alpaca 형식으로 변환 (표준 instruction tuning 형식)"""
        return [
            {
                "instruction": ex.instruction,
                "input": ex.input,
                "output": ex.output,
            }
            for ex in self.examples
        ]

    def to_chat_format(self) -> list[dict]:
        """Chat 형식으로 변환 (ChatML 스타일)"""
        formatted = []
        for ex in self.examples:
            messages = [
                {"role": "system", "content": "당신은 한국의 의료 전문 AI 어시스턴트입니다."},
            ]
            user_content = ex.instruction
            if ex.input:
                user_content += f"\n\n{ex.input}"
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": ex.output})
            formatted.append({"messages": messages})
        return formatted

    def save(self, output_path: str, format: str = "alpaca") -> None:
        """데이터셋 저장"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "alpaca":
            data = self.to_alpaca_format()
        elif format == "chat":
            data = self.to_chat_format()
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"데이터셋 저장: {path} ({len(data)}건, 형식: {format})")


class FineTuner:
    """
    LLM 파인튜닝 실행기

    QLoRA를 사용하여 한국 의료 데이터로 오픈소스 LLM을 파인튜닝합니다.
    Mac Mini (Apple Silicon)에서는 MLX 기반 파인튜닝을 지원합니다.
    """

    def __init__(self, config: Optional[FineTuningConfig] = None):
        self.config = config or FineTuningConfig()

    def train_qlora(self, train_data_path: str, eval_data_path: Optional[str] = None) -> str:
        """
        QLoRA 파인튜닝 실행

        Args:
            train_data_path: 학습 데이터 경로 (JSON)
            eval_data_path: 평가 데이터 경로 (JSON, 선택적)

        Returns:
            파인튜닝된 모델 경로
        """
        try:
            import torch
            from datasets import load_dataset
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                TrainingArguments,
            )
            from trl import SFTTrainer
        except ImportError as e:
            logger.error(
                f"파인튜닝 의존성 미설치: {e}\n"
                "pip install peft trl bitsandbytes 실행 필요"
            )
            raise

        logger.info(f"QLoRA 파인튜닝 시작: {self.config.base_model_id}")

        # 1. 4-bit 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # 2. 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_id,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 3. LoRA 설정
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # 4. 데이터셋 로드
        dataset = load_dataset("json", data_files=train_data_path, split="train")
        eval_dataset = None
        if eval_data_path:
            eval_dataset = load_dataset("json", data_files=eval_data_path, split="train")

        # 5. 학습 설정
        output_dir = self.config.output_dir
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            fp16=self.config.fp16,
            logging_steps=10,
            save_steps=100,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
            save_total_limit=3,
            report_to="none",
        )

        # 6. 트레이너 설정 및 실행
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=self.config.max_seq_length,
        )

        logger.info("학습 시작...")
        trainer.train()

        # 7. 모델 저장
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info(f"파인튜닝 완료. 모델 저장 위치: {output_dir}")
        return output_dir

    def train_mlx(self, train_data_path: str) -> str:
        """
        MLX 기반 파인튜닝 (Apple Silicon Mac Mini 전용)

        Args:
            train_data_path: 학습 데이터 경로

        Returns:
            파인튜닝된 어댑터 경로
        """
        try:
            from mlx_lm import fine_tune
        except ImportError:
            logger.error("mlx-lm 미설치. pip install mlx-lm 실행 필요")
            raise

        output_dir = self.config.output_dir + "_mlx"

        logger.info(f"MLX 파인튜닝 시작: {self.config.base_model_id}")
        logger.info("Apple Silicon GPU 가속 사용")

        # MLX fine-tuning은 별도의 CLI 도구 사용
        import subprocess

        cmd = [
            "mlx_lm.lora",
            "--model", self.config.base_model_id,
            "--data", train_data_path,
            "--train",
            "--batch-size", str(self.config.batch_size),
            "--num-layers", "16",
            "--iters", str(self.config.num_epochs * 1000),
            "--adapter-path", output_dir,
        ]

        logger.info(f"실행 명령: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"MLX 파인튜닝 실패: {result.stderr}")
            raise RuntimeError(f"MLX fine-tuning failed: {result.stderr}")

        logger.info(f"MLX 파인튜닝 완료. 어댑터 경로: {output_dir}")
        return output_dir
