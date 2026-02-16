"""
Korean Medical AI Assistant (KorMedAI)
=====================================

한국형 의료 AI 진단/치료 보조 시스템
- Multi-Agent LangChain 기반 의료 진단 보조
- RAG 기반 한국 의료 가이드라인 검색
- 의료 영상 분석 (X-ray, CT 등)
- 개원가 의사를 위한 진단 및 치료계획 수립 보조

Models:
  - SNUH HARI-Q3-14B: 한국어 의료 LLM (서울대병원)
  - MedGemma-4B: 의료 비전-언어 모델 (Google)
  - Qwen3-8B: 한국어 범용 LLM (Apache 2.0)
"""

__version__ = "0.1.0"
__app_name__ = "KorMedAI"
