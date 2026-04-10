# 환경 설정 가이드

## 현재 상태
- ✅ PyTorch 설치됨 (python3.13 + 3.14)
- ✅ MPS 백엔드 지원
- ✅ powermetrics 작동 확인 (CPU/GPU/ANE 전력 측정)
- ❌ transformers 미설치
- ❌ diffusers 미설치
- ❌ audiocraft 미설치
- ❌ accelerate 미설치

## 설치 명령어 (python3.13 기준)
```bash
# 핵심 패키지
pip3.13 install --break-system-packages transformers accelerate
pip3.13 install --break-system-packages diffusers
pip3.13 install --break-system-packages audiocraft  # MusicGen
pip3.13 install --break-system-packages codecarbon  # 에너지 측정 보조
pip3.13 install --break-system-packages sentencepiece protobuf  # LLaMA tokenizer

# 선택 (Video)
pip3.13 install --break-system-packages imageio[ffmpeg]
```

## 모델 다운로드 (HuggingFace)
실행 시 자동 다운로드되지만, 미리 받아두면 좋음:

### Text (~15GB)
- meta-llama/Llama-3.2-3B-Instruct
- microsoft/Phi-3-mini-4k-instruct

### Image (~12GB)  
- stabilityai/stable-diffusion-xl-base-1.0
- black-forest-labs/FLUX.1-schnell

### Video (~10GB)
- THUDM/CogVideoX-2b

### Music (~7GB)
- facebook/musicgen-small (300M)
- facebook/musicgen-medium (1.5B)
- facebook/musicgen-large (3.3B)

## 디스크 공간 필요
모델 총: ~45GB
현재 여유: ~130GB → 충분!

## Mac mini 스펙
- Apple M4 Pro
- 24GB Unified Memory
- macOS Tahoe
