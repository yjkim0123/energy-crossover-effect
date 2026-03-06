# Nature Communications 제출 계획

## 논문 컨셉 (리프레이밍)
**제목안**: "The Energy-Quality Pareto Frontier of Generative AI: When Edge Hardware Matches Cloud Efficiency"

**핵심 노벨티**: 단순 에너지 비교가 아닌, **에너지 대비 출력 품질**의 Pareto frontier를 최초로 제시
- "같은 품질을 달성하는 데 하드웨어별로 에너지가 얼마나 다른가?"
- "품질을 10% 낮추면 에너지를 몇 배 절약할 수 있는가?"

## Phase 1: 하드웨어 확장 (2-3일)
### 현재 보유
- [x] Mac mini M4 Pro (edge)
- [x] NVIDIA A100 (datacenter)

### 추가 필요
- [ ] NVIDIA H100 (Colab) — 최신 datacenter
- [ ] NVIDIA T4 (Colab) — mid-range (Text/Image/Video만, Music 제외)

## Phase 2: 모델 확장 (3-4일)
### 현재 모델
- Text: Phi-3 (3.8B), Mistral-7B (A100 only)
- Image: SD v1.5 (0.9B), SDXL (3.5B)
- Video: AnimateDiff
- Music: MusicGen-small, MusicGen-medium

### 추가 모델 (Colab에서)
- Text: LLaMA-3-8B-Instruct, Gemma-2-2B
- Image: SD Turbo (distilled, fewer steps)
- Video: (AnimateDiff으로 충분)
- Music: (MusicGen으로 충분)

### 해상도/토큰 sweep 세밀화
- Image: 128², 192², 256², 320², 384², 448², 512², 640², 768², 1024² (10 points)
- Text: 32, 64, 128, 192, 256, 384, 512 (7 points)
- Video: 4, 6, 8, 10, 12, 16 frames (6 points)

## Phase 3: 품질 측정 (핵심 노벨티!) (3-4일)
### Image Quality
- **CLIP Score**: prompt-image alignment (높을수록 좋음)
- **FID**: Fréchet Inception Distance (낮을수록 좋음)
- Steps별 품질 변화: 5, 10, 15, 20, 25, 30, 50 steps

### Text Quality  
- **Perplexity**: 생성 텍스트의 perplexity
- **BLEU/ROUGE**: reference 대비 (선택)
- Token length별 품질 변화

### Video Quality
- **FVD**: Fréchet Video Distance
- Frame consistency (CLIP frame-to-frame similarity)

### Music Quality
- **FAD**: Fréchet Audio Distance
- Spectral quality metrics

### Energy-Quality Pareto Analysis
- 각 (하드웨어, 모델, 설정) 조합에 대해 (Energy, Quality) 점 찍기
- Pareto frontier 추출
- "어떤 설정이 에너지-품질 최적인가?"

## Phase 4: 글로벌 임팩트 분석 (2-3일)
- IEA AI 전력 소비 데이터 인용
- 우리 측정 기반 추론당 에너지 × 글로벌 추론 횟수 추정
- "edge 최적화 시나리오" vs "cloud only 시나리오" 비교
- CO₂ 환산 (탄소 집약도 by 국가)

## Phase 5: 논문 작성 (3-4일)
- Nature Comms format (5,000 words, 8 figures)
- Methods 상세화
- Supplementary materials
- Code/data GitHub repo

## 예상 Figure 구성 (8개)
1. **Overview**: 실험 설계 개요도 (4 modalities × 4 hardware × multiple models)
2. **Energy Scaling**: 모달리티별 에너지 스케일링 (모든 하드웨어)
3. **Crossover Analysis**: 하드웨어별 crossover 포인트 (bootstrap CI)
4. **Energy-Quality Pareto**: ⭐ 핵심 figure — Pareto frontier
5. **Quality vs Steps**: 품질 포화점 분석 (어디서 품질이 수렴하는지)
6. **Model Size Effect**: 모델 크기별 에너지 효율
7. **Batched Inference**: 배치 효율 + per-query 비용
8. **Global Impact**: CO₂ 시나리오 분석

## 측정 방법 통일 방안
- GPU: pynvml (GPU-only power) — 모든 GPU에 동일 적용
- Mac: powermetrics (system power) — 유일한 edge device
- **Sensitivity analysis**: GPU idle power 보정 (40% offset 시나리오)
- 논문에서 명시적으로 비대칭 인정 + 보정 결과 함께 제시

## 타임라인
- Week 1: H100/T4 실험 + 모델 확장
- Week 2: 품질 측정 실험
- Week 3: 분석 + figure 생성 + 글로벌 임팩트
- Week 4: 논문 작성 + 내부 리뷰
- 제출 목표: 3월 중순

## 예상 총 실험 규모
- 4 hardware × 6+ models × 7-10 output sizes × 30 repeats = ~5,000+ runs (energy)
- + 품질 측정 ~2,000 runs
- **총 ~7,000 runs**
