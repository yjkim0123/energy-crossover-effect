# Project Energy — PLAN

## Overview
- **주제**: The Energy Crossover Effect — 4-modal generative AI의 edge vs GPU 에너지 비교
- **타겟**: Nature Communications (IF 16.6, APC $7,350)
- **저자**: Yongjun Kim (Ajou), Seoksoo Kim (Hannam), Yonghyun Kim (Georgia Tech)
- **상태**: 📝 거의 완성 — 용준이 최종 리뷰 후 제출

## 핵심 발견: Energy Crossover Effect
- Diffusion (image/video): edge→GPU로 효율 역전 (Image 493×493, Video 7 frames)
- Autoregressive (text/music): edge가 항상 6-14× 효율적, 역전 없음
- Scaling: edge β>1 (superlinear), GPU β<1 (sublinear)

## 실험 데이터
- **총 1,260 runs** (Mac 600 + A100 660)
- Mac mini M4 Pro (22W TDP, powermetrics)
- NVIDIA A100-SXM4-40GB (400W TDP, pynvml via Colab)
- 5 models: Phi-3-mini, SD-v1.5, AnimateDiff, MusicGen-small, MusicGen-medium
- 22 configurations, 30 repeats each
- Results: `results/exp{1-4}_*.json`, `results/exp{1-4}_*_A100.json`

## 논문 상태
- **파일**: `paper/main.tex` → 15 pages PDF
- **단어 수**: ~4,500 words (Nature Comms 5,000 제한)
- **Abstract**: 176 words (200 제한)
- **레퍼런스**: 30개
- **Figure**: 6개 (`figures/fig{1-6}_*.pdf`)
- **Cover letter**: `paper/cover_letter.tex/pdf`
- **T4 Colab 노트북**: `scripts/colab_t4_experiment.ipynb` (미실행)

## 체크리스트 결과
- 🔴 CRITICAL: 0개 (전부 수정)
- 🟡 WARNING: 0개 (전부 대응)
- 적대적 리뷰 17개 항목 전부 논문에 반영

## 적대적 리뷰 대응 요약
1. 측정 비대칭 → 40% sensitivity analysis, idle offset 대안 인정
2. Colab 비통제 → CV 낮음 명시, bare-metal 필요 인정
3. 3점 scaling → "empirical scaling relationships"로 완화
4. CO₂ 과장 → batching/모델크기/재생에너지 고려, 현실적 범위 제시
5. 1 model/modality → DiT, 7B+ 필요 인정
6. 품질 미측정 → FID/perplexity 필요 인정
7. Text CV 19% → EOS 변동 원인 설명
8. PyTorch 버전 불일치 → Limitations에 추가
9. Crossover 불확실성 → Bootstrap 95% CI 추가
10. β artefact → idle offset 대안 + sensitivity
11. 단일 프롬프트 → Limitations에 추가
12. Author 이니셜 → S.K., Y.H.K. 수정
13-17. Minor → 전부 대응

## TODO
- [ ] 용준이 최종 리뷰
- [ ] T4 Colab 실험 (3번째 하드웨어 포인트, 선택)
- [ ] GitHub 리포 정리 (data availability용)
- [ ] 제출
