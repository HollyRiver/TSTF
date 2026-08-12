# TSTF — 시계열 트랜스포머 전이학습 베이스라인 실험 (PyTorch)

&nbsp;IEEE Access 논문 **"Transfer Learning Based on N-BEATS in Forecasting Univariate Time Series"**
(Lee, Lee, Kim, 2026, *IEEE Access* **14**, 45191–45212, [링크](https://ieeexplore.ieee.org/abstract/document/11443264))의
**PatchTST 계열 베이스라인 실험**을 기록한 리포지토리

&nbsp;저자가 아닌 **Acknowledgement**로 기여했으며, `torch/` 디렉토리는 100% 단독 작업입니다. (`main/`은 원저자 측 레거시 Keras 코드 — 참고용)

## 기여

* 레거시 Keras 실험을 **PyTorch + HuggingFace PatchTST**로 재구현
* M4 소스 데이터 부트스트랩 사전학습 → head 교체(Transformer/LSTM/MLP) 파인튜닝 전이학습 파이프라인
* 손실함수 5종(MSE/MAE/MAPE/sMAPE/MASE) × 100개 모델 = **태스크당 500개 모델**의 median·지수 가중 앙상블
* 21개 실험 태스크를 multiprocessing 큐 + GPU 라운드로빈으로 자동화, 결과 존재 시 스킵(재시작 안전) —
  **L40S 2-GPU 활용률 98% 달성**
* 원본 코드의 결함(MASE 손실 오구현, MAPE 0값 발산, 분할 분포 불일치)을 발견·문서화 (`torch/ISSUE.ipynb`)
* 베이스라인 실험 결과는 논문 최종본에 수록

## 데이터

* 소스: M4 (입력 168스텝 → 출력 24스텝)
* 타겟: AIR·coin·ELE·MET·SOL·TEM·WID 7종 (학습 윈도우 252~724개의 소규모 시계열)

## 사용법

&nbsp;데이터는 `torch/`와 같은 계층(즉, 리포 루트의 `data/` 또는 동일 위치)에 배치합니다. 7개의 타겟 데이터셋 파일(`**_**_7.csv` 형식)과 2개의 소스 데이터셋 파일(`M4_**.csv`)이 필요합니다.

```bash
cd torch
bash run_all.sh   # 전체 실험 매트릭스 자동 실행 (GPU 라운드로빈)
# 또는 직접 실행:
nohup python run_all.py --num_gpus=2 --num_process=8 &
```

&nbsp;멀티프로세싱은 태스크(데이터셋 × 어댑터 알고리즘 = 21개) 단위로 구동합니다. 각 태스크 내 500개 모델은 순차 학습이므로, 추후 최적화 시 태스크 내 병렬화를 검토할 수 있습니다.

* 결과 집계와 앙상블 성능표는 `torch/result_agg.ipynb` 참고.

## 구조

```
├── data/     # M4 소스 + 7개 타겟 데이터셋
├── main/     # [레거시] 원저자 측 TF/Keras 코드 (참고용)
└── torch/    # [본인 작업] PyTorch 재구현
    ├── run_all.py        # 멀티 GPU 실험 오케스트레이션
    ├── pretraining.py    # M4 사전학습
    ├── trTFTF.py / trTFLSTM.py / trTFMLP.py   # 어댑터 헤드 전이학습
    ├── trPatchTST.py / Scratch*.py            # 비교군
    ├── result_agg.ipynb  # 앙상블 집계
    └── ISSUE.ipynb       # 원본 코드 오류 분석 기록
```
