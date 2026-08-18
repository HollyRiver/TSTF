# 시계열 트랜스포머 전이학습 베이스라인 실험 (trTSTF)

&nbsp;프론티어 시계열 베이스라인 모델(PatchTST) 기반 전이학습 성능 실험 연구의 코드베이스. 베이스라인으로써 N-Beats 기반 전이학습 모델과의 차이를 식별하기 위함.

&nbsp;저자가 아닌 **Acknowledgement**로 기여했으며, 실험에 사용된 `torch/` 디렉토리의 모든 코드는 직접 구성함을 알림. (`main/`은 원저자 측 레거시 Keras 코드 — 아카이브용)

> Publication (베이스라인 실험 기여 ― Acknowledgement)
>
> Minwoo Lee, Youngmi Lee, Gwangsu Kim. (2026). Transfer Learning Based on N-BEATS in Forecasting Univariate Time Series. IEEE Access, 14, 45191–45212. [Link](https://ieeexplore.ieee.org/abstract/document/11443264)

## 개요

* 레거시 Keras 실험을 **PyTorch + HuggingFace PatchTST**로 재구현
* M4 소스 데이터 부트스트랩 사전학습 → head 교체(Transformer/LSTM/MLP) 파인튜닝 전이학습 파이프라인
* 손실함수 5종(MSE/MAE/MAPE/sMAPE/MASE) × 100개 모델 = 태스크당 500개 모델의 median·지수 가중 앙상블
* 태스크 단위로 multiprocessing을 통해 21개의 실험 프로세스를 설계
* 원본 코드의 결함(MASE 손실 오구현, MAPE 0값 발산, 분할 분포 불일치)을 발견·문서화 (`torch/ISSUE.ipynb`)
* 베이스라인 실험 결과는 논문 최종본에 수록

## 데이터

* 소스: M4 (입력 168스텝 → 출력 24스텝)
* 타겟: AIR·coin·ELE·MET·SOL·TEM·WID 7종 (학습 윈도우 252~724개의 소규모 시계열)

## 사용법

&nbsp;데이터를 `torch/`와 같은 계층(즉, 리포 루트의 `data/` 또는 동일 위치)에 배치. 7개의 타겟 데이터셋 파일(`**_**_7.csv` 형식)과 2개의 소스 데이터셋 파일(`M4_**.csv`)이 요구됨.

```bash
cd torch
bash run_all.sh   # 전체 실험 매트릭스 자동 실행 (GPU 라운드로빈)
# 또는 직접 실행:
nohup python run_all.py --num_gpus=2 --num_process=8 &
```

&nbsp;멀티프로세싱은 태스크(데이터셋 × 어댑터 알고리즘 = 21개) 단위로 구동. 각 태스크 내 500개 모델은 순차 학습이므로, 태스크 내 모델 단위 병렬화를 통해 추가적인 최적화가 가능할 것으로 보임.

* 결과 집계와 앙상블 성능표는 `torch/result_agg.ipynb` 참고

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
