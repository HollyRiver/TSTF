# TSTF
time series transformer

논문 링크: https://ieeexplore.ieee.org/abstract/document/11443264

> 저자가 아닌 Acknowledgement 항목에 기여도 기재됨

## 사용 방법

* `main` 폴더는 레거시입니다. 제가 짠 코드는 `torch` 폴더에만 위치합니다.
* `torch` 폴더의 `run_all.sh` 파일을 배시로 실행하시면 동작합니다. 또는 그냥 터미널에 `nohup python run_all.py --num_gpus=2 --num_process=8 &`를 입력하세요. (L40S 기준 Utilize 98% 달성)
* 데이터는 `torch` 폴더가 있는 동일한 계층에 넣어주세요. 7개의 타겟 데이터셋 폴더(`**_**_7.csv`)와 2개의 소스 데이터셋 파일(`M4_**.csv`)이 필요합니다.
* 원본 코드를 제대로 수정하는 것이 도저히 엄두가 안나서 멀티프로세싱은 태스크 단위로 수행했습니다. 즉, 7개의 데이터셋 * 3개의 어뎁터 알고리즘 = 21개의 태스크 수행 자체를 멀티프로세싱으로 구동합니다. 각 태스크에서 5 * model_num개의 모델은 독립적으로 학습되므로 해당 레벨에서의 병렬화가 가장 바람직합니다. 추후 코드 최적화 시 참고하세요.

## 기본적인 설명

* IEEE 투고 논문인 **Transfer Learning Based on N-BEATS in Forecasting Univariate Time Series**에서 메인 모델과 비교할 Baseline 모델을 구축하기 위한 리포지토리입니다.
* 시계열 전이학습 모델의 퍼포먼스를 중심으로 구성되며, 사전학습된 백본 모델에서 헤드를 제거한 다음 동일한 트랜스포머/LSTM/MLP를 부착하여 파인튜닝. 해당 과정을 MSE/MAE/MAPE/SMAPE/MASE의 손실함수로 설정한 각 100개의 모델, 총 500개의 모델을 사전학습 -> 파인튜닝하여 얻은 결과를 앙상블하여 최종 추론을 진행합니다.
* 베이스라인으로는 PatchTST 모델을 사용합니다. [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730)
