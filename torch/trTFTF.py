import os, time, json, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## 1. 일단 사용된 코드가 GPU를 이용하지 못하고 있음을 확인. GPU 사용 코드가 있음에도 안됨, 버전 문제 같음
## 2. 백본 모델로는 PatchTST를 사용하면 좋을 것 같음. 논문에서도 전이학습, fine-tuning을 할 경우 잘 작동한다고 나와있음
## 3. 코드 상 freezing 말고 fine-tuning을 사용하는 것 같음. 그리고 현재 모델 그대로 두고 헤드만 부착하는 형태인 것 같음.
## 4. 지금 트랜스포머를 백본으로 한 세 가지 경우가 모두 문제가 있다는 건가? MLP, LSTM, TF 헤드 전부 다? -> 그런 것 같네.
## 5. 그리고 세 가지 코드 전부 구동할 때 지금 Pre-training model을 생성하는 거임? 한 번만 고성능 모델을 저장해두고 그걸 이용하는 게 아니라?
## 6. 데이터셋 로드는 어떻게 한 거임? target_X와 target_X_val을 나눠서 모델에 넣는 거임? 그런데 왜 target_X에 validation data가 포함되는 거임?