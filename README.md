# TFT_train

# 프로젝트 이름

이 프로젝트는 PyTorch와 PyTorch Forecasting을 사용하여 Temporal Fusion Transformer (TFT) 모델을 구현합니다.

## 환경 설정

이 프로젝트는 다음 환경에서 테스트되었습니다:

- Python 3.10.9
- PyTorch 2.0.0+cu117
- PyTorch Forecasting 0.10.3

## 오류 발생 부분

**optimizer 미선언시 오류 발생**

'''python
import pytorch_optimizer as optim
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss

tft = TemporalFusionTransformer.from_dataset(
    training,
    optimizer=optim.Ranger,  # 사용할 최적화 알고리즘
    learning_rate=0.03,  # 최적화 알고리즘의 학습률 (0.001~0.1)
    hidden_size=32,  # 모델의 숨겨진 레이어 크기를 설정
    attention_head_size=3,  # Attention 메커니즘에서 사용되는 헤드 수를 설정
    dropout=0.1,  # 드롭아웃 비율을 설정(0.1~0.3)
    hidden_continuous_size=16,  # 연속 변수를 처리하는 데 사용되는 숨겨진 레이어의 크기
    output_size=7,  # 출력 벡터의 크기를 설정
    loss=QuantileLoss(),  # 손실 함수
    reduce_on_plateau_patience=4,  # 검증 손실이 지정된 에폭 수 동안 개선되지 않을 경우 학습률을 줄이는 옵션
)
'''
