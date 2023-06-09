# TFT_train

# 프로젝트 이름

이 프로젝트는 PyTorch와 PyTorch Forecasting을 사용하여 Temporal Fusion Transformer (TFT) 모델을 구현합니다.

## 환경 설정

이 프로젝트는 다음 환경에서 테스트되었습니다:

- Python == 3.10.9
- PyTorch == 2.0.0+cu117
- PyTorch Forecasting == 0.10.3

## 오류 발생 부분

**optimizer 미선언시 오류 발생**

# 오류 발생 부분

```python
import pytorch_optimizer as optim

tft = TemporalFusionTransformer.from_dataset(
    training,
    # 사용할 최적화 알고리즘
    optimizer=optim.Ranger,
    #  최적화 알고리즘의 학습률 (0.001~0.1)
    learning_rate=0.03,
    #  모델의 숨겨진 레이어 크기를 설정합니다. 이 값은 모델의 복잡성과 성능에 영향을 미칩니다.
    hidden_size=32,
    #  Attention 메커니즘에서 사용되는 헤드 수를 설정합니다. 큰 데이터셋의 경우 최대 4까지 설정할 수 있습니다.
    attention_head_size=3,
    # 드롭아웃 비율을 설정(0.1~0.3)
    dropout=0.1,
    #  연속 변수를 처리하는 데 사용되는 숨겨진 레이어의 크기(hidden_size 이하로 설정)
    hidden_continuous_size=16,
    #  출력 벡터의 크기를 설정합니다. 기본적으로는 7개의 분위수를 출력하도록 설정
    output_size=7,
    # 손실 함수
    loss=QuantileLoss(),
    # 검증 손실이 지정된 에폭 수 동안 개선되지 않을 경우 학습률을 줄이는 옵션
    reduce_on_plateau_patience=4,
    # embedding_sizes=embedding_sizes
)
```
