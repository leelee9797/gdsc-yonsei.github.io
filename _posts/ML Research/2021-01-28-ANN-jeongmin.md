---
published : true
title: "[ML Research]Artificial Neural Network"
date: 2021-01-28
excerpt: "인공신경망의 기초적인 부분에 대해서 다루는 포스트입니다."
use_math: true
categories:
  - MLResearch
author : JeongMin Do
---



# 이론

앞서 [로지스틱 회귀](https://silverstar0727.github.io/ml%20basic/2021/01/05/%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1%ED%9A%8C%EA%B7%80/#)를 통해 이진 분류를 할 수 있다는 것을 참고할 수 있습니다.  
그러나 이러한 로지스틱 회귀는 단일한 직선으로 분류하는 것이기에, 직선으로 분류되지 않은 문제들에 대해서는 무용합니다.

따라서, 이를 해결하기 위해 등장한 것이 ANN입니다.

  
위에서의 로지스틱 회귀를 하나의 perceptron으로 본다면, perceptron은 다양한 독립변수들을 입력 받아 weights와 선형결합을 한 뒤, sigmoid function을 통해 0과 1사이의 결과 값을 반환합니다.

  
ANN은 이러한 perceptron이 단일하지 않고 여러개 사용한다는 관점에서 출발합니다.

언뜻보면 이러한 관점이 이해가 가지 않을 수도 있으나, 하나하나 차근차근히 이해하면 정말 별 것 없다는 것을 배울 수 있습니다.

  
특히 아래의 그림을 보면 보다 직관적으로 이해가 갑니다.

![image](https://user-images.githubusercontent.com/49096513/105300125-165dd100-5bfe-11eb-81fa-b1c272db8f63.png)

위 그림에서 수많은 독립변수인 x의 값들이 주어지고, 각 노드에 이전 layer의 모든 output이 input이 되어 들어가게 됩니다.  
이렇게 들어간 input은 노드에 있는 input의 개수만큼의 weights와 선형 결합을 하게되고, 0과 1 사이의 값을 sigmoid function을 통해 반환하는 것입니다.

어떤가요? perceptron과 정말 똑같지 않나요?

여기서 sigmoid function이외의 다른 함수를 사용한다면, 또 다른 결과의 예측이 나올 것입니다.  
이러한 선형결합이 통과하는 함수를 Activation function이라고 하며, 다음의 링크 [Activation function](https://silverstar0727.github.io/ml%20basic/2021/01/06/Activation_Function/#)을 참조하세요.

> 함수를 통과해야 하는 이유: activation function은 말 그대로 '활성화 정도'를 나타내기 때문.  
> 즉, 해당 노드의 모델 전체에서의 영향이 어느정도인지를 알려준다는 것입니다.  
> (앞으로 ANN에서의 노드는 모두 perceptron에 해당한다).

> layer는 초기 input이 노드를 통과하는 횟수가 같을 때의 노드집단을 가리킵니다. layer는 크게 입력층, 은닉층, 출력층으로 구분됩니다.  
> 입력층은 input의 독립변수의 개수와 동일해야하고, 은닉층은 상관없으나, 출력층은 원하는 class 개수만큼의 유닛으로 이루어져야 합니다.(그래야 결과가 class 개수만큼 0~1 사이로 나올 수 있기 때문입니다.)

## ANN의 매커니즘(Backpropagation)

앞서, ANN의 구조를 살펴보았다. 그렇다면 이러한 node가 뭉쳐 layer가 되고, layer가 모여 ANN이 되는데 이것을 도대체 어떻게 학습시킨다는 걸까요?

  
우선 오차를 정의해야 합니다. 그 이유는 여지껏 우리가 해오던 작업이 '오차의 최소화'이며 이것이 머신러닝 전반을 관통하는 핵심적인 개념이기 때문입니다.

오차는 크게 Regression에서는 MSE, Classification에서는 Crossentropy가 사용되며, 이밖에도 다양한 오차들이 존재합니다. 다양한 오차에 대해서는 별도의 포스트에서 다루는 것이 좋을 것 같습니다.

이렇게 정한 오차를 최적화하귀 위해서는 Backpropagation 알고리즘을 사용하게 됩니다. 이는 각 weight가 결과에 미치는 영향을 반영하기 위함입니다.

  
따라서, Loss를 Weight로 미분한 값을 Gradient Descent 등의 Optimizer로 최적화를 진행하게 됩니다.  
아래의 그림은 이러한 과정을 수식으로 보다 적나라하게 드러내 줍니다.

![image](https://user-images.githubusercontent.com/49096513/105667983-c776ab00-5f1f-11eb-937f-6b3a01240997.png)

이러한 방식이 널리 쓰이는 이유는 chain-rule에 의해서 아주 간편하게 계산이 가능하기 때문입니다. 단순 계산 작업은 컴퓨터에 특화된 영역이며, 이를 통해 최적화 하는 방식이 Deep Learning으로의 시작이라 할 수 있습니다.

# 구현

#### Libraries

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import matplotlib.pyplot as plt
```

#### Data

```
num_data = 1000

noise = init.normal_(torch.FloatTensor(num_data,1),std=1)
x = init.uniform_(torch.Tensor(num_data,1),-15,15)
y = (x**3) + 3*x**2 +3 
y_noise = y + noise
```

#### Model

```
model = nn.Sequential(
          nn.Linear(1,32),
          nn.ReLU(),
          nn.Linear(32,64),
          nn.ReLU(),
          nn.Linear(64,16),
          nn.ReLU(),
          nn.Linear(16,1),
      )

loss_func = nn.L1Loss()
optimizer = optim.SGD(model.parameters(),lr=0.0002)
```

모델은 독립변수가 1개일 때, 1차원의 입력 값에 hidden layer로 32개, 64개, 16개를 배치하였습니다.

#### Training

```
num_epoch = 10000

loss_array = []
for i in range(num_epoch):
    optimizer.zero_grad()
    output = model(x)

    loss = loss_func(output,y_noise)
    loss.backward()
    optimizer.step()

    loss_array.append(loss)
```

#### Loss

```
plt.plot(loss_array)
plt.show()
```

![image](https://user-images.githubusercontent.com/49096513/106087730-f0897c80-6167-11eb-8148-f2b8ee94d94b.png)

#### Visualization

```
plt.figure(figsize=(10,10))
plt.scatter(x.detach().numpy(),y_noise,label="Original Data")
plt.scatter(x.detach().numpy(),output.detach().numpy(),label="Model Output")
plt.legend()
plt.show()
```

![image](https://user-images.githubusercontent.com/49096513/106087721-e8314180-6167-11eb-8ae5-1d73d8fd8976.png)

최종적으로는 3차 함수에 대해 나쁘지 않은 예측 결과를 보인 것을 알 수 있습니다.

## Reference

-   [Fig.1](https://www.researchgate.net/figure/Artificial-neural-network-architecture-ANN-i-h-1-h-2-h-n-o_fig1_321259051)
-   [Fig.2](https://www.youtube.com/watch?v=An5z8lR8asY) Backpropagation: how it works
