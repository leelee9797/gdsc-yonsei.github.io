---
published : true
layout : posts
author : Jiyeon Lee
title : "[지연/1st meetup] Intro to ML 키워드 조사"
---

## What is ML?

> 머신러닝 (Machine Learning), 직역하면 기계 학습

인간이 하나부터 열까지 직접 가르치는 기계를 의미하는 것이 아니라, 학습할 거리를 일단 던져놓으면 이걸 가지고 스스로 학습하는 기계를 의미함.

### KeyPoint!
- 경험(Experience)에 의해 학습(Learning)
- Task에 대해 점점 더 잘 실행되도록!
- 데이터가 많아지면 더욱 잘 될 것이다.


---

## ML vs Rule-based

> 컴퓨터 프로그램을 통해 인공 지능을 구현하는 방법에는 두가지가 있다.
    >> 기계 학습 vs 규칙 기반 학습

### 1. 규칙기반학습(Rule-Based Learning)
주어진 입력에 대해서 결과값을 도출하는 방법으로, if-then 방식이라고도 한다. 확고한 규칙(rule)에 따라 예측을 하는 방법이다.
이 방법은 기계가 인간처럼 사고하기 위해서 Rule을 이용한다. (근데 이것도 머신러닝????)

### 2. 머신 러닝 (Machine Learning)
인간이 컴퓨터에게 규칙을 입력해주는 것이 아니라 기계 스스로 배우도록 한다.


---

## Definition of AI, ML, DeepLearning

![AIMLDL](https://pbs.twimg.com/media/DLVm9nTXkAECS_S.png)

### 1. AI
> Artificial Intelligence is defined as any technology which appears to do something smart. </br> This can be anything from programmed software to deep learning models which mimic human intelligence

![인공지능-현대적접근](https://image.aladin.co.kr/product/7616/19/cover500/k442434518_1.jpg)

전 세계 110여개 나라의 1300여개 대학교에서 교재로 사용하고 있는 '인공지능 - 현대적 접근 (Artificial Intelligence - A Modern Approach)'에는 4가지 관점에서 여러가지 정의가 나온다.
=> 근데 모두가 동의하는 정의는 하나도 없음. 걍 모호함.

### 2. ML


> Machine learning is a specific kind of artificial intelligence but rather than a rule-based approach, the system learns how to do something from rather than being explicitly told what to do.examples

아서 사무엘은 머신러닝을 ‘컴퓨터가 명시적으로 프로그램되지 않고도 학습할 수 있도록 하는 연구 분야'라고 정의함


### 3. Deep Learning

> Deep learning is a specific type of machine learning using a technique known as a neural network which connects multiple models together to solve even more complex types of problems.

> Deep Learning, similar to other ML models, learns via examples. It's unique because it connects models to other models in layers in order to handle more complex types of data like as images.

딥러닝은 최근 유행하는 인공 신경망(Artificial Neural Network)의 새로운 이름!


---

# Type of ML
머신 러닝(Machine Learning)이란 "데이터를 이용해서 컴퓨터를 학습시키는 방법론"이다. 이때, 머신러닝 알고리즘을 지도 학습(Supervised Learning), 비지도 학습(Unsupervised Learning), 강화 학습(Reinforcement Learning)이다.

![TypeofML](http://solarisailab.com/wp-content/uploads/2017/06/supervsied_unsupervised_reinforcement.jpg)

## 지도 학습 (Supervised Learning)
지도학습은 <u>데이터에 대한 레이블(label : 명시적인 정답)이 주어진 상태</u>에서 컴퓨터를 학습시키는 방법이다. 
즉, (데이터 : 레이블) 형태로 학습을 진행하는 방법이다. 이 때, 예측하는 결과값이 <u>이산 값(Discrete value)</u>이면 <u>분류(Classification)</u> 문제이고, 예측하는 결과값이 <u>연속 값(Continuous Value)</u>이면 <u>회귀(Regression)</u> 문제이다. 

> 예시
    > 이산값 : 예상되는 숫자가 1인가 2인가?
    > 연속 값 : 3개월 뒤 예측되는 아파트 가격은 2억 1천인가 2억 2천인가??

### 1. Classification


### 2. Regression


## 비지도 학습(Unsupervised Learning)
비지도학습은 <u>데이터에 대한 레이블(label, 명시적인 정답)이 주어지지 않은 상태</u>에서 컴퓨터를 학습시키는 방법론이다.

즉, (데이터) 형태로 학습을 진행하는 방법이다. 예를 들어 아래와 같이 데이터가 무작위로 분포되어있을 때, 이 데이터를 비슷한 특성을 가진 것을 하나의 부류로 묶는 클러스터링(Clustering) 알고리즘이 있다.

![Clustering](http://solarisailab.com/wp-content/uploads/2017/06/unsupervised_learning.jpg)

### 3. Clustering
샘플들에 대한 유사도(similarity)에 근거하여 Cluster들을 구분. 패턴 공간에 주어진 유한개의 패턴들이 서로 가깝게 모여서 이루고 있는 패턴 집합을 cluster(군집)이라 하고 무리지워 나가는 처리 과정을 <u>clustering</u>이라고 한다.

## 강화 학습 (Reinforce Learning)
강화학습은 앞서 살펴본 지도 학습과 비지도 학습과는 약간 다른 종류의 알고리즘이다.

앞서 살펴본 알고리즘들이 데이터가 주어진 정적인 상태(static environment)에서 학습을 진행했다면, 강화 학습은(Reinforcement Learning)은 에이전트가 주어진 환경(state)에 대해 어떤 행동(action)을 취하고 어떤 보상(reward)을 얻으면서 학습을 진행한다.

이때, 에이전트는 보상을 최대화(maximize)하도록 학습이 진행된다. 즉, 강화 학습은 일종의 동적인 상태(dynamic environment)에서 데이터를 수집하는 과정까지 포함되어있는 알고리즘이다.

![강화학습](http://solarisailab.com/wp-content/uploads/2015/11/%EA%B0%95%ED%99%94%ED%95%99%EC%8A%B5.jpg)

강화학습의 대표적인 알고리즘은 Q-Learning이 있고, 딥러닝과 결합하여 Deep-Q-Network(DQN)방법으로도 사용된다.


### 4. Sequence Prediction
> Sequence prediction is a problem that involves using historical sequence information to predict the next value or values in the sequence. The sequence may be symbols like letters in a sentence or real values like those in a time series of prices.

시퀀스 예측이란 기록 시퀀스 정보를 사용하여 시퀀스 내의 다음 값이나 값을 예측하는 것과 관련된 문제다. 
![sequenceprediction](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/07/Many-to-Many-Sequence-Prediction-Model-1.png)

---

## Kind of Bias 

### 1. Interaction Bias
사용자는 알고리즘과 상호작용하는 방식으로 bias를 갖는다. 
(예) 신발 그려 -> 남자신발만 그림 -> 하이힐은 신발이라는 것을 모름

### 2. Latent Bias
알고리즘은 아이디어를 성별, 인종, 소득 등과 잘못 연관시킨다.
(예) 의사 -> 남자일 것이다

### 3. Selection Bias
알고리즘을 훈련시키는데 사용되는 데이터는 한 모집단을 지나치게 나타내 다른 모집단을 희생시키면서 더 잘 작동하게 한다.
(예) 미인대회를 백인으로만 학습 -> 다른 인종일때는 미인 아니라 함

---
## Reference
- <http://solarisailab.com/archives/1785>
- <https://machinelearningmastery.com/models-sequence-prediction-recurrent-neural-networks/>
- <https://qz.com/1064035/google-goog-explains-how-artificial-intelligence-becomes-biased-against-women-and-minorities/>
