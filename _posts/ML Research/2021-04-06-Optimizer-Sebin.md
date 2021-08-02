---
published : true
title: "[ML Research]최적화 알고리즘(Optimizer)"
date: 2021-04-06
excerpt: "최적화 알고리즘 Optimizer에 대한 포스트입니다."
use_math: true
categories:
  - MLResearch
author : SeBin Oh
---

<br/>

안녕하세요~ 여러분!  
다들 반가워요. 🧚‍♀️

<br/>

오늘은 **최적화 알고리즘 Optimizer** 에 대해 알아보려고 합니다!

<br/>

목차는 다음과 같습니다.

<br/>

### 📖 목차

- [🛤  최적화 알고리즘(Optimizer)이란?](#--최적화-알고리즘optimizer이란)
- [1. 경사하강법(Gradient Descent: GD)](#1-경사하강법gradient-descent-gd)
- [2. 확률적 경사하강법(Stochastic Gradient Descent: SGD)](#2-확률적-경사하강법stochastic-gradient-descent-sgd)
- [3. 미니 배치 확률적 경사하강법(Mini-Batch Gradient Descent)](#3-미니-배치-확률적-경사하강법mini-batch-gradient-descent)
- [4. 관성(Momentum)](#4-관성momentum)
- [5. Nesterov Accelerated Gradient(NAG)](#5-nesterov-accelerated-gradientnag)
- [6\. Adaptive Gradient(Adagrad)](#6-adaptive-gradientadagrad)
- [7\. Root Mean Square PROPagation(RMSprop)](#7-root-mean-square-propagationrmsprop)
- [8. Adaptive Delta(AdaDelta)](#8-adaptive-deltaadadelta)
- [9. Adaptive moment esimation(Adam)](#9-adaptive-moment-esimationadam)
  - [🤝 마치며](#-마치며)
  - [Reference](#reference)
  - [글쓴이](#글쓴이)

<br/>

최적화 알고리즘이란 무엇인가! 하나하나 살펴보시죠. 

<br/>

## 🛤  최적화 알고리즘(Optimizer)이란?  

<br/>

최적화 알고리즘이란 신경망(neural network) 학습에서 손실함수(loss function) 값을 최소화하는 매개변수, neural network의 가중치와 학습률을 업데이트하는 것을 의미합니다.  
주로 Gradient Descent Algorithm를 기반으로 한 SGD에서 변형된 여러 종류의 최적화함수가 사용되고 있습니다.

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FuXBgR%2FbtqVIJwz6Vb%2FFqut2qOU32qSyZ9l87Xp8k%2Fimg.png)

<br/>

아래는 대표적인 Optimizer 기법들이 최적값을 찾아가는 그래프로 각각의 특성이 잘 드러나 있습니다.

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FNXkd0%2FbtqVK8P5Qv3%2F8WmnkPUWNTsPFFKPLGpQvK%2Fimg.gif)  

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbD5etd%2FbtqVTHp8mxY%2FU7lXvhVmWPLk1ShaSW0lvk%2Fimg.gif)  

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FupNYI%2FbtqVSvwJGgh%2FI7lxtFog1r2ommkO9uROKK%2Fimg.gif)  

<br/>

## 1. 경사하강법(Gradient Descent: GD)

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FnhTKN%2FbtqVTG5QSUI%2FIEHqwDs7UoXh3reS749bZk%2Fimg.png)

<br/>

가장 기본적이며 많이 쓰이는 최적화 알고리즘입니다. 전체 데이터에 대하여 손실함수(loss function)의 현 가중치에서의 기울기(gradient)를 구해서 손실(loss)을 줄이는 방향으로 업데이트합니다. 하지만 local minima에 빠질 수 있으며, 전체 데이터를 대상으로 학습하기 때문에 많은 메모리와 시간이 소요된다는 단점이 존재합니다. 그래서 이를 극복하기 위하여 확률적 경사하강법(SGD)이 등장하였습니다.

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F073hY%2FbtqVK7XZHPA%2F6KvtSJjKj4xG4oTSK3EOIk%2Fimg.png)

<br/>

## 2. 확률적 경사하강법(Stochastic Gradient Descent: SGD)

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FxgXbN%2FbtqVSuLohtA%2FhNisa6ooPgxcNP5nBeEX20%2Fimg.png)

<br/>

확률적 경사하강법(SGD)는 매개변수 값을 조정 시 전체 데이터가 아니라 랜덤으로 선택한 하나의 데이터에 대해서만 계산하는 optimizer입니다. 손실함수(loss function)를 계산할 때 전체 데이터가 아닌 mini-batch 크기를 결정하여 그 크기의 데이터마다 기울기(gradient)를 구해서 손실(loss)을 줄이는 방향으로 업데이트합니다. 모델 parameters를 빠르게 찾기 때문에 수렴하는데 적은 메모리와 시간이 소요됩니다. 그래서 GD보다 상대적으로 처리속도가 빠릅니다. 하지만 매 학습마다 parameters를 찾는데 기울기(gradient) 크기의 변동 폭이 커서 분산이 큽니다. 분산이 큰 기울기(gradient)는 확률적 경사하강법(SGD)가 local minimum에서 빠져나오는데 도움이 되는 동시에 수렴을 방해할 수도 있습니다.

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FqzZr5%2FbtqVSt6JUaF%2FJ7RyVwKKsMWeHWKngr1uzK%2Fimg.png)

<br/>

## 3. 미니 배치 확률적 경사하강법(Mini-Batch Gradient Descent)

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FzyXwx%2FbtqVJSGOJZn%2Ffiwd3NwKsvWp7z6ns22GWk%2Fimg.png)

<br/>

미니 배치 확률적 경사하강법(Mini-Batch Gradient Descent)는 경사하강법(GD)과 확률적 경사하강법(SGD)의 장점만 모아서 만든 optimizer입니다. 매 배치 후에 모델 매개변수를 업데이트합니다. 따라서 데이터 세트는 다양한 배치로 분할되고 매 배치 후에 매개변수를 업데이트합니다. 모델 변수들을 자주 업데이트할 수 있고 적은 분산을 가지며 중간 크기의 메모리가 필요합니다. 하지만 학습률(learning rate)이 기울기(gradient)에 비해 작을 경우, 수렴에 많은 시간이 소요됩니다. 그리고 모든 변수들은 고정된 학습률(learning rate)을 가지고 있어 local minimum에 빠질 수 있습니다.

<br/>

## 4. 관성(Momentum)

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdDvQJx%2FbtqVJQWxxsf%2F377kLhqrzxhorRMKXNoMl0%2Fimg.png)

<br/>

Momentum은 관성을 응용한 방법으로, SGD에서 추진력, 여세, 타성 등 물체가 한 방향으로 지속적으로 변화하려는 특징을 이용하여 만든 optimizer입니다. 즉, 이전 학습에서의 가중치 업데이트에 관성을 부여하는 것을 의미합니다. Momentum은 확률적 경사하강법(SGD)에서 계산된 접선의 기울기(gradient)에 한 시점(step) 전의 접선의 기울기(gradient) 값을 일정한 비율만큼 반영합니다. 그래서 예를 들면 언덕에서 공이 내려올 때, 중간에 작은 웅덩이에 빠지더라도 관성의 힘으로 넘어서는 효과를 줄 수 있습니다.

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FRLlh7%2FbtqVRwbzD0C%2FahSHRpTio5I24KWdLfmqa1%2Fimg.png)

<br/>

local minimum에 도달하였을 때, 기울기(gradient)가 0이라서 기존의 경사 하강법(GD)이라면 이를 global minimum으로 잘못 인식하여 계산이 끝났을 수도 있지만, 관성을 적용하면 값이 조절되면서 local minimum에서 탈출하는 효과를 얻을 수도 있습니다.

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FecGCty%2FbtqVQV3IYvx%2FvIjxuKAqDYThASJ8N3HZX1%2Fimg.png)

<br/>

## 5. Nesterov Accelerated Gradient(NAG)

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FQRlGZ%2FbtqVIJJ5eV7%2F41dCpKtKZT3vEJ1kiVpZiK%2Fimg.png)

<br/>

Momentum이 너무 큰 경우, 최적화 알고리즘은 local minima를 지나쳐 계속 커질 수 있습니다. 이를 해결하기 위하여 Nesterov Accelerated Gradient(NAG)가 등장하였습니다. Nesterov Accelerated Gradient(NAG)는 미래를 보고 현재의 관성을 조절하여 업데이트하는 optimizer입니다. 그래서 local minimum을 놓치지 않고 속도를 늦출 수 있습니다. 즉, Momentum의 빠른 이동과 적절한 시점에 제동을 걸 수 있다는 장점이 있습니다. 하지만 변수를 여전히 수동적으로 선택하는 단점도 있습니다.

<br/>

## 6\. Adaptive Gradient(Adagrad)

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FMkAJj%2FbtqVIbGPy89%2FLe2WcP3lkjO3pM03GpyK11%2Fimg.png)

<br/>

Adagrad는 변수의 업데이트 횟수에 따라 학습률(learning rate)을 조절하는 옵션이 추가된 최적화 방법입니다. 지금까지 설명한 optimizer들의 단점은 모든 cycle 및 변수들에 대하여 학습률(learning rate)가 일정하다는 것입니다. 이를 해결하기 위해 Adagrad는 학습률(learning rate)에 변화를 줍니다. Adagard는 많이 변화하지 않은 변수들은 학습률(learning rate)을 크게 하고, 반대로 많이 변화한 변수들에 대해서는 학습률(learning rate)을 적게 합니다. 이는 많이 변화한 변수는 최적값에 근접하였을 것이라는 가정 하에 작은 크기로 이동하면서 세밀한 값을 조정하고, 반대로 적게 변화한 변수들은 학습률(learning rate)을 크게 하여 빠르게 손실(loss) 값을 줄입니다. 이처럼 학습률(learning rate)에 변화를 준다는 점과 수동적으로 학습률(learning rate)을 조정할 필요가 없다는 장점이 있지만, 학습이 너무 빠르게 느려져서 global minimum에 수렴하지 못하는 단점도 존재합니다.

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcVMDKF%2FbtqVIJ4psUx%2FCAJx1EBhTkFw5CP3nvb9Mk%2Fimg.png)

<br/>

## 7\. Root Mean Square PROPagation(RMSprop)

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb0Utbz%2FbtqVQXN1dRP%2FAnZt6s47en5NsriGS6qqSK%2Fimg.png)

<br/>

Adagard의 문제점을 개선하기 위하여 가장 최근 반복에서 비롯된 기울기(gredient)만 누적시키는 optimizer입니다. Adagrad와의 차이는 감마가 있느냐 없느냐입니다. 이는 이전의 기울기와 현재의 기울기를 의미합니다. $G$ 계산식에 지수이동평균을 적용하였습니다. 학습이 진행됨에 따라 parameter 사이 차별화는 유지하되 학습속도가 지속적으로 줄어들어 0에 수렴하는 것을 방지할 수 있습니다.

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbIi1Yq%2FbtqVSuR8Xlo%2FzMZVGYzw41zpAsIXo9heuk%2Fimg.png)

<br/>

## 8. Adaptive Delta(AdaDelta)

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdJZNvD%2FbtqVK7X1Fpj%2F3XkXh8mGmGDb5WwLgl5kl0%2Fimg.png)

<br/>

RMSprop와 마찬가지로 Adagrad의 학습률이 떨어지는 문제를 해결하기 위해 개발된 optimizer입니다. 이전에 제곱된 모든 gradient를 누적하는 대신 AdaDelta는 누적된 이전의 기울기(gradient)의 창을 고정된 크기(window size: w)로 제한하고 지수이동평균이 사용됩니다. 학습률이 떨어지지 않기 때문에 훈련이 중단되지 않습니다. 하지만 계산속도는 상대적으로 느립니다.

<br/>

## 9. Adaptive moment esimation(Adam)

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FqOfgn%2FbtqVOqQx6iT%2FBiWopD5t0Lyv1EPrxR5xYk%2Fimg.png)

<br/>

Adam은 딥러닝에서 가장 일반적으로 사용되며, Momentum과 RMSprop이 결합하여 최솟값을 뛰어넘지 않도록 속도를 줄이기 위해 개발된 optimizer입니다. Momentum에서 관성계수 $\\gamma$과 함께 계산된 $V{t}$로 parameter를 업데이트하지만 기울기 값과 기울기의 제곱 값의 지수이동평균을 활용하여 step 변화량을 조절합니다. Momentum처럼 이전의 gradient 지수 감소 평균을 따르고 RMSprop처럼 이전의 gredient 제곱의 지수 감소된 평균을 따릅니다. Adam은 매우 빠르게 수렴하지만, 계산속도는 느립니다.

<br/>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FIURQ8%2FbtqVK8P76mq%2FQUAEOb0HVeDTrNptOrv440%2Fimg.png)

<br/>

### 🤝 마치며

이상 최적화 알고리즘(Optimizer)에 대해 간략히 알아보았습니다.
다들 긴 글 읽어주셔서 감사하고 다음 포스트도 기대해주세요~ 🎉

<br/>

### Reference

[https://blog.naver.com/i\_am\_sangyun/222165187257](https://blog.naver.com/i_am_sangyun/222165187257)

[https://blog.naver.com/stu5073/222159804653](https://blog.naver.com/stu5073/222159804653)

[https://blog.naver.com/winddori2002/221957205824](https://blog.naver.com/winddori2002/221957205824)

[https://blog.naver.com/youjaeah/222218660537](https://blog.naver.com/youjaeah/222218660537)

[https://onevision.tistory.com/entry/Optimizer-%EC%9D%98-%EC%A2%85%EB%A5%98%EC%99%80-%ED%8A%B9%EC%84%B1-Momentum-RMSProp-Adam](https://onevision.tistory.com/entry/Optimizer-%EC%9D%98-%EC%A2%85%EB%A5%98%EC%99%80-%ED%8A%B9%EC%84%B1-Momentum-RMSProp-Adam)

김수윤(Sooyoon Kim), 정우근(Wookeen Chung), 신성렬(Sungryul Shin). 2019. "Adam Optimizer를 이용한 음향매질 탄성파 완전파형역산." 지구물리와 물리탐사, 22(4) : 202-209

마샤 모라디(Mahsa Moradi), 이태삼(Taesam Lee). 2018. "Comparison of Optimization Algorithms in Deep Learning-Based Neural Networks for Hydrological Forecasting: Case Study of Nam River Daily Runoff." 한국방재학회논문집, 18(6) : 377-384



<br/>

### 글쓴이

DSC Yonsei 오세빈

E-mail: [osb3372@yonsei.ac.kr](http://osb3372@yonsei.ac.kr)
