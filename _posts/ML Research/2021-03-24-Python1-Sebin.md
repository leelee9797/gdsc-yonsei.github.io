---
published : true
layout : posts
title: "[ML Research]Python을 이용한 데이터 분석과 이미지 처리(1)"
date: 2021-03-24
excerpt: "Python을 이용한 데이터 분석과 이미지 처리에 대한 포스트입니다."
comments: true
use_math: true
author : SeBin Oh
---

<br/>

안녕하세요~ 여러분! 😄
오늘은 Python을 이용한 데이터 분석과 이미지 처리에 대해 알아보도록 하겠습니다.

<br/>

목차는 다음과 같습니다.

<br/>

### 📖 목차
- [OpenCV 소개](#opencv-소개)
   * [OpenCV 기본 함수](#opencv의-기본적인-함수)
- [OpenCV 이미지 처리 방법](#opencv-이미지-처리-방법)
   * [OpenCV 이미지 연산](#opencv-이미지-연산)
   * [OpenCV 이미지 변형](#opencv-이미지-변형)
   * [OpenCV 이미지 회전](#opencv-이미지-회전)
   * [OpenCV 임계점 처리하기](#opencv-임계점-처리하기)
   * [OpenCV Tracker](#opencv-tracker)
   * [OpenCV Contours](#opencv-contours)
   * [OpenCV Contours 처리](#opencv-contours-처리)
   * [OpenCV Filtering](#opencv-filtering)
- [KNN Algorithm](#knn-algorithm)
   * [KNN이란?](#knnk-nearest-neighbor이란)
   * [KNN 숫자 인식 예제](#knn-숫자-인식-예제)

<br/>

본문은 **동빈나** 유튜브의 'Python 데이터 분석과 이미지 처리' 강의(7~18)를 참고하여 재구성하였으며 대부분의 실습은 **CoLab** 과 **PyCharm**을 이용하였습니다.

[<span style="color:skyblue">강의링크</span>](https://www.youtube.com/watch?v=F2FRpmh9sQo&list=PLRx0vPvlEmdBx9X5xSgcEk4CEbzEiws8C&index=7)  

<br/>

우선 OpenCV에 대해서 하나씩 살펴보도록 하겠습니다!

<br/>

# 📱 OpenCV 소개

Open Source Computer Vision Library  
컴퓨터 비전을 위한 오픈소스 라이브러리를 말합니다.

- 이미지 처리에 초점
- C, C++, Python 등에서 사용 가능

<br/>

다음으로 OpenCV의 기본적인 사용법에 대해 살펴보겠습니다!

<br/>

## OpenCV의 기본적인 함수

<br/>

OpenCV에서는 다음과 같은 함수를 자주 사용합니다.

<br/>

> cv2.imread(fileName, flag) : 이미지를 읽어 Numpy 객체로 만드는 함수
>   
> cv2.imshow(title, image) : 특정한 이미지를 화면에 출력하는 함수  
>
> cv2.imwrite(fileName, image) : 특정한 이미지를 파일로 저장하는 함수  
>
> cv2.waitKey(time) : 키보드 입력을 처리하는 함수  
>
> cv2.destroyAllWindows() : 화면의 모든 윈도우를 닫는 함수

<br/>

## 💡 보충 ) CoLab에서의 이미지 출력

<br/>

-   CoLab은 Jupyter Notebook을 기반으로 동작하므로 Matplotib를 이용하여 이미지를 출력합니다.
-   OpenCV는 BGR을 기준으로, Matplotlib은 RGB를 기준으로 한다는 차이가 있습니다.

<br/>

```
import cv2
import matplotlib.pyplot as plt

img_basic = cv2.imread('cat.jpg', cv2.IMREAD_COLOR)
plt.imshow(cv2.cvtColor(img_basic, cv2.COLOR_BGR2RGB))
plt.show()

img_basic = cv2.cvtColor(img_basic, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(img_basic, cv2.COLOR_GRAY2RGB))
plt.show()
```

[##_Image|kage@AGlT7/btqXSnqNqco/T03qeYIcvS11Cs4O5d3yNk/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##][##_Image|kage@bgoKdD/btqXQ4ytNut/S9driBO2PnsePPk3GdRAuK/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

# 📷 OpenCV 이미지 처리 방법

이제 OpenCV를 이용하여 기본적인 이미지를 처리하는 방법에 대하여 하나씩 알아보겠습니다!  
글은 전반적으로 간단한 설명과 예제 위주로 서술하였습니다. 자세한 설명은 강의를 참고하시기 바랍니다.

<br/>

## ➕➖ OpenCV 이미지 연산

<br/>

### OpenCV를 활용한 이미지 크기 및 픽셀 정보 확인

<br/>

```
import cv2
image = cv2.imread('cat.jpg')

# 픽셀 수 및 이미지 크기 확인
print(image.shape)
print(image.size)

# 이미지 Numpy 객체의 특정 픽셀을 가리킵니다.
px = image[100, 100]

# B, G, R 순서로 출력됩니다.
# (단, Gray Scale인 경우에는 B, G, R로 구분되지 않습니다.)
print(px)

# R 값만 출력하기
print(px[2])
```

(380, 441, 3)  
502740  
\[111 151 179\]  
179

<br/>

### OpenCV를 활용한 특정 범위 픽셀 변경

<br/>

아래에 두 연산을 비교해보면 동일하게 특정 범위의 값을 바꾸지만 아래의 슬라이싱연산 처리속도가 더 빠르다는 것을 알 수 있습니다.

<br/>

```
import cv2
import matplotlib.pyplot as plt
import time

image = cv2.imread('cat.jpg')

start_time = time.time()
for i in range(0, 100):
    for j in range(0, 100):
        image[i, j] = [255, 255, 255]
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
image[0:100, 0:100] = [0, 0, 0]
print("--- %s seconds ---" % (time.time() - start_time))

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
```

\--- 0.016465187072753906 seconds ---  
\--- 0.0002148151397705078 seconds ---

[##_Image|kage@S83z2/btqXToiFLle/HVOdvjMMKBzbSCN7soRDbk/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

### OpenCV를 활용한 ROI (Region of Interest: 관심 영역) 추출 및 복사

<br/>

관심 영역을 지정하여 그 부분을 복사하여 추출하는 것이 가능합니다.

<br/>

```
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('cat.jpg')

# Numpy Slicing: ROI 처리 가능
roi = image[200:350, 50:200]

# ROI 단위로 이미지 복사하기
image[0:150, 0:150] = roi

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
```

[##_Image|kage@dSDy72/btqXXjOFyCq/aThAQ2jaVC4oQs1tGnEIWk/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

### OpenCV를 활용한 픽셀별 색상 다루기

<br/>

아래 예제는 R 값을 0으로 하여 G, B 값만으로 나타내었습니다.

<br/>

```
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('cat.jpg')
image[:, :, 2] = 0

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
```

[##_Image|kage@wxZIq/btqXU4qJBds/YKZGUXxDUfAvXjksr9QJSk/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

## 🎞 OpenCV 이미지 변형

### 이미지 크기 조절

<br/>

* 💡 **보간법(Interpolation)** 이란 ?

크기가 변할 때 중간값을 가지거나 특정한 분포를 따르게 하여 매끄럽게 픽셀 사이의 값을 조절하는 방법을 말합니다.

[##_Image|kage@wkWhZ/btqXMdbCpKD/0bwkcBSiO4okaShnInR6x0/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

> cv2.resize(image, dsize, fx, fy, interpolation) : 이미지의 크기를 조절합니다.

-   dsize : Manual Size
-   fx : 가로 비율
-   fy : 세로 비율

1) INTER_CUBIC : 사이즈를 크게 할 때 주로 사용합니다.

2) INTER_AREA : 사이즈를 작게 할 때 주로 사용합니다.

<br/>

```
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('cat.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

expand = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(expand, cv2.COLOR_BGR2RGB))
plt.show()

shrink = cv2.resize(image, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
plt.imshow(cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB))
plt.show()
```

[##_Image|kage@BPn9l/btqXTopsCzF/FgkRXpLUCkxl8fgDkZVVAK/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

### 이미지 위치 변경

<br/>

다음 함수를 사용하여 이미지 위치를 변경합니다.

<br/>

> cv2.warpAffine(image, M, dsize) : 이미지의 위치를 변경합니다.

-   M : 변환 행렬
-   dsize : Manual Size

<br/>

#### 💡 보충 ) 변환 행렬과 변환

<br/>

- 변환 행렬의 다음과 같은 형태로 정의되고
    [##_Image|kage@byu9uh/btqXXVUp2sK/hjDoAu1LPjePC0LTsRZVYk/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

- 이미지의 모든 좌표(a, b)는 다음의 좌표로 이동됩니다.  
    [##_Image|kage@uVakk/btqXYpAWPon/hKULvVfNNP7teJx2a01Of0/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('cat.jpg')

# 행과 열 정보만 저장합니다.
height, width = image.shape[:2]

M = np.float32([[1, 0, 50], [0, 1, 10]])
dst = cv2.warpAffine(image, M, (width, height))

plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.show()
```

[##_Image|kage@bNTxIt/btqXXT3mQVC/2qVKqn2rcsSESglezHBrLk/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

## 🌀 이미지 회전

이미지 회전을 위해 다음 함수를 사용합니다.

<br/>

> cv2.getRotationMatrix2D(center, angle, scale) : 이미지 회전을 위한 변환 행렬을 생성합니다.

-   center : 회전 중심
-   angle : 회전 각도
-   scale : Scale Factor

<br/>

#### 💡 보충 ) 회전 변환을 위한 행렬

<br/>

- 회전 변환을 위한 행렬은 다음과 같습니다.  
    [##_Image|kage@bmULCu/btqXXU2jwkm/fK9kaZsaYVtf2F9ZNVJJkk/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

- 이때 무게 중심을 적용할 수 있는 회전 변환 식은 다음과 같이 정의할 수 있습니다.  
    [##_Image|kage@dr6Pm9/btqXQ5df5AH/4KMzRGxFgv4srELnlaNRK0/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('cat.jpg')

# 행과 열 정보만 저장합니다.
height, width = image.shape[:2]

M = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 0.5)
dst = cv2.warpAffine(image, M, (width, height))

plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.show()
```

[##_Image|kage@biA5Se/btqXXkmBXBV/yFWOMkCHIXs1pdNeBF2Avk/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

## 🔰 OpenCV 이미지 합치기

<br/>

다음 두 이미지를 합쳐보도록 하겠습니다.

<br/>

[##_Image|kage@P0XrH/btqXQ4yEaz1/Q0A4BLMlQF2UmXiycjCM9K/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

### 이미지를 합치는 2가지 방법

아래의 두 함수를 이용하여 각각 이미지를 합쳐보도록 하겠습니다.

<br/>

> cv2.add() : Saturation 연산을 수행합니다.  
* 0보다 작은면 0, 255보다 크면 255로 표현

<br/>

[##_Image|kage@cgM0uO/btqXTmL0HXM/VlCg8PiM8ZFR6Trckeyfi1/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

> np.add() : Modulo 연산을 수행합니다.  
* 256은 0, 257은 1로 표현

<br/>

[##_Image|kage@7X7ve/btqXRvJtorB/sAQadrKIkTZFMIV1nUxTo1/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

```
import cv2
import matplotlib.pyplot as plt

image_1 = cv2.imread('image_1.jpg')
image_2 = cv2.imread('image_2.png')

result = cv2.add(image_1, image_2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()

result = image_1 + image_2
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()
```

[##_Image|kage@cxr18w/btqXU27Aryc/FIbLDlKVweZYmtUnJQJw41/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

## 〽 OpenCV 임계점 처리하기

다음과 같은 방법을 통하여 이미지의 임계점을 처리할 수 있습니다.

<br/>

### 이미지의 기본 이진화

<br/>

> cv2.threshold(image, thresh, max\_value, type) : 임계값을 기준으로 흑/백으로 분류하는 함수

- image : 처리할 Gray Scale 이미지
- thresh : 임계 값 (전체 픽셀에 적용)
- max_value : 임계 값을 넘었을 때 적용할 값
- type : 임계점을 처리하는 방식

1) THRESH_BINARY : 임계 값보다 크면 max_value, 작으면 0  
2) THRESH_BINARY_INV : 임계 값보다 작으면 max_value, 크면 0  
3) THRESH_TRUNC : 임계 값보다 크면 임계 값, 작으면 그대로  
4) THRESH_TOZERO : 임계 값보다 크면 그대로, 작으면 0  
5) THRESH_TOZERO_INV : 임계 값보다 크면 0, 작으면 그대로

<br/>

[##_Image|kage@9x8aV/btqXOaMGRHf/3ZRekcvCAwy8RkP1ibAr21/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

```
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('gray_image.jpg', cv2.IMREAD_GRAYSCALE)

images = []
ret, thres1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
ret, thres2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
ret, thres3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
ret, thres4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
ret, thres5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
images.append(thres1)
images.append(thres2)
images.append(thres3)
images.append(thres4)
images.append(thres5)

for i in images:
  plt.imshow(cv2.cvtColor(i, cv2.COLOR_GRAY2RGB))
  plt.show()
```

[##_Image|kage@cBjoX5/btqXYZI1ZT2/gKk2ofwyFjsIEXc76Ebxkk/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##][##_Image|kage@bWTAsa/btqXXUhbgzC/7Y4uYu4kkl2olUVEGjLvGK/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

### 이미지의 적응 임계점 처리

<br/>

하나의 이미지에 다수의 조명 상태가 존재하는 경우 적용하는 방법입니다.

<br/>

[##_Image|kage@cBdbk3/btqXU4K7fDO/4USdGvf75hFtzKw3IiF7pk/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

> cv2.adaptiveThreshold(image, max_value, adaptive_method, type, block_size, C) : 적응 임계점 처리 함수

-   max_value : 임계 값을 넘었을 때 적용할 값
-   adaptive_method : 임계 값을 결정하는 계산 방법  
    ADAPTIVE_THRESH_MEAN_C : 주변영역의 평균값으로 결정  
    ADAPTIVE_THRESH_GAUSSIAN_C
-   type : 임계점을 처리하는 방식
-   block_size : 임계 값을 적용할 영역의 크기
-   C: 평균이나 가중 평균에서 차감할 값

<br/>

Adaptive Threshold를 이용하면, 전체 픽셀을 기준으로 임계 값을 적용하지 않습니다.

<br/>

#### 💡 ADAPTIVE_THRESH_MEAN_C

<br/>

* 적용할 픽셀 (x,y)를 중심으로 하는 Block Size * Block Size 안에 있는 픽셀 값의 평균에서 C를 뺀 값을 임계점으로 설정합니다.

<br/>

[##_Image|kage@tWCbA/btqXXUnI4uS/insOZcBqkWB2PyMtmA7Kj1/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

#### 💡 ADAPTIVE\_THRESH\_GAUSSIAN\_C

<br/>

- 적용할 픽셀 (x,y)를 중심으로 하는 Block Size * Block Size 안에 있는 Gaussian 윈도우 기반의 가중치들의 합에서 C를 뺀 값을 임계점으로 설정합니다.

<br/>

```
import cv2

image = cv2.imread('hand_writing_image.jpg', cv2.IMREAD_GRAYSCALE)

ret, thres1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
thres2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 3)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
plt.show()

plt.imshow(cv2.cvtColor(thres1, cv2.COLOR_GRAY2RGB))
plt.show()

plt.imshow(cv2.cvtColor(thres2, cv2.COLOR_GRAY2RGB))
plt.show()
```

[##_Image|kage@GxAmh/btqXU4dxx90/hi9Rtkx3MWYxw3kjZnYWb0/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

## ⏸ OpenCV Tracker

**Tracker** 란 사용자가 값을 편하게 슬라이드 바를 이용하여 움직여보면서 바꿀 수 있는 기능입니다.

<br/>

### Tracker 사용방법

<br/>

> cv2.createTracker(track_bar, name, window_name, value, count, on_change) : Tracker를 생성하는 함수

-   value : 초기 값
-   count : Max 값 (Min: 0)
-   on_change : 값이 변경될 때 호출되는 Callback 함수

<br/>

> cv2.getTrackerPos(track_bar, name, window_name) : Tracker로부터 값을 얻어 오는 함수

<br/>

이번 실습은 **PyCharm**을 이용합니다.

<br/>

```
import cv2
import numpy as np

def change_color(x):
  r = cv2.getTrackbarPos("R", "Image")
  g = cv2.getTrackbarPos("G", "Image")
  b = cv2.getTrackbarPos("B", "Image")
image = np.zeros((600, 400, 3), np.uint8)
cv2.namedWindow("Image")
  image[:] = [b, g, r]
  cv2.imshow('Image', image)


cv2.createTrackbar("R", "Image", 0, 255, change_color)
cv2.createTrackbar("G", "Image", 0, 255, change_color)
cv2.createTrackbar("B", "Image", 0, 255, change_color)

cv2.imshow('Image', image)
cv2.waitKey(0)
```

[##_Image|kage@ESF4T/btqXXkmTATp/6Tt12T4Nm8lqrIz6jU2jfk/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

## 🔸 OpenCV 도형 그리기

OpenCV에서는 다양한 도형을 그릴 수 있습니다.

<br/>

### 직선 그리기

<br/>

> cv2.line(image, start, end, color, thickness) : 하나의 직선을 그리는 함수

-   start : 시작 좌표 (2차원)
-   end : 종료 좌표 (2차원)
-   thickness : 선의 두께

<br/>

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = np.full((512, 512, 3), 255, np.uint8)
image = cv2.line(image, (0, 0), (255, 255), (255, 0, 0), 3)

plt.imshow(image)
plt.show()
```

[##_Image|kage@wsc58/btqXY1z6P4B/2dNCfkmdYMtfLMtaMZ42lk/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

### 사각형 그리기

<br/>

> cv2.rectangle(image, start, end, color, thickness) : 하나의 사각형을 그리는 함수

-   start : 시작 좌표 (2차원)
-   end : 종료 좌표 (2차원)
-   thickness : 선의 두께 (채우기: -1)

<br/>

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = np.full((512, 512, 3), 255, np.uint8)
image = cv2.rectangle(image, (20, 20), (255, 255), (255, 0, 0), 3)

plt.imshow(image)
plt.show()
```

[##_Image|kage@oiI7o/btqXY1z6UqO/rckd5PJIflzjI9BA6ZV9G1/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

### 원 그리기

<br/>

> cv2.circle(image, center, radian, color, thickness) : 하나의 원을 그리는 함수

-   center : 원의 중심 (2차원)
-   radian : 반지름
-   thickness : 선의 두께 (채우기: -1)

<br/>

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = np.full((512, 512, 3), 255, np.uint8)
image = cv2.circle(image, (255, 255), 30, (255, 0, 0), 3)

plt.imshow(image)
plt.show()
```

[##_Image|kage@TWhvf/btqXU3TfQmR/o3p6GkQ2Y6EN7Jf9tXpujk/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

### 다각형 그리기

<br/>

> cv2.polylines(image, points, is_closed, color, thickness) : 하나의 다각형을 그리는 함수

-   points : 꼭지점들
-   is_closed : 닫힌 도형 여부
-   thickness : 선의 두께 (채우기: -1)

<br/>

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = np.full((512, 512, 3), 255, np.uint8)
points = np.array([[5, 5], [128, 258], [483, 444], [400, 150]])
image = cv2.polylines(image, [points], True, (0, 0, 255), 4)

plt.imshow(image)
plt.show()
```

[##_Image|kage@df3pYt/btqXQ4S7SGX/kCzUHpbgoZKDVdjUyX3VI1/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

### 텍스트 그리기

<br/>

> cv2.putText(image, text, position, font_type, font_scale, color) : 하나의 텍스트를 그리는 함수

-   position : 텍스트가 출력될 위치
-   font_type : 글씨체
-   font_scale : 글씨 크기 가중치

<br/>

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = np.full((512, 512, 3), 255, np.uint8)
image = cv2.putText(image, 'Hello World', (0, 200), cv2.FONT_ITALIC, 2, (255, 0, 0))

plt.imshow(image)
plt.show()
```

[##_Image|kage@pT8mH/btqXXjnXcdc/PSuKcMtONzGVwUClb0a681/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

## 📪 OpenCV Contours

**Contour**란 윤곽, 테두리을 말합니다.

### Contours 찾기

<br/>

> cv2.findContours(image, mode, method) : 이미지에서 Contour들을 찾는 함수

* mode : Contour들을 찾는 방법

1) RETR_EXTERNAL : 바깥쪽 Line만 찾기  
2) RETR_LIST : 모든 Line을 찾지만, Hierarchy 구성 X  
3) RETR_TREE : 모든 Line을 찾으며, 모든 Hierarchy 구성 O

* method : Contour들을 찾는 근사치 방법

1) CHAIN_APPROX_NONE : 모든 Contour 포인트 저장  
2) CHAIN_APPROX_SIMPLE : Contour Line을 그릴 수 있는 포인트만 저장

<br/>

입력 이미지는 Gray Scale Threshold 전처리 과정이 필요합니다.

<br/>

> cv2.drawContours(image, contours, contour_index, color, thickness) : Contour들을 그리는 함수

* contour_index : 그리고자 하는 Contours Line (전체: -1)

<br/>

```
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('gray_image.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(image_gray, 127, 255, 0)

plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
plt.show()

contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
image = cv2.drawContours(image, contours, -1, (0, 255, 0), 4)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
```

[##_Image|kage@ow1i9/btqXSnYX7vF/fqGsTXgR7LFkLkqzVoP5ok/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

## 📬 OpenCV Contours 처리

### Contour의 사각형 외각 찾기

<br/>

> cv2.boundingRect(contour): Contour를 포함하는 사각형을 그립니다.

-   사각형의 X, Y 좌표와 너비, 높이를 반환합니다.

<br/>

```
import cv2  
import matplotlib.pyplot as plt

image = cv2.imread('digit\_image.png')  
image\_gray = cv2.cvtColor(image, cv2.COLOR\_BGR2GRAY)  
ret, thresh = cv2.threshold(image\_gray, 230, 255, 0)  
thresh = cv2.bitwise\_not(thresh)

plt.imshow(cv2.cvtColor(thresh, cv2.COLOR\_GRAY2RGB))  
plt.show()

contours = cv2.findContours(thresh, cv2.RETR\_TREE, cv2.CHAIN\_APPROX\_SIMPLE)\[1\]  
image = cv2.drawContours(image, contours, -1, (0, 0, 255), 4)

plt.imshow(cv2.cvtColor(image, cv2.COLOR\_BGR2RGB))  
plt.show()

contour = contours\[0\]  
x, y, w, h = cv2.boundingRect(contour)  
image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

plt.imshow(cv2.cvtColor(image, cv2.COLOR\_BGR2RGB))  
plt.show()
```

[##_Image|kage@be5ryY/btqYhzi80ZT/wHbobng6xCKzF58Ktb7jP1/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

### Contour의 Convex Hull

<br/>

> cv2.convexHull(contour) : Convex Hull 알고리즘으로 외곽을 구하는 함수

-   대략적인 형태의 Contour 외곽을 빠르게 구할 수 있습니다. (단일 Contour 반환)

<br/>

```
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('digit_image.png')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(image_gray, 230, 255, 0)
thresh = cv2.bitwise_not(thresh)

contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
image = cv2.drawContours(image, contours, -1, (0, 0, 255), 4)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

contour = contours[0]
hull = cv2.convexHull(contour)
image = cv2.drawContours(image, [hull], -1, (255, 0, 0), 4)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
```

[##_Image|kage@b1xDJr/btqX1oD15tM/K5YX2MA7gnCQo746OyBFWK/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

### Contour의 유사 다각형 구하기

<br/>

> cv2.approxPolyDP(curve, epsilon, closed) : 근사치 Contour를 구합니다.

-   curve : Contour
-   epsilon : 최대 거리 (클수록 Point 개수 감소)
-   closed : 폐곡선 여부

<br/>

```
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('digit_image.png')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(image_gray, 230, 255, 0)
thresh = cv2.bitwise_not(thresh)

contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
image = cv2.drawContours(image, contours, -1, (0, 0, 255), 4)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

contour = contours[0]
epsilon = 0.01 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)
image = cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
```

[##_Image|kage@eJQhFN/btqX1pJG26Y/TM3kXjGeDox7TGrVghvegK/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

### Contour의 기본정보

<br/>

> cv2.contourArea(contour) : Contour의 면적을 구합니다.
>
> cv2.arcLength(contour) : Contour의 둘레를 구합니다.
>
> cv2.moments(contour) : Contour의 특징을 추출합니다.

<br/>

```
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('digit_image.png')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(image_gray, 230, 255, 0)
thresh = cv2.bitwise_not(thresh)

contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
image = cv2.drawContours(image, contours, -1, (0, 0, 255), 4)

contour = contours[0]
area = cv2.contourArea(contour)
print(area)

length = cv2.arcLength(contour, True)
print(length)

M = cv2.moments(contour)
print(M)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
```

9637.5  
1112.1046812534332  
{'m00': 9637.5, 'm10': 2328654.1666666665, 'm01': 525860.6666666666, 'm20': 592439950.25, 'm11': 125395340.54166666, 'm02': 32616659.75, 'm30': 157199366984.05002, 'm21': 31597487112.5, 'm12': 7677332730.433333, 'm03': 2223038890.5, 'mu20': 29780523.227014065, 'mu11': -1665373.5978347063, 'mu02': 3923591.96819859, 'mu30': -339915780.7390442, 'mu21': 76375946.41720533, 'mu12': -21905836.49518633, 'mu03': 15169233.760740757, 'nu20': 0.3206295471760697, 'nu11': -0.01793010748946005, 'nu02': 0.04224302932750429, 'nu30': -0.03727866486560947, 'nu21': 0.008376172780476334, 'nu12': -0.0024024196097321344, 'nu03': 0.001663614382378067}

[##_Image|kage@m9Yps/btqX6gS3VIC/eyaOcV3XR8rObkQqz3FFt1/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

## 💈 OpenCV Filtering

### 필터링이란?

<br/>

이미지에 커널을 적용하여 이미지를 흐리게(Blurring = Smoothing) 처리하는 것을 말합니다.
이러한 과정을 통해 이미지를 흐리게 만들면 노이즈 및 손상을 줄일 수 있습니다.

<br/>

### 컨볼루션 계산

<br/>

특정한 이미지에서 커널(Kernel)을 적용해 컨볼루션 계산하여 필터링을 수행할 수 있습니다.

<br/>

\[##\_Image|kage@bP1ouz/btqYgGpoiXB/SipOFemM5IszTHCiHNdLwk/img.png|alignCenter|width="100%"|\_##

[##_Image|kage@bdtvd2/btqX32U17GU/zgTfsvxEzE34kb5R0I5y8k/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

### 직접 커널을 생성하여 필터 적용하기

<br/>

```
import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('gray_image.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

size = 4
kernel = np.ones((size, size), np.float32) / (size ** 2)
print(kernel)

dst = cv2.filter2D(image, -1, kernel)
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.show()
```

[##_Image|kage@b2n9Mp/btqX32AF7lj/z88Fsunzmv9g3xLOPVN8MK/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]


\[\[0.0625 0.0625 0.0625 0.0625\]  
\[0.0625 0.0625 0.0625 0.0625\]  
\[0.0625 0.0625 0.0625 0.0625\]  
\[0.0625 0.0625 0.0625 0.0625\]\]

[##_Image|kage@NFoai/btqYfXkANnL/CemAIjyboHy9KMPvx9R6X1/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

### Basic Blurring

<br/>

```
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('gray_image.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

dst = cv2.blur(image, (4, 4))
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.show()
```

[##_Image|kage@DzTMn/btqYgGixBY1/n2wX61wqUTw9ueyzvSmkY0/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

### Gaussian Blur

<br/>

```
import cv2

image = cv2.imread('gray_image.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

# kernel_size: 홀수
dst = cv2.GaussianBlur(image, (5, 5), 0)
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.show()
```

[##_Image|kage@bHIGmp/btqYhzpVtpI/w2zS4aWdUAmndcfx3ZfQvK/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

이제까지 OpenCV의 기본적인 이미지처리에 대해 알아보았습니다!  
다음은 OpenCV에서 간단한 머신러닝 알고리즘인 KNN이 어떻게 적용되는지 살펴보겠습니다.

<br/>

# 🧮 KNN Algorithm

## KNN(K-Nearest Neighbor)이란?

KNN은 머신러닝 알고리즘으로  
비지도학습(Unsupervised Learning)의 가장 간단한 예시입니다. 다양한 레이블의 데이터 중에서, 자신과 가까운 데이터를 찾아 자신의 레이블을 결정하는 방식입니다.

<br/>

## 예시

<br/>

-   K가 3일 때, 주위에 빨간색 2개, 파란색 1개를 찾아서 별 모양을 빨간색으로 결정하게 됩니다.

<br/>

[##_Image|kage@vUvYh/btqYiCmpaxv/aAtMklEBbdK92F45PTkcm0/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

```
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 각 데이터의 위치: 25 X 2 크기에 각각 0 ~ 100
trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)
# 각 데이터는 0 or 1
response = np.random.randint(0, 2, (25, 1)).astype(np.float32)

# 값이 0인 데이터를 각각 (x, y) 위치에 빨간색으로 칠합니다.
red = trainData[response.ravel() == 0]
plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')
# 값이 1인 데이터를 각각 (x, y) 위치에 파란색으로 칠합니다.
blue = trainData[response.ravel() == 1]
plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')

# (0 ~ 100, 0 ~ 100) 위치의 데이터를 하나 생성해 칠합니다.
newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)
plt.scatter(newcomer[:, 0], newcomer[:, 1], 80, 'g', 'o')

knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, response)
ret, results, neighbours, dist = knn.findNearest(newcomer, 3)

# 가까운 3개를 찾고, 거리를 고려하여 자신을 정합니다.
print("result : ", results)
print("neighbours :", neighbours)
print("distance: ", dist)
plt.show()
```

result : \[\[1.\]\]  
neighbours : \[\[1. 0. 1.\]\]  
distance: \[\[ 25. 97. 148.\]\]

[##_Image|kage@cgyakI/btqX32Hs9dv/mahTlSY5hgMJZDFViAqQak/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]

<br/>

## KNN 숫자 인식 예제

### 숫자 이미지 분류하여 저장하기

<br/>

```
import cv2
import numpy as np

img = cv2.imread('digits.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 세로로 50줄, 가로로 100줄로 사진을 나눕니다.
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
x = np.array(cells)

# 각 (20 X 20) 크기의 사진을 한 줄(1 X 400)으로 바꿉니다.
train = x[:, :].reshape(-1, 400).astype(np.float32)

# 0이 500개, 1이 500개, ... 로 총 5,000개가 들어가는 (1 x 5000) 배열을 만듭니다.
k = np.arange(10)
train_labels = np.repeat(k, 500)[:, np.newaxis]

np.savez("trained.npz", train=train, train_labels=train_labels)
```

<br/>

```
import matplotlib.pyplot as plt

# 다음과 같이 하나씩 글자를 출력할 수 있습니다.
plt.imshow(cv2.cvtColor(x[0, 0], cv2.COLOR_GRAY2RGB))
plt.show()

# 다음과 같이 하나씩 글자를 저장할 수 있습니다.
cv2.imwrite('test_0.png', x[0, 0])
cv2.imwrite('test_1.png', x[5, 0])
cv2.imwrite('test_2.png', x[10, 0])
cv2.imwrite('test_3.png', x[15, 0])
cv2.imwrite('test_4.png', x[20, 0])
cv2.imwrite('test_5.png', x[25, 0])
cv2.imwrite('test_6.png', x[30, 0])
cv2.imwrite('test_7.png', x[35, 0])
cv2.imwrite('test_8.png', x[40, 0])
cv2.imwrite('test_9.png', x[45, 0])
```

[##_Image|kage@T6LOG/btqYfWy9hf6/jJlI2Kk87JaeGXCKEdTwQ1/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]


Out\[43\]: True

<br/>

## KNN 숫자 인식

<br/>

```
import cv2
import numpy as np
import glob

FILE_NAME = 'trained.npz'

# 파일로부터 학습 데이터를 불러옵니다.
def load_train_data(file_name):
  with np.load(file_name) as data:
    train = data['train']
    train_labels = data['train_labels']
  return train, train_labels

# 손 글씨 이미지를 (20 x 20) 크기로 Scaling합니다.
def resize20(image):
  img = cv2.imread(image)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_resize = cv2.resize(gray, (20, 20))
  plt.imshow(cv2.cvtColor(gray_resize, cv2.COLOR_GRAY2RGB))
  plt.show()
  # 최종적으로는 (1 x 400) 크기로 반환합니다.
  return gray_resize.reshape(-1, 400).astype(np.float32)

def check(test, train, train_labels):
  knn = cv2.ml.KNearest_create()
  knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
  # 가장 가까운 5개의 글자를 찾아, 어떤 숫자에 해당하는지 찾습니다.
  ret, result, neighbours, dist = knn.findNearest(test, k=5)
  return result

train, train_labels = load_train_data(FILE_NAME)

for file_name in glob.glob('./test_*.png'):
  test = resize20(file_name)
  result = check(test, train, train_labels)
  print(result)
```

[##_Image|kage@m1rCH/btqX821eI1z/bnr2BQKPYduoYzT8l0CLN1/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]


\[\[5.\]\]

[##_Image|kage@bHzIs1/btqYhzDvu5Z/DVP9sKIZtMhPVt9a22a9wK/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]


\[\[3.\]\]

[##_Image|kage@cscDZ6/btqYebi2bYI/9eh8ciNuYUdWzypnesMKZk/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]


\[\[6.\]\]

[##_Image|kage@SkrXJ/btqX321KNaZ/bbgv0zQQU95wCqAmkvdEKK/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]


\[\[8.\]\]

[##_Image|kage@cPIxzu/btqX1oD28T6/j2lv1SwYcL970Tv285UMVK/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]


\[\[4.\]\]

[##_Image|kage@bOGjEf/btqYbH3BQrR/EY8wSwu2xQfFmK4bh86YT0/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]


\[\[2.\]\]

[##_Image|kage@rRxjF/btqYeaYOucu/BZk3KVQKEJKhkfrIdUFCT1/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]


\[\[7.\]\]

[##_Image|kage@KmcKX/btqYiBHO0Gf/HP5g6aTum62WLLAFvF10Ok/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]


\[\[0.\]\]

[##_Image|kage@KiqfR/btqX32HtgMJ/wTCyDciY39PAGmZ0kwt27k/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]


\[\[1.\]\]

[##_Image|kage@lMz0r/btqYgFRsase/LuSQDNYKhW9kdbTiKBlCc0/img.png|alignCenter|width="100%" data-origin-width="0" data-origin-height="0" data-ke-mobilestyle="widthContent"|||_##]


\[\[9.\]\]

<br/>

# 🤝 마치며

이상 Python을 이용한 데이터 분석과 이미지 처리에 대해 알아보았습니다.
이 포스트 내용 이외에 더 궁금하신 분은 아래의 글을 참고해주시길 바랍니다.

<br/>

### Reference

[https://www.youtube.com/watch?v=F2FRpmh9sQo&list=PLRx0vPvlEmdBx9X5xSgcEk4CEbzEiws8C&index=7](https://www.youtube.com/watch?v=F2FRpmh9sQo&list=PLRx0vPvlEmdBx9X5xSgcEk4CEbzEiws8C&index=7)

<br/>

### 글쓴이

DSC Yonsei 오세빈

E-mail:[osb3372@yonsei.ac.kr](http://osb3372@yonsei.ac.kr/)
