---
published : true
layout : posts
title: "[ML Research]파이썬과 OpenCV를 이용한 컴퓨터 비전 학습(1)"
date: 2021-04-06
excerpt: "OpenCV를 이용한 컴퓨터 비전의 간단한 기능에 대한 포스트입니다."
comments: true
use_math: true
author : SeBin Oh
---

<br/>

안녕하세요~ 여러분!  
다들 반가워요. 🤗  

<br/>

오늘은 OpenCV를 이용한 컴퓨터 비전의 간단한 기능에 대해 살펴보려고 합니다! ✍

<br/>

본 글은 **'파이썬과 OpenCV를 이용한 컴퓨터 비전 학습'** 서적을 참고하여 작성하였으며,  
여러 포스트로 나누어 소개드릴 예정입니다. 

<br/>

제가 앞으로 소개해드릴 내용 이외에 자세한 내용이 궁금하신 분은 공식 Document 또는 서적을 읽어보시길 바랍니다. 

---

<br/>

이번 포스트의 주제는 <u> *입출력과 GUI* </u> 입니다.   

입출력 기능과 관련된 기본적인 OpenCV의 기능을 설명해드리고 다양한 소스에서 이미지를 가져와 표시하고 이미지와 비디오를 저장하는 방법, 나아가 OpenCV UI 시스템에 대해 배워보도록 하겠습니다. 🔥

<br/>

이제 하나씩 살펴보시죠! 

<br/>

## **입출력과 GUI**  📱  

<br/>

우선 이미지를 읽어오는 것부터 시작해야겠죠?

<br/>

> ### 파일에서 이미지 읽어오기
---
<br/>

OpenCV에서는 PNG, JPEG, TIFF 등 다양한 형식의 이미지 읽기를 지원를 하고 있습니다.  
그래서 다음 함수를 통해 이미지를 불러오게 됩니다.

<br/>

* **cv2.imread** : 이미지의 경로와 선택적 플래그를 받는 함수

<br/>

예시> 이미지 읽기

```
img = cv2.imread(params.path)
```

<br/>

이미지를 불러왔으면 다음은 간단한 이미지를 처리할 시간입니다.

<br/>

> ### 간단한 이미지 변환: 크기 조절 및 뒤집기
---
<br/>

이번에 설명해드리는 크기 조절과 뒤집기는 일반적으로 복잡한 컴퓨터 비전 알고리즘의 예비단계에 사용됩니다.  

먼저 이미지 크기를 조절하는 함수입니다.

<br/>

* **cv2.resize** : 이미지 크기 조절 함수

<br/>

예시> 두 번째 매개변수로 대상 크기를 픽셀 단위로 설정

```
width, height = 128, 256
resized_img = cv2.resize(img, (width, height))
print('resized to 128x256 image shape: ', resized_img.shape)
```

<br/>

예시> 기본 값 대신 최근접 이웃 보간법을 사용해 크기를 조절

```
w_mult, h_mult = 2, 4 
resized_img = cv2.resize(img, (0, 0), resized_img, w_mult, h_mult, cv2.INTER_NEAREST)
print('half sized image shape: ', resized_img.shape)
```

<br/>

다음은 이미지를 뒤집는 함수입니다.  

<br/>

* **cv2.flip** : 이미지를 뒤집는 함수

<br/>

이 함수는 이미지 크기를 변경하지 않고
픽셀을 바꾸기만 합니다.

<br/>

예시> cv2.flip 함수 마지막 인자로 0을 전달해 수평인 x 축을 따라 이미지 뒤집기

```
img_flip_along_x = cv2.flip(img, 0)
```

<br/>

이제 이미지를 저장하는 방법을 알려드리겠습니다.

<br/>

> ### 손실 및 무손실 압축을 사용한 이미지 저장
---
<br/>

이미지 결과를 디스크에 저장하는 이유는 컴퓨터 비전 알고리즘으로부터 피드백을 받기 위함입니다.  
피드백은 윤곽선, 메트릭 등과 같은 추가 정보가 있는 이미지이거나 복잡한 파이프라인의 개별 단계의 결과일 수 있습니다.  

<br/>

다음은 이미지를 저장하는 함수입니다.

<br/>

* **cv2.imwirte** : 출력파일의 경로, 이미지, 저장 매개변수 인자를 받아서 이미지를 저장하는 함수

<br/>

이 때 파일의 형식은 함수에 의해 결정됩니다. 이미지를 저장하는 두 가지 옵션으로 일부 정보의 유실 여부를 결정할 수 있습니다.

<br/>

1) **IMWRITE\_PNG\_COMPRESSION**

- PNG 형식으로 압축 수준을 지정할 수 있습니다.
- (0, 9) 사이의 값을 갖습니다.
- 숫자가 클수록 디스크의 파일 크기는 작지만 디코딩 프로세스가 느려집니다.

<br/>

1) **IMWRITE\_JPEG\_QUALITY**

- JPEG 형식으로 압축 프로세스를 관리할 수 있습니다.
- 0 ~ 100 사이의 값을 설정할 수 있습니다.
- 큰 값일수록 결과의 품질이 좋아지고 JPEG 아티팩트(artifact)양이 감소합니다.

<br/>

예시> 낮은 압축률로 이미지 저장 - 파일 크기는 크지만 디코딩 신속

```
cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
```

<br/>

다음은 이미지를 보여주는 방법을 다룹니다.

<br/>

> ### OpenCV 창에 이미지 표시
---
<br/>

OpenCV의 뛰어난 기능 중 하나는 작은 노력으로 이미지를 시각화할 수 있는다는 점입니다.

<br/>

다음 함수를 사용하여 이미지를 보여주게 됩니다.

<br/>

* **cv2.imshow** : 이미지를 보여주는 함수

* **cv2.waitKey** : 창(window)의 표시 시간을 제어하는 함수

<br/>

예시> 이미지 표시를 위해 cv2.imshow와 cv2.waitKey 함수 호출

```
cv2.imshow("Original image", img)
cv2.waitKey(2000)
```

<br/>

> ### OpenCV 창에서 버튼 및 탐색바와 같은 UI 요소로 작업
---
<br/>

이번에는 OpenCV 창에 UI 요소를 추가하는 방법을 알려드리겠습니다.

<br/>

그 중 트랙바는 다음과 같은 경우 유용한 UI입니다. 

* 값이 정의된 범위 내에 있다고 가정한 정수 변수의 값 표시
* 탐색 바 위치 변경을 통한 반응형의 값 변경하기

<br/>

예시> 이미지의 채우기 색상을 지정하는 프로그램 중

~~~
def trackbar_callback(idx, value):
       fill_val[idx] = value

cv2.createTrackbar('R', 'window', 255, 255, lambda v:
      trackbar_callback(2, v))
cv2.createTrackbar('G', 'window', 255, 255, lambda v:
      trackbar_callback(1, v))
cv2.createTrackbar('B', 'window', 255, 255, lambda v:
      trackbar_callback(0, v))
~~~

<br/>

> ### 2D 프리미티브 그리기: 마커, 선, 타원, 사각형 및 테스트
---
<br/>

OpenCV에는 이미지의 모든 기능을 강조할 수 있는 수많은 그리기 기능이 있습니다.

<br/>

그래서 원, 선, 화살표, 사각형, 타원, 텍스트을 그리는 함수에 대해 소개해드리겠습니다.

<br/>

* **cv2.circle** : 원을 그리는 함수

* **cv2.line** : 선을 그리는 함수

* **cv2.arrowedLine** : 화살표를 그리는 함수

* **cv2.rectangle** : 사각형을 그리는 함수

* **cv2.ellipse** : 타원을 그리는 함수

* **cv2.putText** : 텍스트를 표시하는 함수

<br/>

예시> 텍스트 표시
~~~
cv2.putText(image, 'OpenCV', rand_pt(), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
~~~

<br/>

> ### 사용자의 키보드 입력 처리
---
<br/>

OpenCV에는 키보드 입력을 간단하고 명확하게 처리하는 방법을 제공하기도 합니다.  
이러한 기능은 **cv2.waitKey** 함수에 내장되어 있습니다.

예시> 
~~~
finish = False
while not finish:
    cv2.imshow("result", image_to_show)
    key = cv2.waitKey(0)
    if key == ord('p'):
        for pt in [rand_pt() for _ in range(10)]:
            cv2.circle(image_to_show, pt, 3, (255, 0, 0), -1)
    elif key == ord('t'):
        image_to_show = np.copy(image)
    elif key == 27:
        finish = True            
~~~

<br/>

> ### 비디오 프레임 스트림 재생
---
<br/>

이번에는 비디오 파일을 여는 방법과 열린 비디오의 프레임을 다시 재생하는 방법입니다.  

<br/>

다음 예시는 **cv2.VideoCapture** 클래스에서 수행되었으며, while 무한 루프에서 비디오 파일을 열고 **capture.read** 함수를 통해 프레임을 가져옵니다.  
이 함수는 불리언 값과 프레임으로 구성된 페어를 반환합니다. cv2.imshow 함수를 호출하고 cv2.waitKey 함수에서 50ms을 기다리는데, 이미지를 표시하고 비디오를 디코딩하는 데 소용되는 시간을 무시할 수 있다고 가정하면 비디오는 20FPS 이하의 속도로 재생될 수 있음을 알 수 있습니다.

<br/>

예시>
~~~
capture = cv2.VideoCapture(path)

while True:
    has_frame, frame = capture.read()
    if not has_frame:
        print('Reached the end of the video')
        break

    cv2.imshow('frame', frame)
    key = cv2.waitKey(50)
    if key == 27:
        print('Pressed Esc')
        break

    cv2.destroyAllWindows()        
~~~

<br/>

### 🤝 마치며

이상 OpenCV에서 다루는 <입출력과 GUI>에 대해 간략히 알아보았습니다.
다들 이 글을 읽어주셔서 감사하고 다음 포스트도 기대해주세요~ 🎉

<br/>

### 글쓴이

DSC Yonsei 오세빈

E-mail: [osb3372@yonsei.ac.kr](http://osb3372@yonsei.ac.kr)