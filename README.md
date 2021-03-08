## DSC Yonsei Tech Blog 필독 해주세요!!

#### 글 작성 방법
1. _posts 폴더에서 다음과 같은 이름으로 markdown 폴더를 하나 만듭니다.
    * <날짜>-<제목>-<본인이름영어>.md
    * ex) 2021-03-08-OptimalPolicy-jeongmin.md
2. 설정을 아래와 같이 합니다.
~~~
---
published : true # 건드리지 마세요
layout : posts # 건드리지 마세요
title: "[ML Research]optimal policy의 수학적 증명과 Monte Carlo method" # [category 명]제목  <= 해당 양식에 맞춰주세요
date: 2020-02-24 # 날짜를 적어주세요
excerpt: "Q-learning에서 Markov Decision Process, Bellman equation을 통해 optimal policy를 증명하고 MC 방법에 대해 알아본다." # 요약을 적어주세요
comments: true # 건드리지 마세요
use_math: true # 건드리지 마세요
author : JeongMin Do # 본인 프로필 이름을 입력해주세요
---
~~~

 *  Example

~~~
---
published : true
layout : posts
title: "[ML Research]optimal policy의 수학적 증명과 Monte Carlo method"
date: 2020-02-24
excerpt: "Q-learning에서 Markov Decision Process, Bellman equation을 통해 optimal policy를 증명하고 MC 방법에 대해 알아본다."
comments: true
use_math: true
author : JeongMin Do
---
~~~


#### 사진 넣기
마크다운에 사진을 넣고 싶으면, 아래와 같이 해주세요.

1. /assets/ 현재 제목을 입력하고 이미지 파일을 넣어주세요.
2. 마크 다운에서 아래와 같은 명령어를 입력해주세요.

~~~
[!image](/assets/<제목>/<이미지명>)
~~~

* Example
~~~
[!image](/assets/OptimalPolicy/image1.png)
~~~


#### 마크다운 사용법
[이 링크를 참조하여 글을 작성해주세요](https://gist.github.com/danggai/7c2ba1c5e3f923e85411fb740bf514fa)


## ISSUE
* 수식 중 "$$", "$$" 가 제대로 작동하지 않습니다. "$", "$"를 사용해주세요. 원인을 파악중입니다.
