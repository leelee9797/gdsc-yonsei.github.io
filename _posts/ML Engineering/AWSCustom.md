# amazon-sagemaker-custom-container

해당 글은 AWS SageMaker에서 custom model을 docker container로 빌드하고 이것을 ECR에 푸쉬하여 사용하는 것을 다룹니다.

[https://github.com/aws-samples/amazon-sagemaker-custom-container](https://github.com/aws-samples/amazon-sagemaker-custom-container) 해당 링크의 원 글에서 코드를 적절하게 수정하여 재배포하였습니다.

1.  노트북 인스턴스 생성
2.  git clone
3.  스크립트 실행
4.  모델 생성 & 배포

## 노트북 인스턴스 생성

노트북 인스턴스를 생성할 때, 볼륨 크기를 50GB로 변경해서 생성합니다.

InService상태가 되면 jupyterLab을 실행하고, terminal을 실행해야 합니다.  
커서가 깜빡인다면 여기까지 바르게 잘 수행한 것.

## git clone

아래 명령어를 차례로 입력하여 해당 레포를 clone 하도록 합니다.

```
# 클론
git clone https://www.github.com/silverstar0727/amazon-sagemaker-custom-container
# 클론 한 디렉토리로 위치를 변경하자.
cd amazon-sagemaker-custom-container
```

## 스크립트 실행

아래의 명령어를 통해 스크립트를 실행합니다. 이때 뒤에 오는 'test' 명령어는 ecr에 생성되는 이미지 이름으로 본인에 맞게 변경이 가능합니다.

```
# 스크립트를 실행이 가능하도록 설정
chmod +x build-and-push.sh
# 스크립트 실행
./build-and-push.sh test
```

## 모델 생성 & 배포

1.  model 버튼을 아래 그림과 같이 클릭합니다.

![image](https://user-images.githubusercontent.com/49096513/108234073-c8e66c80-7187-11eb-83cb-7bf83f418590.png)  
![image](https://user-images.githubusercontent.com/49096513/108234102-d1d73e00-7187-11eb-95d9-38522f584549.png)

2.  새로운 IAM role을 만들고 Image와 s3 bucket에서 모델파일을 가져옵니다.

![image](https://user-images.githubusercontent.com/49096513/108234118-d69bf200-7187-11eb-8f23-516d958f0e90.png)  
![image](https://user-images.githubusercontent.com/49096513/108234135-dbf93c80-7187-11eb-91e6-40e05394161e.png)

## 엔드포인트 구성

1.  SageMaker에서 Endpoint 구성에 들어갑니다.
    
2.  Endpoint 구성에서 새 구성을 생성합니다.
    

![image](https://user-images.githubusercontent.com/49096513/108234181-e582a480-7187-11eb-9bb5-ba5bdf522823.png)  
![image](https://user-images.githubusercontent.com/49096513/108234208-eb788580-7187-11eb-9ae3-48366fe6f869.png)  
![image](https://user-images.githubusercontent.com/49096513/108234237-f206fd00-7187-11eb-9536-9f129de42592.png)  
![image](https://user-images.githubusercontent.com/49096513/108234263-f7644780-7187-11eb-99f6-2b3a20e65d65.png)

## 엔드포인트 생성

1.  SageMaker에서 엔드포인트에 들어갑니다.
    
2.  엔드포인트 생성을 아래의 그림과 같이 진행하면 됩니다.
    

![image](https://user-images.githubusercontent.com/49096513/108234293-fd5a2880-7187-11eb-9748-f1237ea1c39d.png)  
![image](https://user-images.githubusercontent.com/49096513/108234303-021edc80-7188-11eb-83c2-a4acaeb242af.png)
