---
published : true
title: "[ML Engineering]AWS SageMaker에서 pre-built된 모델 이용하기"
date: 2021-02-04
excerpt: "AWS SageMaker에서 pre-built된 모델을 이용하여 머신러닝 작업을 수행하는 것을 다룹니다.."
use_math: true
categories:
  - MLEngineering
author : YeongHyun Park 
---

## AWS SageMaker에서 노트북 인스턴스 생성하는 방법

1.  AWS SageMaker 창 좌측에 위치한 노트북 인스턴스를 통해 인스턴스 창으로 이동합니다.
2.  창 우측에 위치한 '노트북 인스턴스 생성' 을 누르고 이동한 창에서 노트북 인스턴스 이름을 지정해줍니다.
3.  노트북 인스턴스 유형은 과금을 피하기 위해 ml.t2.medium 또는 ml.t3.medium 인스턴스를 이용합니다.  
    AWS 프리티어 계정의 경우 서버 및 시간에 따른 AWS SageMaker 요금은 [다음 페이지](https://aws.amazon.com/ko/sagemaker/pricing/)에서 확인하실 수 있습니다.
4.  권한 및 암호화 부분에서 IAM 역할을 '새 역할 생성'-'지정하는 S3 버킷 : 모든 S3 버킷' 으로 선택한 후, 역할을 생성합니다.
5.  Git 리포지토리를 연결하려면 미리 Git 리포지토리에 저장해놓은 것을 연결하거나, 노트북 인스턴스를 먼저 만든 후에 Jupyter 창에서 연결할 수 있습니다.
6.  선택이 완료되었다면 노트북 인스턴스를 생성하면 됩니다.

---

#### 1\. DATA load 및 Library import

```
    data_bucket = 'bucket_name'     # bucket_name 부분을 Amazon s3 에서 사용할 버킷이름으로 변경
    subfolder = 'ch03' #파일 위치 지정해줄 지시어
    dataset = 'churn_data.csv' #파일 이름 지정

    import sys
    import pandas as pd
    from time import sleep
    import boto3

    import sagemaker
    import s3fs       #s3버킷을 aws sagemaker 내에서 열 수 있게 해주는 시스템
    from sklearn.model_selection import train_test_split

    role = sagemaker.get_execution_role()
    s3 = s3fs.S3FileSystem(anon=False)



```

role과 s3fs를 통해서 s3를 설정해줘야 합니다.  
s3fs는 s3를 filesystem 처럼 사용할 수 있게하는 패키지입니다.  
s3fs를 통해 s3를 mount 하는 것에 대한 설명은 [다음 페이지](https://bluese05.tistory.com/22)에서 자세히 확인하실 수 있습니다.

---

#### SageMaker와 Python SDK 버전 확인

```
    if int(sagemaker.__version__.split('.')[0]) == 2:
        print("Version is good")
    else:
        !{sys.executable} -m pip install --upgrade sagemaker
        print("Installing latest SageMaker Version. Please restart the kernel")

```

SageMaker와 Python SDK 버전이 서로 지원하는 버전인지 알 수 있는 코드입니다.  
버전이 맞다면 "Version is good"이 출력되고,

그렇지 않다면 "Installing latest SageMaker Version. Please restart the kernel"이 출력됩니다.

```
    df = pd.read_csv(f's3://{data_bucket}/{subfolder}/{dataset}')
    df.head()

    print(f'Number of rows in dataset: {df.shape[0]}')
    print(df['churned'].value_counts())

```

---

![image](https://user-images.githubusercontent.com/72022988/106865666-d410b580-670e-11eb-88e0-74b5627738ad.png)

s3 버킷에서 csv dataset을 dataframe으로 불러오는 코드입니다.  
s3에 지정한 버킷이 존재해야 불러올 수 있습니다.  
에러 메시지가 뜬다면 AWS Service 중 s3 속 지정한 버킷안에 데이터가 있는지 확인해볼 필요가 있습니다.

---

#### 2\. data 인코딩을 위해 원하는 모양으로 전환

```
    columns = df.columns.tolist()
    encoded_data = df.drop(['id', 'customer_code', 'co_name'], axis=1)
    encoded_data.head()


```

---

![image](https://user-images.githubusercontent.com/72022988/106865694-de32b400-670e-11eb-91f3-85c930a1fcee.png)

---

#### 3\. training 및 test 용 dataset 생성

```
    #전체 데이터셋에서 30% 를 검증 및 테스트용으로 사용하기 위해 데이터 분할하기
    #train_test_spilt 함수는 데이터를 분할할 때 사용

    y = encoded_data['churned']
    train_df, test_and_val_data, _, _ = train_test_split(
        encoded_data, 
        y, 
        test_size=0.3, 
        stratify=y, 
        random_state=0)   

        #sample을 추출하여 학습용과 검증 및 테스트용 데이터셋에 할당

    y = test_and_val_data['churned']
    val_df, test_df, _, _ = train_test_split(
        test_and_val_data, 
        y, 
        test_size=0.333, 
        stratify=y, 
        random_state=0)  

        #테스트 및 검증용 데이터를 다시 분할하여 테스트용 데이터셋과 검증용 데이터셋을 생성

    print(train_df.shape, val_df.shape, test_df.shape)
    print()
    print('Number of rows in Train dataset: {train_df.shape[0]}')
    print(train_df['churned'].value_counts())
    print()
    print('Number of rows in Validate dataset: {val_df.shape[0]}')
    print(val_df['churned'].value_counts())
    print()
    print('Number of rows in Test dataset: {test_df.shape[0]}')
    print(test_df['churned'].value_counts())


```

---

![image](https://user-images.githubusercontent.com/72022988/106865703-e2f76800-670e-11eb-89aa-ecca0cdf4535.png)

```
    #데이터셋을 csv로 변환하고 s3에 저장

    train_data = train_df.to_csv(None, header=False, index=False).encode()
    val_data = val_df.to_csv(None, header=False, index=False).encode()
    test_data = test_df.to_csv(None, header=True, index=False).encode()



    with s3.open(f'{data_bucket}/{subfolder}/processed/train.csv', 'wb') as f:
        f.write(train_data)

    with s3.open(f'{data_bucket}/{subfolder}/processed/val.csv', 'wb') as f:
        f.write(val_data) 

    with s3.open(f'{data_bucket}/{subfolder}/processed/test.csv', 'wb') as f:
        f.write(test_data) 

    train_input = sagemaker.TrainingInput(s3_data=f's3://{data_bucket}/{subfolder}/processed/train.csv', content_type='csv')
    val_input = sagemaker.TrainingInput(s3_data=f's3://{data_bucket}/{subfolder}/processed/val.csv', content_type='csv')  

```

---

#### 4\. 머신러닝 모델 학습

```
    sess = sagemaker.Session()
    region_name = boto3.Session().region_name

    container = sagemaker.image_uris.retrieve(
                    framework = 'xgboost',
                    rdgion = region_name,
                    version = 'latest')

    estimator = sagemaker.estimator.Estimator(
                            image_uri = container, 
                            role = role,
                            instance_count=1, 
                            instance_type='ml.m4.xlarge',    #sagemaker가 모델을 실행할 때 사용하는 서버 유형(인스턴스)를 설정
                            output_path=f's3://{data_bucket}/{subfolder}/output',    #결과를 출력할 s3 위치를 지정
                            sagemaker_session=sess)

    estimator.set_hyperparameters(
                            max_depth=3,
                            subsample=0.7,
                            objective='binary:logistic',
                            eval_metric='auc',
                            num_round=100,   #학습의 반복수를 지정
                            early_stopping_rounds=10,    #학습에 진전이 없을 때 중지시킬 학습 반복 수
                            scale_pos_weight=17)    #양의 가중치 지정

    #model 배포하기
    estimator.fit({'train': train_input, 'validation': val_input})


```

estimator.fit 함수를 통해 모델을 배포할 수 있습니다.  
estimator 함수에 대한 구체적인 설명은 [다음 페이지](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html)에서 확인하실 수 있습니다.

---

#### 5\. model 실행하기

앞서 학습모델을 생성했다면, 학습된 모델을 호스팅(데이터를 입력받아 추론을 수행한 뒤 결과를 반환하는 서버를 설정)할 수 있습니다.

```
    endpoint_name = 'customer-churn'

    try:
        sess.delete_endpoint(endpoint_name)
        print('Warning: Existing endpoint deleted to make way for your new endpoint.')
        sleep(30)
    except:
        pass


    predictor = estimator.deploy(
                    initial_instance_count=1,
                    instance_type='ml.m4.xlarge',     #결과 반환 서버 설정
                    endpoint_name=endpoint_name)



    # 데이터를 CSV 포맷 문자열로 직렬화
    from sagemaker.serializers import CSVSerializer
    predictor.serializer = CSVSerializer()

```

---

#### 6\. Test the model

모델의 엔드포인트를 설정하고 호스팅하였다면 의사결정을 하는 추론 서비스를 할 수 있습니다.

```
    def get_prediction(row):
        prob = float(predictor.predict(row[1:]).decode('utf-8'))
        return 1 if prob > 0.5 else 0

    with s3.open(f'{data_bucket}/{subfolder}/processed/test.csv') as f:
        test_data = pd.read_csv(f)

    test_data['prediction'] = test_data.apply(get_prediction, axis=1)
    test_data[:10]
```

---

![image](https://user-images.githubusercontent.com/72022988/106865722-e7238580-670e-11eb-8581-7a4f39cdead1.png)

```
    print(test_data['churned'].value_counts())
    print(test_data['prediction'].value_counts())
    print(metrics.accuracy_score(test_data['churned'],test_data['prediction']))



    print(metrics.confusion_matrix(test_data['churned'],test_data['prediction']))


    y = [1,0,0,0,0,0,0,0,0,2]
    pred = [0,0,0,0,0,0,0,0,1,2]
    print(metrics.confusion_matrix(y,pred))

```

---

#### Remove the Endpoint

```
    sess.delete_endpoint(endpoint_name)

```

---

#### AWS SageMaker 학습후기

\-[비즈니스 머신러닝](http://www.yes24.com/Product/Goods/96433954?OzSrank=1)이라는 책에 대한 예제를 진행했는데, SageMaker서버 업데이트에 따라 최근에 다시 코드도 업데이트 되었는데 s3와 role을 지정하는 부분이 누락되어 오류가 났었다. 이전 버전 코드를 찾아 누락된 부분을 작성하였더니 결과가 정상적으로 출력되었다.  
\-s3에 데이터를 넣는 방법을 모르고 코드를 실행했을 때 데이터를 찾을 수 없다는 에러가 자꾸 떴는데, AWS s3 버킷에 데이터를 저장하였더니 해결되었다.  
\-aws 콘솔에서 서비스를 프리티어 신분으로 이용하는 것이기 때문에 노트북 인스턴스 및 머신러닝 모델링 과정에서 엔드포인트를 지정하는 서버를 이용하는 시간이 정해져있는 것을 항상 생각해야 한다.

### 참조

* [AWS sagemaker document](https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#)  
* [ml4biz/ch03 예제](https://github.com/K9Ns/ml4biz)
