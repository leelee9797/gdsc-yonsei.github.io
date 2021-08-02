---
published : true
title: "[ML Engineering]Azure에서 ML training, deploy 기초"
date: 2021-03-04
excerpt: "Azure에서의 ML을 training, deploy 과정에 대한 기초에 대해 소개하는 포스트입니다."
use_math: true
categories:
  - MLEngineering
author : YeongHyun Park
---

> Jupyter notebook에서 Azure Machine Learning 작업 영역을 만드는 방법은 [다음페이지](https://docs.microsoft.com/ko-kr/azure/machine-learning/tutorial-1st-experiment-sdk-setup) 에서 확인하실 수 있습니다.

MNIST : 글자 구분하는 데이터

### 앞서 Azure에서 Jupyter notebook 작업 영역을 만들어줬다면 개발 환경을 설정해줘야 합니다.

---

```
import azureml.core
from azureml.core import Workspace
```

다음과 같은 코드를 입력하여 azure 패키지를 import 하고 workspace를 지정해줍니다.

---

작업 영역에서 config.json 파일을 ws 개체에 세부정보를 로드해줍니다.

```
ws = Workspace.from_config()
```

---

### 작업 영역 내에서 실행 추적하는 실험을 만듭니다.

```
from azureml.core import Experiment
experiment_name = 'Tutorial-sklearn-mnist'

exp = Experiment(workspace=ws, name=experiment_name)
```

---

컴퓨팅 대상을 만들거나 이미 컴퓨팅 리소스가 있을 때 해당 리소스를 사용합니다.  
(여기서 만드는 컴퓨팅 리소스는 모델 훈련 시 사용하는 컴퓨팅입니다.)

MNIST 데이터 세트를 다운로드합니다.

```
from azureml.core import Dataset
from azureml.opendatasets import MNIST
```

---

### 클러스터에 작업을 제출하기 위해 학습 스크립트를 만듭니다.

```
%%writefile $script_folder/train.py    #train.py라는 파이썬 파일을 생성하는 코드

import argparse
import os
import numpy as np
import glob

from sklearn.linear_model import LogisticRegression
import joblib

from azureml.core import Run
from utils import load_data

# 필요한 argument 설정하기
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regularization rate')
args = parser.parse_args()

data_folder = args.data_folder
print('Data folder:', data_folder)

# data load
X_train = load_data(glob.glob(os.path.join(data_folder, '**/train-images-idx3-ubyte.gz'), recursive=True)[0], False) / 255.0
X_test = load_data(glob.glob(os.path.join(data_folder, '**/t10k-images-idx3-ubyte.gz'), recursive=True)[0], False) / 255.0
y_train = load_data(glob.glob(os.path.join(data_folder, '**/train-labels-idx1-ubyte.gz'), recursive=True)[0], True).reshape(-1)
y_test = load_data(glob.glob(os.path.join(data_folder, '**/t10k-labels-idx1-ubyte.gz'), recursive=True)[0], True).reshape(-1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep = '\n')

# fit 하고 모델 완성하기
run = Run.get_context()

print('Train a logistic regression model with regularization rate of', args.reg)

#Logistic Regression 모델 불러오기
clf = LogisticRegression(C=1.0/args.reg, solver="liblinear", multi_class="auto", random_state=42)
clf.fit(X_train, y_train)

print('Predict the test set')
y_hat = clf.predict(X_test)

# calculate accuracy on the prediction
acc = np.average(y_hat == y_test)
print('Accuracy is', acc)


run.log('regularization rate', np.float(args.reg))  # 어느정도로 정규화할 건지 나타내는 수치
run.log('accuracy', np.float(acc))    #정확도

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=clf, filename='outputs/sklearn_mnist_model.pkl')   
#pkl 형식의 파일로 저장하는 이유는 모델이 scikit-learn이기 때문
```

---

### train.py를 실행시키기 위해 스크립트를 실행하는데 필요한 라이브러리가 포함된 환경을 만듭니다.

```
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

# 필요한 패키지 설치
env = Environment('tutorial-env')
cd = CondaDependencies.create(pip_packages=['azureml-dataset-runtime[pandas,fuse]', 'azureml-defaults'], conda_packages=['scikit-learn==0.22.1'])

env.python.conda_dependencies = cd

```

---

### 학습 스크립트에서 만들었던 argument를 설정하여 ScriptRunConfig를 만들고 실행합니다.

```
from azureml.core import ScriptRunConfig

args = ['--data-folder', mnist_file_dataset.as_mount(), '--regularization', 0.5]

src = ScriptRunConfig(source_directory=script_folder,
                      script='train.py', 
                      arguments=args,
                      compute_target=compute_target,
                      environment=env)

# config 실행하기
run = exp.submit(config=src)
run
```

---

## 모델 배포하기

### package import

```
import azureml.core
```

---

모델 사용법을 보기 위해 채점 스크립트를 만들어줍니다.

```
%%writefile score.py
import json
import numpy as np
import os
import pickle
import joblib

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'sklearn_mnist_model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    y_hat = model.predict(data)
    return y_hat.tolist()
```

---

### 배포를 위해 구성파일을 만듭니다.

ACI를 임포트하고 ACI 컨테이너에 필요한 RAM 기가바이트와 CPU 개수를 지정해줍니다.

```
from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "MNIST",  "method" : "sklearn"}, 
                                               description='Predict MNIST with sklearn')
```

---

### ACI에서 배포합니다.

```
%%time
from azureml.core.webservice import Webservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core import Workspace
from azureml.core.model import Model

ws = Workspace.from_config()
model = Model(ws, 'sklearn_mnist')


myenv = Environment.get(workspace=ws, name="tutorial-env", version="1")
inference_config = InferenceConfig(entry_script="score.py", environment=myenv)

service = Model.deploy(workspace=ws, 
                       name='sklearn-mnist-svc3', 
                       models=[model], 
                       inference_config=inference_config, 
                       deployment_config=aciconfig)

service.wait_for_deployment(show_output=True)

#엔드포인트 가져오기
print(service.scoring_uri)
```

