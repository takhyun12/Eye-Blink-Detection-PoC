# Anomaly-Detection-of-Eye-blinks-Pattern
#### EAR(Eye Aspect Ratio) 알고리즘 기반의 눈 깜빡임 측정 및 간단한 이상탐지 PoC

Author: Tackhyun Jung

Status: 완료

![1](https://user-images.githubusercontent.com/41291493/109273272-50ea0780-7855-11eb-86f2-8b38194d2d17.png)
![2](https://user-images.githubusercontent.com/41291493/109273283-534c6180-7855-11eb-9e46-c6b297ea6b8a.png)
![3](https://user-images.githubusercontent.com/41291493/109273285-53e4f800-7855-11eb-8863-4398504291eb.png)

### 핵심목표
1) CNN 모델 구현 및 학습 메커니즘 구현 `retrain.py`
2) CNN 모델 기반의 이미지 분류 `prediction.py`
3) CNN 모델의 학습 결과 시각화 `tensorboard.py`
4) 클라이언트/서버 환경애서 해당 모델 활용 방안 예시 `server.py` `client.py` 
5) python 코드를 C# windows 소프트웨어에서 활용하는 방안에 대한 예시 `CNN_UI`

---

### 사용된 기술
* CNN(Convolutional Neural Networks)
* Socket
* Multi Processing, Multi threading
* google Inception v3

#### What is eye aspect ratio (EAR)?
From the last project - Detecting facial landmarks - we know that we can apply facial landmark detection to localize important regions of the face, including eyes, eyebrows, nose, ears, and mouth. This also implies that we can extract specific facial structures by knowing the indexes of the particular face parts. In terms of blink detection, we are only interested in two sets of facial structures — the eyes.

---

### Requirement
* Python 3x
* tensorflow
* keras
* numpy
* tarfile
* threading
* socket

`CNN_UI`에만 해당
* C# (.NET FRAMEWORK 3.5 이상)

---

### Usage

```
$ Python {script}.py
```

---

### References
1. Tereza Soukupova and Jan ´ Cech, Real-Time Eye Blink Detection using Facial Landmarks, https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
