# Detection of Eye blinks Pattern
#### `EAR(Eye Aspect Ratio)` 알고리즘 기반의 눈 깜빡임 측정 및 간단한 이상탐지 PoC

Author: Tackhyun Jung

Status: 완료

![1](https://user-images.githubusercontent.com/41291493/109273272-50ea0780-7855-11eb-86f2-8b38194d2d17.png)
![2](https://user-images.githubusercontent.com/41291493/109273283-534c6180-7855-11eb-9e46-c6b297ea6b8a.png)
![3](https://user-images.githubusercontent.com/41291493/109273285-53e4f800-7855-11eb-8863-4398504291eb.png)

### 핵심목표
1) 실시간 동영상에서 프레임 단위로 눈 깜빡임 횟수 추적
2) 눈 깜빡임 횟수에서의 이상 탐지
3) 화면에 눈 깜빡임 횟수, EAR의 수치 등을 표시

---

### 사용된 기술
* EAR(Eye Aspect Ratio)
* Face Detection
* Eye blinks Tracker

#### What is eye aspect ratio (EAR)?

From the last project - Detecting facial landmarks - we know that we can apply facial landmark detection to localize important regions of the face, including eyes, eyebrows, nose, ears, and mouth. This also implies that we can extract specific facial structures by knowing the indexes of the particular face parts. In terms of blink detection, we are only interested in two sets of facial structures — the eyes.

---

### Requirement
* Python 3x
* distance
* FileVideoStream
* VideoStream
* imutils
* numpy
* cv2
* dlib
* openface
* threading
---

### Usage

```
$ Python main.py
```

---

### References
1. Tereza Soukupova and Jan ´ Cech, Real-Time Eye Blink Detection using Facial Landmarks, https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
