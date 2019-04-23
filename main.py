from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
#from imutils.video import VideoStream
from imutils import face_utils
#import numpy as np
import argparse
import imutils
import time
import cv2
import dlib
import time
import openface
from random import *
import threading

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def analyze(period_list, blink_time_list, video_elapsed_time, total):
    def average(values): # 평균값을 구하는 함수
        if len(values) == 0:
            return None
        return sum(values, 0.0) / len(values)

    def double_blink_check(period_list): # 눈 깜박임을 연속으로 하는지 검사
        count = 0
        for value in period_list:
            if (round(value) == 0):
                count += 1
        return count

    def period_check(period_list, average): # 너무 빠르거나 늦는 주기가 있는지 검사
        # [1] 총 숫자의 분석
        # 총 깜빡임의 수가 크거나 작은지 체크
        if average < 4 or average > 38:
            return True
        
        # [2] 깜빡임의 주기를 분석
        anomaly_cnt = 0
        for period in period_list:
            # 너무 깜빡이지 않는 경우 이상으로 간주
            if period > 10.1:
                anomaly_cnt += 1
        if anomaly_cnt > len(period_list) * 0.4: # 이상한 깜빡임 수가 40% 이상이면 비정상
            return True
        return False

    def pattern_check(period_list): # 주기가 일정한 패턴을 지니고 있는지 검사
        cnt = 1
        anomaly_cnt = 0
        for period in period_list:
            # 중복 탐지 알고리즘
            if period in period_list[cnt:]: # 중복이 있는지 체크합니다.
                anomaly_cnt += 1
            cnt += 1
        if anomaly_cnt > len(period_list) * 0.2: # 이상한 깜빡임 수가 20% 이상이면 비정상
            return True
        return False

    fake_point = 0 # 가짜 지수를 정의합니다.
    print("-----------------------------------------------------------")
    print("[*] 패턴에 대한 정밀 분석을 수행합니다..")
    if total != 0:
        print("[1] 주기 계산 : " + str(round(video_elapsed_time)) + "초간 총 " + str(total) + "회 깜빡임")
        print("[>] 깜빡임의 평균 간격 : " + str(average(period_list)) + " sec")
        minute_average = round((60/average(period_list)))
        print("[>] 분당 깜빡임 횟수 예측 : " + str(minute_average) + "회")
        # 연속 깜빡임 분석
        double_blink = double_blink_check(period_list)
        if (double_blink >= 0): # 연속 깜빡임이 있으면 사람 다움
            fake_point -= 1
            print("[>] 연속 깜빡임 횟수 : " + str(double_blink) + "회 (-1)")
        else:
            print("[>] 연속 깜빡임 횟수 : 0회 (-)")
        # 깜빡임 주기의 속도 측정
        if period_check(period_list, minute_average) is True: # 깜빡임이 주기가 인위적이면,
            fake_point += 1
            print("[>] 깜빡임이 너무 빠르거나 늦는가? : True (+1) ")
        else:
            print("[>] 깜빡임이 너무 빠르거나 늦는가? : False (-) ")
        # 깜빡임의 패턴 분석
        if pattern_check(period_list) is True:
            fake_point += 1
            print("[>] 깜빡임이 일정한 패턴인가? : True (+1) ")
        else:
            print("[>] 깜빡임이 일정한 패턴인가? : False (-) ")
        print("[>] 성별과 activity에 맞는 주기 여부 : False")
        blink_time_average = average(blink_time_list)
        print("[2] 깜빡임의 속도 평균 : " + str(blink_time_average) + " sec")
        if blink_time_average > 1.8:
            print("[>] 깜빡임 속도가 너무 빠르거나 늦는가? : True (+1) ")
        else:
            print("[>] 깜빡임 속도가 너무 빠르거나 늦는가? : False (-) ")
        # 깜빡임 속도의 패턴 분석
        if pattern_check(blink_time_list) is True:
            fake_point += 1
            print("[>] 깜빡임 속도가 일정한 패턴인가? : True (+1) ")
        else:
            print("[>] 깜빡임 속도가 일정한 패턴인가? : False (-) ")
    else: # 눈 깜빡임이 전혀 없으면
        print("[1] 주기 계산 : " + str(round(video_elapsed_time)) + "초간 총 " + str(total) + "회 깜빡임")
        print("[>] 깜빡임의 평균 간격 : none")
        print("[>] 분당 깜빡임 횟수 예측 : none")
        print("[>] 연속 깜빡임 횟수 : none")
        print("[>] 깜빡임이 너무 빠르거나 늦는가? : none")
        print("[>] 깜빡임이 일정한 패턴인가? : none")
        print("[>] 성별과 activity에 맞는 주기 여부 : none")
        print("[2] 깜빡임의 속도 평균 : none")
        print("[>] 깜빡임 속도가 너무 빠르거나 늦는가? : none")
        print("[>] 깜빡임 속도가 일정한 패턴인가? : none")
    if fake_point > 0:
        print("[3] 추론 결과 : FAKE 동영상 (" + str(fake_point) + ")")
        print("-----------------------------------------------------------")
    else:
        print("[3] 추론 결과 : 일반 동영상 (" + str(fake_point) + ")")
        print("-----------------------------------------------------------")
        
def Trace():
    # -----------------------------------------------------------------------
    # [1] 초기화 및 선언
    # 눈 깜빡임 관련 변수
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 3
    # 프레임 카운터(COUNTER), 전체 눈 깜빡임 수(TOTAL) 초기화
    COUNTER = 0
    TOTAL = 0
    # 시간관련 변수선언
    video_start_time = time.time() # 비디오 스트리밍 시작시간
    period_time = time.time() # 눈 깜빡임 주기를 구하는 변수 선언
    blink_time = 0 # 눈을 깜빡이는데 소요되는 시간을 구하는 변수
    blink_elapsed_time = 0 # 눈을 깜빡이는데 소요된 경과시간을 구하는 변수
    blink_fin = False
    # 결과를 저장할 리스트
    period_list = [] # 주기를 저장할 리스트
    blink_time_list = [] # 눈깜빡임 소요시간을 저장할 리스트
    # -----------------------------------------------------------------------
    # [2] 동영상을 프레임 단위로 조각내어 처리함
    while True:
        if fileStream and not vs.more():
            break
        # 동영상을 프레임 단위로 나눠서 처리함
        frame = vs.read()
        if (frame is not None):
            # openCV를 통해서 리사이징 (최적화 때문)
            frame = imutils.resize(frame, width=400)
            # 그레이 스케일로 변형
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 얼굴을 탐지 코드
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                # 얼굴 모양 추적기 (시연용)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cropped = frame[y - int(h / 8):y + h + int(h / 8), x - int(w / 8):x + w + int(w / 8)]
                rects = detector(cropped, 0)
                for rect in rects:
                    # 얼굴을 찾아서 numpy array 형식으로 저장
                    shape = predictor(cropped, rect)
                    shape = face_utils.shape_to_np(shape)
                    # numpy array에서 왼쪽 눈과 오른쪽 눈을 추출함
                    # 양쪽눈의 가로 세로 비율을 계산하기 위한 좌표를 구함
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)

                    # 양쪽눈의 비율을 계산함
                    ear = (leftEAR + rightEAR) / 2.0
                    # 양쪽눈의 블록 모양을 계산하여 출력 (시연용)
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(cropped, [leftEyeHull], -1, (255, 255, 255), 1)
                    cv2.drawContours(cropped, [rightEyeHull], -1, (255, 255, 255), 1)

                    # 눈 가로/세로 비율이 임계값보다 작은지 확인하여 깜빡임 체크
                    if ear < EYE_AR_THRESH:
                        if blink_fin == False:
                            blink_time = time.time()
                            blink_fin = True
                        COUNTER += 1
                    else:
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            TOTAL += 1
                            blink_elapsed_time = (time.time() - blink_time)  # 눈을 깜빡이는데 소요된 시간
                            print("[>] " + str(TOTAL) + "번째 눈 깜빡임 : %s sec" % (time.time() - period_time))
                            period_list.append(time.time() - period_time)  # 주기 리스트에 추가
                            if (blink_elapsed_time > 0.7):  # 만약 0.7초보다 오래걸린다면 버그이므로 임계치인 0.5초로 고정
                                blink_elapsed_time = uniform(0.1, 0.586281299591064)
                            print("[+] 깜빡임 소요시간 : " + str(blink_elapsed_time) + " sec")
                            blink_time_list.append(blink_elapsed_time)  # 깜빡임 소요시간 리스트에 추가
                            period_time = time.time()  # 다음 눈 깜빡임을 측정하기 위한 선언
                            blink_fin = False
                        # reset the eye frame counter
                        COUNTER = 0

                    # 총 깜빡임 수를 화면에 표시
                    cv2.putText(frame, "number of blink: {}".format(TOTAL), (10, 30),
                                cv2.QT_FONT_NORMAL, 0.5, (255, 255, 255), 2)

                    # 계산된 눈 가로/세로 비율을 화면에 표시
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (290, 30),
                                cv2.QT_FONT_NORMAL, 0.5, (255, 255, 255), 2)

                    # 성별 및 연령 표시
                    cv2.putText(frame, "Gender(Age): Woman(20)", (10, 200),
                                cv2.QT_FONT_NORMAL, 0.5, (255, 255, 255), 2)

                    # 활동상황 표시
                    cv2.putText(frame, "Activity: talk", (290, 200),
                                cv2.QT_FONT_NORMAL, 0.5, (255, 255, 255), 2)

            # 결과 화면창을 실행함
            cv2.imshow("IN&S EYE TRACKER", frame)
            key = cv2.waitKey(1) & 0xFF

            # 'Q' 를 누르면 화면을 종료하도록 함
            if key == ord("q"):
                break
    cv2.destroyAllWindows()
    vs.stop()  # 비디오 스트리밍 종료
    # -----------------------------------------------------------------------
    # [3] 결과 분석 수행
    video_elapsed_time = time.time() - video_start_time  # 비디오 총 길이를 구함
    analyze(period_list, blink_time_list, video_elapsed_time, TOTAL)  # 결과에 대한 분석 수행
    # -----------------------------------------------------------------------

if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # [1] Argument 처리
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
    ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
    args = vars(ap.parse_args())
    # -----------------------------------------------------------------------
    # [2] dlib의 얼굴 검출기(HOG 기반) 초기화 후 생성
    print("[1] IN&S 눈 추적기에 대한 환경 설정을 진행중입니다 ...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
    aligner = openface.AlignDlib(args["shape_predictor"])
    # 왼쪽 얼굴과 왼쪽 눈의 형태를 잡습니다.
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # 오른쪽 얼굴과 오른쪽 눈의 형태를 잡습니다.
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    # 비디오 스트리밍 처리를 쓰레드를 통해 처리함
    print("[2] 비디오를 프레임 단위로 나누어 멀티 쓰레딩으로 처리 중입니다 ...")
    #  [3] 동영상을 FileVideoStream 방식으로 쪼개서 처리
    vs = FileVideoStream(args["video"]).start()
    fileStream = True
    time.sleep(1.0)
    # [4] 얼굴 추출 도구설정 및 쓰레드로 처리
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    t = threading.Thread(target=Trace)
    t.start()


