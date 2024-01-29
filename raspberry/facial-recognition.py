import cv2
import numpy as np
from os import listdir, makedirs
from os.path import isdir, isfile, join
import shutil
import serial


CAM_ID = 0

opencode = "o"

ser = serial.Serial("com3", 9600, timeout=1)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# 여러 사용자 학습


def trains():
    user_count = 0

    # faces 폴더의 하위 폴더를 학습

    data_path = "faces/"

    # 폴더만 색출

    model_dirs = [f for f in listdir(data_path) if isdir(join(data_path, f))]

    # 학습 모델 저장할 딕셔너리

    models = {}

    # 각 폴더에 있는 얼굴들 학습

    for model in model_dirs:
        # 학습 시작

        result = train(model)

        # 학습이 안되었다면 패스!

        if result is None:
            continue

        user_count = user_count + 1

        # 학습되었으면 저장

        models[model] = result

    # 학습된 모델 딕셔너리 리턴

    return models, user_count


# 사용자 얼굴 학습


def train(name):
    data_path = "faces/" + name + "/"

    # 파일만 리스트로 만듬

    face_pics = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    Training_Data, Labels = [], []

    for i, files in enumerate(face_pics):
        image_path = data_path + face_pics[i]

        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 이미지가 아니면 패스

        if images is None:
            continue

        Training_Data.append(np.asarray(images, dtype=np.uint8))

        Labels.append(i)

    if len(Labels) == 0:
        print("There is no data to train.")

        return None

    Labels = np.asarray(Labels, dtype=np.int32)

    # 모델 생성

    model = cv2.face.LBPHFaceRecognizer_create()

    # 학습

    model.train(np.asarray(Training_Data), np.asarray(Labels))

    # 학습 모델 리턴

    return model


models, users_count = trains()


# 얼굴 검출 함수


def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # 얼굴이 없으면 패스!

    if faces is ():
        return None

    # 얼굴이 있으면 얼굴 부위만 이미지로 만들고

    for x, y, w, h in faces:
        cropped_face = img[y : y + h, x : x + w]

    # 리턴!

    return cropped_face


# 얼굴만 저장


def take_pictures(user_count):
    # global screen_heigt, screen_width

    user_name = user_count + 1

    while True:
        if not isdir("face/" + str(user_name)):
            makedirs("faces/" + str(user_name))

            break

    cap = cv2.VideoCapture(CAM_ID)

    count = 0

    while True:
        ret, frame = cap.read()

        if face_extractor(frame) is not None:
            count += 1

            face = cv2.resize(face_extractor(frame), (200, 200))

            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = (
                "faces/"
                + str(user_name)
                + "/"
                + str(user_name)
                + "-"
                + str(count)
                + ".jpg"
            )

            cv2.imwrite(file_name_path, face)

            cv2.putText(
                face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
            )

            cv2.imshow("Face Cropper", face)

        else:
            pass

        cv2.waitKey(1)

        if count == 100:
            break

    cap.release()

    cv2.destroyAllWindows()


# 얼굴 검출


def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return img, []

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        roi = img[y : y + h, x : x + w]

        roi = cv2.resize(roi, (200, 200))

    return img, roi  # 검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달


def run():
    global users_count

    while True:
        op = input()

        if op == "":
            continue

        if op == "a":
            addUser()

        elif op == "r":
            resetUser()

        elif op == "d":
            if users_count == 0:
                pass

            else:
                Use_Face_Recognition()

        elif op == "s":
            Use_Face_Recognition()

        ser.reset_input_buffer()

        # Tk().mainloop()


def input():
    if not ser.readable():
        return ""

    data = ser.read()

    if data == b"":
        return ""

    return chr(data[0])


def addUser():
    global models, users_count

    take_pictures(users_count)

    models, users_count = trains()


def resetUser():
    global models, users_count

    for i in range(users_count):
        i = i + 1

        shutil.rmtree("faces/" + str(i) + "/")

    models, users_count = trains()


def Use_Face_Recognition():
    cap = cv2.VideoCapture(0)

    # 카메라로 부터 사진 한장 읽기

    while True:
        ret, frame = cap.read()

        # 얼굴 검출 시도

        image, face = face_detector(frame)

        try:
            min_score = 999  # 가장 낮은 점수로 예측된 사람의 점수

            min_score_name = ""  # 가장 높은 점수로 예측된 사람의 이름

            # 검출된 사진을 흑백으로 변환

            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # 위에서 학습한 모델로 예측시도

            for key, model in models.items():
                result = model.predict(face)

                if min_score > result[1]:
                    min_score = result[1]

                    min_score_name = key

            # min_score 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다.

            if min_score < 500:
                confidence = int(100 * (1 - (min_score) / 300))

                display_string = (
                    str(confidence) + "% Confidence it is " + min_score_name
                )

                cv2.putText(
                    image,
                    display_string,
                    (100, 120),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (250, 120, 255),
                    2,
                )

            # 75 보다 크면 동일 인물로 간주해 UnLocked!

            if confidence > 75:
                ser.write(opencode.encode())

                cv2.putText(
                    image,
                    "user count : " + str(users_count),
                    (380, 50),
                    cv2.cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )

                cv2.putText(
                    image,
                    "Unlocked : " + min_score_name,
                    (250, 450),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Face Cropper", image)

                # open door

                break

            else:
                # 75 이하면 타인.. Locked!!!

                cv2.putText(
                    image,
                    "user count : " + str(users_count),
                    (380, 50),
                    cv2.cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )

                cv2.putText(
                    image,
                    "Locked",
                    (250, 450),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

                cv2.imshow("Face Cropper", image)

        except:
            # 얼굴 검출 안됨

            cv2.putText(
                image,
                "user count : " + str(users_count),
                (380, 50),
                cv2.cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )

            # cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("Face Cropper", image)

            pass

        cv2.waitKey(1)

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
