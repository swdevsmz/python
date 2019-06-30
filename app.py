from flask import Flask
import cv2
import numpy as np

app = Flask(__name__)


@app.route('/')
def hello_world():

    # img = cv2.imread('shutterstock_1055756639.jpg', 0)
    # 顔検出クラス
    face_cascade = cv2.CascadeClassifier(
        'C:\\00_data\\90_repos\\python\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    # 目検出クラス
    eye_cascade = cv2.CascadeClassifier(
        'C:\\00_data\\90_repos\\python\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml')

    # 画像読み込み
    img = cv2.imread('shutterstock_1055756639.jpg')
    # 顔検出用にグレースケールの画像作成
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 顔検出
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        #顔にマーキング
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi_gray = gray[y:y + h, x:x + w]
        # roi_color = img[y:y + h, x:x + w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # 画像を表示
    cv2.imshow('img', img)
    # キーボード入力処理
    cv2.waitKey(0)
    # 全てのウィンドウを閉じる
    cv2.destroyAllWindows()

    return 'Hello World!'

if __name__ == '__main__':
    app.run()
