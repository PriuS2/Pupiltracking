import cv2
import numpy as np
import mediapipe as mp
import time


def GetImgPos(index, landMarkArr):
    landmark = landmarks.landmark[index]
    pixel_x = int(landmark.x * imgSize_x)
    pixel_y = int(landmark.y * imgSize_y)
    return pixel_x, pixel_y

if __name__ == '__main__':

    #cap = cv2.VideoCapture('test.mp4')
    #delay = int(1000 / 25)  # 25frame/sec

    cap = cv2.VideoCapture(0)
    delay = 1


    mp_faceMesh = mp.solutions.face_mesh
    with mp_faceMesh.FaceMesh(False, 1, 0.5, 0.5) as faceMesh:
        while True:
            ret, frame = cap.read()



            # r = 640. / img.shape[1]
            # dim = (640, int(img.shape[0] * r))
            # image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            imgSize_y, imgSize_x, _ = frame.shape

            # Process
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = faceMesh.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                cv2.putText(image, "Found Face", (20, 20), 0, 0.6, (255, 30, 30))
            else:
                cv2.putText(image, "Lost Face", (20, 20), 0, 0.6, (30, 30, 255))
                cv2.imshow("image", image)
                key = cv2.waitKey(30)
                if key == 27:
                    break
                continue

            for landmarks in results.multi_face_landmarks:
                for i in range(0, 468):
                    pixelPos = GetImgPos(i, landmarks)
                    #cv2.circle(image, (pixelPos[0], pixelPos[1]), 2, (100, 0, 0), 0)

            # Gaze recognition
            for landmarks in results.multi_face_landmarks:
                leftEye_left = GetImgPos(130, landmarks)
                leftEye_right = GetImgPos(133, landmarks)
                leftEye_top = GetImgPos(27, landmarks)
                leftEye_bottom = GetImgPos(23, landmarks)

                cv2.line(image,
                         leftEye_left,
                         leftEye_right,
                         [255, 30, 30], thickness=2)

                cv2.line(image,
                         leftEye_top,
                         leftEye_bottom,
                         [255, 30, 30], thickness=2)

                # 크롭
                leftEye_crop = frame[leftEye_top[1]:leftEye_bottom[1], leftEye_left[0]:leftEye_right[0]]

                r = 200. / leftEye_crop.shape[1]
                dim = (200, int(leftEye_crop.shape[0] * r))
                leftEye_crop = cv2.resize(leftEye_crop, dim, interpolation=cv2.INTER_AREA)

                # 그레이
                leftEye_crop_gray = cv2.cvtColor(leftEye_crop, cv2.COLOR_BGR2GRAY)
                # 블러
                #leftEye_crop_gray = cv2.GaussianBlur(leftEye_crop_gray, (7, 7), 0)
                leftEye_crop_gray = cv2.bilateralFilter(leftEye_crop_gray, 9, 75, 75)

                # 이진화
                _, threshold = cv2.threshold(leftEye_crop_gray, 80, 255, cv2.THRESH_BINARY_INV)

                # 외곽선 추출
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
                rows, cols, _ = leftEye_crop.shape
                for cnt in contours:
                    cv2.drawContours(leftEye_crop, [cnt], -1, (0, 0, 255), 1)
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    #cv2.rectangle(leftEye_crop, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    print(h)
                    if (h<=15) | (h/w < 0.2): break
                    centerPosition = (x + int(w / 2),y + int(h / 2))
                    cv2.circle(leftEye_crop, centerPosition, int(w/2), (255, 0, 0), 2)
                    cv2.line(leftEye_crop, (x + int(w / 2), 0), (x + int(w / 2), rows), (0, 255, 0), 2)
                    cv2.line(leftEye_crop,
                             (0, y + int(h / 2)),
                             (cols, y + int(h / 2)),
                             (0, 255, 0), 2)
                    break;

            cv2.imshow("Crop", leftEye_crop)
            cv2.imshow("Threshold", threshold)
            cv2.imshow("leftEye", leftEye_crop_gray)
            cv2.imshow("image", image)

            key = cv2.waitKey(delay)
            if key == 27:
                break
