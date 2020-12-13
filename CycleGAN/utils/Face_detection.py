import cv2
import math
import numpy as np
import face_alignment

class FaceDetection:
    def __init__(self, device, detector):
        # 使用face_alignment 检测脸部区域
        self.face = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector=detector)

    def align(self, image):
        landmarks = self.__get_max_face_landmarks(image)

        if landmarks is None:
            return None
        else:
            return self.__rotate(image, landmarks)

    def __get_max_face_landmarks(self, image):
        landmarks = self.face.get_landmarks(image)
        if landmarks is None:
            return None

        elif len(landmarks) == 1:
            return landmarks[0]

        else:
            areas = []
            for pred in landmarks:
                landmarks_top = np.min(pred[:, 1])
                landmarks_bottom = np.max(pred[:, 1])
                landmarks_left = np.min(pred[:, 0])
                landmarks_right = np.max(pred[:, 0])
                areas.append((landmarks_bottom - landmarks_top) * (landmarks_right - landmarks_left))
            max_face_index = np.argmax(areas)
            return landmarks[max_face_index]

    @staticmethod
    def __rotate(image, landmarks):
        # 图像预处理
        left_eye_corner = landmarks[36]
        right_eye_corner = landmarks[45]
        radian = np.arctan((left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0]))

        height, width, _ = image.shape
        cos = math.cos(radian)
        sin = math.sin(radian)
        new_w = int(width * abs(cos) + height * abs(sin))
        new_h = int(width * abs(sin) + height * abs(cos))

        Tx = new_w // 2 - width // 2
        Ty = new_h // 2 - height // 2

        M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx],
                      [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])

        image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))

        landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
        landmarks_rotate = np.dot(M, landmarks.T).T
        return image_rotate, landmarks_rotate


if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread('3989161_1.jpg'), cv2.COLOR_BGR2RGB)
    face_detection = FaceDetection(device='gpu')
    face_info = face_detection.align(img)
    if face_info is not None:
        image_align, landmarks_align = face_info

        for i in range(landmarks_align.shape[0]):
            cv2.circle(image_align, (int(landmarks_align[i][0]), int(landmarks_align[i][1])), 2, (255, 0, 0), -1)

        cv2.imwrite('image_align.png', cv2.cvtColor(image_align, cv2.COLOR_RGB2BGR))
