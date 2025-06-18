import math

import numpy as np
import cv2
from faces_parts import *

# Constantes
ACCEPTANCE_RATIO = 0.0


def summary_mat(mat):
    total = 0
    # print "Mat received : ", mat
    dimensions = mat.shape
    # print "Mat dimensions : ", dimensions, " size : ", len(dimensions)
    if len(dimensions) == 1:
        for i in mat:
            total += mat[i]
    else:
        for i in mat:
            # print "Invocando novo summary, ", i
            total += summary_mat(i)
    return total


def distancia_euclidiana_normalizada(p1, p2, normal):
    # print "P1 : ", p1, "p1[0] : ", p1[0], "p1[1] : ", p1[1]
    # print "P2 : ", p2, "p2[0] : ", p2[0], "p2[1] : ", p2[1]
    d0 = float(p1[0] - p2[0]) / float(normal[0])
    d1 = float(p1[1] - p2[1]) / float(normal[1])
    dist = math.sqrt(d0 * d0 + d1 * d1)
    return dist


def create_circular_roi(roi, x, y, w, h):
    pass


# noinspection PyPep8Naming
def get_eyes_affline_info(left_eye, right_eye, image):
    print("Received Left Eye : ", left_eye, "  Right Eye : ", right_eye)
    eyes_center = {'x': (float(left_eye.x) + float(right_eye.x)) * 0.5,
                   'y': (float(left_eye.y) + float(right_eye.y)) * 0.5}
    # Get the angle between the 2 eyes.

    dy = (float(right_eye.y) - float(left_eye.y))
    dx = (float(right_eye.x) - float(left_eye.x))
    lenght = np.sqrt(dx * dx + dy * dy)
    # Convert Radians to Degrees.

    angle = math.atan2(dy, dx) * 180.0 / math.pi
    # Hand measurements shown that the left eye center should
    # ideally be roughly at(0.16, 0.14) of a scaled face image.

    DESIRED_LEFT_EYE_X = 0.16
    DESIRED_LEFT_EYE_Y = 0.26
    DESIRED_RIGHT_EYE_X = (1.0 - DESIRED_LEFT_EYE_X)
    # DESIRED_RIGHT_EYE_Y = (1.0 - DESIRED_LEFT_EYE_Y)
    # Get the amount we need to scale the image to be the desired
    # fixed size we want

    # DESIRED_FACE_WIDTH = 70
    # DESIRED_FACE_HEIGHT = 70
    DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH = image.shape
    desired_len = (DESIRED_RIGHT_EYE_X - 0.16)
    scale = desired_len * DESIRED_FACE_WIDTH / lenght
    # print "Eyes Center : ", eyes_center, "  Angle : ", angle, "   Scale : ", scale
    # Get the transformation matrix for the desired angle & size.
    rot_mat = cv2.getRotationMatrix2D((eyes_center['x'], eyes_center['y']), angle, scale / 2)
    # Shift the center of the eyes to be the desired center.
    ex = DESIRED_FACE_WIDTH * 0.5 - eyes_center['x']
    ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyes_center['y']
    # print "Rotation Mat : ", rot_mat
    # print "ex, ey : ", ex, ey
    rot_mat[0, 2] += ex
    rot_mat[1, 2] += ey
    # Transform the face image to the desired angle & size &
    # position! Also clear the transformed image background to a
    # default grey.
    # warped = cv2.cv.CreateMat(DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH, cv2.CV_8U)  # cv2.cv.Scalar(128)
    # warped = np.zeros((DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH), dtype=np.uint8)

    # M = cv2.getRotationMatrix2D((DESIRED_FACE_WIDTH/2, DESIRED_FACE_HEIGHT/2), 90, 0.5)
    # print "Summary gray image : ", summary_mat(gray_image)
    # dst = cv2.warpAffine(gray_image, M, (DESIRED_FACE_WIDTH, DESIRED_FACE_HEIGHT))

    # print "Summary destination image : ", dst
    return cv2.warpAffine(image, rot_mat, (DESIRED_FACE_WIDTH, DESIRED_FACE_HEIGHT))


class Face(object):
    def __init__(self, roi, roi_original, x, y, w, h):
        """
        :param x: int
        :param y :int
        :param w: int
        :param h: int
        """
        self.face_id = -1
        self.face_group = None
        self.left_eye = None
        self.right_eye = None
        self.mouth = None
        self.roi = roi
        # self.roi_circle = create_circular_roi(roi, x, y, w, h)
        self.roi_original = roi_original
        self.nose = None
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.parts_detected = []
        self.marked = False

    def can_append_part(self, face_part):
        for part in self.parts_detected:
            if part.__class__.__name__ == "CirclePart" and face_part.__class__.__name__ == "CirclePart":
                if not (part.getCircleColision(face_part) > (1 - ACCEPTANCE_RATIO)):
                    return False
            elif part.__class__.__name__ == "EllipsePart" and face_part.__class__.__name__ == "EllipsePart":
                if part.getBoxColision(face_part):
                    return False
            elif part.__class__.__name__ == "BoxPart" and face_part.__class__.__name__ == "BoxPart":
                if part.getBoxColision(face_part):
                    return False
            elif part.__class__.__name__ == "BoxPart" and face_part.__class__.__name__ == "EllipsePart":
                if part.getBoxColision(face_part):
                    return False
            elif part.__class__.__name__ == "EllipsePart" and face_part.__class__.__name__ == "BoxPart":
                if part.getBoxColision(face_part):
                    return False
            elif part.__class__.__name__ == "BoxPart" and face_part.__class__.__name__ == "CirclePart":
                if face_part.get_box_circle_colision(part):
                    return False
            elif part.__class__.__name__ == "CirclePart" and face_part.__class__.__name__ == "BoxPart":
                if part.getBoxCircleColision(face_part):
                    return False
            elif part.__class__.__name__ == "CirclePart" and face_part.__class__.__name__ == "EllipsePart":
                if part.getBoxCircleColision(face_part):
                    return False
            elif part.__class__.__name__ == "EllipsePart" and face_part.__class__.__name__ == "CirclePart":
                if face_part.get_box_circle_colision(part):
                    return False
        return True

    def append_part(self, face_part):
        # if (self.can_append( face_part )):
        self.parts_detected.append(face_part)

    def draw_all(self, frame, font=cv2.FONT_HERSHEY_SIMPLEX):
        text = "ID : None"
        if self.face_id is not None:
            text = "ID : " + str(self.face_id)
        cv2.putText(frame, text, org=(self.x, self.y), color=(0, 255, 0), thickness=1, fontFace=font, fontScale=1)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)
        for part in self.parts_detected:
            part.draw(frame)

    def get_all_parts(self):
        eyes = []
        eyes_detected = 0
        nose = None
        mouth = None
        for part in self.parts_detected:
            if part.nome == "mouth":
                mouth = part
            if part.nome == "nose":
                nose = part
            if part.nome == "eye":
                eyes_detected += 1
                eyes.append(part)
        if (eyes_detected == 2) and mouth is not None and nose is not None:
            if eyes[0].x > eyes[1].x:
                self.left_eye = eyes[1]
                self.right_eye = eyes[0]
            else:
                self.left_eye = eyes[0]
                self.right_eye = eyes[1]
            self.nose = nose
            self.mouth = mouth

            resultado = [eyes[0], eyes[1], mouth, nose]
        else:
            resultado = []
        # if (mouth and nose and (eyes == 2)):
        #    print "Boca : ", mouth, " Nariz : ", nose, " Olhos : ", eyes, "    Resultado : ", resultado
        return resultado

    def is_valid_face(self):
        face_parts = self.get_all_parts()
        if ((face_parts is not None and len(face_parts) >= 4 and face_parts[0] is not None) and
                (face_parts[1] is not None) and face_parts[2] is not None and face_parts[3] is not None):
            return True
        else:
            return False

    def get_face_coordinates(self):
        return self.x, self.y, self.w, self.h

    def get_info(self):
        if self.is_valid_face():
            left_eye_info = self.left_eye.get_part_info()
            right_eye_info = self.right_eye.get_part_info()
            nose_info = self.nose.get_part_info()
            mouth_info = self.mouth.get_part_info()
            # print "Left Eye : ", left_eye_info
            # print "Right Eye : ", right_eye_info
            # print "Nose Eye : ", nose_info
            # print "Mouth Eye : ", mouth_info
            size = (self.w, self.h)
            eyes_distance = distancia_euclidiana_normalizada(left_eye_info["center"],
                                                             right_eye_info["center"], size)
            eye_left_mouth_distance = distancia_euclidiana_normalizada(mouth_info["center"],
                                                                       left_eye_info["center"], size)
            eye_right_mouth_distance = distancia_euclidiana_normalizada(mouth_info["center"],
                                                                        right_eye_info["center"], size)
            eye_left_nose_distance = distancia_euclidiana_normalizada(nose_info["center"],
                                                                      left_eye_info["center"], size)
            eye_right_nose_distance = distancia_euclidiana_normalizada(nose_info["center"],
                                                                       right_eye_info["center"], size)
            nose_mouth_distance = distancia_euclidiana_normalizada(nose_info["center"],
                                                                   mouth_info["center"], size)

            return [eyes_distance, eye_left_mouth_distance, eye_right_mouth_distance,
                    eye_left_nose_distance, eye_right_nose_distance, nose_mouth_distance]
        return []


class FaceDetectionEngine(object):
    def __init__(self, face_path="./haar_cascade/haarcascade_frontalface_default.xml",
                 eye_path="./haar_cascade/haarcascade_eye.xml", mouth_path="./haar_cascade/haarcascade_mcs_mouth.xml",
                 nose_path="./haar_cascade/haarcascade_mcs_nose.xml"):
        self.face_cascade = cv2.CascadeClassifier(face_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_path)
        self.mouth_cascade = cv2.CascadeClassifier(mouth_path)
        self.nose_cascade = cv2.CascadeClassifier(nose_path)

    def detect_all_parts(self, face):
        self.detect_eyes(face=face)
        self.detect_mouth(face=face)
        self.detect_nose(face=face)

    def detect_faces(self, image, image_gray, scale_factor=1.3, min_neighbors=5, min_size=(100, 100),
                     flags=cv2.CASCADE_SCALE_IMAGE):

        list_faces = []
        faces = self.face_cascade.detectMultiScale(
            image_gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=flags
        )
        for (fx, fy, fw, fh) in faces:
            # print "Imagem original : ", image
            roi = image_gray[fy:fy + fh, fx:fx + fw]
            roi_original = image[fy:fy + fh, fx:fx + fw]
            # print "Face Roi : ", roi_original
            face = Face(roi=roi, roi_original=roi_original, x=fx, y=fy, w=fw, h=fh, )
            list_faces.append(face)
        return list_faces

    def detect_eyes(self, face, scale_factor=1.3, min_neighbors=5, min_size=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE):
        list_eyes = []
        eyes = self.eye_cascade.detectMultiScale(
            face.roi,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=flags)
        for (x, y, w, h) in eyes:
            # print "Olho detectado : ", x, y, w, h
            eye = EllipsePart(nome='eye', x=int(face.x + x), y=int(face.y + y),
                              w=w, h=h, color=(0, 0, 255), thik=2)
            face.append_part(eye)
            list_eyes.append(eye)
        return list_eyes

    def detect_mouth(self, face, scale_factor=1.3, min_neighbors=5, min_size=(30, 30),
                     flags=cv2.CASCADE_SCALE_IMAGE):
        list_mouth = []
        mouth = self.mouth_cascade.detectMultiScale(
            face.roi,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=flags)
        for (x, y, w, h) in mouth:
            # print "Boca detectada : ", x, y, w, h
            mou = EllipsePart(nome='mouth', x=int(face.x + x), y=int(face.y + y),
                              w=w, h=h, color=(255, 0, 0), thik=2)
            face.append_part(mou)
            list_mouth.append(mou)
        return list_mouth

    def detect_nose(self, face, scale_factor=1.3, min_neighbors=5, min_size=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE):
        list_nose = []
        nose = self.nose_cascade.detectMultiScale(
            face.roi,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=flags)
        for (x, y, w, h) in nose:
            # print "Nariz detectado : ", x, y, w, h
            nos = CirclePart(nome='nose', x=int(face.x + x + w * 0.5), y=int(face.y + y + h * 0.5),
                             radius=round((w + h) * 0.25),
                             color=(0, 255, 255), thik=2)
            face.append_part(nos)
            list_nose.append(nos)
        return list_nose
