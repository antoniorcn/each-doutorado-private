import cv2
import numpy as np
import tensorflow as tf
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, RoundedRectangle
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.config import Config
from kivy.core.window import Window
from deepface_controller import DeepFaceRecognitionController
from custom_model_controller import CustomModelController
from face_detector import FaceDetector
from face_detection_engine import FaceDetectionEngine

FACE_FOUND_DELAY = 50
ANOTHER_FRAME_DELAY = 10
CAPTURE_CAMERA_SIZE = (380, 500)
DETECTED_FACE_SIZE = (200, 200)

class ColoredBox(BoxLayout):
    def __init__(self, background_color=(0.1, 0.5, 0.7, 1), radius=20, **kwargs):
        super().__init__(**kwargs)

        with self.canvas.before:
            self.background_color = Color(*background_color)  # Aqui fica armazenada a cor atual
            self.bg_rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[radius])

        self.bind(pos=self._update_bg, size=self._update_bg)

    def _update_bg(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size

    def set_background_color(self, r, g, b, a=1):
        """Altera a cor de fundo dinamicamente"""
        self.background_color.rgba = (r, g, b, a)


class FaceDetectionApp(App):
    def __init__(self, **kwargs):
        super(FaceDetectionApp, self).__init__(**kwargs)
        self.deepface_controller = DeepFaceRecognitionController()
        self.custom_model_controller = CustomModelController()
        # Define tamanho da janela
        Config.set('graphics', 'width', '600')
        Config.set('graphics', 'height', '400')

        # Desativa maximização/redimensionamento (opcional)
        Config.set('graphics', 'resizable', '0')
        self.ticks = 0
        self.face_detected = False
        self.delay_another_frame = False
        # self.face_detector = FaceDetector("./haar_cascade")
        self.face_detector = FaceDetectionEngine()

    def build(self):
        # Posicionando a tela no canto superior esquerdo
        # Window.top = 0  # Altura total da tela - altura da janela
        # Window.left = 0  # Colado na esquerda
        # Window.borderless = True # Remove os frames laterais
        Clock.schedule_interval(self.update, 1.0 / 15.0)

        self.capture = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
        # self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        # self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # === Layout Raiz ===
        root = ColoredBox(orientation='horizontal', padding=20, spacing=20,
                          background_color=(1, 0.7, 0.6, 1), radius=40)

        # === Painel Esquerdo (Imagem do rosto) ===
        left_panel = ColoredBox(orientation='vertical', padding=10, spacing=10,
                                background_color=(0, 0.4, 0.5, 1), radius=30)

        self.face_image = Image(size_hint=(1, 0.9))
        left_panel.add_widget(self.face_image)

        # === Painel Direito ===
        right_panel = ColoredBox(orientation='vertical', spacing=20)

        # ----- Indicadores de Spoofing -----
        spoofing_box = ColoredBox(orientation='horizontal', padding=10,
                                 spacing=10, size_hint_y=0.3,
                                 background_color=(0, 0.4, 0.5, 1))

        self.deepface_indicator = ColoredBox(orientation='vertical', spacing=5, padding=5, size_hint=(0.5, 1))
        lbl_deepface = Label(text="Facebook\nDeepface", size_hint=(1, 1),
                                  color=(1, 1, 1, 1), bold=True)
        self.deepface_indicator.add_widget(lbl_deepface)

        self.custom_indicator = ColoredBox(orientation='vertical', spacing=5, padding=5, size_hint=(0.5, 1))
        lbl_custom = Label(text="Modelo\nCustomizado", size_hint=(1, 1),
                                      color=(1, 1, 1, 1), bold=True)
        self.custom_indicator.add_widget(lbl_custom)

        spoofing_box.add_widget(self.deepface_indicator)
        spoofing_box.add_widget(self.custom_indicator)
        right_panel.add_widget(spoofing_box)

        # ----- Painel com a foto -----
        self.extracted_face_image = Image(size_hint=(0.5, 0.5))
        right_panel.add_widget(self.extracted_face_image)

        # ----- Painel de Identificação -----
        identification_box = ColoredBox(orientation='vertical', padding=10,
                                       spacing=10, size_hint_y=0.7,
                                       background_color=(0, 0.4, 0.5, 1))

        label_title = Label(text="Identificação", font_size=24, bold=True,
                            size_hint=(None, None), size=(200, 40),
                            # pos_hint={"x": 0.05, "top": 1}
                            )

        self.label_user_id = Label(text="Id:", font_size=18,
                                   size_hint=(None, None), size=(200, 30),
                                   # pos_hint={"x": 0.05, "top": 0.75}
                                   )

        self.label_user_name = Label(text="Nome:", font_size=18,
                                     size_hint=(None, None), size=(200, 30),
                                     # pos_hint={"x": 0.05, "top": 0.6}
                                     )

        self.btn_delete = Button(background_normal='trash.png',
                                 background_down='trash.png',
                                 size_hint=(None, None), size=(50, 60),
                                 # pos_hint={'right': 0.98, 'y': 0.02},
                                 background_color=(1, 0, 0, 1))
        self.btn_delete.bind(on_press=self.delete_user)
        identification_box.add_widget(label_title)
        identification_box.add_widget(self.label_user_id)
        identification_box.add_widget(self.label_user_name)
        identification_box.add_widget(self.btn_delete)

        right_panel.add_widget(identification_box)

        # === Montagem dos painéis ===
        root.add_widget(left_panel)
        root.add_widget(right_panel)

        return root

    def update(self, _):
        self.ticks += 1
        ret, frame = self.capture.read()

        if ret:
            # Inverte o frame da camera na horizontal e na vertical
            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)

            # face_detection_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # face_detection_frame = cv2.equalizeHist(face_detection_frame)
            face_detection_frame = frame
            # color_format = "bgr"
            # frame = face_detection_frame
            color_format = "rgb"

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # faces_detected_frame = self.face_detector.extract_image_objects(face_detection_frame)

            faces_detected = self.face_detector.detect_faces(image=face_detection_frame,
                                                             image_gray=gray_frame)

            # Atualiza a imagem da janela com o frame original da camera
            if not self.delay_another_frame or self.ticks % ANOTHER_FRAME_DELAY == 0:
                self.delay_another_frame = False
                camera_buf = cv2.resize(face_detection_frame, CAPTURE_CAMERA_SIZE).tobytes()
                texture_camera = Texture.create(size=CAPTURE_CAMERA_SIZE, colorfmt=color_format)
                texture_camera.blit_buffer(camera_buf, colorfmt=color_format, bufferfmt='ubyte')
                self.face_image.texture = texture_camera
           
            # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # faces_df = self.extract_faces_deepface( rgb_frame )


            # faces_haar = self.face_cascade.detectMultiScale(face_detection_frame, scaleFactor=1.07,
            #                                         minNeighbors=4,
            #                                         #minSize=(100, 100)
            #                                         )
            # faces_detected_qty = len(faces_haar)

            faces_detected_qty = len(faces_detected)
            
            if faces_detected_qty > 0:
                print("Faces detectadas: ", faces_detected)
                self.ticks = 0
                self.face_detected = True
                self.delay_another_frame = True
                # (h, w) = frame.shape[:2]
                # (x, y, bw, bh) = faces[0]
                # y_adjust = int(h * 0.15)
                # h_adjust = int(h * 0.85)
                # x1 = max(x, 0)
                # y1 = max(y - y_adjust, 0)
                # x2 = min(x + bw, w)
                # y2 = min(y + h_adjust, h)
                # face_img = frame[y1:y2, x1:x2].copy()
                face_img = None
                # for face in faces_df:
                #     x, y = face["facial_area"]['x'], face["facial_area"]['y']
                #     w, h = face["facial_area"]['w'], face["facial_area"]['h']

                for face in faces_detected:
                    print("Analisando face detectada")
                    (x, y, w, h) = face.get_face_coordinates()
                    print(f"Face localizada: POS: ({x}, {y})   SIZE: ({w}, {h})")
                    # haar_frame = frame[y:y + h, x:x + w]
                    # gray = cv2.cvtColor(haar_frame, cv2.COLOR_BGR2GRAY)
                    # gray = cv2.equalizeHist(gray)
                    # faces_haar = self.face_cascade.detectMultiScale(gray, scaleFactor=1.03,
                    #                     minNeighbors=9, minSize=(150, 100))
                    frame_drawn = frame.copy()
                    face_img = frame[y:y + h, x:x + w].copy()
                    center = (x + w // 2, y + h // 2)
                    axes = (w // 2, h // 2)
                    cv2.ellipse(frame_drawn, center, axes, angle=0,
                                startAngle=0, endAngle=360, color=(0, 255, 0), thickness=2)

                    # Atualiza a tela com o frame da camera com as faces circuladas
                    camera_buf = cv2.resize(frame_drawn, CAPTURE_CAMERA_SIZE).tobytes()
                    texture_camera = Texture.create(size=CAPTURE_CAMERA_SIZE, colorfmt=color_format)
                    texture_camera.blit_buffer(camera_buf, colorfmt=color_format, bufferfmt='ubyte')
                    self.face_image.texture = texture_camera


                if face_img is not None and\
                (not self.face_detected or self.ticks > FACE_FOUND_DELAY):
                    # Desenha apenas o rosto no frame detectado
                    display_frame = cv2.resize(face_img, DETECTED_FACE_SIZE)
                    buf = display_frame.tobytes()
                    texture = Texture.create(size=DETECTED_FACE_SIZE, colorfmt=color_format)
                    texture.blit_buffer(buf, colorfmt=color_format, bufferfmt='ubyte')
                    self.extracted_face_image.texture = texture

                    # Prepara o frame para fazer os testes no deepface e o no modelo customizado
                    face_img_resized = cv2.resize(frame, (244, 244))
                    face_img_resized_rgb = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB)

                    # === Verifica spoofing ===
                    fb_result = self.check_spoof_deepface(face_img_resized_rgb)
                    tf_result = self.check_spoof_tensorflow(face_img_resized_rgb)

                    if fb_result == 1:
                        self.deepface_indicator.set_background_color(0, 0.7, 0, 1)  # Verde
                    else:
                        self.deepface_indicator.set_background_color(1, 0, 0, 1)  # Vermelho

                    if tf_result == 1:
                        self.custom_indicator.set_background_color(0, 0.7, 0, 1)  # Verde
                    else:
                        self.custom_indicator.set_background_color(1, 0, 0, 1)  # Vermelho

                    # === Preenche dados se não for spoofing ===
                    if fb_result and tf_result == 1:
                        self.label_user_id.text = "Id: 123"
                        self.label_user_name.text = "Nome: Antonio"
                    else:
                        self.label_user_id.text = "Id:"
                        self.label_user_name.text = "Nome:"
            else:
                if self.ticks > FACE_FOUND_DELAY:
                    self.extracted_face_image.texture = None
                    self.label_user_id.text = "Id:"
                    self.label_user_name.text = "Nome:"

    def delete_user(self, instance):
        print("Removendo usuário do Face Detection...")
        print(instance)
        self.label_user_id.text = "Id:"
        self.label_user_name.text = "Nome:"
        # Aqui você coloca sua lógica real de remoção no Facebook Face Detection

    def on_stop(self):
        self.capture.release()

    def check_spoof_deepface(self, face_image):
        # Simula: retorna 0 se for spoofing e 1 se for válido
        return 0 if not self.deepface_controller.is_real_face( face_image ) else 1

    def extract_faces_deepface(self, face_image):
        return self.deepface_controller.extract_faces( face_image )
    
    def check_spoof_tensorflow(self, face_image):
        return self.custom_model_controller.is_real_face(face_image)


if __name__ == '__main__':
    FaceDetectionApp().run()
