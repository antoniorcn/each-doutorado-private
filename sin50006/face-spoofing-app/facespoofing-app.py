import cv2
import numpy as np
import tensorflow as tf
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, RoundedRectangle, Rectangle
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

CAMERA_FRAME_SIZE=(380, 500)
DETECTED_FACE_FRAME_SIZE=(300, 500)
FACE_DETECTION_TIMEOUT = 50


def create_color_texture(size, rgba):
    """
    Cria uma textura colorida preenchida com a cor definida em rgba (valores entre 0 e 1).
    """
    width, height = size
    # Converte de float (0 a 1) para int (0 a 255)
    r = int(rgba[0] * 255)
    g = int(rgba[1] * 255)
    b = int(rgba[2] * 255)
    a = int(rgba[3] * 255)

    # Cria uma matriz numpy com os valores RGBA
    data = np.full((height, width, 4), (r, g, b, a), dtype=np.uint8)

    # Cria a textura
    texture = Texture.create(size=(width, height), colorfmt='rgba')
    texture.blit_buffer(data.flatten(), colorfmt='rgba', bufferfmt='ubyte')
    texture.flip_vertical()

    return texture

class ImageWithBackground(Image):
    def __init__(self, background_color=Color(0.1, 0.5, 0.7, 1), **kwargs):
        super().__init__(**kwargs)

        with self.canvas.before:
            self.bg_color = background_color  # Cor de fundo
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)

        self.bind(pos=self._update_bg, size=self._update_bg)

    def _update_bg(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size

    def set_background_color(self, r, g, b, a=1):
        self.bg_color.rgba = (r, g, b, a)

class ColoredBox(BoxLayout):
    def __init__(self, background_color=(0.1, 0.5, 0.7, 1), radius=20, **kwargs):
        super().__init__(**kwargs)

        with self.canvas.before:
            Color(*background_color)
            self.bg = RoundedRectangle(pos=self.pos, size=self.size, radius=[radius])

        self.bind(pos=self._update_bg, size=self._update_bg)

    def _update_bg(self, *_):
        self.bg.pos = self.pos
        self.bg.size = self.size


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

        self.face_detected = False
        self.face_detection_counter = 0
        self.detected_face_empty_texture = create_color_texture(DETECTED_FACE_FRAME_SIZE, (0.1, 0.5, 0.7, 1))

    def build(self):
        # Posicionando a tela no canto superior esquerdo
        # Window.top = 0  # Altura total da tela - altura da janela
        # Window.left = 0  # Colado na esquerda
        # Window.borderless = True # Remove os frames laterais
        Clock.schedule_interval(self.update, 1.0 / 15.0)

        self.capture = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # === Layout Raiz ===
        root = ColoredBox(orientation='horizontal', padding=20, spacing=20,
                          background_color=(1, 0.7, 0.6, 1), radius=40)

        # === Painel Esquerdo (Imagem do rosto) ===
        left_panel = ColoredBox(orientation='vertical', padding=10, spacing=10,
                                background_color=(0, 0.4, 0.5, 1), radius=30)

        self.face_image = ImageWithBackground(size_hint=(1, 0.9),
                                              pos_hint = {'center_x': 0.5, 'center_y': 0.5})
        left_panel.add_widget(self.face_image)

        # === Painel Direito ===
        right_panel = ColoredBox(orientation='vertical', spacing=20)

        # ----- Indicadores de Spoofing -----
        spoofing_box = ColoredBox(orientation='horizontal', padding=10,
                                 spacing=10, size_hint_y=0.3,
                                 background_color=(0, 0.4, 0.5, 1))

        self.fb_indicator = Label(text="Facebook\nDeepFace", size_hint=(0.5, 1),
                                  color=(1, 1, 1, 1), bold=True)
        self.custom_indicator = Label(text="Modelo\nCustomizado", size_hint=(0.5, 1),
                                      color=(1, 1, 1, 1), bold=True)

        spoofing_box.add_widget(self.fb_indicator)
        spoofing_box.add_widget(self.custom_indicator)
        right_panel.add_widget(spoofing_box)

        # ----- Painel com a foto -----
        # image_float_container = FloatLayout()
        self.extracted_face_image = ImageWithBackground(size_hint=(0.5, 0.5),
                                                        pos_hint = {'center_x': 0.5, 'center_y': 0.5})
        # image_float_container.add_widget( self.extracted_face_image )
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

        self.btn_delete = Button(size_hint=(None, None), size=(50, 60),
                                 text="Apagar",
                                #  background_normal='trash.png',
                                #  background_down='trash.png',
                                # pos_hint={'right': 0.98, 'y': 0.02},
                                 background_color=(1, 0, 0, 1))
        self.btn_delete.bind(on_press=self.delete_user)
        identification_box.add_widget(label_title)
        identification_box.add_widget(self.label_user_id)
        identification_box.add_widget(self.label_user_name)
        identification_box.add_widget(self.btn_delete)

        # right_panel.add_widget(identification_box)

        # === Montagem dos painéis ===
        root.add_widget(left_panel)
        root.add_widget(right_panel)

        return root

    def update(self, _):
        ret, frame = self.capture.read()
        self.face_detection_counter += 1
        if self.face_detection_counter > FACE_DETECTION_TIMEOUT:
            self.face_detection_counter = 0
            self.face_detected = False

        if ret:
            # Inverte o frame da camera na horizontal e na vertical
            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)

            # Atualiza a imagem da janela com o frame original da camera
            camera_buf = cv2.resize(frame, CAMERA_FRAME_SIZE).tobytes()
            texture_camera = Texture.create(size=CAMERA_FRAME_SIZE, colorfmt='bgr')
            texture_camera.blit_buffer(camera_buf, colorfmt='bgr', bufferfmt='ubyte')
            self.face_image.texture = texture_camera

            if not self.face_detected:
                # Limpa os quadro para cor padrão
                self.fb_indicator.canvas.before.clear()
                with self.fb_indicator.canvas.before:
                    Color(0.1, 0.5, 0.7, 1)  # Verde
                    RoundedRectangle(pos=self.fb_indicator.pos,
                                    size=self.fb_indicator.size, radius=[10])
                    
                self.custom_indicator.canvas.before.clear()
                with self.custom_indicator.canvas.before:
                    Color(0.1, 0.5, 0.7, 1)  # Verde
                    RoundedRectangle(pos=self.custom_indicator.pos,
                                    size=self.custom_indicator.size, radius=[10])
                    
                self.extracted_face_image.texture = self.detected_face_empty_texture
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05,
                                                        minNeighbors=6, minSize=(200, 100))
                faces_detected_qty = len(faces)
                # faces_detected_qty = 1

                if faces_detected_qty > 0:
                    (x, y, w, h) = faces[0]
                    face_img = frame[y:y + h, x:x + w]
                    for (x, y, w, h) in faces:
                        center = (x + w // 2, y + h // 2)
                        axes = (w // 2, h // 2)
                        cv2.ellipse(frame, center, axes, angle=0,
                                    startAngle=0, endAngle=360, color=(0, 255, 0), thickness=2)

                    # Atualiza a tela com o frame da camera com as faces circuladas
                    camera_buf = cv2.resize(frame, CAMERA_FRAME_SIZE).tobytes()
                    texture_camera = Texture.create(size=CAMERA_FRAME_SIZE, colorfmt='bgr')
                    texture_camera.blit_buffer(camera_buf, colorfmt='bgr', bufferfmt='ubyte')
                    self.face_image.texture = texture_camera

                    # Desenha apenas o rosto no frame de detecção
                    display_frame = cv2.resize(face_img, DETECTED_FACE_FRAME_SIZE)
                    buf = display_frame.tobytes()
                    texture = Texture.create(size=DETECTED_FACE_FRAME_SIZE, colorfmt='bgr')
                    texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                    self.extracted_face_image.texture = texture

                    # Prepara o frame para fazer os testes no deepface e o no modelo customizado
                    face_img_resized = cv2.resize(face_img, (244, 244))
                    face_img_resized_rgb = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB)

                    # === Verifica spoofing ===
                    fb_result = self.check_spoof_facebook(face_img_resized_rgb)
                    tf_result = self.check_spoof_tensorflow(face_img_resized_rgb)

                    if fb_result == 1:
                        self.fb_indicator.canvas.before.clear()
                        with self.fb_indicator.canvas.before:
                            Color(0, 0.7, 0, 1)  # Verde
                            RoundedRectangle(pos=self.fb_indicator.pos,
                                            size=self.fb_indicator.size, radius=[10])
                    else:
                        with self.fb_indicator.canvas.before:
                            Color(1, 0, 0, 1)  # Vermelho
                            RoundedRectangle(pos=self.fb_indicator.pos,
                                            size=self.fb_indicator.size, radius=[10])

                    if tf_result == 1:
                        self.custom_indicator.canvas.before.clear()
                        with self.custom_indicator.canvas.before:
                            Color(0, 0.7, 0, 1)  # Verde
                            RoundedRectangle(pos=self.custom_indicator.pos,
                                            size=self.custom_indicator.size, radius=[10])
                    else:
                        with self.custom_indicator.canvas.before:
                            Color(1, 0, 0, 1)  # Vermelho
                            RoundedRectangle(pos=self.custom_indicator.pos,
                                            size=self.custom_indicator.size, radius=[10])

                    # === Preenche dados se não for spoofing ===
                    if fb_result and tf_result == 1:
                        self.label_user_id.text = "Id: 123"
                        self.label_user_name.text = "Nome: Antonio"
                    else:
                        self.label_user_id.text = "Id:"
                        self.label_user_name.text = "Nome:"

                    self.face_detection_counter = 0
                    self.face_detected = True
                else:
                    self.extracted_face_image.texture = self.detected_face_empty_texture
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

    def check_spoof_facebook(self, face_image):
        # Simula: retorna 0 se for spoofing e 1 se for válido
        return 0 if not self.deepface_controller.is_real_face( face_image ) else 1

    def check_spoof_tensorflow(self, face_image):
        input_data = face_image.astype('float32') / 255.0
        input_data = np.expand_dims(input_data, axis=0)  # (1, 244, 244, 3)

        return self.custom_model_controller.is_real_face(input_data)


if __name__ == '__main__':
    FaceDetectionApp().run()
