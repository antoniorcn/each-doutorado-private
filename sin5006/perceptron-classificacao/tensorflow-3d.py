import os
import sys
import math
import threading
import pygame
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Layer, Input, Conv2D, Dense, MaxPooling2D
from tensorflow.keras.models import Sequential
from pygame import Surface
from typing import Dict, List, Tuple


class TensorFlowAnimation():

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Tensorflow Animation")
        print("TensorFlow Animation iniciado")
        self.clock = pygame.time.Clock()

        # ========================
        # Câmera
        # ========================
        self.camera_pos = [0, -300, -1000]
        self.yaw = math.radians(-20)
        self.pitch = math.radians(15)
        self.fov = 500

        self.dragging = False
        self.rotating = False
        self.last_mouse = (0, 0)
        print("Dados da cmaera iniciados")

        self.imagens = []

        self.layers : Dict [ str, Layer ] = {}
        self.elementos = []
        print("Listas iniciais criadas")

    class ObservedLayer(Layer):
        def __init__(self, externa_ref, layer, **kwargs):
            super().__init__(**kwargs)
            self.layer = layer
            self.externa_ref = externa_ref
            # self.on_pass = on_pass

        def call(self, inputs):
            outputs = self.layer(inputs)
            if self.externa_ref:
                self.externa_ref.update(self.layer.name, inputs, outputs)
            return outputs

    def observe(self, layer : Layer) -> Layer:
        print(f"observe() invocado - Layer: {layer.name} ")
        self.layers[ layer.name ] = self.ObservedLayer( self, layer )
        return self.layers[ layer.name ]


    def update(self, layer_name : str, inputs : any, outputs : any):
        print("Update() invocado")
        print(f"Layer: {layer_name}\tInput Shape: {inputs.shape}\tOutput Shape: {outputs.shape}")

    # ========================
    # Vetores e projeção
    # ========================
    def rotate_yaw_pitch(self, point):
        # print("rotate_yaw_pitch() invocado")
        x, y, z = point

        # Rotação em Y (yaw)
        x1 = x * math.cos(self.yaw) - z * math.sin(self.yaw)
        z1 = x * math.sin(self.yaw) + z * math.cos(self.yaw)

        # Rotação em X (pitch)
        y2 = y * math.cos(self.pitch) - z1 * math.sin(self.pitch)
        z2 = y * math.sin(self.pitch) + z1 * math.cos(self.pitch)

        return [x1, y2, z2]

    def project(self, x, y, z):
        # print("project() invocado")
        relative = [x - self.camera_pos[0], y - self.camera_pos[1], z - self.camera_pos[2]]
        rotated = self.rotate_yaw_pitch(relative)
        if rotated[2] <= 1:
            return None
        local_scale = self.fov / rotated[2]
        local_x_proj = int(400 + rotated[0] * local_scale)
        local_y_proj = int(300 - rotated[1] * local_scale)
        return local_x_proj, local_y_proj, local_scale

    # ========================
    # Carregar imagens
    # ========================

    @staticmethod
    def processar_imagem(image_path : str, tamanho: Tuple[int, int]=(244, 244)) -> np.ndarray:
        # logger.debug(f"Lendo imagem de: {image_path}")
        imagem = Image.open(image_path)

        # Garantir que a imagem tenha 3 canais (RGB)
        imagem = imagem.convert("RGB")

        # Redimensionar a imagem para 244x244
        imagem = imagem.resize(tamanho)

        # Converter a imagem em um array numpy
        pixels = np.array(imagem, dtype=np.float32)

        # Normalizar os pixels para o intervalo [0, 1]
        pixels /= 255.0
        return pixels
    
    @staticmethod
    def processar_imagem_tf(image_path, label, tamanho=(224, 224)):
        image = tf.io.read_file(image_path)  # lê o conteúdo do arquivo
        # Detecta a extensão
        is_jpg = tf.strings.regex_full_match(image_path, ".*\\.jpe?g")

        # Condicional de decodificação
        if is_jpg:
            image = tf.image.decode_jpeg(image, channels=3)
        else:
            image = tf.image.decode_png(image, channels=3)
        print(f"Imagem: {image}  decode usado {'JPG' if is_jpg else 'PNG'} redimensionando para o tamanho {tamanho}")
        image = tf.image.resize(image, tamanho)  # redimensiona
        print("Imagem redimensionada")
        image = tf.cast(image, tf.float32) / 255.0  # normaliza
        print("Imagem normalizada")
        print(f"Retornando Imagem: {image}   Label: {label}")
        return image, label
    
    @staticmethod
    def criar_dataset(
        image_paths_labels : List[Tuple[str, int]],
        tamanho: Tuple[int, int] = (224, 224),
        batch_size: int = 32
    ):
        caminhos, labels = zip(*image_paths_labels)

        caminhos = list(caminhos)
        labels = list(labels)

        ds = tf.data.Dataset.from_tensor_slices((caminhos, labels))
        print("dataset criado")

        # Aplica o pré-processamento
        ds = ds.map(lambda x, y: TensorFlowAnimation.processar_imagem_tf(x, y, tamanho), num_parallel_calls=tf.data.AUTOTUNE)

        # Agrupa em batches
        ds = ds.batch(batch_size)
        print("dataset agrupado em batches")

        # Prefetch para performance
        ds = ds.prefetch(tf.data.AUTOTUNE)
        print("aplicado prefetch para performance")

        return ds

    def carregar_imagens(self, pasta='images'):
        print("carregar_imagens() invocado")
        for nome in sorted(os.listdir(pasta)):
            if nome.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(pasta, nome)
                print("Carregando imagem: ", img_path)
                pixels = TensorFlowAnimation.processar_imagem(img_path, tamanho=(244, 244))
                # surface = pygame.surfarray.make_surface(np.transpose(pixels, (1, 0, 2)))
                # img = pygame.image.load(os.path.join(pasta, nome)).convert_alpha()
                # img = pygame.transform.scale(img, (200, 150))
                self.imagens.append(pixels)
        return self.imagens
    
    @staticmethod
    def generate_surface_from_nparray( pixels ):
        surface = pygame.surfarray.make_surface(np.transpose(pixels * 255, (1, 0, 2)))
        return pygame.transform.scale(surface, (200, 150))

    # ========================
    # Gerar esfera
    # ========================
    def gerar_esfera(self, passo=10, raio=80):
        print("gerar_esfera() invocado")
        pontos = []
        linhas = []
        for phi in range(-90, 91, passo):
            for theta in range(0, 361, passo):
                p1 = self.ponto_esfera(raio, math.radians(theta), math.radians(phi))
                pontos.append(p1)

        # Conectar linhas de latitude e longitude
        n_theta = len(range(0, 361, passo))
        for i in range(len(pontos)):
            if (i + 1) % n_theta != 0:
                linhas.append((pontos[i], pontos[i + 1]))
            if i + n_theta < len(pontos):
                linhas.append((pontos[i], pontos[i + n_theta]))

        return pontos, linhas

    def ponto_esfera(self, raio, theta, phi):
        x = raio * math.cos(phi) * math.cos(theta)
        y = raio * math.sin(phi)
        z = raio * math.cos(phi) * math.sin(theta)
        return [x, y, z]

    def inicializar_cena(self):
        print("inicializar_cena() invocado")
        # ========================
        # Inicialização da cena
        # ========================
        for i, img in enumerate(self.imagens):
            self.elementos.append({
                'tipo': 'quadro',
                'img': TensorFlowAnimation.generate_surface_from_nparray(img),
                'pos': [0, 0, i * 300]
            })
            if (i + 1) % 3 == 0:
                pontos, linhas = self.gerar_esfera()
                self.elementos.append({
                    'tipo': 'esfera',
                    'pontos': pontos,
                    'linhas': linhas,
                    'centro': [0, 0, (i + 1) * 300 + 150]
                })

    def draw(self):
        # print("draw() invocado")
        self.screen.fill((25, 25, 40))

        # Ordenar os elementos por profundidade (usando a escala da projeção como proxy)
        elementos_render = []

        for elem in self.elementos:
            if elem['tipo'] == 'quadro':
                proj = self.project(*elem['pos'])
                if proj:
                    elementos_render.append((proj[2], elem, proj))
            elif elem['tipo'] == 'esfera':
                proj = self.project(*elem['centro'])
                if proj:
                    elementos_render.append((proj[2], elem, proj))

        # Ordena do mais distante (menor escala) para o mais próximo (maior)
        elementos_render.sort(reverse=False)

        # Renderiza na ordem correta
        for _, elem, proj in elementos_render:
            if elem['tipo'] == 'quadro':
                x_proj, y_proj, scale = proj
                img_scaled = pygame.transform.scale(elem['img'], (
                    int(200 * scale),
                    int(150 * scale)
                ))
                self.screen.blit(img_scaled, (x_proj - img_scaled.get_width() // 2, y_proj - img_scaled.get_height() // 2))

            elif elem['tipo'] == 'esfera':
                cx, cy, cz = elem['centro']
                for a, b in elem['linhas']:
                    p1 = self.project(cx + a[0], cy + a[1], cz + a[2])
                    p2 = self.project(cx + b[0], cy + b[1], cz + b[2])
                    if p1 and p2:
                        pygame.draw.line(self.screen, (100, 200, 255), p1[:2], p2[:2], 1)
                for p in elem['pontos']:
                    pp = self.project(cx + p[0], cy + p[1], cz + p[2])
                    if pp:
                        pygame.draw.circle(self.screen, (255, 255, 255), pp[:2], 2)

        pygame.display.flip()
        self.clock.tick(60)

    def execute(self):
        print("running() invocado")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.dragging = True
                    elif event.button == 3:
                        self.rotating = True
                    self.last_mouse = pygame.mouse.get_pos()

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False
                    elif event.button == 3:
                        self.rotating = False

                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging or self.rotating:
                        mx, my = event.pos
                        dx = mx - self.last_mouse[0]
                        dy = my - self.last_mouse[1]
                        self.last_mouse = (mx, my)

                        if self.dragging:
                            self.camera_pos[0] -= dx * 1.5
                            self.camera_pos[1] += dy * 1.5
                        elif self.rotating:
                            self.yaw += dx * 0.01
                            self.pitch += dy * 0.01
                            self.pitch = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, self.pitch))

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.camera_pos[2] += 20
            if keys[pygame.K_s]:
                self.camera_pos[2] -= 20

            self.draw()
        pygame.quit()



# ========================
# Loop principal
# ========================
tfa = TensorFlowAnimation()



input_images = tfa.carregar_imagens()
tfa.inicializar_cena()
# print("Shape: ", input_images.shape)
y = [0 for i in input_images]
print("Saida: ", y)
model = Sequential()
model.add(Input(shape=(12, 244, 244, 3)) )
model.add(Conv2D(64, (7, 7), strides=2, padding='same', activation='relu') )
model.add(MaxPooling2D((3, 3), strides=2, padding='same') )
model.add(Dense(1, activation="sigmoid"))
model.compile( optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"] )
model.fit( input_images, y, epochs=3 )

def run_model():
    pass
    

tf_thread = threading.Thread(target=run_model)
pg_thread = threading.Thread(target=tfa.execute)

tf_thread.start()
pg_thread.start()

tf_thread.join()
pg_thread.join()

