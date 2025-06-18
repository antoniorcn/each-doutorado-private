from deepface import DeepFace
import chromadb
from chromadb.config import Settings
import numpy as np
import uuid

class DeepFaceRecognitionController:
    def __init__(self, **kwargs):
        self.model_names = ["VGG-Face", "Facenet", "ArcFace"]
        self.metrics = ["cosine", "euclidean"]
        self.todas_faces = {}
        self.load_database_to_memory()

    def load_database_to_memory(self):
        self.client = chromadb.PersistentClient(path="db/")
        self.collection = self.client.get_or_create_collection(
            name='facedb',
            metadata={
                "hnsw:space": 'cosine',
            },
        )

    def cadastrar_imagem(self, image_name, pil_img):
        embedding_objs = DeepFace.represent(np.array(pil_img), enforce_detection=False)

        self.collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding_objs[0]["embedding"]],
            metadatas=[{'name': image_name}]
        )

    def extract_faces(self, pil_img):
        face_objs = DeepFace.extract_faces(np.array(pil_img),
                                           anti_spoofing = True, enforce_detection = False)
        return face_objs

    def is_real_face(self, pil_img):
        face_objs = self.extract_faces(pil_img)
        return not all(face_obj["is_real"] is True for face_obj in face_objs)
    
    def pesquisar_imagem(self, pil_img):
        faces_represented = DeepFace.represent(np.array(pil_img), enforce_detection=False)
        print("Faces Represented: ", faces_represented)
        faces = {}
        for face in faces_represented:
            if "facial_area" in face and "embedding" in face:
                results = self.collection.query(
                    query_embeddings=[faces_represented[0]["embedding"]],  # Replace with your unknown embeddings
                    n_results=5
                )
                print("Results: ", results)
                print("Results Metadatas: ", results["metadatas"])
                print("Results Metadatas[0]: ", results["metadatas"][0])
                print("Results Metadatas[0][0]: ", results["metadatas"][0][0])
                if len(results["metadatas"]) > 0 and len(results["distances"]) > 0:
                    faces[results["metadatas"][0][0]["name"]] = face["facial_area"]
        return faces