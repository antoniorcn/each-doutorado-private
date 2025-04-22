import tensorflow as tf
import numpy as np
import PIL
from PIL import Image
from tensorflow.keras.layers import Embedding
from spoofing.edu.ic.logger import get_logger

logger = get_logger(__name__)

def normalize_images(images, normalization=255.0):
    images = images / normalization
    return images

def resize_images_to_tensors(images, image_size):
    # Convert to tf.float64 and expand the dimensions
    images_tensor = tf.cast(images, tf.float64)
    # Resize image
    images_tensor = tf.image.resize(images_tensor, size=image_size)
    return images_tensor

def create_embeddings_from_tensors(images_tensor, model):
    # Generate the image embedding
    embeddings = model(images_tensor)
    return embeddings

def create_embeddings_from_images(images, model, model_type="mobilenet"):
    if model_type == "mobilenet":
        images = normalize_images(images, normalization=255.0)
        image_size = (224, 224)
    images_tensor = resize_images_to_tensors(images, image_size)
    embeddings = create_embeddings_from_tensors(images_tensor, model)
    return embeddings


if __name__ == "__main__":
    # Create embeddings from a randomly selected number of images
    logger.info(f"Pillow Version: {PIL.__version__}")
    logger.info(f"Executando ...")
    image = Image.open("../../resources/man-photo.jpg")
    # convert image to numpy array
    data = np.asarray(image)
    logger.debug(f"Image data Type: {type(data)}")
    logger.debug(f"Image data Shape: {data.shape}")

    #Generate Embedding
    max_sequence_length = data.shape[0] * data.shape[1] * data.shape[2]
    embed = Embedding(max_sequence_length, 64)
    #
    embeddings  = create_embeddings_from_images(data, model=embed, model_type="mobilenet")
    logger.info(f"Embeddings.shape: {embeddings.shape}, images.shape: {data.shape}")