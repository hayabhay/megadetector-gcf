import base64
from io import BytesIO
import logging

from PIL import Image
from dotenv import dotenv_values
from google.cloud import storage
import numpy as np
import requests

# Setup logger & environment variables
logger = logging.getLogger(__name__)
env = dotenv_values(".env")

PROJECT_ID = env.get("PROJECT_ID", "")
storage_client = storage.Client(PROJECT_ID)

BUCKETS = {}

# Function to get a PIL Image from google cloud storage from a gs:// url
def load_gs_image(image_gs_loc: str) -> Image:
    """Function to load an image directly from a google storage link"""
    # First get the bucket name from the url.
    bucket_name = image_gs_loc[5:].split("/")[0]
    file_path = "/".join(image_gs_loc[5:].split("/")[1:])

    if bucket_name not in BUCKETS:
        BUCKETS[bucket_name] = storage_client.get_bucket(bucket_name)

    blob = BUCKETS[bucket_name].get_blob(file_path)
    binary_string = blob.download_as_string()

    image = Image.open(BytesIO(binary_string))

    return image


# Function to get a PIL Image from an HTTP url
def load_web_image(image_url: str) -> Image:
    """Function to load an image from a web url"""
    # Retrieve the image content
    response = requests.get(image_url)

    # Directly try to load the returned data with pillow. If it fails, handle it downstream.
    image = Image.open(BytesIO(response.content))

    return image


# Function to get a PIL Image from a string
def load_image_from_binary_string(image_byte_string: str) -> Image:
    """Function to load an image directly from a binary string"""
    # Directly try to load the returned data with pillow. If it fails, handle it downstream.
    image_binary = base64.decodebytes(image_byte_string.encode())

    image = Image.open(BytesIO(image_binary))

    return image


# Function to load an image from the web, gs or a binary string
def load_image(image_src: str) -> Image:
    """Function that takes in a string, determines the source and returns a correctly formatted PIL Image"""
    # Load the image depending on the type
    if image_src.startswith(("http://", "https://")):
        image = load_web_image(image_src)
    elif image_src.startswith("gs://"):
        image = load_gs_image(image_src)
    else:
        image = load_image_from_binary_string(image_src)

    # ==========================================================================================
    # THIS CODE IS FROM
    # https://github.com/microsoft/CameraTraps/blob/master/visualization/visualization_utils.py
    if image.mode not in ("RGBA", "RGB", "L", "I;16"):
        raise AttributeError(f"Image {image_src} uses unsupported mode {image.mode}")

    if image.mode == "RGBA" or image.mode == "L":
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode="RGB")
    # ==========================================================================================

    return image
