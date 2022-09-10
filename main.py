import functions_framework
from google.cloud import storage

from image_manager import load_image
from model_manager import get_megadetector_model


def detect(request):
    """Endpoint to run MegaDetector model inference on an image

    Request payload shape, fields & descriptions:
    NOTE: This could be multiple endpoints but since GCF allows a single endpoint per function,
    the endpoints are mushed with internal branching.
    {
        # The value of "image" is a string
        # If it startswith http/https it represents a web url,
        # If it startswith gs:// it is treated as a google cloud storage url
        # If neither is true, it is treated as a string representing the image binary
        "image": "",

        # Optional field specifying the format. If unspecified, the image is treated as a jpg
        "format": "jpeg/png/",

        # Optional: MegaDetector model address. Note, must be in an accessible google storage bucket
        # Defaults to MegaDetector v4.1
        "model_uri": "",

        # Optional: Detection threshold for megadetector
        "detection_threshold": 0.1,
    }

    Args:
        request: http post request

    Returns:
        Dictionary with Megadetector results

    """
    # Get request params
    params = request.get_json()

    # Get or load MegaDetector model
    # If model is not specified, set the default model value
    model_uri = params.get("model", "")
    if not model_uri:
        # TODO: Update the default model location when deployment is finalized
        model_uri = "gs://wildepod_models/md_v4.1.0.pb"

    model = get_megadetector_model(model_uri)

    # Load the image depending on the source type
    image = load_image(params.get("image"))
    # Use the first 100 character as a filename. This isn't used downstream at the moment so it doesn't really matter what it is
    image_fname = params.get("image")[:100]

    # There are a lot of extremely low confidence detections (~e-3 to e-5) that are not useful
    # and must be ignored regardless of the value passed in
    DEFAULT_DETECTION_THRESHOLD = 0.1
    detection_threshold = float(params.get("detection_threshold", DEFAULT_DETECTION_THRESHOLD))
    detection_threshold = max(DEFAULT_DETECTION_THRESHOLD, detection_threshold)

    result = model.generate_detections_one_image(image, image_fname, detection_threshold=detection_threshold)

    return result
