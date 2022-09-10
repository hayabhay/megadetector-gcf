import logging
import math
import time

import numpy as np
import tensorflow.compat.v1 as tf

logger = logging.getLogger(__name__)


# ==================================================================================
# Code directly used from here
# https://github.com/Microsoft/CameraTraps/blob/master/detection/run_tf_detector.py


def truncate_float(x, precision=3):
    """
    Function for truncating a float scalar to the defined precision.
    For example: truncate_float(0.0003214884) --> 0.000321
    This function is primarily used to achieve a certain float representation
    before exporting to JSON

    Args:
    x         (float) Scalar to truncate
    precision (int)   The number of significant digits to preserve, should be
                      greater or equal 1
    """

    if np.isclose(x, 0):
        return 0
    else:
        # Determine the factor, which shifts the decimal point of x
        # just behind the last significant digit
        factor = math.pow(10, precision - 1 - math.floor(math.log10(abs(x))))
        # Shift decimal point by multiplicatipon with factor, flooring, and
        # division by factor
        return math.floor(x * factor) / factor


class TFDetector:
    """
    A detector model loaded at the time of initialization. It is intended to be used with
    the MegaDetector (TF). The inference batch size is set to 1; code needs to be modified
    to support larger batch sizes, including resizing appropriately.
    """

    # Number of decimal places to round to for confidence and bbox coordinates
    CONF_DIGITS = 3
    COORD_DIGITS = 5

    # An enumeration of failure reasons
    FAILURE_TF_INFER = "Failure TF inference"

    DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD = 0.1  # to include in the output json file

    DEFAULT_DETECTOR_LABEL_MAP = {"1": "animal", "2": "person", "3": "vehicle"}  # available in megadetector v4+

    NUM_DETECTOR_CATEGORIES = 4  # animal, person, group, vehicle - for color assignment

    def __init__(self, model_path):
        """Loads model from model_path and starts a tf.Session with this graph. Obtains
        input and output tensor handles."""
        detection_graph = TFDetector.__load_model(model_path)
        self.tf_session = tf.Session(graph=detection_graph)

        self.image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
        self.box_tensor = detection_graph.get_tensor_by_name("detection_boxes:0")
        self.score_tensor = detection_graph.get_tensor_by_name("detection_scores:0")
        self.class_tensor = detection_graph.get_tensor_by_name("detection_classes:0")

    @staticmethod
    def round_and_make_float(d, precision=4):
        return truncate_float(float(d), precision=precision)

    @staticmethod
    def __convert_coords(tf_coords):
        """Converts coordinates from the model's output format [y1, x1, y2, x2] to the
        format used by our API and MegaDB: [x1, y1, width, height]. All coordinates
        (including model outputs) are normalized in the range [0, 1].
        Args:
            tf_coords: np.array of predicted bounding box coordinates from the TF detector,
                has format [y1, x1, y2, x2]
        Returns: list of Python float, predicted bounding box coordinates [x1, y1, width, height]
        """
        # change from [y1, x1, y2, x2] to [x1, y1, width, height]
        width = tf_coords[3] - tf_coords[1]
        height = tf_coords[2] - tf_coords[0]

        new = [tf_coords[1], tf_coords[0], width, height]  # must be a list instead of np.array

        # convert numpy floats to Python floats
        for i, d in enumerate(new):
            new[i] = TFDetector.round_and_make_float(d, precision=TFDetector.COORD_DIGITS)
        return new

    @staticmethod
    def convert_to_tf_coords(array):
        """From [x1, y1, width, height] to [y1, x1, y2, x2], where x1 is x_min, x2 is x_max
        This is an extraneous step as the model outputs [y1, x1, y2, x2] but were converted to the API
        output format - only to keep the interface of the sync API.
        """
        x1 = array[0]
        y1 = array[1]
        width = array[2]
        height = array[3]
        x2 = x1 + width
        y2 = y1 + height
        return [y1, x1, y2, x2]

    @staticmethod
    def __load_model(model_path):
        """Loads a detection model (i.e., create a graph) from a .pb file.
        Args:
            model_path: .pb file of the model.
        Returns: the loaded graph.
        """
        logger.info("TFDetector: Loading graph...")
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")
        logger.info("TFDetector: Detection graph loaded.")

        return detection_graph

    def _generate_detections_one_image(self, image):
        np_im = np.asarray(image, np.uint8)
        im_w_batch_dim = np.expand_dims(np_im, axis=0)

        # need to change the above line to the following if supporting a batch size > 1 and resizing to the same size
        # np_images = [np.asarray(image, np.uint8) for image in images]
        # images_stacked = np.stack(np_images, axis=0) if len(images) > 1 else np.expand_dims(np_images[0], axis=0)

        # performs inference
        (box_tensor_out, score_tensor_out, class_tensor_out) = self.tf_session.run(
            [self.box_tensor, self.score_tensor, self.class_tensor], feed_dict={self.image_tensor: im_w_batch_dim}
        )

        return box_tensor_out, score_tensor_out, class_tensor_out

    def generate_detections_one_image(self, image, image_id, detection_threshold=DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD):
        """Apply the detector to an image.
        Args:
            image: the PIL Image object
            image_id: a path to identify the image; will be in the "file" field of the output object
            detection_threshold: confidence above which to include the detection proposal
        Returns:
        A dict with the following fields, see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
            - 'file' (always present)
            - 'max_detection_conf'
            - 'detections', which is a list of detection objects containing keys 'category', 'conf' and 'bbox'
            - 'failure'
        """
        result = {"file": image_id}
        try:
            b_box, b_score, b_class = self._generate_detections_one_image(image)

            # our batch size is 1; need to loop the batch dim if supporting batch size > 1
            boxes, scores, classes = b_box[0], b_score[0], b_class[0]

            detections_cur_image = []  # will be empty for an image with no confident detections
            max_detection_conf = 0.0
            for b, s, c in zip(boxes, scores, classes):
                if s > detection_threshold:
                    detection_entry = {
                        "category": TFDetector.DEFAULT_DETECTOR_LABEL_MAP[
                            str(int(c))
                        ],  # use string type for the numerical class label, not int
                        "conf": truncate_float(float(s), precision=TFDetector.CONF_DIGITS),
                        "bbox": TFDetector.__convert_coords(b),
                    }
                    detections_cur_image.append(detection_entry)
                    if s > max_detection_conf:
                        max_detection_conf = s

            result["max_detection_conf"] = truncate_float(float(max_detection_conf), precision=TFDetector.CONF_DIGITS)
            result["detections"] = detections_cur_image

        except Exception as e:
            result["failure"] = TFDetector.FAILURE_TF_INFER
            logger.info("TFDetector: image {} failed during inference: {}".format(image_id, str(e)))

        return result


# ==================================================================================


# -------- Model loader -------
# Global variable to save loaded MegaDetector models to serve subsequent cloud function calls
# This packages the above TF Detector code for use inside a cloud function
# Indexed by model path
MODELS = {}


def get_megadetector_model(model: str) -> TFDetector:
    """Downloads & initializes the MegaDetector Tensorflow Models

    Args:
        model: Google storage path where the megadetector model lives

    Returns:
        TFDetector model

    """
    # global so the config can be updated
    global MODELS

    # Look up cache, if it doesn't exist, get model, persist to disk, load & return
    if model not in MODELS:
        logger.info(f"{model} isn't loaded into memory!")
        start = time.perf_counter()
        MODELS[model] = TFDetector(model)
        logger.info(f"{model} loaded in {time.perf_counter()-start}s")

    else:
        logger.info(f"{model} already in memory!")

    # Return the model
    return MODELS[model]
