from segment_anything import SamPredictor, SamAutomaticMaskGenerator
from ts.torch_handler.base_handler import BaseHandler
import torch
import base64
import io
from abc import ABC

import torch
from PIL import Image

from torchvision import transforms as T
import numpy as np
import pickle as pkl


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        super().__init__()
        # actually sam code uses np arrays
        # self.image_processing = T.Compose(
        #     [
        #         T.Resize(1440),
        #         T.CenterCrop(1024),
        #         T.ToTensor(),
        #         # handled by sam itself
        #         # T.Normalize(
        #         #     mean=[0.485, 0.456, 0.406],
        #         #     std=[0.229, 0.224, 0.225]
        #         # )
        #     ]
        # )

    def preprocess(self, data):
        # copied from visionhandler
        images = []
        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            print(row)  # for debugging ...
            # image = row.get("body") or row.get("data")
            # d = row.get("body") or row.get("data")
            # image = d["image"]
            image = row["image"]
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                # image = self.image_processing(image)
                image = np.asarray(image)

            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            prompt_points, prompt_labels = [
                row[s] for s in ["prompt_points", "prompt_labels"]
            ]
            prompt_points, prompt_labels = [
                pkl.loads(x) for x in [prompt_points, prompt_labels]
            ]
            prompt_points, prompt_labels = [
                np.asarray(x) for x in [prompt_points, prompt_labels]
            ]

            images.append([image, prompt_points, prompt_labels])

        # current inferrence code expects a single np array
        return images

    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.

        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.

        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        sp = SamPredictor(self.model)
        results = []
        with torch.no_grad():
            for row in data:
                image, prompt_points, prompt_labels = row
                sp.set_image(image)
                masks, scores, logits = sp.predict(
                    point_coords=prompt_points,
                    point_labels=prompt_labels,
                    multimask_output=True,
                )
                result = [x.tolist() for x in [masks, scores, logits]]
                results.append(result)
        return results

    def postprocess(self, data):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.

        Args:
            data (Torch Tensor): The torch tensor received from the prediction output of the model.

        Returns:
            List: The post process function returns a list of the predicted output.
        """

        # return data.tolist()
        return data
