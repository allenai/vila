from typing import List, Union, Dict, Any, Tuple

import torch
import torch.nn.parallel
from effdet import create_model
from effdet.data.transforms import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    transforms_coco_eval,
)
from PIL import Image
import layoutparser as lp


class InputTransform:
    def __init__(
        self,
        image_size,
        config=None,
        device=None,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):

        self.mean = mean
        self.std = std

        self.transform = transforms_coco_eval(
            image_size,
            interpolation="bilinear",
            use_prefetcher=True,
            fill_color="mean",
            mean=self.mean,
            std=self.std,
        )

        self.mean_tensor = torch.tensor([x * 255 for x in mean]).view(1, 3, 1, 1)
        self.std_tensor = torch.tensor([x * 255 for x in std]).view(1, 3, 1, 1)

    def preprocess(self, image: Image.Image):

        image = image.convert("RGB")
        image_info = {"img_size": image.size}

        input, image_info = self.transform(image, image_info)
        image_info = {
            key: torch.tensor(val).unsqueeze(0) for key, val in image_info.items()
        }

        input = torch.tensor(input).unsqueeze(0)
        input = input.float().sub_(self.mean_tensor).div_(self.std_tensor)

        return input, image_info


class EfficientDetLayoutModel:
    def __init__(
        self,
        model_name,
        model_path,
        num_classes,
        label_map=None,
        device="cpu",
        output_confidence_threshold=0.25,
    ):

        self.model = create_model(
            model_name,
            num_classes=num_classes,
            bench_task="predict",
            pretrained=True,
            checkpoint_path=model_path,
        )
        self.model.to(device)
        self.model.eval()

        self.config = self.model.config
        self.device = device
        self.label_map = label_map if label_map is not None else {}
        self.output_confidence_threshold = output_confidence_threshold

        self.preprocessor = InputTransform(self.config.image_size)

    def detect(self, image: Image.Image):

        input, image_info = self.preprocessor.preprocess(image)

        output = self.model(
            input.to(self.device),
            {key: val.to(self.device) for key, val in image_info.items()},
        )

        bbox_pred = self._fetch_model_prediction(output)
        layout = self._convert_pred_to_layout(bbox_pred)
        return layout

    def _fetch_model_prediction(self, output):

        output = output.cpu().detach()
        results = []
        for index, sample in enumerate(output):
            sample[:, 2] -= sample[:, 0]
            sample[:, 3] -= sample[:, 1]
            for det in sample:
                score = float(det[4])
                if (
                    score < self.output_confidence_threshold
                ):  # stop when below this threshold, scores in descending order
                    break
                coco_det = dict(
                    bbox=det[0:4].tolist(), score=score, category_id=int(det[5])
                )
                results.append(coco_det)

        return results

    def _convert_pred_to_layout(self, pred_bbox: List[List]):
        
        layout = lp.Layout()

        for idx, ele in enumerate(pred_bbox):

            x, y, w, h = ele["bbox"]

            layout.append(
                lp.TextBlock(
                    block=lp.Rectangle(x, y, w + x, h + y),
                    type=self.label_map.get(ele["category_id"], ele["category_id"]),
                    score=ele["score"],
                    id=idx,
                )
            )

        return layout