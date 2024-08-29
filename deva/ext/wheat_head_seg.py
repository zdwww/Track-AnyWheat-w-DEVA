# Reference: https://github.com/IDEA-Research/Grounded-Segment-Anything
import os
from typing import Dict, List
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn.functional as F
import torchvision

from segment_anything import sam_model_registry, SamPredictor
from deva.ext.MobileSAM.setup_mobile_sam import setup_model as setup_mobile_sam
import numpy as np
import torch

from deva.inference.object_info import ObjectInfo

class Detector():
    pass

def get_wheat_detection_model(config: Dict, device: str):

    # Building SAM Model and SAM Predictor
    variant = config['sam_variant'].lower()
    if variant == 'mobile':
        MOBILE_SAM_CHECKPOINT_PATH = config['MOBILE_SAM_CHECKPOINT_PATH']

        # Building Mobile SAM model
        checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
        mobile_sam = setup_mobile_sam()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=device)
        sam = SamPredictor(mobile_sam)
    elif variant == 'original':
        SAM_ENCODER_VERSION = config['SAM_ENCODER_VERSION']
        SAM_CHECKPOINT_PATH = config['SAM_CHECKPOINT_PATH']

        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
            device=device)
        sam = SamPredictor(sam)

    return sam


def segment_with_yolo(config: Dict, sam: SamPredictor,
                      image: np.ndarray, detection_df,
                      min_side: int) -> (torch.Tensor, List[ObjectInfo]):

    sam.set_image(image, image_format='RGB')
    
    # YOLO detection
    detections = Detector()
    num_detect = len(detection_df)
    xyxy = detection_df.iloc[:, :4].to_numpy()
    confidence = detection_df.iloc[:, 4].to_numpy()
    class_id = detection_df.iloc[:, 5].to_numpy()
    nms_idx = torchvision.ops.nms(torch.from_numpy(xyxy),
                              torch.from_numpy(confidence),
                              1.0).numpy().tolist()
    assert num_detect == len(nms_idx)
    detections.xyxy = xyxy[nms_idx]
    detections.confidence = confidence[nms_idx]
    detections.class_id = class_id[nms_idx]
    print(f"Total of {num_detect} wheat heads detected.")

    result_masks = []
    for box in detections.xyxy:
        masks, scores, _ = sam.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])

    detections.mask = np.array(result_masks)

    h, w = image.shape[:2]
    if min_side > 0:
        scale = min_side / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        new_h, new_w = h, w

    output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device='cuda')
    curr_id = 1
    segments_info = []

    # sort by descending area to preserve the smallest object
    # for i in np.flip(np.argsort(detections.area)):
    for i in range(len(detections.mask)):
        mask = detections.mask[i]
        confidence = detections.confidence[i]
        class_id = detections.class_id[i]
        mask = torch.from_numpy(mask.astype(np.float32))
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (new_h, new_w), mode='bilinear')[0, 0]
        mask = (mask > 0.5).float()

        if mask.sum() > 0:
            output_mask[mask > 0] = curr_id
            segments_info.append(ObjectInfo(id=curr_id, category_id=class_id, score=confidence))
            curr_id += 1

    return output_mask, segments_info
