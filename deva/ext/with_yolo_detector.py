import os
from os import path
from typing import Dict, List

import cv2
import torch
import numpy as np
import pandas as pd

from deva.inference.object_info import ObjectInfo
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.frame_utils import FrameInfo
from deva.inference.result_utils import ResultSaver
from deva.inference.demo_utils import get_input_frame_for_deva
from deva.ext.wheat_head_seg import segment_with_yolo
from segment_anything import SamPredictor


def make_segmentation_with_yolo(cfg: Dict, image_np: np.ndarray, detection_df, 
                                sam_model: SamPredictor,
                                min_side: int) -> (torch.Tensor, List[ObjectInfo]):
    mask, segments_info = segment_with_yolo(cfg, sam_model, image_np, detection_df, min_side)
    return mask, segments_info


@torch.inference_mode()
def process_frame_with_yolo(deva: DEVAInferenceCore,
                            sam_model: SamPredictor,
                            frame_path: str,
                            result_saver: ResultSaver,
                            ti: int,
                            image_np: np.ndarray = None) -> None:
    # image_np, if given, should be in RGB
    if image_np is None:
        image_np = cv2.imread(frame_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cfg = deva.config
    raw_prompt = cfg['prompt']
    prompts = raw_prompt.split('.')

    h, w = image_np.shape[:2]
    new_min_side = cfg['size']
    need_resize = new_min_side > 0
    image = get_input_frame_for_deva(image_np, new_min_side)

    frame_name = path.basename(frame_path)
    frame_info = FrameInfo(image, None, None, ti, {
        'frame': [frame_name],
        'shape': [h, w],
    })

    # Load saved yolo detection results
    detection_path = frame_path.replace("images", "yolo_results").replace(".png", ".pkl")
    assert os.path.exists(detection_path), f"Detection results not found in {detection_path}"
    detection_df = pd.read_pickle(detection_path)

    if cfg['temporal_setting'] == 'semionline':
        if ti + cfg['num_voting_frames'] > deva.next_voting_frame:
            mask, segments_info = make_segmentation_with_yolo(cfg, image_np, detection_df, sam_model,
                                                              new_min_side)
            frame_info.mask = mask
            frame_info.segments_info = segments_info
            frame_info.image_np = image_np  # for visualization only
            # wait for more frames before proceeding
            deva.add_to_temporary_buffer(frame_info)

            if ti == deva.next_voting_frame:
                # process this clip
                this_image = deva.frame_buffer[0].image
                this_frame_name = deva.frame_buffer[0].name
                this_image_np = deva.frame_buffer[0].image_np

                _, mask, new_segments_info = deva.vote_in_temporary_buffer(
                    keyframe_selection='first')
                prob = deva.incorporate_detection(this_image, mask, new_segments_info)
                deva.next_voting_frame += cfg['detection_every']

                result_saver.save_mask(prob,
                                       this_frame_name,
                                       need_resize=need_resize,
                                       shape=(h, w),
                                       image_np=this_image_np,
                                       prompts=prompts)

                for frame_info in deva.frame_buffer[1:]:
                    this_image = frame_info.image
                    this_frame_name = frame_info.name
                    this_image_np = frame_info.image_np
                    prob = deva.step(this_image, None, None)
                    result_saver.save_mask(prob,
                                           this_frame_name,
                                           need_resize,
                                           shape=(h, w),
                                           image_np=this_image_np,
                                           prompts=prompts)

                deva.clear_buffer()
        else:
            # standard propagation
            prob = deva.step(image, None, None)
            result_saver.save_mask(prob,
                                   frame_name,
                                   need_resize=need_resize,
                                   shape=(h, w),
                                   image_np=image_np,
                                   prompts=prompts)

    elif cfg['temporal_setting'] == 'online':
        if ti % cfg['detection_every'] == 0:
            # incorporate new detections
            mask, segments_info = make_segmentation_with_yolo(cfg, image_np, detection_df, sam_model,
                                                              new_min_side)
            frame_info.segments_info = segments_info
            prob = deva.incorporate_detection(image, mask, segments_info)
        else:
            # Run the model on this frame
            prob = deva.step(image, None, None)
        result_saver.save_mask(prob,
                               frame_name,
                               need_resize=need_resize,
                               shape=(h, w),
                               image_np=image_np,
                               prompts=prompts)
