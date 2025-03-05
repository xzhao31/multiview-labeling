import argparse
import numpy as np
import yaml

from efficientvit.apps.utils import parse_unknown_args
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model

# set model args
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="efficientvit-sam-xl1")
parser.add_argument("--weight_url", type=str, default=None)
parser.add_argument("--multimask", action="store_true")
parser.add_argument("--mode", type=str, default="point")
parser.add_argument("--point", type=str, default=None)
args, opt = parser.parse_known_args()
opt = parse_unknown_args(opt)

# build model
efficientvit_sam = create_efficientvit_sam_model(args.model, True, args.weight_url).cuda().eval()
efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)


def cat_images(image_list: list[np.ndarray], axis=1, pad=20) -> np.ndarray:
    shape_list = [image.shape for image in image_list]
    max_h = max([shape[0] for shape in shape_list]) + pad * 2
    max_w = max([shape[1] for shape in shape_list]) + pad * 2

    for i, image in enumerate(image_list):
        canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        h, w, _ = image.shape
        crop_y = (max_h - h) // 2
        crop_x = (max_w - w) // 2
        canvas[crop_y : crop_y + h, crop_x : crop_x + w] = image
        image_list[i] = canvas

    image = np.concatenate(image_list, axis=axis)
    return image


def mask(raw_image):
    H, W, _ = raw_image.shape
    # points = yaml.safe_load(f"[[{W // 2},{H // 2},{1}]]" if args.point is None else args.point) # if no input assume center is positive
    points = yaml.safe_load(f"[[{W // 2},{H // 2},{1}]]")
    point_coords = [(x, y) for x, y, _ in points]
    point_labels = [l for _, _, l in points]

    efficientvit_sam_predictor.set_image(raw_image)
    masks, _, _ = efficientvit_sam_predictor.predict(
        point_coords=np.array(point_coords),
        point_labels=np.array(point_labels),
        multimask_output=args.multimask,
    )
    plots = [np.expand_dims(binary_mask, axis=2)*255 for binary_mask in masks]
    plots = cat_images(plots, axis=1)
    return plots

# ### debugging
# import time
# import cv2
# ortho = cv2.imread('debug/ortho.png')
# start_time = time.time()
# ortho_mask = mask(ortho)
# print(f'{time.time()-start_time} seconds to segment')
# cv2.imwrite('debug/ortho_mask.png',ortho_mask)