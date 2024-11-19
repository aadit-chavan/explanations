from structured_attention_graphs.generation import generate_sag
from structured_attention_graphs.generation import generate_sag_and_subex
from structured_attention_graphs.utils import load_model_new
import os
import numpy as np
import json

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

model_list = ['resnet50', 'vgg19', 'resnet50_c1', 'resnet50_c2', 'resnet50d', 'swin_t', 'deit_s', 'pit_s', 'deit_s_distilled', 'pit_s_distilled', 'levit_256', 'convnext_t']

final_dict = {}

image_dir = '/users/k24085355/data/imagenet_val/imagenet_val/images'
image_dir_list = os.listdir(image_dir)

for modelnm in model_list:
    final_dict[modelnm] = {}
    model = load_model_new(use_cuda=1, model_name=modelnm)
    for img_nm in image_dir_list[:10]:
        img_path = image_dir + '/' + img_nm
        model_results = generate_sag_and_subex(input_img=img_path, model=model)
        final_dict[modelnm][img_nm] = model_results

with open('results10.json', 'w') as json_file:
    json.dump(final_dict, json_file, default=convert_to_serializable, indent=4)


