import  numpy as np
import itertools
import random
import math
import os
import time
import scipy.io as scio
import datetime
import re
import matplotlib.pyplot as plt
import pylab
import csv
from skimage import transform, filters
from textwrap import wrap
import cv2
import sys
from PIL import Image
from tqdm import tqdm

from .utils import *
from .get_perturbation_mask import *
from .search import *
from .diverse_subset_selection import *
from .patch_deletion_tree import *




# HYPERPARAMS
ups = 30
prob_thresh = 0.9 # note that this is prob factor. So we are considering 0.9 * full_image_probability
numCategories = 1
node_prob_thresh = 40 # minimum score threshold to expand a node in the sag

beam_width = 5 # suggested values [3,5,10,15]
max_num_roots = 10 # upper limit on number of roots obtained via search - suggested values [10,20,30]
overlap_thresh = 1 # number of patches allowed to overlap in roots - suggested values [0,1,2]
numSuccessors = 15 # should be greater or equal to beam_width - 'q' hyperparam in the paper
num_roots_sag = 3 # max number of roots to be displayed in the sag

input_folder = 'Images'

maxRootSize = 49 # max number of patches allowed for a root

# enable cuda
use_cuda = 0
if torch.cuda.is_available():
    use_cuda = 1
# load DNN model
#model = load_model_new(use_cuda=use_cuda, model_name='vgg19')

def generate_sag(input_img, model, img_label=-1):

    # get blurred image
    img, blurred_img = Get_blurred_img(
                                    input_img,
                                    img_label,
                                    model,
                                    resize_shape=(224, 224),
                                    Gaussian_param=[51, 50],
                                    Median_param=11,
                                    blur_type='Black',
                                    use_cuda=use_cuda)
    
    # get top "numCategories" predicted categories with their probabilities
    top_cp = get_topn_categories_probabilities_pairs(img, model, numCategories, use_cuda=use_cuda)


    for category, probability in tqdm(top_cp, desc="Processing Roots"):

        # get the ground truth label for the given category
        f_groundtruth = open('./GroundTruth1000.txt')
        category_name = f_groundtruth.readlines()[category]
        category_name = category_name[:-2]
        f_groundtruth.close()

        # get perturbation mask
        mask, upsampled_mask = Integrated_Mask(
                                            ups,
                                            img,
                                            blurred_img,
                                            model,
                                            category,
                                            max_iterations=2,
                                            integ_iter=20,
                                            tv_beta=2,
                                            l1_coeff=0.01 * 100,
                                            tv_coeff=0.2 * 100,
                                            size_init=28,
                                            use_cuda=use_cuda)

        # get all DISTINCT roots found via beam search
        roots_mp = beamSearch_topKSuccessors_roots(mask, beam_width, numSuccessors, img, blurred_img, model, category, prob_thresh, probability, max_num_roots, maxRootSize, use_cuda=use_cuda)

        numRoots = len(roots_mp)
        print('numRoots_all = ', numRoots)
        # get maximal set of non-overlapping roots
        maximal_Overlap_mp = []
        numRoots_Overlap = 0
        if numRoots > 0:
            maximal_Overlap_mp = maximal_overlapThresh_set(roots_mp, overlap_thresh)
        else:
            images_no_roots_found += 1
            numRoots_Overlap = len(maximal_Overlap_mp)
            print('numRoots_Overlap = ', numRoots_Overlap)

        # prune number of roots to be shown in the sag
        if numRoots_Overlap > num_roots_sag:
            maximal_Overlap_mp = maximal_Overlap_mp[:num_roots_sag]
            numRoots_Overlap = num_roots_sag


    return maximal_Overlap_mp


def generate_sag_and_subex(input_img, model, img_label=-1, thresholds=[0.8, 0.7, 0.6, 0.5], use_cuda=False):
    # get blurred image
    # input_img here will be path to input img
    img, blurred_img = Get_blurred_img(
        input_img,
        img_label,
        model,
        resize_shape=(224, 224),
        Gaussian_param=[51, 50],
        Median_param=11,
        blur_type='Black',
        use_cuda=use_cuda
    )
    
    # get top "numCategories" predicted categories with their probabilities
    top_cp = get_topn_categories_probabilities_pairs(img, model, numCategories, use_cuda=use_cuda)

    results = {}
    images_no_roots_found = 0  # Initialize this variable

    for category, probability in tqdm(top_cp, desc="Processing Roots"):
        # get the ground truth label for the given category
        with open('./GroundTruth1000.txt') as f_groundtruth:
            category_name = f_groundtruth.readlines()[category].strip()

        # get perturbation mask
        mask, upsampled_mask = Integrated_Mask(
            ups,
            img,
            blurred_img,
            model,
            category,
            max_iterations=2,
            integ_iter=20,
            tv_beta=2,
            l1_coeff=0.01 * 100,
            tv_coeff=0.2 * 100,
            size_init=28,
            use_cuda=use_cuda
        )

        # get all DISTINCT roots found via beam search
        roots_mp = beamSearch_topKSuccessors_roots(
            mask, beam_width, numSuccessors, img, blurred_img, model, category, prob_thresh, probability, max_num_roots, maxRootSize, use_cuda=use_cuda
        )

        numRoots = len(roots_mp)
        print('numRoots_all = ', numRoots)

        # get maximal set of non-overlapping roots
        maximal_Overlap_mp = []
        if numRoots > 0:
            maximal_Overlap_mp = maximal_overlapThresh_set(roots_mp, overlap_thresh)
        else:
            images_no_roots_found += 1

        # prune number of roots to be shown in the sag
        if len(maximal_Overlap_mp) > num_roots_sag:
            maximal_Overlap_mp = maximal_Overlap_mp[:num_roots_sag]

        # Count sub-explanations for each root
        for root in maximal_Overlap_mp:
            mask, _, _ = root
            sub_explanations = generate_sub_explanations(mask)
            sub_explanation_counts = {threshold: 0 for threshold in thresholds}
            sub_explanation_details = []

            for sub_mask in tqdm(sub_explanations, desc="Counting Sub-Explanations", leave=False):
                sub_confidence = get_mask_insertion_prob(sub_mask, img, blurred_img, model, category, use_cuda=use_cuda)
                sub_explanation_details.append((sub_mask, sub_confidence))

                for threshold in thresholds:
                    if sub_confidence >= threshold * probability:
                        sub_explanation_counts[threshold] += 1

            results[root] = {
                'sub_explanations': sub_explanation_details,
                'sub_explanation_counts': sub_explanation_counts
            }

    return results



def generate_sub_explanations(mask):
    # Generate all possible sub-masks by removing one patch at a time
    sub_explanations = []
    for i in range(mask.shape[2]):
        for j in range(mask.shape[3]):
            if mask[0, 0, i, j] == 1:
                sub_mask = mask.copy()
                sub_mask[0, 0, i, j] = 0
                sub_explanations.append(sub_mask)
    return sub_explanations