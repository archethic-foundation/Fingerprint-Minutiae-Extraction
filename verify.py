import cv2 as cv
from glob import glob
import os
import numpy as np
from poincare import calculate_singularities
from segmentation import create_segmented_and_variance_images
from normalization import normalize
from gabor_filter import gabor_filter
from frequency import ridge_freq
import orientation
from crossing_number import calculate_minutiaes
from tqdm import tqdm
from skeletonize import skeletonize
import pathlib
import json
from collections import Counter

import math

file = open('test.txt', 'w')

def calculate_minutiae_singularity_angle(minutiae, singularities):
    """
    Calculates the angle between two vectors: 
    - Vector 1: Minutia → Top-left singularity
    - Vector 2: Minutia → Bottom-right singularity

    :param minutiae: List of lists, where each list contains a minutia defined by [x, y]
    :param singularities: List of tuples, where each tuple is a rectangle defined by ((top-left x, top-left y), (bottom-right x, bottom-right y))
    :return: A list of lists, where each sublist contains a single angle (in degrees) for each minutia
    """


    if len(singularities) != 2:
        return []

    singularities_center = []
    for singularity in singularities:
        top_left, bottom_right = singularity
        center_x = (top_left[0] + bottom_right[0]) / 2
        center_y = (top_left[1] + bottom_right[1]) / 2

        singularities_center.append([center_x, center_y])

    midpoint = [(singularities_center[0][0]+singularities_center[1][0]) / 2, (singularities_center[0][1]+singularities_center[1][1]) / 2]

    distanceSingularities = math.sqrt(
        (singularities_center[0][0] - singularities_center[1][0]) ** 2
        + (singularities_center[0][1] - singularities_center[1][1]) ** 2
    )

    angles_ratio_distances = []
    for minutia in minutiae:
        minutia_x, minutia_y = minutia  # Unpack [x, y]

        vectorAB = [singularities_center[0][0] - minutia_x, singularities_center[0][1] - minutia_y]
        vectorAC = [midpoint[0] - minutia_x, midpoint[1] - minutia_y]

        dot_product = (vectorAB[0] * vectorAC[0]) + (vectorAB[1] * vectorAC[1])
        mag_AB = math.sqrt(vectorAB[0]**2 + vectorAB[1]**2)
        mag_AC = math.sqrt(vectorAC[0]**2 + vectorAC[1]**2)

        cos_theta = dot_product / (mag_AB * mag_AC)
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)

        if angle_deg == 0:
            continue

        distance = math.sqrt((minutia_x - midpoint[0]) ** 2 + (minutia_y-midpoint[1]) **2)
        distance_ratio = distance / distanceSingularities

        angles_ratio_distances.append((round(angle_deg), round(distance_ratio, 1)))

    return  angles_ratio_distances

def f(input_img):
    # normalization -> orientation -> frequency -> mask -> filtering
    block_size = 16

    # normalization - removes the effects of sensor noise and finger pressure differences.
    normalized_img = normalize(input_img.copy(), float(100), float(100))

    # ROI and normalisation
    (segmented_img, normim, mask) = create_segmented_and_variance_images(
        normalized_img, block_size, 0.2
    )

    # orientations
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    orientation_img = orientation.visualize_angles(
        segmented_img, mask, angles, W=block_size
    )

    # find the overall frequency of ridges in Wavelet Domain
    freq = ridge_freq(
        normim,
        mask,
        angles,
        block_size,
        kernel_size=5,
        minWaveLength=5,
        maxWaveLength=15,
    )

    # create gabor filter and do the actual filtering
    gabor_img = gabor_filter(normim, angles, freq)

    # thinning oor skeletonize
    thin_image = skeletonize(gabor_img)

    # singularities
    singularities_img, singularities = calculate_singularities(thin_image, angles, 1, block_size, mask)

     # minutias
    minutias_img, minutias = calculate_minutiaes(thin_image)

    ending = minutias.get("ending")
    bifurcation = minutias.get("bifurcation")

    angles = { 
        "ending": calculate_minutiae_singularity_angle(ending, singularities), 
        "bifurcation": calculate_minutiae_singularity_angle(bifurcation, singularities)
    }

    return angles

import sys
import itertools
from hashlib import sha256

def generate_combinations(data, sample_size):
    # """Generate random combinations from the given list."""
    # return [random.sample(data, sample_size) for _ in range(num_combinations)]
    """Generate all possible unique combinations from the given list."""
    return itertools.combinations(data, sample_size)

if __name__ == "__main__":
    if len(sys.argv) == 0:
        print("Missing argument")
        exit()

    image = sys.argv[1]
    # open images
    image_path = f"./input/{image}.png"

    def process_image(img_path):
      path = pathlib.PurePath(img_path)
      img = cv.imread(img_path, 0)
      if img is None:
          print(f"Failed to load image: {img_path}")
      else:
        angles = f(img)

        bifurcations_combinations = generate_combinations(angles.get("bifurcation"), 3)

        bifurcations_combinations_id = {}
        for i, combination in enumerate(bifurcations_combinations, 1):
            print(combination)
            digest = sha256(str(combination).encode('utf-8')).hexdigest()
            bifurcations_combinations_id[digest] = combination

        json_file_path =  './verify_combinations.json' 
        with open(json_file_path, 'w') as json_file:
            json.dump(bifurcations_combinations_id, json_file)


    process_image(image_path)