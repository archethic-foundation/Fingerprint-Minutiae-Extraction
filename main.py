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

def get_angles_min_frequenties(images_angles, threshold):
    frequencies = {}
    for i, singularity_angles in enumerate(images_angles):
        for singularity_angle in singularity_angles:
                angle, ratio = singularity_angle

                # if angle > 0:
                frequency = frequencies.get((angle, ratio))
                if frequency == None:
                    frequencies[(angle, ratio)] = { "images": [i] }
                else:
                    existing_images = frequencies[(angle, ratio)]["images"]
                    
                    if i not in existing_images:
                        frequencies[(angle, ratio)]["images"].append(i)


    angles = []

    for angle_singularity, frequency in frequencies.items():
        images = frequency['images']

        if len(images) >= threshold:
            print(f"{angle_singularity} present in {len(images)} images: {images}")
            angles.append(angle_singularity)

#     # for image, frequencies in angles_per_image.items():
#     #     print(f"Image #{image} contains {frequencies} minutiae which where present in more than {threshold} images")


    return angles

# def euclidean_distance(point1, point2):
#     return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

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

        # distance = math.sqrt((minutia-vectorAC[0]**2 + vectorAC[1]**2)
        # print((round(angle_deg), nearest_ten(distance)))

    return  angles_ratio_distances


    # for singularity in singularities:
    #     top_left, bottom_right = singularity
    #     minutiae_angles = []

    #     # # Compute the midpoint of the singularity (center of the rectangle)
    #     # center_x = (top_left[0] + bottom_right[0]) / 2
    #     # center_y = (top_left[1] + bottom_right[1]) / 2

    #     for minutia in minutiae:
    #         minutia_x, minutia_y = minutia  # Unpack [x, y]

            # distance = math.sqrt(center_x ** 2 + center_y ** 2)

            # diff_adding_translation_as_origin = [0 - minutia_x, 0 - minutia_y]
            # v1_x = top_left[0] + diff_adding_translation_as_origin[0]
            # v1_y = top_left[1] + diff_adding_translation_as_origin[1]

            # v2_x = bottom_right[0] + diff_adding_translation_as_origin[0]
            # v2_y = bottom_right[1] + diff_adding_translation_as_origin[1]

            # dot_product = v1_x * v2_x + v1_y * v2_y

            # # Compute magnitudes
            # mag_v1 = math.sqrt(v1_x**2 + v1_y**2)
            # mag_v2 = math.sqrt(v2_x**2 + v2_y**2)

            # # Avoid division by zero
            # if mag_v1 == 0 or mag_v2 == 0:
            #     angle_deg = 0  # If one vector is zero, angle is undefined (set to 0)
            # else:
            #     # Compute the cosine of the angle, ensuring it's within [-1,1]
            #     cos_theta = dot_product / (mag_v1 * mag_v2)
            #     cos_theta = max(-1, min(1, cos_theta))  # Clamp value to valid range

            #     # Compute angle
            #     angle_rad = math.acos(cos_theta)
            #     angle_deg = math.degrees(angle_rad)

            # print(f"{top_left[0] ** 2} + {top_left[1] ** 2}")

            # Pytaghore
            # AC = math.sqrt((v1_x ** 2) + (v1_y ** 2))
            # BC = math.sqrt((v2_x ** 2) + (v2_y ** 2))
            # AB = math.sqrt(abs(BC**2 - AC**2))

            # roundedAB = round(distance)
            # modulo = roundedAB % 10
            # if modulo > 5:
            #     roundedAB = roundedAB + (10 - modulo)
            # else:
            #     roundedAB = roundedAB - modulo

            # print(f"{round(angle_deg)}° dist: {roundedAB}")

            # distance_12 = euclidean_distance((0, 0), (v1_x, v1_y))
            # distance_13 = euclidean_distance((0, 0), (v2_x, v2_y))
            # distance_23 = euclidean_distance(top_left, bottom_right)

    #         minutiae_angles.append((singularity, (minutia_x, minutia_y), 0, (round_modulo(distance_12), round_modulo(distance_13))))

    #     minutiae_angle_all_singularities.append(minutiae_angles)

    # return minutiae_angle_all_singularities

def nearest_ten(number):
    number = round(number)
    modulo = number % 10
    if modulo > 5:
        number = number + (10 - modulo)
    else:
        number = number - modulo

    return number

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

    # visualize pipeline stage by stage
    output_imgs = [
        input_img,
        normalized_img,
        segmented_img,
        orientation_img,
        gabor_img,
        thin_image,
        minutias_img,
        singularities_img,
    ]

    for i in range(len(output_imgs)):
        if len(output_imgs[i].shape) == 2:
            output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)

    results = np.concatenate(
        [np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]
    ).astype(np.uint8)

    return (results, minutias, angles)

import sys
import itertools
from hashlib import sha256

def generate_combinations(data, sample_size):
    # """Generate random combinations from the given list."""
    # return [random.sample(data, sample_size) for _ in range(num_combinations)]
    """Generate all possible unique combinations from the given list."""
    return itertools.combinations(data, sample_size)

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 0 else "*"
    # open images
    img_dir = f"./input/{folder}/*.png"
    output_dir = "./output/"

    def process_images(directory):
        images_paths = glob(directory)

        angles_acc = { "ending": [], "bifurcation": []}

        for i, img_path in enumerate(tqdm(images_paths)):
            path = pathlib.PurePath(img_path)
            img = cv.imread(img_path, 0)
            if img is None:
                print(f"Failed to load image: {img_path}")
            else:
                results, minutias, angles = f(img)

                # angles_acc["ending"].append(angles.get("ending"))

                if len(angles.get("bifurcation")) > 0:
                    angles_acc["bifurcation"].append(angles.get("bifurcation"))

                if len(angles.get("ending")) > 0:
                    angles_acc["ending"].append(angles.get("ending"))
             
                output_parent_dir = output_dir + path.parent.name
                
                if pathlib.Path(output_parent_dir).is_dir() == False:
                    os.mkdir(output_parent_dir)

                output_parent_dir + '/'

                if cv.imwrite(output_parent_dir + '/' + path.name, results) == False :
                    print(f"Failed to load image: {output_parent_dir + '/' + path.name}")


        # print(angles_acc.get("bifurcation"))
        bifurcations_angles = get_angles_min_frequenties(angles_acc.get("bifurcation"), 2)
        ending_angles = get_angles_min_frequenties(angles_acc.get("ending"), 2)

        print(bifurcations_angles)
        print(ending_angles)

        bifurcations_combinations = generate_combinations(bifurcations_angles + bifurcations_angles, 3)

        bifurcations_combinations_id = {}
        for i, combination in enumerate(bifurcations_combinations, 1):
            print(combination)
            digest = sha256(str(combination).encode('utf-8')).hexdigest()
            bifurcations_combinations_id[digest] = combination

        # ending_combinations_id = {}
        # for i, combination in enumerate(ending_combinations, 1):
        #     digest = sha256(str(combination).encode('utf-8')).hexdigest()
        #     ending_combinations_id[digest] = combination

        json_file_path = output_parent_dir + '/combinations.json' 
        with open(json_file_path, 'w') as json_file:
            json.dump(bifurcations_combinations_id, json_file)

        # print(f"Number of bifurcations angles: {len(bifurcations_angles)}")
        # print(f"Number of terminations angles: {len(ending_angles)}")

        # angles = np.concatenate([
        #     np.array(bifurcations_angles, dtype=object),
        #     np.array(ending_angles, dtype=object)
        # ]).tolist()

        # # Extract unique first tuples
        # singularities = set(item[0] for item in angles)

        # # Print the result
        # print(f"Number of unique singularities (core/delta) {len(singularities)}")


    process_images(img_dir)