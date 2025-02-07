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

def get_angles_min_frequenties(images_angles, threshold = 2):
    frequencies = {}
    for i, image_singularities in enumerate(images_angles):
        for singularity_angles in image_singularities:
            for singularity_angle in singularity_angles:
                singularity, minutia, distance, angle = singularity_angle
                rounded_angle = round(angle)
                rounded_distance = round(distance)
                modulo = rounded_distance % 5

                if modulo > 2.5:
                    rounded_distance = rounded_distance + (5 - modulo)
                else:
                    rounded_distance = rounded_distance - modulo

                if rounded_angle > 0:
                    frequency = frequencies.get((singularity, rounded_angle, rounded_distance))
                    if frequency == None:
                        frequencies[(singularity, rounded_angle, rounded_distance)] = { "images": [i] }
                    else:
                        # frequencies[(singularity, rounded_angle)]["f"] += 1
                        existing_images = frequencies[(singularity, rounded_angle, rounded_distance)]["images"]
                        if i not in existing_images:
                            frequencies[(singularity, rounded_angle, rounded_distance)]["images"].append(i)


    angles = []
    angles_per_image = {}
    singularies_per_image = {}
    for angle_singularity, frequency in frequencies.items():
        singularity, angle, distance = angle_singularity
        images = frequency['images']

        if len(images) >= threshold:
            print(f"{angle_singularity} present in {len(images)} images: {images}")
            angles.append(angle_singularity)

            for i in images:
                angles_per_image[i] = angles_per_image.get(i, 0) + 1

                image_singularities = singularies_per_image.get(i)
                if image_singularities == None:
                    singularies_per_image[i] = set((singularity))
                else:
                    image_singularities.add(singularity)
            

    for image, frequencies in angles_per_image.items():
        print(f"Image #{image} contains {frequencies} minutiae which where present in more than {threshold} images")

    for image, singularities in singularies_per_image.items():
        # print(singularities)
        print(f"Image #{image} contains {len(singularities)} singularities")


    return angles

def calculate_minutiae_singularity_angle(minutiae, singularities):
    """
    Calculates the angle between two vectors: 
    - Vector 1: Minutia → Top-left singularity
    - Vector 2: Minutia → Bottom-right singularity

    :param minutiae: List of lists, where each list contains a minutia defined by [x, y]
    :param singularities: List of tuples, where each tuple is a rectangle defined by ((top-left x, top-left y), (bottom-right x, bottom-right y))
    :return: A list of lists, where each sublist contains a single angle (in degrees) for each minutia
    """
    minutiae_angle_all_singularities = []

    for singularity in singularities:
        top_left, bottom_right = singularity
        minutiae_angles = []

        # Compute the midpoint of the singularity (center of the rectangle)
        center_x = (top_left[0] + bottom_right[0]) / 2
        center_y = (top_left[1] + bottom_right[1]) / 2

        for minutia in minutiae:
            minutia_x, minutia_y = minutia  # Unpack [x, y]

            # Define vectors
            v1_x, v1_y = top_left[0] - minutia_x, top_left[1] - minutia_y
            v2_x, v2_y = bottom_right[0] - minutia_x, bottom_right[1] - minutia_y

            # Compute dot product
            dot_product = v1_x * v2_x + v1_y * v2_y

            # Calculate Euclidean distance
            distance = math.sqrt((minutia_x - center_x) ** 2 + (minutia_y - center_y) ** 2)

            # Compute magnitudes
            mag_v1 = math.sqrt(v1_x**2 + v1_y**2)
            mag_v2 = math.sqrt(v2_x**2 + v2_y**2)

            # Avoid division by zero
            if mag_v1 == 0 or mag_v2 == 0:
                angle_deg = 0  # If one vector is zero, angle is undefined (set to 0)
            else:
                # Compute the cosine of the angle, ensuring it's within [-1,1]
                cos_theta = dot_product / (mag_v1 * mag_v2)
                cos_theta = max(-1, min(1, cos_theta))  # Clamp value to valid range

                # Compute angle
                angle_rad = math.acos(cos_theta)
                angle_deg = math.degrees(angle_rad)

            minutiae_angles.append((singularity, (minutia_x, minutia_y), distance, angle_deg))

        minutiae_angle_all_singularities.append(minutiae_angles)

    return minutiae_angle_all_singularities




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

    # minutias
    minutias_img, minutias = calculate_minutiaes(thin_image)

    ending = minutias.get("ending")
    bifurcation = minutias.get("bifurcation")

    # singularities
    singularities_img, singularities = calculate_singularities(thin_image, angles, 1, block_size, mask)

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

if __name__ == "__main__":
    # open images
    img_dir = "./input/1/*.png"
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

                angles_acc["ending"].append(angles.get("ending"))
                angles_acc["bifurcation"].append(angles.get("bifurcation"))
             
                output_parent_dir = output_dir + path.parent.name
                
                if pathlib.Path(output_parent_dir).is_dir() == False:
                    os.mkdir(output_parent_dir)

                output_parent_dir + '/'
                # with open(json_file_path, 'w') as json_file:
                #     json.dump(data_list, json_file)

                if cv.imwrite(output_parent_dir + '/' + path.name, results) == False :
                    print(f"Failed to load image: {output_parent_dir + '/' + path.name}")


        bifurcations_angles = get_angles_min_frequenties(angles_acc.get("bifurcation"), 4)
        ending_angles = get_angles_min_frequenties(angles_acc.get("ending"))

        # print(f"Number of bifurcations angles: {len(bifurcations_angles)}")
        print(f"Number of terminations angles: {len(ending_angles)}")

        # angles = np.concatenate([
        #     np.array(bifurcations_angles, dtype=object),
        #     np.array(ending_angles, dtype=object)
        # ]).tolist()

        # # Extract unique first tuples
        # singularities = set(item[0] for item in angles)

        # # Print the result
        # print(f"Number of unique singularities (core/delta) {len(singularities)}")


    process_images(img_dir)