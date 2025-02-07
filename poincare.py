import orientation
import math
import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN

def poincare_index_at(i, j, angles, tolerance):
    """
    compute the summation difference between the adjacent orientations such that the orientations is less then 90 degrees
    """
    cells = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ]

    angles_around_index = [math.degrees(angles[i - k][j - l]) for k, l in cells]
    index = 0
    for k in range(0, 8):
        # calculate the difference
        difference = angles_around_index[k] - angles_around_index[k + 1]
        if difference > 90:
            difference -= 180
        elif difference < -90:
            difference += 180

        index += difference

    if 180 - tolerance <= index <= 180 + tolerance:
        return "delta"
    if -180 - tolerance <= index <= -180 + tolerance:
        return "loop"
    if 360 - tolerance <= index <= 360 + tolerance:
        return "whorl"
    return "none"

def merge_singularities(singularities, eps=20):
    """
    Merge close singularities using DBSCAN clustering.
    
    :param singularities: List of (x, y) singularity points.
    :param eps: Maximum distance between points to be considered a single cluster.
    :return: List of merged bounding boxes.
    """
    if not singularities:
        return []

    singularity_points = np.array(singularities)
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=1).fit(singularity_points)
    labels = clustering.labels_

    merged_rectangles = []
    unique_labels = set(labels)

    for label in unique_labels:
        cluster_points = singularity_points[labels == label]

        # Compute bounding rectangle around the cluster
        x_min, y_min = np.min(cluster_points, axis=0)
        x_max, y_max = np.max(cluster_points, axis=0)

        merged_rectangles.append(((x_min, y_min), (x_max, y_max)))

    return merged_rectangles

def calculate_singularities(im, angles, tolerance, W, mask):
    result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)

    # LOOP: RED, DELTA: ORANGE, whorl:INK
    # colors = {"loop": (0, 0, 255), "delta": (0, 128, 255), "whorl": (255, 153, 255)}

    # singularities = { "loop": [], "delta": [], "whorl": [] }

    detected_singularities = []  # List to store singularity locations

    for i in range(3, len(angles) - 2):  # Y
        for j in range(3, len(angles[i]) - 2):  # x
            # mask any singularity outside of the mask
            mask_slice = mask[(i - 2) * W : (i + 3) * W, (j - 2) * W : (j + 3) * W]
            mask_flag = np.sum(mask_slice)
            if mask_flag == (W * 5) ** 2:
                singularity = poincare_index_at(i, j, angles, tolerance)
                if singularity != "none":
                    top_left = ((j + 0) * W, (i + 0) * W)
                    bottom_right = ((j + 1) * W, (i + 1) * W)

                    center = ((j + 0.5) * W, (i + 0.5) * W)
                    detected_singularities.append(center)

                    # singularities[singularity].append(
                    #     (top_left, bottom_right)
                    # )

    merged_rectangles = merge_singularities(detected_singularities, eps=20)
    # Draw final merged singularity rectangles
    for top_left, bottom_right in merged_rectangles:
        cv.rectangle(result, tuple(map(int, top_left)), tuple(map(int, bottom_right)), (0, 255, 0), 2)

    return (result, merged_rectangles)
