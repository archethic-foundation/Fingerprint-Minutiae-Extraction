import cv2 as cv
import numpy as np


def minutiae_at(pixels, i, j, kernel_size, max_i, max_j):
    """
    Crossing number methods is a really simple way to detect ridge endings and ridge bifurcations.
    Then the crossing number algorithm will look at 3x3 pixel blocks:

    if middle pixel is black (represents ridge):
    if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
    if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation
    """
    # if middle pixel is black (represents ridge)
    if pixels[i][j] == 1:
        if kernel_size == 3:
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
        else:
            cells = [
                (-2, -2),
                (-2, -1),
                (-2, 0),
                (-2, 1),
                (-2, 2),
                (-1, 2),
                (0, 2),
                (1, 2),
                (2, 2),
                (2, 1),
                (2, 0),
                (2, -1),
                (2, -2),
                (1, -2),
                (0, -2),
                (-1, -2),
                (-2, -2),
            ]

        values = [pixels[i + l][j + k] for k, l in cells]

        # count crossing how many times it goes from 0 to 1
        crossings = 0
        for k in range(0, len(values) - 1):
            crossings += abs(values[k] - values[k + 1])
        crossings //= 2

        # if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
        # if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation
        if crossings == 1:
            if i < 100: # Top side
                return "none"
            if j < 60: # Left side
                return "none"
            if i > (max_i-70): # Bottom side 
                return "none"
            if j > (max_j-30): # Right side
                return "none"
            
            return "ending"
        if crossings == 3:

            if i < 100:
                return "none"
            if j < 50:
                return "none"
            if i > (max_i-100):
                return "none"
            if j > (max_j-30):
                return "none"

            return "bifurcation"

    return "none"


def calculate_minutiaes(im, kernel_size=3):
    biniry_image = np.zeros_like(im)
    biniry_image[im < 10] = 1.0
    biniry_image = biniry_image.astype(np.int8)

    (y, x) = im.shape
    result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    colors = {"ending": (150, 0, 0), "bifurcation": (0, 150, 0)}

    minutiaes_list = { "ending": [], "bifurcation": [] }

    max_i = x - kernel_size // 2
    max_j = y - kernel_size // 2

    # iterate each pixel minutia
    for i in range(1, x - kernel_size // 2):
       
        for j in range(1, y - kernel_size // 2):
            minutiae = minutiae_at(biniry_image, j, i, kernel_size, max_j, max_i)
            if minutiae != "none":
                minutiaes_list[minutiae].append([i, j])
                cv.circle(result, (i, j), radius=2, color=colors[minutiae], thickness=2)

    return (result, minutiaes_list)
