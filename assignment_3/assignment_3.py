"""this module provides functions as assigned in task assignment 3"""

import cv2 as cv
import numpy as np


def sobel_edge_detection(image):
    """
    applys sobel edge detection for input image
    Parameters:
    - image: The input image on which to perform Sobel edge detection.
    Returns:
    - The image with Sobel edges detected.
    """
    if image is None:
        raise ValueError("Input image is None")

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blured_gausian = cv.GaussianBlur(gray_image, (3, 3), 0)
    sobel_xy_gausian = cv.Sobel(blured_gausian, cv.CV_64F, 1, 1, ksize=1)
    return sobel_xy_gausian


def improved_sobel_edge_detection(image):
    """
    applys improved sobel edge detection for input image
    Parameters:
    - image: The input image on which to perform Sobel edge detection.
    Returns:
    - The image with Sobel edges detected.
    """
    if image is None:
        raise ValueError("Input image is None")

    image = cv.medianBlur(image, 5)
    image = cv.bilateralFilter(image, 20, 60, 75)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blured_gausian = cv.GaussianBlur(gray_image, (3, 3), 0)
    sobel_xy_gausian = cv.Sobel(blured_gausian, cv.CV_64F, 1, 1, ksize=1)
    return sobel_xy_gausian


def canny_edge_detection(image, low_threshold=100, high_threshold=200):
    """
    applys canny edge detection for input image
    Parameters:
    - image: The input image on which to perform Canny edge detection.
    - low_threshold: The lower threshold for the hysteresis procedure. Default is 100.
    - high_threshold: The upper threshold for the hysteresis procedure. Default is 200.
    Returns:
    - The image with Canny edges detected.
    """
    if image is None:
        raise ValueError("Input image is None")
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    smooth_image_gausian = cv.GaussianBlur(gray_image, (3, 3), 0)
    canny_edges = cv.Canny(smooth_image_gausian, low_threshold, high_threshold)
    return canny_edges


def remove_background_lambo(image):
    """
    removes the background of the lambo image
    Parameters:
    - image: The input image from which to remove the background.
    Returns:
    - The image with the background removed.
    """
    if image is None:
        raise ValueError("Input image is None")

    original = image.copy()

    for line in image:
        for pixel in line:
            if pixel[2]-(pixel[0]/2+pixel[1]/2) > 20:
                pixel[0] = 0
                pixel[1] = 0
                pixel[2] = 255

    blurred_image = cv.bilateralFilter(image, 100, 100, 100)
    blurred_image = cv.GaussianBlur(blurred_image, (9, 9), 0)
    edges = cv.Canny(blurred_image, 200, 230)

    for _ in range(7):
        edges = cv.dilate(edges, None)

    for _ in range(7):
        edges = cv.erode(edges, None)

    contours, _ = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv.contourArea)
    mask = np.zeros_like(edges)
    cv.drawContours(mask, [main_contour], -1, 255, thickness=cv.FILLED)
    result = cv.bitwise_and(original, original, mask=mask)

    return result


def template_match(image, template):
    """
    applies template matching to find the template in the image
    and draws rectangles around the found templates
    Parameters:
    - image: The input image in which to search for the template.
    - template: The template image to be matched.
    Returns:
    - The image with rectangles drawn around the matched templates.
    """
    if image is None:
        raise ValueError("Input image")
    if template is None:
        raise ValueError("Input template is None")

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    result = cv.matchTemplate(gray_image, gray_template, cv.TM_CCOEFF_NORMED)
    threshold = 0.9
    locations = np.where(result >= threshold)
    locations = [(x, y) for x, y in zip(locations[1], locations[0])]
    for pt in locations:
        cv.rectangle(image, pt, (pt[0] + template.shape[1],
                     pt[1] + template.shape[0]), (0, 0, 255), 1)
    return image

def resize(image, scale_factor: int = 1, up_or_down: str = "up"):
    """
    Resizes the input image by a given scale factor.

    Parameters:
    - image: The input image to be resized.
    - scale_factor: The factor by which to scale the image. Default is 1.
    - up_or_down: A string indicating whether to scale 'up' or 'down'. Default is 'up'.

    Returns:
    - The resized image.
    """
    if image is None:
        raise ValueError("Input image is None")

    if scale_factor is None:
        raise ValueError("Scale factor is None or")

    if scale_factor <= 0:
        raise ValueError("Scale factor must be a positive integer")

    if up_or_down not in ["up", "down"]:
        raise ValueError("up_or_down must be either 'up' or 'down'")

    if up_or_down == "up":  # zoom in
        output_image = cv.pyrUp(image,
                                 (int(image.shape[1] * scale_factor),
                                  int(image.shape[0] * scale_factor)),
                                 )
    else:  # down; zoom out
        output_image = cv.pyrDown(image,
                                 (int(image.shape[1] // scale_factor),
                                  int(image.shape[0] // scale_factor)),
                                 )

    return output_image

def zoom(image, scale_factor: int = 1, up_or_down: str = "up"):
    """
    Resizes the input image by a given scale factor.

    Parameters:
    - image: The input image to be resized.
    - scale_factor: The factor by which to scale the image. Default is 1.
    - up_or_down: A string indicating whether to scale 'up' or 'down'. Default is 'up'.

    Returns:
    - The resized image.
    """
    if image is None:
        raise ValueError("Input image is None")

    if scale_factor is None:
        raise ValueError("Scale factor is None or")

    if scale_factor <= 0:
        raise ValueError("Scale factor must be a positive integer")

    if up_or_down not in ["up", "down"]:
        raise ValueError("up_or_down must be either 'up' or 'down'")

    if up_or_down == "up":  # zoom in
        output_image = cv.pyrUp(image,
                                dstsize=(
                                    int(image.shape[1] * scale_factor),
                                    int(image.shape[0] * scale_factor)
                                )
                                )
        output_image = output_image[
            (output_image.shape[0]-image.shape[0])//2:-(output_image.shape[0]-image.shape[0])//2,
            (output_image.shape[1]-image.shape[1])//2:-(output_image.shape[1]-image.shape[1])//2
        ]

    else:  # down; zoom out
        output_image = cv.pyrDown(image,
                                  dstsize=(
                                      int(image.shape[1] // scale_factor),
                                      int(image.shape[0] // scale_factor)
                                  )
                                  )
        output_image = cv.copyMakeBorder(
            output_image,
            top=(image.shape[0]-output_image.shape[0])//2,
            bottom=(image.shape[0]-output_image.shape[0])//2,
            left=(image.shape[1]-output_image.shape[1])//2,
            right=(image.shape[1]-output_image.shape[1])//2,
            borderType=cv.BORDER_CONSTANT,
            value=[0, 255, 0]
        )

    return output_image


if __name__ == "__main__":
    lambo_image = cv.imread("assignment_3/images/lambo.png")
    shapes_template_image = cv.imread(
        "assignment_3/images/shapes_template.jpg")
    shapes_image = cv.imread("assignment_3/images/shapes-1.png")

    # assignment tasks
    cv.imwrite("assignment_3/solutions/sobel_lambo.png",
               sobel_edge_detection(lambo_image.copy()))
    cv.imwrite("assignment_3/solutions/canny_lambo.png",
               canny_edge_detection(lambo_image.copy(), 50, 50))
    cv.imwrite("assignment_3/solutions/matched_shapes.png",
               template_match(shapes_image.copy(), shapes_template_image.copy()))
    cv.imwrite("assignment_3/solutions/resize_up_lambo.png",
               resize(lambo_image.copy(), 2, "up"))
    cv.imwrite("assignment_3/solutions/resize_down_lambo.png",
               resize(lambo_image.copy(), 2, "down"))

    # playing around
    
    cv.imwrite("assignment_3/playing/zoom_up_lambo.png",
                zoom(lambo_image.copy(), 2, "up"))
    cv.imwrite("assignment_3/playing/zoom_down_lambo.png",
                zoom(lambo_image.copy(), 2, "down"))
    cv.imwrite("assignment_3/playing/improved_sobel_lambo.png",
               improved_sobel_edge_detection(lambo_image.copy()))
    cv.imwrite("assignment_3/playing/lambo_background_removal.png",
               remove_background_lambo(lambo_image.copy()))
