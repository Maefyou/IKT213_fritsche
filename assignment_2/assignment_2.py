import cv2
import numpy as np

def padding(image, border_with: int) -> np.ndarray:
    padded_image = cv2.copyMakeBorder(image, border_with, border_with, border_with, border_with, cv2.BORDER_REPLICATE)
    return padded_image

def crop(image, x_0,x_1,y_0,y_1) -> np.ndarray:
    hight, width = image.shape[:2]
    croped_image = image[y_0:hight-y_1, x_0:width-x_1]
    return croped_image

def resize(image, new_width: int, new_height: int) -> np.ndarray:
    hight, width = image.shape[:2]

    hight_ratio = new_height / hight
    width_ratio = new_width / width
    match (hight_ratio, width_ratio):
        case (hr, wr) if hr < 1 and wr < 1:
            interpolation = cv2.INTER_AREA
        case (hr, wr) if hr > 1 and wr > 1:
            interpolation = cv2.INTER_CUBIC
        case _:
            interpolation = cv2.INTER_LINEAR
    
    resized_image = cv2.resize(image, (new_width, new_height), interpolation)
    return resized_image

def copy(image) -> np.ndarray:
    hight, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    copy = np.zeros((hight, width, channels), dtype=np.uint8)
    copy[:] = image[:]
    return copy

def grayscale(image) -> np.ndarray:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def hsv(image) -> np.ndarray:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def hue_shift(image, shift_value: int) -> np.ndarray:
    hue_shifted = image.copy()
    hue_shifted[:,:,0:3] = (hue_shifted[:,:,0:3] + shift_value) % 255
    return hue_shifted

def smoothing(image, kernel_size: int) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1  # Kernel size must be odd
    smoothed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size),0 ,borderType=cv2.BORDER_DEFAULT)
    return smoothed_image

def rotation(image, angle: float) -> np.ndarray:
    hight, width = image.shape[:2]
    center = (width // 2, hight // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, hight), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

if __name__ == "__main__":
    image_path = "lena.png"
    image = cv2.imread(image_path)
    cv2.imwrite("solutions/lena_padded.png", padding(image, 100))
    cv2.imwrite("solutions/lena_croped.png", crop(image, 80,130,80,130))
    cv2.imwrite("solutions/lena_resized.png", resize(image, 200, 200))
    cv2.imwrite("solutions/lena_copy.png", copy(image))
    cv2.imwrite("solutions/lena_grayscale.png", grayscale(image))
    cv2.imwrite("solutions/lena_hsv.png", hsv(image))
    cv2.imwrite("solutions/lena_hue_shifted.png", hue_shift(image, 50))
    cv2.imwrite("solutions/lena_smoothed.png", smoothing(image, 15))
    cv2.imwrite("solutions/lena_rotated.png", rotation(image, 45))