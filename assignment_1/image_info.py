mport numpy as np
import cv2

def print_image_information(image):
    print(f"Height: {image.shape[0]}")
    print(f"Width: {image.shape[1]}")
    print(f"Channels: {image.shape[2]}")
    print(f"Size: {image.size}")
    print(f"Data type: {image.dtype}")

def write_webcam_info():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Error: Could not open webcam.")
    
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Error: Could not read frame from webcam.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Webcam FPS: {fps}")
    print_image_information(frame)
    cap.release()

    with open('assignment_1/camera_outputs.txt', 'w+') as f:
        f.write(f"Webcam FPS: {fps}\n")
        f.write(f"Image Height: {frame.shape[0]}\n")
        f.write(f"Image Width: {frame.shape[1]}\n")
        f.write(f"Image Channels: {frame.shape[2]}\n")
        f.write(f"Image Size: {frame.size}\n")
        f.write(f"Image Data Type: {frame.dtype}\n")

if __name__ == "__main__":
    image_path = 'assignment_1/lena-1.png'  # Update with your image path
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Could not read the image.")
    print_image_information(image)
    write_webcam_info()