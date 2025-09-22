import cv2
import time

def find_ORB_keypoints_and_descriptors(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def find_SIFT_keypoints_and_descriptors(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_FLANN(des1, des2):
    index_params = dict(algorithm=1, trees=5)  # Using KDTree for SIFT
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    return good_matches

def match_BF(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Using Hamming for ORB
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def match_SIFT_FLANN(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)  # Reads as grayscale array
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)  # Reads as grayscale array
    
    # At this point, img1 and img2 are NumPy arrays, not JPG or PNG files
    # The original file format doesn't matter anymore
    
    kp1, des1 = find_SIFT_keypoints_and_descriptors(img1)
    kp2, des2 = find_SIFT_keypoints_and_descriptors(img2)
    
    good_matches = match_FLANN(des1, des2)
    
    # Draw matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_img = cv2.resize(match_img, (600,300))  # Resize for better visibility if needed

    return len(good_matches), match_img  # match_img is also a NumPy array

def match_ORB_BF(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)  # Reads as grayscale array
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)  # Reads as grayscale array

    kp1, des1 = find_ORB_keypoints_and_descriptors(img1)
    kp2, des2 = find_ORB_keypoints_and_descriptors(img2)
    
    matches = match_BF(des1, des2)
    
    # Draw matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_img = cv2.resize(match_img, (600,300))  # Resize for better visibility if needed


    return len(matches), match_img  # match_img is also a NumPy array


same1 = 'same_1/101_6.tif'
same2 = 'same_1/101_7.tif'
different1 = 'different_10/101_6.tif'
different2 = 'different_10/104_7.tif'
uia1 = 'UiA_front1.png'
uia2 = 'UiA_front3.jpg'

images = [(same1, same2), (different1, different2), (uia1, uia2)]
for img1, img2 in images:
    elappsed_time_sf = 0
    elappsed_time_ob = 0
    for i in range(100):
        start_time_sf = time.time()
        num_matches_sf, matched_image_sf = match_SIFT_FLANN(img1, img2)
        end_time_sf = time.time()
        start_time_ob = time.time()
        num_matches_ob, matched_image_ob = match_ORB_BF(img1, img2)
        end_time_ob = time.time()

        elappsed_time_sf += (end_time_sf - start_time_sf) * 1000  # Convert to milliseconds
        elappsed_time_ob += (end_time_ob - start_time_ob) * 1000  # Convert to milliseconds
    
    avg_time_sf = elappsed_time_sf / 100
    avg_time_ob = elappsed_time_ob / 100
    print(f"elappsed_time SIFT FLANN: {avg_time_sf:.2f} ms, ORB BF: {avg_time_ob:.2f} ms")

    cv2.imshow('Matched Image SIFT FLANN', matched_image_sf)
    cv2.imshow('Matched Image ORB BF', matched_image_ob)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

#SIFT FLANN gives very good accuracy on the fingerprints but is pretty slow
#ORB BF is much faster and while finding a lot of matches on the fingerprints most to all are wrong
#on the uia pictures SF is still 8x slower than OB but the results are both pretty bad