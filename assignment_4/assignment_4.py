import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def convert_to_grayscale(img):    
    """
    convert input image to grayscale
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    return gray


def harris_corner_detection(img):
    """
    detect edges an mark them on the image
    """
    gray = convert_to_grayscale(img)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]

    os.makedirs('output', exist_ok=True)
    cv2.imwrite('output/harris_corners.png', img)
    
    return img


def align_sift(align_img, ref_img, max_features=1000, good_match_percent=0.7):
    MIN_MATCH_COUNT = 10
 
    img1 = convert_to_grayscale(ref_img)
    img2 = convert_to_grayscale(align_img)
    
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    sorted_matches = sorted(matches, key=lambda x: x[0].distance)
    matches = sorted_matches
    for m,n in matches:
        if m.distance < good_match_percent*n.distance:
            good.append(m)
        if len(good) > max_features:
            break
    
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # Find homography from align_img (dst) to reference_img (src)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # Use inverse homography for visualization
        M_inv = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]
        dst = cv2.perspectiveTransform(pts, M_inv)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,10, cv2.LINE_AA)

        # save aligned image
        aligned_img = cv2.warpPerspective(align_img, M, (ref_img.shape[1], ref_img.shape[0]))
        os.makedirs('output', exist_ok=True)
        cv2.imwrite('output/aligned_image.png', aligned_img)

    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    # show detected matches
    os.makedirs('output', exist_ok=True)
    cv2.imwrite('output/sift_matches.png', img3)

# Main execution

image_to_align = cv2.imread('align_this.jpg')
reference_img = cv2.imread('reference_img.png')
harris_corner_detection(reference_img)
align_sift(image_to_align, reference_img, max_features=10, good_match_percent=0.7)