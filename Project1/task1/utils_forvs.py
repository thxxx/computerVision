import numpy as np
import cv2


def calculate_rms(img1, img2):
    """
    Calculates RMS error between two images. Two images should have same sizes.
    """
    if (img1.shape[0] != img2.shape[0]) or (img1.shape[1] != img2.shape[1]) or (img1.shape[2] != img2.shape[2]):
        raise Exception("img1 and img2 should have sime sizes.")

    diff = np.abs(img1.astype(np.int) - img2.astype(np.int))
    return np.sqrt(np.mean(diff ** 2))


"""
Calculates RMS error between two cropped images. Two images should have same sizes.
"""


def calculate_rms_cropped(img1, img2):
    H, W, C = img1.shape
    cut_size = 20

    img1 = img1[cut_size:H - cut_size, cut_size:W - cut_size]
    img2 = img2[cut_size:H - cut_size, cut_size:W - cut_size]

    diff = np.abs(img1 - img2)
    diff = np.abs(img1.astype(dtype=np.int) - img2.astype(dtype=np.int))

    return np.sqrt(np.mean(diff ** 2))


test2_img = cv2.imread('./inputs/test2.png', cv2.IMREAD_COLOR)
test2_c_img = cv2.imread('./inputs/test2_clean.png', cv2.IMREAD_COLOR)

test1_img = cv2.imread('./inputs/test1.png', cv2.IMREAD_COLOR)
test1_c_img = cv2.imread('./inputs/test1_clean.png', cv2.IMREAD_COLOR)

test1_ria_5 = cv2.imread('./outputs/test1_ria_5.png', cv2.IMREAD_COLOR)
test1_ria_3 = cv2.imread('./outputs/test1_ria_3.png', cv2.IMREAD_COLOR)  #
rim_3 = cv2.imread('./outputs/rim_3.png', cv2.IMREAD_COLOR)  #


rib_200200 = cv2.imread('./outputs/rib_200200.png', cv2.IMREAD_COLOR)  #
rib_100100 = cv2.imread('./outputs/rib_100100.png', cv2.IMREAD_COLOR)  #
rib_5 = cv2.imread('./outputs/rib_5.png', cv2.IMREAD_COLOR)  #
rib_2100 = cv2.imread('./outputs/rib_2100.png', cv2.IMREAD_COLOR)  #
rib_22 = cv2.imread('./outputs/rib_22.png', cv2.IMREAD_COLOR)  #
rib_7575 = cv2.imread('./outputs/rib_7575.png', cv2.IMREAD_COLOR)  #


print("에버리지 필터 쓴거", calculate_rms(test1_c_img, test1_ria_5))
print("에버리지 필터 쓴거", calculate_rms_cropped(test1_c_img, test1_ria_3))
print("바이레터럴ㅁ", calculate_rms_cropped(test1_c_img, rib_5))
print("바이레터럴ㅁ", calculate_rms_cropped(test1_c_img, rib_7575))

print(f" 원래 : {calculate_rms_cropped(test1_c_img, test1_img)}")
