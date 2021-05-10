import numpy as np
import cv2
import sys
import math


def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return (1.0 / np.sqrt(2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s, c):
    hl = diameter/2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = int(x - (hl - i))
            neighbour_y = int(y - (hl - j))
            if neighbour_x >= len(source):
                neighbour_x = len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y = len(source[0])
            gi = gaussian(source[neighbour_x]
                          [neighbour_y][c] - source[x][y][c], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y][c] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y][c] = int(round(i_filtered))


def bilateral_filter_own(source, filter_diameter, sigma_i, sigma_s):
    filtered_image = np.zeros(source.shape)

    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(
                source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s, 0)
            apply_bilateral_filter(
                source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s, 1)
            apply_bilateral_filter(
                source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s, 2)
            j += 1
        i += 1
    return filtered_image


test_img = cv2.imread(
    'inputs/test2.png', cv2.IMREAD_COLOR)
test_c_img = cv2.imread(
    '/Users/gimhojin/Downloads/CV-Assignment1/inputs/test2_clean.png', cv2.IMREAD_COLOR)

print("doing!")
"""
result_img_bilateral_5 = apply_bilateral_filter(
    test1_img, 3, 2.0, 200.0)  # 현재까지 제일 좋다
print("3 끝")
cv2.imwrite('./outputs/rib_5.png', result_img_bilateral_5)

해본거 : 커널사이즈는 3 - 100,100 | 2,100 | 2,2 | 200,200

result_img_bilateral_2100 = apply_bilateral_filter(test1_img, 3, 2.0, 300.0)
print("4 끝")
cv2.imwrite('./outputs/rib_2100.png', result_img_bilateral_2100)
"""


def calculate_rms_cropped(img1, img2):
    H, W, C = img1.shape
    cut_size = 20

    img1 = img1[cut_size:H - cut_size, cut_size:W - cut_size]
    img2 = img2[cut_size:H - cut_size, cut_size:W - cut_size]

    diff = np.abs(img1 - img2)
    diff = np.abs(img1.astype(dtype=np.int) - img2.astype(dtype=np.int))

    return np.sqrt(np.mean(diff ** 2))


# cv2.imwrite('outputs/median_3of4.png', apply_median_filter(test_img, 3))
"""
print("메디안2 : ", calculate_rms_cropped(
    test_c_img, apply_median_filter2(test_img, 3)))

print("가우시안 : ", calculate_rms_cropped(
    test_c_img, cv2.GaussianBlur(test_img, (3, 3), 3)))

print("메디안1 : ", calculate_rms_cropped(
    test_c_img, apply_median_filter(test_img, 3)))


print("에버리지 : ", calculate_rms_cropped(
    test_c_img, apply_average_filter(test_img, 3)))
"""

print("바이 1, 600 : ", calculate_rms_cropped(
    test_c_img, bilateral_filter_own(test_img, 3, 1.0, 600.0)))
print("바이 10, 40 : ", calculate_rms_cropped(
    test_c_img, bilateral_filter_own(test_img, 3, 10.0, 40.0)))
print("바이 200, 2 : ", calculate_rms_cropped(
    test_c_img, bilateral_filter_own(test_img, 3, 200.0, 2.0)))


print(f"원래값 : ", calculate_rms_cropped(
    test_c_img, test_img))
