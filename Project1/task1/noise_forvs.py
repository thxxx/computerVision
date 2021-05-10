import numpy as np
import cv2
import matplotlib.pyplot as plt  # 주피터 환경에서 이미지 열어주는거
import statistics
import math


def task1(src_img_path, clean_img_path, dst_img_path):
    """
    This is main function for task 1.
    It takes 3 arguments,
    'src_img_path' is path for source image.
    'clean_img_path' is path for clean image.
    'dst_img_path' is path for output image, where your result image should be saved.

    You should load image in 'src_img_path', and then perform task 1 of your assignment 1,
    and then save your result image to 'dst_img_path'.

    find out what is optimal filter and kernel_size for this image
    """
    noisy_img = cv2.imread(src_img_path)
    clean_img = cv2.imread(clean_img_path)
    result_img = None

    # do noise removal

    cv2.imwrite(dst_img_path, result_img)
    pass


def apply_average_filter(img, kernel_size):  # kernel_size 3 means 3x3 matrix.
    """
    You should implement average filter convolution algorithm in this function.
    It takes 2 arguments,

    'img' is source image, and you should perform convolution with average filter.

    'kernel_size' is a int value, which determines kernel size of average filter.

    You should return result image.
    """

    k = int(kernel_size/2)

    new_img = np.zeros([img.shape[0], img.shape[1],
                        img.shape[2]], dtype='uint8')

    # add more zeros to each edges
    apply_img = np.zeros(
        [img.shape[0]+2*k, img.shape[1]+2*k, img.shape[2]], dtype='uint8')
    apply_img[k:img.shape[0]+k, k:img.shape[1]+k] = img

    mask = np.ones([3, 3, 3], dtype=int)
    mask = mask/9

    temp = []  # temp는 한 픽셀당 rgb값의 리스트이다. ex. [143,52,211]

    for i in range(k, len(img)-k):
        for j in range(k, len(img[0])-k):
            t = [0, 0, 0]
            for row in range(kernel_size):
                for col in range(kernel_size):
                    t += img[i-k+row, j-k+col]  # * mask[row,col]

            new_img[i, j] = t/kernel_size**2
    return new_img


"""
    for i in range(k, len(apply_img)-k):
        for j in range(k, len(apply_img[0])-k):
            t = [0, 0, 0]
            for row in range(kernel_size):
                for col in range(kernel_size):
                    t += apply_img[i-k+row, j-k+col]  # * mask[row,col]

            new_img[i-k, j-k] = t/(kernel_size*kernel_size)
"""


def apply_median_filter(img, kernel_size):
    """
    You should implement median filter convolution algorithm in this function.
    It takes 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is a int value, which determines kernel size of median filter.

    You should return result image.

    """

    k = int(kernel_size/2)

    new_img = np.zeros([img.shape[0], img.shape[1],
                        img.shape[2]], dtype='uint8')

    mask = np.ones([3, 3], dtype=int)
    mask = mask/9

    temp = []  # temp는 한 픽셀당 rgb값의 리스트이다. ex. [143,52,211]

    def mid_of(color):
        return sorted(color)[int(kernel_size*kernel_size/2)]

    for i in range(1, len(img)-k):
        for j in range(1, len(img[0])-k):
            red, green, blue = [], [], []
            for row in range(kernel_size):
                for col in range(kernel_size):
                    t = img[i-k+row, j-k+col]
                    red.append(t[0])
                    green.append(t[1])
                    blue.append(t[2])

            new_img[i, j] = [mid_of(red), mid_of(green), mid_of(blue)]

            ###
    return new_img


def apply_bilateral_filter1(img, kernel_size, sigma_s, sigma_r):
    """
    You should implement convolution with additional filter.
    You can use any filters for this function, except average, median filter.
    It takes at least 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is a int value, which determines kernel size of average filter.
    'sigma_s' is a int value, which is a sigma value for G_s
    'sigma_r' is a int value, which is a sigma value for G_r

    You can add more arguments for this function if you need.

    You should return result image.
    """

    # 범위가 1) by 2) 사이즈가 된다. 3)칸으로 나눈다.
    rad = np.linspace(-kernel_size, kernel_size, kernel_size)
    a, b = np.meshgrid(rad, rad)

    s_s = 2

    # 가우시안 커널 생성.
    kernel = (1/(2*math.pi*(s_s**2)))*np.exp(-1*(a**2+b**2)/(2*s_s*s_s))

    k = int(kernel_size/2)

    apply_img = np.zeros(
        [img.shape[0]+2*k, img.shape[1]+2*k, img.shape[2]], dtype='uint8')
    apply_img[k:img.shape[0]+k, k:img.shape[1]+k] = img

    new_img = np.zeros([img.shape[0], img.shape[1],
                        img.shape[2]], dtype='uint8')

    for i in range(1, len(img)-k):
        for j in range(1, len(img[0])-k):
            t = [0, 0, 0]
            for row in range(kernel_size):
                for col in range(kernel_size):
                    t += apply_img[i-k+row, j-k+col]  # *kernel[row,col]

            new_img[i, j] = t

            ###
    return new_img


def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (int(x) ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s, color):
    hl = int(diameter/2)
    i_filtered = 0
    Wp = 0
    i = 0
    sigma_i = int(sigma_i)
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            gi = gaussian(source[neighbour_x][neighbour_y]
                          [color] - source[x][y][color], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y][color] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y][color] = int(round(i_filtered))


def bilateral_filter_own(source, filter_diameter, sigma_i, sigma_s):
    filtered_image = np.zeros(source.shape, dtype='uint8')
    k = int(filter_diameter/2)
    i = k
    while i < len(source):
        j = k
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


# task1('../inputs/test1.png','../outputs/test1.png','../outputs/test1.png' )


def calculate_rms_cropped(img1, img2):
    H, W, C = img1.shape
    cut_size = 20

    img1 = img1[cut_size:H - cut_size, cut_size:W - cut_size]
    img2 = img2[cut_size:H - cut_size, cut_size:W - cut_size]

    diff = np.abs(img1 - img2)
    diff = np.abs(img1.astype(dtype=np.int) - img2.astype(dtype=np.int))

    return np.sqrt(np.mean(diff ** 2))


test4_img = cv2.imread('./inputs/test4.png', cv2.IMREAD_COLOR)
test4_c_img = cv2.imread('./inputs/test4_clean.png', cv2.IMREAD_COLOR)


def calculate_rms(img1, img2):
    """
    Calculates RMS error between two images. Two images should have same sizes.
    """
    if (img1.shape[0] != img2.shape[0]) or (img1.shape[1] != img2.shape[1]) or (img1.shape[2] != img2.shape[2]):
        raise Exception("img1 and img2 should have sime sizes.")

    diff = np.abs(img1.astype(np.int) - img2.astype(np.int))
    return np.sqrt(np.mean(diff ** 2))


print("doing!")

print(f"미디안레터럴 : ", calculate_rms(
    test4_c_img, apply_median_filter(test4_img, 3)))

print(f"원래값 : ", calculate_rms_cropped(
    test4_c_img, test4_img))


print("end!")
print("end!")
