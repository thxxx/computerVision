import numpy as np
import cv2
import math
import matplotlib.pyplot as plt  # 주피터 환경에서 이미지 열어주는거
import statistics


def distance(x, y, i, j):
    return ((x-i)**2 + (y-j)**2)**0.5


def gaussian(x, sigma):
    return (1.0 / (2 * np.pi * (sigma ** 2))**0.5) * np.exp(- (abs(int(x)) ** 2) / (2 * sigma ** 2))


def abf(img, filtered_image, x, y, kernel_size, sigma_s, sigma_r, c):
    i_filtered = 0
    Wp = 0

    k = int(kernel_size/2)

    for row in range(kernel_size):
        for col in range(kernel_size):
            n_x = x - k + row
            n_y = y - k + col
            if n_x >= len(img):
                n_x = len(img)-1
            if n_y >= len(img[0]):
                n_y = len(img[0])-1
            if n_x <= 0:
                n_x = 0
            if n_y <= 0:
                n_y = 0

            gi = gaussian(img[n_x][n_y]
                          [c] - img[x][y][c], sigma_r)

            gs = (1 / (2 * np.pi * (sigma_s ** 2)) ** 0.5) * \
                np.exp(-((((x-n_x) ** 2 + (y-n_y) ** 2) ** 0.5)
                         ** 2) / (2 * (sigma_s ** 2)))
            w = gi * gs  # 최종 bilateral

            i_filtered += img[n_x][n_y][c] * w
            Wp += w

    i_filtered = i_filtered / Wp  # normalize?
    filtered_image[x][y][c] = int(round(i_filtered))


"""
#      sigma_i = int(sigma_i)
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (k - i)
            neighbour_y = y - (k - j)
            if neighbour_x >= len(border_add_img):
                neighbour_x -= len(border_add_img)
            if neighbour_y >= len(border_add_img[0]):
                neighbour_y -= len(border_add_img[0])

            gi = gaussian(border_add_img[neighbour_x][neighbour_y]
                          [color] - border_add_img[x][y][color], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs  # 최종 bilateral
            i_filtered += border_add_img[neighbour_x][neighbour_y][color] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp  # normalize?
    filtered_image[x][y][color] = int(round(i_filtered))
"""


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

    # new_img = np.zeros([img.shape[0], img.shape[1], img.shape[2]], dtype='uint8')
    a_img = img.copy()
    new_img = img.copy()

    # add more zeros to each edges

    apply_img = np.ones([img.shape[0]+2*k, img.shape[1] +
                         2*k, img.shape[2]], dtype='uint8')*100
    apply_img[k:img.shape[0]+k, k:img.shape[1]+k] = img

    # 테두리에 무슨 값 넣지?


#     for i in range(k,len(img)-k):
#         for j in range(k,len(img[0])-k):
#             t= [0,0,0]
#             for row in range(kernel_size):
#                 for col in range(kernel_size):
#                         t += img[i-k+row,j-k+col] # * mask[row,col]

#             new_img[i, j]= t/(kernel_size*kernel_size)

    for i in range(k, len(apply_img)-k):
        for j in range(k, len(apply_img[0])-k):
            t = [0, 0, 0]
            for row in range(kernel_size):
                for col in range(kernel_size):
                    t += apply_img[i-k+row, j-k+col]  # * mask[row,col]

            new_img[i-k, j-k] = t/kernel_size**2

    return new_img


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

    temp = []  # temp는 한 픽셀당 rgb값의 리스트이다. ex. [143,52,211]

    def mid_of(color):
        return sorted(color)[int(len(color)/2)]

    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            red, green, blue = [], [], []
            for row in range(kernel_size):
                for col in range(kernel_size):
                    if i-k+row >= len(img) or i-k+row < 0:
                        continue
                    if j-k+col >= len(img[0]) or i-k+col < 0:
                        continue
                    t = img[i-k+row, j-k+col]
                    red.append(t[0])
                    green.append(t[1])
                    blue.append(t[2])

            new_img[i, j] = [mid_of(red), mid_of(green), mid_of(blue)]

            ###
    return new_img


def apply_median_filter2(img, kernel_size):
    """
    You should implement median filter convolution algorithm in this function.
    It takes 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is a int value, which determines kernel size of median filter.

    You should return result image.

    """
    new_img = img.copy()

    k = int(kernel_size/2)

    new_img = np.zeros([img.shape[0], img.shape[1],
                        img.shape[2]], dtype='uint8')

    temp = []  # temp는 한 픽셀당 rgb값의 리스트이다. ex. [143,52,211]

    def mid_of(color):
        return sorted(color)[int(len(color)/2)]
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            red, green, blue = [], [], []
            if img[i, j].sum() == 0 or img[i, j].sum() == 765:
                for row in range(kernel_size):
                    for col in range(kernel_size):
                        if i-k+row >= len(img) or i-k+row < 0:
                            continue
                        if j-k+col >= len(img[0]) or i-k+col < 0:
                            continue
                        t = img[i-k+row, j-k+col]
                        blue.append(t[0])
                        green.append(t[1])
                        red.append(t[2])

                new_img[i, j] = [mid_of(blue), mid_of(green), mid_of(red)]
            else:
                new_img[i, j] = img[i, j]
    return new_img


def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
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

    #  filtered_image = img.copy()

    filtered_image = img.copy()

    k = int(kernel_size/2)

    apply_img = np.zeros([img.shape[0]+2*k, img.shape[1] +
                          2*k, img.shape[2]], dtype='uint8')
    apply_img[k:img.shape[0]+k, k:img.shape[1]+k] = img

    for i in range(k, apply_img.shape[0]-k):
        for j in range(k, apply_img.shape[1]-k):
            Ws0, Ws1, Ws2 = 0, 0, 0
            f0, f1, f2 = 0, 0, 0
            for row in range(kernel_size):
                for col in range(kernel_size):
                    n_x = i - k + row
                    n_y = j - k + col

                    gaussian_s = (1 / (2 * np.pi * (sigma_s ** 2)) ** 0.5) * np.exp(-(
                        (((i-n_x) ** 2 + (j-n_y) ** 2) ** 0.5) ** 2) / (2 * (sigma_s ** 2)))

                    gaussian_r = (1.0 / np.sqrt(2 * np.pi * (sigma_r ** 2))) * np.exp(- (
                        (apply_img[n_x][n_y][0] - apply_img[i][j][0]) ** 2) / (2 * (sigma_r ** 2)))

                    gaussian_r1 = (1.0 / np.sqrt(2 * np.pi * (sigma_r ** 2))) * np.exp(- (
                        (apply_img[n_x][n_y][1] - apply_img[i][j][1]) ** 2) / (2 * (sigma_r ** 2)))

                    gaussian_r2 = (1.0 / np.sqrt(2 * np.pi * (sigma_r ** 2))) * np.exp(- (
                        (apply_img[n_x][n_y][2] - apply_img[i][j][2]) ** 2) / (2 * (sigma_r ** 2)))

                    w0 = gaussian_s * gaussian_r
                    Ws0 = Ws0 + w0
                    w1 = gaussian_s * gaussian_r1
                    Ws1 = Ws1 + w1
                    w2 = gaussian_s * gaussian_r2
                    Ws2 = Ws2 + w2

                    f0 = f0 + apply_img[n_x][n_y][0] * w0
                    f1 = f1 + apply_img[n_x][n_y][1] * w1
                    f2 = f2 + apply_img[n_x][n_y][2] * w2

            filtered_image[i-k][j-k][0] = int(round(f0 / Ws0))
            filtered_image[i-k][j-k][1] = int(round(f1 / Ws1))
            filtered_image[i-k][j-k][2] = int(round(f2 / Ws2))

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
    test_c_img, apply_bilateral_filter(test_img, 3, 0.7, 900.0)))


print(f"원래값 : ", calculate_rms_cropped(
    test_c_img, test_img))


print("end!")
