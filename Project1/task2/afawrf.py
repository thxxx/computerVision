import cv2
import matplotlib.pyplot as plt
import numpy as np

##### To-do #####


def fm_spectrum(img):
    complexes = np.fft.fft2(img)

    # 복소수 배열이어야 똑같이 복소수(complex한 형)를 넣을때 오차가 발생하지 않는다.
    new_boksosu = complexes.copy()

    w, h = img.shape

    # gather to the center
    for i in range(w):
        for j in range(h):
            if i < w/2 and j < h/2:
                new_boksosu[i][j] = complexes[i+int(w/2)][j + int(h/2)]

            if i >= w/2 and j < h/2:
                new_boksosu[i][j] = complexes[i - int(w/2)][j + int(h/2)]

            if i < w/2 and j >= h/2:
                new_boksosu[i][j] = complexes[i + int(w/2)][j - int(h/2)]

            if i >= w/2 and j >= h/2:
                new_boksosu[i][j] = complexes[i - int(w/2)][j - int(h/2)]

    # because the number is too small now.
    img = np.log(np.abs(new_boksosu))*10

    return img


def low_pass_filter(img, th=20):

    complexes = np.fft.fft2(img)

    new_complexes = complexes.copy()

    w, h = img.shape

    for i in range(w):
        for j in range(h):
            if i < w/2 and j < h/2:
                new_complexes[i][j] = complexes[i+int(w/2)][j + int(h/2)]
            if i >= w/2 and j < h/2:
                new_complexes[i][j] = complexes[i - int(w/2)][j + int(h/2)]

            if i < w/2 and j >= h/2:
                new_complexes[i][j] = complexes[i + int(w/2)][j - int(h/2)]

            if i >= w/2 and j >= h/2:
                new_complexes[i][j] = complexes[i - int(w/2)][j - int(h/2)]

    # gather to the center but do not apply abs and log.

    copy_Fspectrum = new_complexes.copy()

    k = new_complexes.shape[0]/2
    l = new_complexes.shape[1]/2

    for a in range(new_complexes.shape[0]):
        for b in range(new_complexes.shape[1]):
            if np.sqrt((k-a)**2 + (l-b)**2) >= th:
                # calculate distance from the center.
                copy_Fspectrum[a][b] = copy_Fspectrum[a][b]*0

    new_img = np.fft.ifft2(copy_Fspectrum)

    new_img = np.abs(new_img).astype(np.uint8)

    return new_img


def high_pass_filter(img, th=30):
    freq_domain = np.fft.fft2(img)

    w, h = img.shape

    copy_Fspectrum = freq_domain.copy()

    # filter적용
    for i in range(copy_Fspectrum.shape[0]):
        for j in range(copy_Fspectrum.shape[1]):

            if i < w/2 and j < h/2:
                if 0**2 < (i**2 + j**2) < th**2:
                    copy_Fspectrum[i][j] = copy_Fspectrum[i][j]*0
            if i >= w/2 and j < h/2:
                if 0**2 < ((w-i)**2 + j**2) < th**2:
                    copy_Fspectrum[i][j] = copy_Fspectrum[i][j]*0

            if i < w/2 and j >= h/2:
                if 0**2 < (i**2 + (h-j)**2) < th**2:
                    copy_Fspectrum[i][j] = copy_Fspectrum[i][j]*0

            if i >= w/2 and j >= h/2:
                if 0**2 < ((w-i)**2 + (h-j)**2) < th**2:
                    copy_Fspectrum[i][j] = copy_Fspectrum[i][j]*0

    img = np.abs(np.fft.ifft2(copy_Fspectrum))

    return img


def denoise1(img):

    freq_domain = np.fft.fft2(img)

    # 복소수 배열이어야 똑같이 복소수(complex한 형)를 넣을때 오차가 발생하지 않는다.
    copy_Fspectrum = freq_domain.copy()

    jugi = 100

    # filter적용
    for a in range(copy_Fspectrum.shape[0]):
        for b in range(copy_Fspectrum.shape[1]):
            for i in range(6):
                for k in range(6):
                    if 47+jugi*i < a < 66+jugi*i and 45+jugi*k < b < 64+jugi*k:
                        copy_Fspectrum[a][b] = freq_domain[a-15][b-15]

                    if 97+jugi*i < a < 114+jugi*i and 97+jugi*k < b < 114+jugi*k:
                        copy_Fspectrum[a][b] = freq_domain[a-15][b-15]

    new_img = np.abs(np.fft.ifft2(copy_Fspectrum))

    return new_img


def denoise2(img):

    freq_domain = np.fft.fft2(img)

    # 복소수 배열이어야 똑같이 복소수(complex한 형)를 넣을때 오차가 발생하지 않는다.
    copy_Fspectrum = freq_domain.copy()

    w, h = img.shape

    # filter적용
    for i in range(copy_Fspectrum.shape[0]):
        for j in range(copy_Fspectrum.shape[1]):

            if i < w/2 and j < h/2:
                if 40**2 < (i**2 + j**2) < 42**2:
                    copy_Fspectrum[i][j] = freq_domain[i+5][j+5]
            if i >= w/2 and j < h/2:
                if 40**2 < ((w-i)**2 + j**2) < 42**2:
                    copy_Fspectrum[i][j] = freq_domain[i-5][j+5]

            if i < w/2 and j >= h/2:
                if 40**2 < (i**2 + (h-j)**2) < 42**2:
                    copy_Fspectrum[i][j] = freq_domain[i+5][j-5]

            if i >= w/2 and j >= h/2:
                if 40**2 < ((w-i)**2 + (h-j)**2) < 42**2:
                    copy_Fspectrum[i][j] = freq_domain[i-5][j-5]

    new_img = np.abs(np.fft.ifft2(copy_Fspectrum))

    return new_img


def discreteFourierTransform(img):
    """
    This function performs the same function as discrete fourier transform
    """

    new_img = np.zeros(img.shape, dtype='complex')
    w, h = img.shape

    for i in range(w):
        for k in range(h):
            temp = img[i][k]*0
            for x in range(w):
                for y in range(h):
                    temp += img[x][y]*np.exp(-2j * np.pi * (i*x/w + k*y/h))

            new_img[i][k] = temp/w*h

    return new_img


def inverseDiscreteFourierTransform(img):
    """
    This function performs the same function as inverse of discrete fourier transform.
    I assume that input 'img' is fourier transformed data.
    """

    print(np.fft.ifft2(img))

    new_img = np.zeros(img.shape, dtype='complex')

    w, h = img.shape

    for x in range(w):
        for y in range(h):
            temp = img[x][y]*0
            for u in range(w):
                for v in range(h):
                    temp += (img[u][v]*np.exp(2j * np.pi *
                                              (x*u/w + y*v/h))) / (w*h)

            new_img[x][y] = temp
            print(new_img[x][y])

    return new_img


#################


if __name__ == '__main__':
    img = cv2.imread('task2_sample.png', cv2.IMREAD_GRAYSCALE)
    cor1 = cv2.imread('task2_corrupted_1.png', cv2.IMREAD_GRAYSCALE)
    cor2 = cv2.imread('task2_corrupted_2.png', cv2.IMREAD_GRAYSCALE)

    plt.show()


cor2 = cv2.imread(
    '/Users/gimhojin/Downloads/CV-Assignment1/task2/task2_corrupted_2.png', cv2.IMREAD_GRAYSCALE)
cor1 = cv2.imread(
    '/Users/gimhojin/Downloads/CV-Assignment1/task2/task2_corrupted_1.png', cv2.IMREAD_GRAYSCALE)

print(fm_spectrum(cor2))
print(cor2)


cv2.imwrite(
    '/Users/gimhojin/Downloads/CV-Assignment1/task2/task21.png', fm_spectrum(cor2))

cv2.imwrite(
    '/Users/gimhojin/Downloads/CV-Assignment1/task2/task2123123.png', fm_spectrum(denoise2(cor2)))

cv2.imwrite(
    '/Users/gimhojin/Downloads/CV-Assignment1/task2/task2123.png', denoise2(cor2))
