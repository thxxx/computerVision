import os
import sys
import glob
import cv2
import numpy as np

# 폴더와 파일 생성해서 결과 작성
STUDENT_CODE = '2017147505'
FILE_NAME = 'output.txt'
if not os.path.exists(STUDENT_CODE):
    os.mkdir(STUDENT_CODE)
f = open(os.path.join(STUDENT_CODE, FILE_NAME), 'w')

# 우선 이미지를 전부 읽어와서 vectorize한 다음 하나로 합친다.
# (39, 192 * 168)
train_images = []

image = cv2.imread("faces_training/face01.pgm", cv2.IMREAD_GRAYSCALE)
train_images.append(image)

length, height = image.shape
image = image.reshape((length * height * 1, 1))

for i in range(2, 10):
    x = cv2.imread(f"faces_training/face0{i}.pgm", cv2.IMREAD_GRAYSCALE)
    train_images.append(x)
    temp = x.reshape((length * height * 1, 1))
    image = np.concatenate([image, temp], axis=1)

for i in range(10, 40):
    x = cv2.imread(f"faces_training/face{i}.pgm", cv2.IMREAD_GRAYSCALE)
    train_images.append(x)
    temp = x.reshape((length * height * 1, 1))
    image = np.concatenate([image, temp], axis=1)

train_all_image = image


def computeDimensions(img, percent=0.95):
    """
    전체 이미지 행렬과 퍼센트를 인자로 받아서 reconstruction된 이미지와
    selected Dimension의 갯수를 반환하는 함수
    """
    # computing eigenvalues and eigenvectors of covariance matrix
    img_mean = np.mean(img.T, axis=1)
    img = (img - img_mean)

    U, Sv, Vt = np.linalg.svd(img, full_matrices=False)

    # Sv 는 고유값 리스트이다.
    Sv = sorted(Sv)
    Sv = Sv[::-1]  # 순서를 높은게 앞에 오도록 변경
    eigvals = [x**2 for x in Sv]

    sum_of = 0
    numPc = 0
    sum_all = sum(eigvals)

    for e in eigvals:
        sum_of += e
        numPc += 1
        if(sum_of/sum_all >= percent):
            break

    U = U[:, :numPc]  # 입력된 pc 개수에 따라 선택한다.
    Sv = np.diag(Sv[:numPc])
    Vt = Vt[:numPc, :]
    reconstruct = np.dot(U, np.dot(Sv, Vt)) + img_mean

    return reconstruct, numPc, U


# 목표로 할 퍼센트
perc = float(sys.argv[1])

# 값을 전달해서 추출
reconstruct, numPc, truncU = computeDimensions(train_all_image, perc)


f.write("######### STEP 1 #########\n")
f.write(f"Input Percentage: {perc}\n")
f.write(f"Selected Dimansion : {numPc}\n")


f.write("\n######### STEP 2 #########\n")
f.write("Reconstruction error\n")

# Reconstruction 에러를 담는 리스트
re_error = []

# Reconstructione 된 이미지를 담는 리스트
re_imgs = []

for n in range(39):
    re_err_sum = 0
    # 각 행별로 벡터를 가져와서 다시 이미지화 한다.
    re_img = reconstruct[:, n].reshape(length, height)
    for i in range(length):
        for j in range(height):
            re_err_sum += np.abs(train_images[n][i][j] - re_img[i][j])**2
    # 전체 에러를 픽셀 개수로 나눈다.
    re_error.append(re_err_sum/(length*height))
    re_imgs.append(re_img)

# 에러 작성하고 이미지 저장.
f.write(f"average : {round(sum(re_error)/len(re_error),4)}\n")
for n in range(1, 10):
    f.write(f"0{n}: {round(re_error[n-1],4)}\n")
    cv2.imwrite(f"{STUDENT_CODE}/face0{n}.png", re_imgs[n-1])

for n in range(10, 40):
    f.write(f"{n}: {round(re_error[n-1],4)}\n")
    cv2.imwrite(f"{STUDENT_CODE}/face{n}.png", re_imgs[n-1])


f.write("\n######### STEP 3 ##########")

test_images = []
# 테스트 데이터세트의 이미지들도 벡터화해서 매트릭스로 바꿔준다.
test_all_imgs = cv2.imread("faces_test/test01.pgm", cv2.IMREAD_GRAYSCALE)
test_images.append(test_all_imgs)

img_size = test_all_imgs.shape
test_all_imgs = test_all_imgs.reshape((length * height * 1, 1))

for i in range(2, 6):
    x = cv2.imread(f"faces_test/test0{i}.pgm", cv2.IMREAD_GRAYSCALE)
    test_images.append(x)
    temp = x.reshape((length * height * 1, 1))
    test_all_imgs = np.concatenate([test_all_imgs, temp], axis=1)

# Step 1 에서 추출해서 저장해두었던 truncated U matrix로
# 각 reconstructed 된 이미지와 테스트 이미지를 porjection한다.
face_proj = truncU.T @ reconstruct
test_proj = truncU.T @ test_all_imgs


def similarity_between_faces(face_t, face2_r):
    """
    두 이미지의 번호를 인자로 받아 Euclidean Distance를 반환한다.
    """
    face_t -= 1
    face2_r -= 1

    face_diff = test_proj[:, face_t] - face_proj[:, face2_r]

    return np.linalg.norm(face_diff)


def find_most_similar(face):
    """
    테스트 이미지의 번호를 입력으로 받아서 가장 비슷한 이미지 번호를 반환한다.
    """
    distances = []
    for face_n in range(39):
        sim = similarity_between_faces(face, face_n)
        distances.append(sim)

    best_score = min(distances)  # 가장 낮은 점수 = 가장 가까운 거리 = 가장 비슷한 이미지
    best_face_num = distances.index(min(distances))

    return best_face_num


for a in range(1, 6):
    found = find_most_similar(a)

    if found >= 10:
        f.write(f"\ntest0{a}.pgm ==> face{found}.pgm")
    else:
        f.write(f"\ntest0{a}.pgm ==> face0{found}.pgm")

f.close()
