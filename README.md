# computerVision Projects

## Project1 Image processing

### Task1
Noise Removal Filter 만들고 이미지에 적용하기
By using convolution with various filters, we can reduce noise contained in images.
<img width="500" alt="스크린샷 2021-05-12 오후 3 27 13" src="https://user-images.githubusercontent.com/27672442/117928452-896a8f00-b336-11eb-94a4-e19343fdd7ed.png">

1. Average filter
2. Median filter
3. Bilateral filter <br/>
[Bilateral fiflet란?]: https://redstarhong.tistory.com/57

Goal : RMS error (Root Mean Square) boundary 안에 들기

### Task2
Fourier Tranform 을 이용 이미지를 주파수단에서 processing하기
Fourier transform is a way that we can transfer image into frequency domains. So we can apply frequency domain filtering to image processing.
1. fm_spectrum
  • Get frequency magnitude spectrum image of input Image.
  • Spectrum image should be shifted to center.
2. low_pass_filter
  • Get filtered image that pass through with low-pass filter.
  • User could be set frequency threshold.
3. high_pass_filter
  • Get filtered image that pass through with high- pass filter.
  • User could be set frequency threshold.
4. 특정한 패턴의 노이즈 없애기
<img width="300" alt="스크린샷 2021-05-12 오후 3 33 18" src="https://user-images.githubusercontent.com/27672442/117929125-612f6000-b337-11eb-8d3d-9ff430c64544.png">
5. 특정한 패턴의 노이즈 없애기
<img width="300" alt="스크린샷 2021-05-12 오후 3 33 53" src="https://user-images.githubusercontent.com/27672442/117929189-75735d00-b337-11eb-82aa-ed60557f7740.png">

.__Extra__ : fast fourier transform 과 inverse 를 직접 구현.

## Project2 

### Task1 Face Recongnition
Given two datasets : train 39, test 5
train dataset을 이용해서 Principle Components Analysis 실행.

__Step 1__
1. Using __SVD algorithm__, compute principal components of the‘train’ dataset.
2. Given a percentage of the variance as an input, select the number of principal components you use for this data.

__Step 2__
1. Save the reconstructed image
2. Output the reconstruction loss of each train image

__Step 3__
1. Recognize images in the test dataset using a simple nearest neighbor algorithm. We’ll use 'l2 distance’. (Compute distance between two vectors.)
l2 distance(Euclidean distance)
2. __Find the closest identity of each image in the test dataset__ among the identities in the train dataset.

## Task 2 Simple Back Propagation
pytorch와 Google Colab을 써서 간단한 Back Propagation 실행.
구성 - forward pass, backworad pass, weight update


