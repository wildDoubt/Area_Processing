# Area_Processing
디지털 영상 처리 두 번째 과제

opencv 라이브러리를 사용하여 Area processing을 진행

# 1. Smoothing Filter
- Gaussian Filter
- Median Filter
- Average Filter
위의 3가지 필터를 사용하고 각 마스크의 크기는 3, 5, 7로 테스트함.

# 2. Sharpening

![그림1](https://user-images.githubusercontent.com/64261939/118076798-d493a900-b3ed-11eb-9294-db24e79a475e.png)

![그림2](https://user-images.githubusercontent.com/64261939/118076802-d52c3f80-b3ed-11eb-8d6d-7ed2be0df9bb.png)

위와 같은 High-boost filter를 사용하였고 A가 1.2, 1.5일 때 테스트함.

# 3. Edge Detection
- Sobel
- Prewitt
- LoG
위의 3가지 필터를 사용하였고 LoG 필터의 sigma값은 1.4로 고정함.
