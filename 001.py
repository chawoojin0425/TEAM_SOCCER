import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

# ========================
# 1) 이미지 읽기
# ========================
img_path = "/content/video_01_000037.jpg"
img = cv2.imread(img_path)
if img is None:
    raise Exception("이미지를 찾을 수 없습니다. 경로를 확인하세요.")

# ========================
# 2) 경기장 영역 추출 (잔디)
# ========================
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 초록색 범위 (조정 가능)
lower_green = np.array([35, 40, 40])
upper_green = np.array([90, 255, 255])
field_mask = cv2.inRange(hsv, lower_green, upper_green)

# 형태학적 연산
kernel = np.ones((5, 5), np.uint8)
field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel)
field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel)

# 경기장 외 영역 검은색 처리
field_only = img.copy()
field_only[field_mask == 0] = 0

# ========================
# 3) 밝기 히스토그램
# ========================
gray_field = cv2.cvtColor(field_only, cv2.COLOR_BGR2GRAY)

# 전체 히스토그램 (경기장 영역만)
hist = cv2.calcHist([gray_field], [0], field_mask, [256], [0, 256])

plt.figure(figsize=(10,5))
plt.plot(hist, color='green')
plt.title("Histogram of Green Field Area (Brightness)")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.xlim([0, 255])
plt.grid(True)
plt.show()

# ========================
# 4) 결과 출력
# ========================
cv2_imshow(field_only)
