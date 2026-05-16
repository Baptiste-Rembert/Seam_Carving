import cv2 as cv
import numpy as np

img = np.zeros((100, 100, 3), dtype=np.uint8)
cv.imshow('test', img)
print("Press Left, Right, Up, Down, ESC to quit")
while True:
    k = cv.waitKeyEx(0)
    print(f"Key pressed: {k}")
    if k == 27:
        break
cv.destroyAllWindows()
