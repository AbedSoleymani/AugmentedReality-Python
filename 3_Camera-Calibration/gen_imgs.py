import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

num = 0

while True:

    ret, img = cap.read()

    if not ret:
            print("Error: Could not read frame.")
            break

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite("./3_Camera-Calibration/imgs/img" + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()