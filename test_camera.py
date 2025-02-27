import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to read frame.")
        break

    cv2.imshow('Face Recognition', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
