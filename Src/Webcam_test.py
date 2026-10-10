import cv2  # OpenCV: the library for working with video and images

# Open the default webcam (0 = first camera)
cap = cv2.VideoCapture(0)

# Check that the camera actually opened
if not cap.isOpened():
    print("Could not open webcam")
    exit()

while True:
    # Read one frame. 'ok' is True if it worked, 'frame' is the image
    ok, frame = cap.read()
    if not ok:
        print("Could not read frame")
        break

    # Show the frame in a window
    cv2.imshow("Webcam test", frame)

    # Wait 1 ms for a key press; quit if it's 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up: release the camera and close the window
cap.release()
cv2.destroyAllWindows()
