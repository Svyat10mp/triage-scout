import time
import cv2  # OpenCV: video and drawing
import mediapipe as mp  # MediaPipe: the pose model
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "/Users/Svyat/Projects/pose_landmarker_full.task"

# Set up the pose model. VIDEO mode is for frames from a video or webcam.
options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,  # track one person
)
landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

start_time = time.time()
prev = start_time  # time of the previous frame, for FPS

while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read frame")
        break

    # OpenCV frames are BGR, but MediaPipe expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # The model needs a timestamp in milliseconds that always increases
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # Draw pose connections and a dot at each landmark (if a person was found)
    h, w = frame.shape[:2]
    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS:
            start_lm = landmarks[connection.start]
            end_lm = landmarks[connection.end]
            start_xy = (int(start_lm.x * w), int(start_lm.y * h))
            end_xy = (int(end_lm.x * w), int(end_lm.y * h))
            cv2.line(frame, start_xy, end_xy, (0, 255, 0), 2)

        for lm in landmarks:
            # lm.x and lm.y are 0-1 fractions, so scale to pixels
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    # FPS = 1 divided by the time this frame took
    now = time.time()
    fps = 1 / (now - prev)
    prev = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Pose live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()