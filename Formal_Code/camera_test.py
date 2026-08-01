import cv2

RTSP_URL = "rtsp://192.168.180.226:5543/live/channel0?tcp"

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Could not open RTSP stream")
    exit()

print("✅ Connected to camera!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    cv2.imshow("CP Plus Camera", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()