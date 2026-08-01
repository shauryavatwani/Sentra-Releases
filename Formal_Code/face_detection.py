import cv2
import threading
import time
from insightface.app import FaceAnalysis


# ==============================
# CONFIGURATION
# ==============================

#RTSP_URL = "rtsp://192.168.180.226:5543/live/channel0?tcp"
RTSP_URL = "rtsp://192.168.1.9:5543/live/channel0"

# ==============================
# LOAD AI MODEL
# ==============================

print("Loading InsightFace...")

app = FaceAnalysis(
    allowed_modules=['detection'],
    providers=['CPUExecutionProvider']
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

print("✅ AI Model Loaded")


# ==============================
# CAMERA THREAD
# ==============================

class VideoStream:

    def __init__(self, url):

        self.cap = cv2.VideoCapture(
            url,
            cv2.CAP_FFMPEG
        )

        # Reduce buffering
        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        self.frame = None
        self.lock = threading.Lock()
        self.running = True


    def start(self):

        thread = threading.Thread(
            target=self.update,
            daemon=True
        )

        thread.start()

        return self


    def update(self):

        if not self.cap.isOpened():
            print("❌ Camera failed")
            self.running = False
            return


        print("✅ Camera Connected")


        while self.running:

            ret, frame = self.cap.read()

            if ret:

                with self.lock:
                    # Always replace old frame
                    self.frame = frame


    def read(self):

        with self.lock:

            if self.frame is None:
                return None

            return self.frame.copy()


    def stop(self):

        self.running = False
        self.cap.release()



# ==============================
# START CAMERA
# ==============================

camera = VideoStream(
    RTSP_URL
).start()


time.sleep(2)



# ==============================
# AI VARIABLES
# ==============================

faces = []

ai_lock = threading.Lock()

running = True



# ==============================
# AI THREAD
# ==============================

def ai_worker():

    global faces

    while running:

        frame = camera.read()


        if frame is None:
            continue


        h, w = frame.shape[:2]


        # Smaller image for AI
        small = cv2.resize(
            frame,
            (480,270)
        )


        detected = app.get(
            small
        )


        scaled_faces = []


        sx = w / 480
        sy = h / 270


        for face in detected:


            x1,y1,x2,y2 = face.bbox.astype(int)


            scaled_faces.append(
                (
                    int(x1*sx),
                    int(y1*sy),
                    int(x2*sx),
                    int(y2*sy)
                )
            )


        with ai_lock:
            faces = scaled_faces


        # AI frequency
        time.sleep(0.15)



ai_thread = threading.Thread(
    target=ai_worker,
    daemon=True
)

ai_thread.start()



# ==============================
# DISPLAY LOOP
# ==============================

previous_time = time.time()

fps = 0



while True:


    frame = camera.read()


    if frame is None:
        continue



    # Get latest detections

    with ai_lock:
        current_faces = faces.copy()



    # Draw faces

    for box in current_faces:


        x1,y1,x2,y2 = box


        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )



    # FPS

    now = time.time()

    fps = 1/(now-previous_time)

    previous_time = now



    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )


    cv2.putText(
        frame,
        f"Faces: {len(current_faces)}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )


    cv2.imshow(
        "AI Face Detection",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



# ==============================
# CLEANUP
# ==============================

running = False

camera.stop()

cv2.destroyAllWindows()