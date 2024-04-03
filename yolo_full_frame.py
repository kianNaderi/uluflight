import cv2
from ultralytics import YOLO
from supervision.video.dataclasses import VideoInfo
import time

# Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Load the YOLOv8 mode,l
model = YOLO("best.pt")
# model.export(format='onnx' , dynamic = True , optimize = True , simplify = True)
# Open the video file
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# Video information
print(VideoInfo.from_video_path(video_path))
print(model.__class__)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    start_time = time.time()
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame,persist=True, conf=0.3, iou=0.5,  imgsz=(320, 320), tracker="botsort.yaml", max_det=2 , device='cpu')

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # end = time.time()
        # Display the annotated frame
        cv2.putText(annotated_frame, f"FPS: {1.0 / (time.time() - start_time):.2f}", (10, 15),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break


cap.release()
cv2.destroyAllWindows()
