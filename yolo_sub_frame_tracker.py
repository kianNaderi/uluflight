import cv2
import math
from ultralytics import YOLO
from sort import Sort
import numpy as np
import os
import time
import cProfile

def predict_on_subframes(frame, start_x, start_y, end_x, end_y, model,confident ,org=True ):
    sub_frame = frame[start_y:end_y, start_x:end_x]
    output = model(sub_frame, conf=confident, iou=0.3, max_det=1, imgsz=320, device=device)
    xyxy = output[0].boxes.xyxy.cpu().numpy().astype(int)
    conf = output[0].boxes.conf.cpu().numpy()
    if xyxy.size > 0 and conf.size > 0:
        if org:
            xyxy[:, 0] += start_x
            xyxy[:, 2] += start_x
            xyxy[:, 1] += start_y
            xyxy[:, 3] += start_y
    return xyxy, conf


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
model = YOLO("Models/Plane2-320/best.pt")
cap = cv2.VideoCapture("test.mp4")
imgsz = (320, 320)
device = 'cpu'
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
num_frames_width = math.ceil(w / imgsz[0])
num_frames_height = math.ceil(h / imgsz[1])
center_point = (int(w / 2), int(h / 2))
# should be calculated
pixel_per_meter = 10
blue, green, red = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
track_tresh = 5
sub_tresh = 50

# tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

# PROBABLE TARGET FINDING
while True:
    start_time = time.time()
    results = []
    for i in range(num_frames_width):
        # Calculate sub-frame coordinates
        for j in range(num_frames_height):
            ret, im0 = cap.read()
            if not ret:
                continue

            start_x = i * imgsz[0]
            end_x = min(start_x + imgsz[0], w)
            start_y = j * imgsz[1]
            end_y = min(start_y + imgsz[1], h)
            # Passing sun-frame to Model
            sub_result_xyxy, sub_result_conf = predict_on_subframes(im0, start_x, start_y, end_x, end_y, model,
                                                                    confident=0.35, org=True )

            cv2.rectangle(im0, (start_x, start_y), (end_x, end_y), red, thickness=5)

            if sub_result_xyxy.size > 0:

                results.append(sub_result_xyxy[0])
                cord = [int(i) for i in sub_result_xyxy[0]]

                cv2.rectangle(im0, (cord[0], cord[1]), (cord[2], cord[3]), red, 4)
                cv2.putText(im0, f"Confidence: {sub_result_conf[0]:.2f}", (cord[0], cord[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, red, 2)
                cv2.putText(im0, f"FPS: {1.0 / (time.time() - start_time):.2f}", (10, 15),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("iha_detection", im0)
            cv2.waitKey(2)

    # PROBABLE TARGET FOLLOWING
    if len(results) > 0:
        sum_run = []
        for i in range(5):
            for index, p_target in enumerate(results):
                ret, im0 = cap.read()
                if not ret:
                    continue
                start_x = max(0, int(p_target[0] - imgsz[0] / 2))
                start_y = max(0, int(p_target[1] - imgsz[1] / 2))
                end_x = start_x + imgsz[0]
                end_y = start_y + imgsz[1]
                sub_result_xyxy, sub_result_conf = predict_on_subframes(im0, start_x, start_y, end_x, end_y, model,
                                                                        confident=0.40, org=True)
                if sub_result_xyxy.size > 0:
                    results[index] = sub_result_xyxy[0]
                    if i == 0:
                        sum_run.append(sub_result_conf[0]*100)
                    else:
                        sum_run[index] += sub_result_conf*100
                else:
                    if i == 0:
                        sum_run.append(0)
                    else:
                        sum_run[index] += 0
                cv2.rectangle(im0, (p_target[0], p_target[1]), (p_target[2], p_target[3]), blue, 4)
                cv2.putText(im0, f"Probable Target", (p_target[0], p_target[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, blue, 2)
                cv2.rectangle(im0, (start_x, start_y), (end_x, end_y), blue, thickness=5)
                cv2.putText(im0, f"FPS: {1.0 / (time.time() - start_time):.2f}", (10, 15),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("iha_detection", im0)
                cv2.waitKey(1)


        # TARGET FOLLOWING
        if max(sum_run) > 120:
            target = results[sum_run.index(max(sum_run))]
            tracker = cv2.TrackerCSRT_create()
            #tracker = cv2.TrackerGOTURN_create()
            j = 0
            while True:
                i = 0
                #tracker = cv2.TrackerGOTURN_create()
                target[0] -= track_tresh
                target[1] -= track_tresh
                target[2] = int(target[2]-target[0])+track_tresh
                target[3] = int(target[3] - target[1])+track_tresh
                target = tuple(target)
                ok = tracker.init(im0,target)
                while True:
                    start_time = time.time()
                    i += 1
                    color = green
                    ret, im0 = cap.read()
                    if not ret:
                        continue
                    ok, target = tracker.update(im0)
                    target = list(target)
                    if ok:
                        target[2] = int(target[0] + target[2])
                        target[3] = int(target[1] + target[3])

                    if i > 10:
                        i = 0
                        start_x = max(0, int(target[0] - imgsz[0] / 2))
                        start_y = max(0, int(target[1] - imgsz[1] / 2))
                        end_x = start_x + imgsz[0]
                        end_y = start_y + imgsz[1]
                        sub_result_xyxy, sub_result_conf = predict_on_subframes(im0, start_x, start_y, end_x, end_y,
                                                                                model, confident=0.3, org=True)
                        if sub_result_xyxy.size > 0:
                            j = 0
                            if (abs(sub_result_xyxy[0][0] - target[0]) > 10 or
                                    abs(sub_result_xyxy[0][3] - target[3]) > 10):
                                color = (100, 100, 100)
                                target = sub_result_xyxy[0]
                                target = list(target)
                                cv2.rectangle(im0, (target[0], target[1]), (target[2], target[3]), color, 4)
                                cv2.putText(im0, f"Target", (target[0], target[1]),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
                                cv2.rectangle(im0, (start_x, start_y), (end_x, end_y), color, thickness=5)
                                cv2.putText(im0, f"FPS: {1.0 / (time.time() - start_time):.2f}", (10, 15),
                                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.imshow("iha_detection", im0)
                                cv2.waitKey(1)
                                break
                        else:
                            j += 1
                            break

                    cv2.rectangle(im0, (target[0], target[1]), (target[2], target[3]), color, 4)
                    cv2.putText(im0, f"Target", (target[0], target[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
                    cv2.rectangle(im0, (start_x, start_y), (end_x, end_y), color, thickness=5)
                    cv2.putText(im0, f"FPS: {1.0 / (time.time() - start_time):.2f}", (10, 15),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow("iha_detection", im0)
                    cv2.waitKey(1)
                if j > 1 or all(element == 0 for element in target):
                    break
        else:
            continue
    else:
        continue

cap.release()
cv2.destroyAllWindows()
