from ultralytics import YOLO
import cv2
import math
import numpy as np
import mediapipe as mp
from datetime import datetime
import time
from PIL import Image
from collections import defaultdict
from base.com.vo.detection_vo import DetectionVO
from base.com.dao.detection_dao import DetectionDAO
from base import app
from io import BytesIO
import json
import uuid
import os
import seaborn as sns
import matplotlib.pyplot as plt


def start_timer():
    return time.time()


def stop_timer(start_time):
    return time.time() - start_time


from ultralytics import YOLO

output_folder = r"base\static\outputs"

model = YOLO(r"base\static\models\yolov8s.pt")
graph_output_folder = "base\static\graphs"

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
wep = YOLO(r"base\static\models\best_20.pt")


class InnerCamera:
    def __init__(self, path_x) -> None:
        self.person_timers = {}
        self.data_dict = defaultdict(lambda: {"covered": 0, "uncovered": 0})
        
        self.data = defaultdict(
            lambda: {"entry_time": 0, "exit_time": 0, "covered": [],
                     "uncovered": []})
        self.first_person = True
        self.occupancy_anomaly = True
        self.occupancy_anomaly_time = []
        self.detection_source = 0
        self.camera_location = "ATM premises camera"

        self.video_capture = path_x
        if self.video_capture == 0:
            self.detection_source = "WebCamera"

        else:
            self.detection_source = "Video"
        self.cap = cv2.VideoCapture(self.video_capture)

        self.weapon_is_present = False
        self.detection_starting_time = 0
        self.detection_completion_time = 0

    def inner_camera_detection(self):
        self.covered_uncovered = defaultdict(
            lambda: {"covered": 0, "uncovered": 0})
        self.created_on_current_time = time.localtime()
        self.epoch_time = time.mktime(self.created_on_current_time)
        self.created_on = self.epoch_time
        output_video_name = uuid.uuid1()

        self.output_video_path = f'{output_folder}\{output_video_name}_out.mp4'
        codec = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = cv2.VideoWriter_fourcc(*chr(codec & 0xFF),
                                        chr((codec >> 8) & 0xFF),
                                        chr((codec >> 16) & 0xFF),
                                        chr((codec >> 24) & 0xFF))
        self.out = cv2.VideoWriter(self.output_video_path, fourcc,
                                   int(self.cap.get(cv2.CAP_PROP_FPS)),
                                   (1280, 720))

        while True:
            self.detection_completion_time_current_time = time.localtime()
            self.epoch_time = time.mktime(
                self.detection_completion_time_current_time)
            self.detection_completion_time = self.epoch_time

            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            results = model.track(frame, persist=True, classes=[0],
                                  tracker="bytetrack.yaml")
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id

            weapon = wep(frame)[0]
            for result in weapon.boxes.data.tolist():
                wx1, wy1, wx2, wy2, score, class_id = result
                if wep.names[int(class_id)] == "Weapon":
                    if score >= 0.5:
                        self.weapon_is_present = True
                        cv2.rectangle(frame, (int(wx1), int(wy1)),
                                      (int(wx2), int(wy2)), (0, 0, 255), 4)
                        cv2.putText(frame, "Weapon", (int(wx1), int(wy1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255),
                                    3, cv2.LINE_AA)

            if track_ids is not None:
                if self.first_person == True:
                    self.detection_starting_time_current_time = time.localtime()
                    self.epoch_time = time.mktime(
                        self.detection_starting_time_current_time)
                    self.detection_starting_time = self.epoch_time
                    self.first_person = False
                if len(track_ids) > 1:
                    if self.occupancy_anomaly == True:
                        self.occupancy_anomaly = False
                        self.occupancy_anomaly_current_time = time.localtime()
                        self.occupancy_time = time.mktime(
                            self.occupancy_anomaly_current_time)
                        self.occupancy_anomaly_time.append(self.occupancy_time)
                else:
                    self.occupancy_anomaly = True

                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    px1, py1, px2, py2 = box
                    px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)
                    if track_id not in self.person_timers:
                        self.person_timers[track_id] = start_timer()
                    person = frame[py1:py2, px1:px2]

                    frame_rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
                    with mp_face_detection.FaceDetection(
                            min_detection_confidence=0.5) as face_detection:
                        results = face_detection.process(frame_rgb)
                        if results.detections:
                            for detection in results.detections:
                                bboxC = detection.location_data.relative_bounding_box
                                ih, iw, _ = person.shape
                                x, y, w, h = int(bboxC.xmin * iw), int(
                                    bboxC.ymin * ih), \
                                             int(bboxC.width * iw), int(
                                    bboxC.height * ih)
                                x2 = x + w
                                y2 = y + h
                                if x < 0:
                                    x = 0
                                if y < 0:
                                    y = 0
                                if x2 > person.shape[1]:
                                    x2 = person.shape[1]
                                if y2 > person.shape[0]:
                                    y2 = person.shape[0]
                                face = person[y:y2, x:x2]
                                ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
                                min_y, min_cr, min_cb = (0, 133,
                                                         78)  # Lower bound for darker skin tones

                                max_y, max_cr, max_cb = (255, 180,
                                                         133)  # Upper bound for lighter skin tones

                                # Create mask for skin pixels
                                mask = cv2.inRange(ycrcb,
                                                   (min_y, min_cr, min_cb),
                                                   (max_y, max_cr, max_cb))

                                # Calculate total number of pixels in the image
                                total_pixels = face.shape[0] * face.shape[1]

                                # Count the number of white pixels (skin pixels) in the mask
                                skin_pixels = cv2.countNonZero(mask)

                                # Calculate the percentage of skin visible
                                self.skin_percentage = (
                                                                   skin_pixels / total_pixels) * 100
                                if self.skin_percentage >= 30:
                                    self.covered_uncovered[track_id][
                                        "uncovered"] += 1
                                    if self.data[track_id]["entry_time"] == 0:
                                        self.current_time = time.localtime()
                                        self.epoch_time = time.mktime(
                                            self.current_time)

                                        self.data[track_id][
                                            "entry_time"] = int(self.epoch_time)

                                    if self.data_dict[track_id][
                                        "uncovered"] == 0:
                                        self.current_time = time.localtime()
                                        self.epoch_time = time.mktime(
                                            self.current_time)

                                        self.data_dict[track_id][
                                            "uncovered"] += 1
                                        self.data[track_id][
                                            "uncovered"].append(int(self.epoch_time))
                                        self.data[track_id][
                                            "exit_time"] = self.epoch_time
                                        # self.data[track_id]["uncovered"].append(self.current_time)
                                        self.data_dict[track_id]["covered"] = 0
                                        cv2.rectangle(frame,
                                                      (int(px1), int(py1)),
                                                      (int(px2), int(py2)),
                                                      (0, 255, 0), 4)
                                        cv2.putText(frame,
                                                    f'Person {track_id}: person is uncoverd',
                                                    (int(px1), int(py1 - 10)),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    1.3,
                                                    (0, 255, 0), 3,
                                                    cv2.LINE_AA)
                                    else:
                                        self.current_time = time.localtime()
                                        self.epoch_time = time.mktime(
                                            self.current_time)

                                        self.data[track_id][
                                            "exit_time"] = int(self.epoch_time)

                                        self.data_dict[track_id]["covered"] = 0
                                        cv2.rectangle(frame,
                                                      (int(px1), int(py1)),
                                                      (int(px2), int(py2)),
                                                      (0, 255, 0), 4)
                                        cv2.putText(frame,
                                                    f'Person {track_id}: person is uncoverd',
                                                    (int(px1), int(py1 - 10)),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    1.3,
                                                    (0, 255, 0), 3,
                                                    cv2.LINE_AA)



                                else:
                                    self.covered_uncovered[track_id][
                                        "covered"] += 1
                                    if self.data[track_id]["entry_time"] == 0:
                                        self.current_time = time.localtime()
                                        self.epoch_time = time.mktime(
                                            self.current_time)

                                        self.data[track_id][
                                            "entry_time"] = int(self.epoch_time)

                                    self.data_dict[track_id]["covered"] += 1
                                    if self.data_dict[track_id][
                                        "covered"] >= 10:
                                        if self.data_dict[track_id][
                                            "covered"] == 10:
                                            self.current_time = time.localtime()
                                            self.epoch_time = time.mktime(
                                                self.current_time)
                                            self.data[track_id][
                                                "covered"].append( int(self.epoch_time))
                                               
                                            self.data[track_id][
                                                "exit_time"] = self.epoch_time
                                            cv2.rectangle(frame,
                                                          (int(px1), int(py1)),
                                                          (int(px2), int(py2)),
                                                          (0, 0, 255), 4)
                                            cv2.putText(frame,
                                                        f'Person {track_id}: person is coverd',
                                                        (int(px1),
                                                         int(py1 - 10)),
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        1.3,
                                                        (0, 0, 255), 3,
                                                        cv2.LINE_AA)
                                            self.data_dict[track_id][
                                                "uncovered"] = 0
                                        else:
                                            self.current_time = time.localtime()
                                            self.epoch_time = time.mktime(
                                                self.current_time)

                                            self.data[track_id][
                                                "exit_time"] = int(self.epoch_time)
                                            cv2.rectangle(frame,
                                                          (int(px1), int(py1)),
                                                          (int(px2), int(py2)),
                                                          (0, 0, 255), 4)
                                            cv2.putText(frame,
                                                        f'Person {track_id}: person is coverd',
                                                        (int(px1),
                                                         int(py1 - 10)),
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        1.3,
                                                        (0, 0, 255), 3,
                                                        cv2.LINE_AA)
                                            self.data_dict[track_id][
                                                "uncovered"] = 0


                                    else:
                                        self.current_time = time.localtime()
                                        self.epoch_time = time.mktime(
                                            self.current_time)

                                        self.data[track_id][
                                            "exit_time"] = int(self.epoch_time)
                                        cv2.rectangle(frame,
                                                      (int(px1), int(py1)),
                                                      (int(px2), int(py2)),
                                                      (0, 255, 0), 4)
                                        cv2.putText(frame,
                                                    f'Person {track_id}: person is uncoverd',
                                                    (int(px1), int(py1 - 10)),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    1.3,
                                                    (0, 255, 0), 3,
                                                    cv2.LINE_AA)


                        else:
                            self.covered_uncovered[track_id]["covered"] += 1
                            if self.data[track_id]["entry_time"] == 0:
                                self.current_time = time.localtime()
                                self.epoch_time = time.mktime(
                                    self.current_time)

                                self.data[track_id][
                                    "entry_time"] = int(self.epoch_time)

                            self.data_dict[track_id]["covered"] += 1
                            if self.data_dict[track_id]["covered"] >= 10:
                                if self.data_dict[track_id]["covered"] == 10:
                                    self.current_time = time.localtime()
                                    self.epoch_time = time.mktime(
                                        self.current_time)
                                    self.data[track_id]["covered"].append(int(self.epoch_time))
                                        
                                    self.data[track_id][
                                        "exit_time"] = int(self.epoch_time)
                                    cv2.rectangle(frame, (int(px1), int(py1)),
                                                  (int(px2), int(py2)),
                                                  (0, 0, 255), 4)
                                    cv2.putText(frame,
                                                f'Person {track_id}: person is coverd',
                                                (int(px1), int(py1 - 10)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                                                (0, 0, 255), 3, cv2.LINE_AA)
                                    self.data_dict[track_id]["uncovered"] = 0
                                else:
                                    self.current_time = time.localtime()
                                    self.epoch_time = time.mktime(
                                        self.current_time)

                                    self.data[track_id][
                                        "exit_time"] = int(self.epoch_time)
                                    cv2.rectangle(frame, (int(px1), int(py1)),
                                                  (int(px2), int(py2)),
                                                  (0, 0, 255), 4)
                                    cv2.putText(frame,
                                                f'Person {track_id}: person is coverd',
                                                (int(px1), int(py1 - 10)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                                                (0, 0, 255), 3, cv2.LINE_AA)
                                    self.data_dict[track_id]["uncovered"] = 0


                            else:
                                self.current_time = time.localtime()
                                self.epoch_time = time.mktime(
                                    self.current_time)

                                self.data[track_id][
                                    "exit_time"] = int(self.epoch_time)
                                cv2.rectangle(frame, (int(px1), int(py1)),
                                              (int(px2), int(py2)),
                                              (0, 255, 0), 4)
                                cv2.putText(frame,
                                            f'Person {track_id}: person is uncoverd',
                                            (int(px1), int(py1 - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                                            (0, 255, 0), 3, cv2.LINE_AA)

            self.out.write(frame)
            yield frame

    def stop_inner_camera(self):
        self.out.release()
        self.cap.release()
        covered_count = 0
        uncovered_count = 0
        for key, value in self.covered_uncovered.items():
            if value['covered'] > value['uncovered']:
                covered_count += 1
            elif value['uncovered'] > value['covered']:
                uncovered_count += 1

        # Data dictionary
        graph_data = {'category': ['Covered', 'Uncovered'],
                        'count': [covered_count, uncovered_count]}

        # Create the bar chart with custom colors
        sns.barplot(x='category', y='count', data=graph_data,
                    palette=['blue', 'orange'], legend=False)

        # Add title and labels
        plt.title("Number of Covered and Uncovered Persons")
        plt.xlabel("Category")
        plt.ylabel("Count")

        graph_name = uuid.uuid1()

        # Save the graphs using 'savefig'
        plt.savefig(f"{graph_output_folder}/{graph_name}.png",
                    format='png')  # Replace 'png' with desired format
        plt.clf()
        self.graph_path = f"{graph_output_folder}/{graph_name}.png"




        self.total_detected_person = 0

        for key in self.data:
            self.total_detected_person += 1
        self.modified_on = self.created_on
        self.occupancy_anomaly_list = json.dumps(self.occupancy_anomaly_time)

        self.detection_stats = json.dumps(self.data)
    
        return str(self.video_capture), str(
            self.output_video_path), self.camera_location, self.detection_source, int(
            self.detection_starting_time), self.total_detected_person, self.detection_stats, self.occupancy_anomaly_list, self.weapon_is_present ,int(
            self.detection_completion_time), str(self.graph_path), int(
            self.created_on), int(self.modified_on)


class OuterCamera:
    def __init__(self, path_x) -> None:
        self.person_timers = {}
        self.data_dict = defaultdict(lambda: {"covered": 0, "uncovered": 0})
        self.covered_uncovered = defaultdict(
            lambda: {"covered": 0, "uncovered": 0})
        self.data = defaultdict(
            lambda: {"entry_time": 0, "exit_time": 0, "covered": [],
                     "uncovered": []})
        self.first_person = True
        self.occupancy_anomaly = True
        self.occupancy_anomaly_time = []
        self.detection_source = 0
        self.camera_location = "Outside ATM premises camera"
        self.area=[(323,502),(818,466),(1276,704),(9,711)]
        # self.area=[(0,0),(0,0),(0,0),(0,0)]


        self.video_capture = path_x
        if self.video_capture == 0:
            self.detection_source = "WebCamera"
        else:
            self.detection_source = "Video"
        self.cap = cv2.VideoCapture(self.video_capture)

        self.weapon_is_present = False
        self.detection_starting_time = 0
        self.detection_completion_time = 0

    def outside_camera_detection(self):
        self.created_on_current_time = time.localtime()
        self.epoch_time = time.mktime(self.created_on_current_time)
        self.created_on = self.epoch_time
        output_video_name = uuid.uuid1()
        

        self.output_video_path = f'{output_folder}\{output_video_name}_out.mp4'
        codec = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = cv2.VideoWriter_fourcc(*chr(codec & 0xFF),
                                        chr((codec >> 8) & 0xFF),
                                        chr((codec >> 16) & 0xFF),
                                        chr((codec >> 24) & 0xFF))
        self.out = cv2.VideoWriter(self.output_video_path, fourcc,
                                   int(self.cap.get(cv2.CAP_PROP_FPS)),
                                   (1280, 720))
        while True:
            self.detection_completion_time_current_time = time.localtime()
            self.epoch_time = time.mktime(
                self.detection_completion_time_current_time)
            self.detection_completion_time = self.epoch_time

            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            results = model.track(frame, persist=True, classes=[0],
                                  tracker="bytetrack.yaml")
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id

           
            if track_ids is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                for box, track_id in zip(boxes, track_ids):
                    px1, py1, px2, py2 = box
                    px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)
                    result = cv2.pointPolygonTest(np.array(self.area, np.int32), ((px1, py2)),
                        False)  # checks if the x1,y1 point is within the ROI
                    if result >= 0:

                        weapon = wep(frame)[0]
                        for result in weapon.boxes.data.tolist():
                            wx1, wy1, wx2, wy2, score, class_id = result
                            if wep.names[int(class_id)] == "Weapon":
                                if score >= 0.5:
                                    self.weapon_is_present = True
                                    cv2.rectangle(frame, (int(wx1), int(wy1)),
                                                    (int(wx2), int(wy2)), (0, 0, 255), 4)
                                    cv2.putText(frame, "Weapon", (int(wx1), int(wy1 - 10)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255),
                                                3, cv2.LINE_AA)

                       
                            if self.first_person == True:
                                self.detection_starting_time_current_time = time.localtime()
                                self.epoch_time = time.mktime(
                                    self.detection_starting_time_current_time)
                                self.detection_starting_time = self.epoch_time
                                self.first_person = False
                            if len(track_ids) > 1:
                                if self.occupancy_anomaly == True:
                                    self.occupancy_anomaly = False
                                    self.occupancy_anomaly_current_time = time.localtime()
                                    self.occupancy_time = time.mktime(
                                        self.occupancy_anomaly_current_time)
                                    self.occupancy_anomaly_time.append(self.occupancy_time)
                            else:
                                self.occupancy_anomaly = True
                            # boxes = results[0].boxes.xyxy.cpu()

                            # track_ids = results[0].boxes.id.int().cpu().tolist()

                            for box, track_id in zip(boxes, track_ids):
                                px1, py1, px2, py2 = box
                                px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)
                                if track_id not in self.person_timers:
                                    self.person_timers[track_id] = start_timer()
                                person = frame[py1:py2, px1:px2]

                                frame_rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
                                with mp_face_detection.FaceDetection(
                                        min_detection_confidence=0.5) as face_detection:
                                    results = face_detection.process(frame_rgb)
                                    if results.detections:
                                        for detection in results.detections:
                                            bboxC = detection.location_data.relative_bounding_box
                                            ih, iw, _ = person.shape
                                            x, y, w, h = int(bboxC.xmin * iw), int(
                                                bboxC.ymin * ih), \
                                                            int(bboxC.width * iw), int(
                                                bboxC.height * ih)
                                            x2 = x + w
                                            y2 = y + h
                                            if x < 0:
                                                x = 0
                                            if y < 0:
                                                y = 0
                                            if x2 > person.shape[1]:
                                                x2 = person.shape[1]
                                            if y2 > person.shape[0]:
                                                y2 = person.shape[0]
                                            face = person[y:y2, x:x2]
                                            ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
                                            min_y, min_cr, min_cb = (0, 133,
                                                                        78)  # Lower bound for darker skin tones

                                            max_y, max_cr, max_cb = (255, 180,
                                                                        133)  # Upper bound for lighter skin tones

                                            # Create mask for skin pixels
                                            mask = cv2.inRange(ycrcb,
                                                                (min_y, min_cr, min_cb),
                                                                (max_y, max_cr, max_cb))

                                            # Calculate total number of pixels in the image
                                            total_pixels = face.shape[0] * face.shape[1]

                                            # Count the number of white pixels (skin pixels) in the mask
                                            skin_pixels = cv2.countNonZero(mask)

                                            # Calculate the percentage of skin visible
                                            self.skin_percentage = (skin_pixels / total_pixels) * 100
                                            if self.skin_percentage >= 30:
                                                self.covered_uncovered[track_id]["uncovered"] += 1
                                                if self.data[track_id]["entry_time"] == 0:
                                                    self.current_time = time.localtime()
                                                    self.epoch_time = time.mktime(
                                                        self.current_time)

                                                    self.data[track_id]["entry_time"] = int(self.epoch_time)

                                                if self.data_dict[track_id]["uncovered"] == 0:
                                                    self.current_time = time.localtime()
                                                    self.epoch_time = time.mktime(
                                                        self.current_time)

                                                    self.data_dict[track_id]["uncovered"] += 1
                                                    self.data[track_id]["uncovered"].append(int(self.epoch_time))
                                                    self.data[track_id]["exit_time"] = int(self.epoch_time)
                                                    # self.data[track_id]["uncovered"].append(self.current_time)
                                                    self.data_dict[track_id]["covered"] = 0
                                                    cv2.rectangle(frame,
                                                                    (int(px1), int(py1)),
                                                                    (int(px2), int(py2)),
                                                                    (0, 255, 0), 4)
                                                    cv2.putText(frame,
                                                                f'Person {track_id}: person is uncoverd',
                                                                (int(px1), int(py1 - 10)),
                                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                                1.3,
                                                                (0, 255, 0), 3,
                                                                cv2.LINE_AA)
                                                else:
                                                    self.current_time = time.localtime()
                                                    self.epoch_time = time.mktime(
                                                        self.current_time)

                                                    self.data[track_id][
                                                        "exit_time"] = int(self.epoch_time)

                                                    self.data_dict[track_id]["covered"] = 0
                                                    cv2.rectangle(frame,
                                                                    (int(px1), int(py1)),
                                                                    (int(px2), int(py2)),
                                                                    (0, 255, 0), 4)
                                                    cv2.putText(frame,
                                                                f'Person {track_id}: person is uncoverd',
                                                                (int(px1), int(py1 - 10)),
                                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                                1.3,
                                                                (0, 255, 0), 3,
                                                                cv2.LINE_AA)



                                            else:
                                                self.covered_uncovered[track_id][
                                                    "covered"] += 1
                                                if self.data[track_id]["entry_time"] == 0:
                                                    self.current_time = time.localtime()
                                                    self.epoch_time = time.mktime(
                                                        self.current_time)

                                                    self.data[track_id][
                                                        "entry_time"] = int(self.epoch_time)

                                                self.data_dict[track_id]["covered"] += 1
                                                if self.data_dict[track_id][
                                                    "covered"] >= 10:
                                                    if self.data_dict[track_id][
                                                        "covered"] == 10:
                                                        self.current_time = time.localtime()
                                                        self.epoch_time = time.mktime(
                                                            self.current_time)
                                                        self.data[track_id][
                                                            "covered"].append(int(self.epoch_time))
                                                            
                                                        self.data[track_id][
                                                            "exit_time"] = int(self.epoch_time)
                                                        cv2.rectangle(frame,
                                                                        (int(px1), int(py1)),
                                                                        (int(px2), int(py2)),
                                                                        (0, 0, 255), 4)
                                                        cv2.putText(frame,
                                                                    f'Person {track_id}: person is coverd',
                                                                    (int(px1),
                                                                        int(py1 - 10)),
                                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                                    1.3,
                                                                    (0, 0, 255), 3,
                                                                    cv2.LINE_AA)
                                                        self.data_dict[track_id][
                                                            "uncovered"] = 0
                                                    else:
                                                        self.current_time = time.localtime()
                                                        self.epoch_time = time.mktime(
                                                            self.current_time)

                                                        self.data[track_id][
                                                            "exit_time"] = int(self.epoch_time)
                                                        cv2.rectangle(frame,
                                                                        (int(px1), int(py1)),
                                                                        (int(px2), int(py2)),
                                                                        (0, 0, 255), 4)
                                                        cv2.putText(frame,
                                                                    f'Person {track_id}: person is coverd',
                                                                    (int(px1),
                                                                        int(py1 - 10)),
                                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                                    1.3,
                                                                    (0, 0, 255), 3,
                                                                    cv2.LINE_AA)
                                                        self.data_dict[track_id][
                                                            "uncovered"] = 0


                                                else:
                                                    self.current_time = time.localtime()
                                                    self.epoch_time = time.mktime(
                                                        self.current_time)

                                                    self.data[track_id][
                                                        "exit_time"] = int(self.epoch_time)
                                                    cv2.rectangle(frame,
                                                                    (int(px1), int(py1)),
                                                                    (int(px2), int(py2)),
                                                                    (0, 255, 0), 4)
                                                    cv2.putText(frame,
                                                                f'Person {track_id}: person is uncoverd',
                                                                (int(px1), int(py1 - 10)),
                                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                                1.3,
                                                                (0, 255, 0), 3,
                                                                cv2.LINE_AA)


                                    else:
                                        self.covered_uncovered[track_id]["covered"] += 1
                                        if self.data[track_id]["entry_time"] == 0:
                                            self.current_time = time.localtime()
                                            self.epoch_time = time.mktime(
                                                self.current_time)

                                            self.data[track_id][
                                                "entry_time"] = int(self.epoch_time)

                                        self.data_dict[track_id]["covered"] += 1
                                        if self.data_dict[track_id]["covered"] >= 10:
                                            if self.data_dict[track_id]["covered"] == 10:
                                                self.current_time = time.localtime()
                                                self.epoch_time = time.mktime(
                                                    self.current_time)
                                                self.data[track_id]["covered"].append(int( self.epoch_time))
                                                   
                                                self.data[track_id][
                                                    "exit_time"] = int(self.epoch_time)
                                                cv2.rectangle(frame, (int(px1), int(py1)),
                                                                (int(px2), int(py2)),
                                                                (0, 0, 255), 4)
                                                cv2.putText(frame,
                                                            f'Person {track_id}: person is coverd',
                                                            (int(px1), int(py1 - 10)),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                                                            (0, 0, 255), 3, cv2.LINE_AA)
                                                self.data_dict[track_id]["uncovered"] = 0
                                            else:
                                                self.current_time = time.localtime()
                                                self.epoch_time = time.mktime(
                                                    self.current_time)

                                                self.data[track_id][
                                                    "exit_time"] = int(self.epoch_time)
                                                cv2.rectangle(frame, (int(px1), int(py1)),
                                                                (int(px2), int(py2)),
                                                                (0, 0, 255), 4)
                                                cv2.putText(frame,
                                                            f'Person {track_id}: person is coverd',
                                                            (int(px1), int(py1 - 10)),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                                                            (0, 0, 255), 3, cv2.LINE_AA)
                                                self.data_dict[track_id]["uncovered"] = 0


                                        else:
                                            self.current_time = time.localtime()
                                            self.epoch_time = time.mktime(
                                                self.current_time)

                                            self.data[track_id][
                                                "exit_time"] = int(self.epoch_time)
                                            cv2.rectangle(frame, (int(px1), int(py1)),
                                                            (int(px2), int(py2)),
                                                            (0, 255, 0), 4)
                                            cv2.putText(frame,
                                                        f'Person {track_id}: person is uncoverd',
                                                        (int(px1), int(py1 - 10)),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                
                                                        (0, 255, 0), 3, cv2.LINE_AA)
                
                    
                
            cv2.polylines(frame,[np.array(self.area,np.int32)],True,(255,0,255),2)
            self.out.write(frame)
            yield frame

    def stop_outer_camera(self):
        self.out.release()
        self.cap.release()
        covered_count = 0
        uncovered_count = 0
        for key, value in self.covered_uncovered.items():
            if value['covered'] > value['uncovered']:
                covered_count += 1
            elif value['uncovered'] > value['covered']:
                uncovered_count += 1

        # Data dictionary
        graph_data = {'category': ['Covered', 'Uncovered'],
                        'count': [covered_count, uncovered_count]}

        # Create the bar chart with custom colors
        sns.barplot(x='category', y='count', data=graph_data,
                    palette=['blue', 'orange'], legend=False)

        # Add title and labels
        plt.title("Number of Covered and Uncovered Persons")
        plt.xlabel("Category")
        plt.ylabel("Count")

        graph_name = uuid.uuid1()

        # Save the graphs using 'savefig'
        plt.savefig(f"{graph_output_folder}/{graph_name}.png",
                    format='png')  # Replace 'png' with desired format
        plt.clf()
        self.graph_path = f"{graph_output_folder}/{graph_name}.png"


        self.total_detected_person = 0

        for key in self.data:
            self.total_detected_person += 1
        self.modified_on = self.created_on
        self.occupancy_anomaly_list = json.dumps(self.occupancy_anomaly_time)


        self.detection_stats = json.dumps(self.data)

        return str(self.video_capture), str(
            self.output_video_path), self.camera_location, self.detection_source, int(
            self.detection_starting_time), self.total_detected_person, self.detection_stats, self.occupancy_anomaly_list, self.weapon_is_present,int(
            self.detection_completion_time), str(self.graph_path), int(
            self.created_on), int(self.modified_on)


class DisplayObject:
    class PersonStatistics:
        id: int
        entry_time: datetime
        exit_time: datetime
        average_covered_time: int
        average_uncovered_time: int

    def __init__(self, detection) -> None:
        self.detection_id = detection.detection_id
        self.input_video = self.filename(str(detection.input_video_path))
        self.output_video = self.filename(str(detection.output_video_path))
        self.camera_location = detection.camera_location
        self.source = detection.detection_source
        self.detection_starting_time = datetime.fromtimestamp(
            detection.detection_starting_time)
        self.total_detected_person = detection.total_detected_person
        self.detection_statistics = self.average_covered_uncovered_time(
            json.loads(detection.detection_statistics))
        self.occupancy_anomaly = self.occupancy_anomaly_time(
            json.loads(detection.occupancy_anomaly))
        self.weapon=detection.weapon_detected
        self.detection_completion_time = datetime.fromtimestamp(
            detection.detection_completion_time)
        self.graph_path = self.filename(str(detection.graph_path))

        self.created_on = datetime.fromtimestamp(detection.created_on)
        self.modified_on = datetime.fromtimestamp(detection.modified_on)

    def filename(self, file):
        try:

            file_path = os.path.join(*file.split(os.path.sep))
            filename = os.path.basename(file_path)
            return filename
        except Exception as e:
            self.input_video = "0"



    def occupancy_anomaly_time(self, timestamps):
        return [datetime.fromtimestamp(timestamp) for timestamp in timestamps]

    def average_covered_uncovered_time(self, statistics):
        person_list = []
        for key, value in statistics.items():
            person_statistics_object = DisplayObject.PersonStatistics()
            person_statistics_object.id = key
            person_statistics_object.entry_time = datetime.fromtimestamp(
                value["entry_time"])
            person_statistics_object.exit_time = datetime.fromtimestamp(
                value["exit_time"])
            try:
                covered_time = value["covered"]
                uncovered_time = value["uncovered"]
                if (len(covered_time)==0) and (len(uncovered_time)==0):
                    person_statistics_object.average_covered_time = 0
                    person_statistics_object.average_uncovered_time = 0

            
                # else:
                average_list = covered_time + uncovered_time
                average_list.append(value["exit_time"])
                average_list.sort()
                length = len(average_list)
                average_time_1 = 0
                average_time_2 = 0

                for i in range(length):
                    if i % 2 == 0:
                        if i == length - 1:

                            break

                        else:

                            average_1 = average_list[i + 1] - average_list[i]
                            average_time_1 += average_1

                    else:
                        if i == length - 1:
                            break

                        else:
                            average_2 = average_list[i + 1] - average_list[i]
                            average_time_2 += average_2
                if len(covered_time) == 0:
                    person_statistics_object.average_uncovered_time = average_time_1
                    person_statistics_object.average_covered_time = 0

                elif len(uncovered_time) == 0:
                    person_statistics_object.average_covered_time = average_time_1
                    person_statistics_object.average_uncovered_time = 0
                else:
                    if covered_time[0] < uncovered_time[0]:
                        person_statistics_object.average_covered_time = average_time_1
                        person_statistics_object.average_uncovered_time = average_time_2
                    else:
                        person_statistics_object.average_covered_time = average_time_2
                        person_statistics_object.average_uncovered_time = average_time_1

            except Exception as e:
                print(e)

                person_statistics_object.average_covered_time = 0
                person_statistics_object.average_uncovered_time = 0
            person_list.append(person_statistics_object)
        return person_list

    @staticmethod
    def list_from_json(json):
        return [DisplayObject(data) for data in json]


















