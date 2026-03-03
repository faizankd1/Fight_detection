import os
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO for the real time analysis
model = YOLO("yolov8n.pt")

# Path to videos folder
video_folder = r"C:\Users\Faizan\OneDrive\Desktop\fight\archive\data"

output_folder = "detected_frames"
os.makedirs(output_folder, exist_ok=True)

def distance(box1, box2):
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2

    c1 = (x1+w1//2, y1+h1//2)
    c2 = (x2+w2//2, y2+h2//2)

    return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

saved_count = 0

for filename in os.listdir(video_folder):

    if filename.endswith(".avi"):

        video_path = os.path.join(video_folder, filename)
        cap = cv2.VideoCapture(video_path)

        print("Processing:", filename)

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            persons = []

            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:  # person
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        persons.append((x1,y1,x2-x1,y2-y1))
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            fight = False

            if len(persons) >= 2:
                for i in range(len(persons)):
                    for j in range(i+1,len(persons)):
                        if distance(persons[i],persons[j]) < 120:
                            fight = True
                            break

            if fight:
                cv2.putText(frame,"FIGHT DETECTED",(40,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

                save_path = os.path.join(
                    output_folder,
                    f"{filename}_frame_{saved_count}.jpg"
                )

                cv2.imwrite(save_path, frame)
                saved_count += 1

        cap.release()

print("Total saved frames:", saved_count)