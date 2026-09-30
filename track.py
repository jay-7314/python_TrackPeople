import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import cvzone

model = YOLO("yolo26n.pt")
names = model.names
line_x = 500

track_history = {}

in_count =0
out_count =0

cap = cv2.VideoCapture(0)           #0일때는 웹캠, 동영상을 넣으면 동영상 나옴

def RGB(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"mouse move to : [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 2 == 0:
        continue
    frame = cv2.resize(frame, (1020,600))
    results = model.track(frame, persist=True, classes = [0])

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1,x2,y2 = box
            name = names[class_id]
            cx = int(x1+x2)//2
            cy = int(y1+y2)//2
            if track_id in track_history:
                prev_cx, prev_cy = track_history[track_id]
                if(prev_cx <line_x <=cx):
                    in_count += 1
                    cv2.circle(frame, (cx,cy), 4, (255,0,0), -1)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1,y1), 1,1)
                if(prev_cx >line_x >=cx):
                    out_count += 1
                    cv2.circle(frame, (cx,cy), 4, (0,0,255), -1)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1,y1), 1,1)

            track_history[track_id] = (cx,cy)



    cvzone.putTextRect(frame, f'In: {in_count}', (40,60), scale=2, thickness=2, colorT=(255,255,255), colorR=(0,128,0))
    cvzone.putTextRect(frame, f'Out: {out_count}', (40,100), scale=2, thickness=2, colorT=(255,255,255), colorR=(0,0,255))
    cv2.line(frame, (line_x,0) ,(line_x,frame.shape[0]),(255,255,255),2)
    cv2.imshow("RGB", frame)
    

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

