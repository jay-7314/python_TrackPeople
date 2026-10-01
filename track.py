import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 로드 에러 방지

import cv2
from ultralytics import YOLO
import cvzone

model = YOLO("yolo26n.pt")
names = model.names

line_x = 500  # 세로 기준선의 x좌표

# 트랙 ID별 이전 프레임의 중심 좌표
track_history = {}

in_count = 0   # 왼쪽 -> 오른쪽으로 선을 넘은 수
out_count = 0  # 오른쪽 -> 왼쪽으로 선을 넘은 수

cap = cv2.VideoCapture("Sample.mp4")  # 0이면 웹캠, 영상 파일 경로를 넣으면 영상


def RGB(event, x, y, flags, param):
    # 마우스 좌표 확인용 (기준선 위치 잡을 때 사용)
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"mouse move to : [{x}, {y}]")


cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 속도를 위해 프레임을 하나씩 건너뜀
    frame_count += 1
    if frame_count % 2 == 0:
        continue

    frame = cv2.resize(frame, (1020, 600))

    # 사람(class 0)만 추적
    results = model.track(frame, persist=True, classes=[0])

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            name = names[class_id]

            # 박스 중심점
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if track_id in track_history:
                prev_cx, prev_cy = track_history[track_id]

                # 이전엔 선 왼쪽, 지금은 선 오른쪽 -> In
                if prev_cx < line_x <= cx:
                    in_count += 1
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1, y1), 1, 1)

                # 이전엔 선 오른쪽, 지금은 선 왼쪽 -> Out
                if prev_cx > line_x >= cx:
                    out_count += 1
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1, y1), 1, 1)

            # 다음 프레임 비교를 위해 현재 위치 저장
            track_history[track_id] = (cx, cy)

    # 카운트 표시
    cvzone.putTextRect(frame, f'In: {in_count}', (40, 60), scale=2, thickness=2,
                       colorT=(255, 255, 255), colorR=(0, 128, 0))
    cvzone.putTextRect(frame, f'Out: {out_count}', (40, 100), scale=2, thickness=2,
                       colorT=(255, 255, 255), colorR=(0, 0, 255))

    # 기준선
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 255, 255), 2)

    cv2.imshow("RGB", frame)

    # q 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()