from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import cv2
import mediapipe as mp
import asyncio
import numpy as np

app = FastAPI()

templates = Jinja2Templates(directory="templates")

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

JUMP_THRESHOLD = 0.05
CROUCH_THRESHOLD = 0.8
SQUAT_THRESHOLD = 0.4 

def detect_hand_orientation(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    avg_fingertip_z = (index_fingertip.z + middle_fingertip.z + ring_fingertip.z + pinky_fingertip.z) / 4

    return "Palm Facing Forward" if avg_fingertip_z < wrist.z else "Palm Facing Backward" #not optimized

def process_frame(frame, pose, hands, prev_hip_y):
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = pose_results.pose_landmarks.landmark
        nose_x = int(landmarks[mp_pose.PoseLandmark.NOSE].x * width)
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]

        upper_body_center = (nose_x + int(left_shoulder.x * width) + int(right_shoulder.x * width)) // 3
        lower_body_center = (int(left_hip.x * width) + int(right_hip.x * width)) // 2

        cv2.line(frame, (upper_body_center, 0), (lower_body_center, height), (0, 255, 255), 3)

        avg_hip_y = (left_hip.y + right_hip.y) / 2
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        avg_knee_y = (left_knee.y + right_knee.y) / 2

        if upper_body_center < lower_body_center - 20:
            cv2.putText(frame, "Bending Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif upper_body_center > lower_body_center + 20:
            cv2.putText(frame, "Bending Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Standing Center", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

  
        chest_y = int(avg_shoulder_y * height)
        cv2.line(frame, (0, chest_y), (width, chest_y), (0, 0, 255), 2)

        
        jump_ref_y = int(0.1 * height)
        crouch_ref_y = int(CROUCH_THRESHOLD * height)
        cv2.line(frame, (0, jump_ref_y), (width, jump_ref_y), (255, 0, 0), 2)
        cv2.line(frame, (0, crouch_ref_y), (width, crouch_ref_y), (0, 255, 0), 2)

    
        if avg_shoulder_y * height > CROUCH_THRESHOLD * height:
            cv2.putText(frame, "Crouching", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if prev_hip_y is not None and avg_hip_y < prev_hip_y - JUMP_THRESHOLD:
            cv2.putText(frame, "Jumping", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 20, 0), 2)

        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_orientation = detect_hand_orientation(hand_landmarks)
                cv2.putText(frame, hand_orientation, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        return frame

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    prev_hip_y = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            processed_frame = process_frame(frame, pose, hands, prev_hip_y)

            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            await websocket.send_bytes(frame_bytes)

            await asyncio.sleep(0.03) 
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        cap.release()

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
