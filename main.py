from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import cv2
import mediapipe as mp
import asyncio
import numpy as np
import base64
import time

app = FastAPI()

templates = Jinja2Templates(directory="templates")

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

JUMP_THRESHOLD = 0.08
CROUCH_THRESHOLD = 0.8
T_POSE_THRESHOLD = 0.2
BEND_THRESHOLD = 0.2
CHEST_THRESHOLD = 0.5
PUSH_UP_THRESHOLD = 0.1

jump_count = 0
crouch_count = 0
t_pose_count = 0
bend_left_count = 0
bend_right_count = 0
hands_joined_count = 0

crouch_hold_time = 0.4
crouch_delay = 1.0
crouch_start_time = 0 
crouch_started = False
last_crouch_time = 0

def detect_hand_orientation(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

def draw_threshold_lines(frame, height, width):
    cv2.line(frame, (0, int(CROUCH_THRESHOLD * height)), (width, int(CROUCH_THRESHOLD * height)), (0, 0, 255), 2)
    cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 0), 2)

def process_frame(frame, pose, hands, prev_hip_y):
    global last_crouch_time, crouch_start_time, crouch_started
    global jump_count, crouch_count, t_pose_count, bend_left_count, bend_right_count

    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    hand_results = hands.process(rgb_frame)
    draw_threshold_lines(frame, height, width)

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = pose_results.pose_landmarks.landmark

        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        avg_hip_y = (left_hip.y + right_hip.y) / 2

        chest_y = int(avg_shoulder_y * height)
        crouch_ref_y = int(CROUCH_THRESHOLD * height)
        cv2.line(frame, (0, chest_y), (width, chest_y), (0, 0, 255), 2)

        upper_body_center = (left_shoulder.x + right_shoulder.x) / 2
        lower_body_center = (left_hip.x + right_hip.x) / 2

        # if upper_body_center < lower_body_center - BEND_THRESHOLD:
        #     bend_left_count += 1
        #     cv2.putText(frame, f"Bending Left: {bend_left_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # elif upper_body_center > lower_body_center + BEND_THRESHOLD:
        #     bend_right_count += 1
        #     cv2.putText(frame, f"Bending Right: {bend_right_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # else:
        #     cv2.putText(frame, "Standing Center", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Squat detection based on chest position crossing threshold
        if chest_y >= crouch_ref_y:
            if not crouch_started:
                crouch_started = True
                crouch_start_time = time.time()
        elif crouch_started and chest_y < crouch_ref_y:
            if time.time() - crouch_start_time >= crouch_hold_time:
                if time.time() - last_crouch_time > crouch_delay:
                    crouch_count += 1
                    last_crouch_time = time.time()
            crouch_started = False

        cv2.putText(frame, f"Squats: {crouch_count}", (width - 250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Jump detection
        if prev_hip_y is not None and avg_hip_y < prev_hip_y - JUMP_THRESHOLD:
            jump_count += 1
            cv2.putText(frame, f"Jumping: {jump_count}", (width - 250, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 20, 0), 2)

        # T-pose detection
        if abs(left_wrist.y - left_shoulder.y) < T_POSE_THRESHOLD and abs(right_wrist.y - right_shoulder.y) < T_POSE_THRESHOLD:
            t_pose_count += 1
            cv2.putText(frame, f"T-Pose: {t_pose_count}", (width - 250, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

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
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            await websocket.send_json({
                "image": base64_image,
                "jump_count": jump_count,
                "crouch_count": crouch_count,
                "tpose_count": t_pose_count,
                "bend_left_count": bend_left_count,
                "bend_right_count": bend_right_count
            })

            await asyncio.sleep(0.05)
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        cap.release()

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
