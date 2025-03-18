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
import threading
import serial
import queue

left_shoulder_positions = []
right_shoulder_positions = []
DISPLACEMENT_THRESHOLD = 15 
SMOOTHING_FACTOR = 0.2 
app = FastAPI()
templates = Jinja2Templates(directory="templates")
frame_queue = queue.Queue(maxsize=5)  


# def detect_circular_motion(positions, threshold=5):
#     if len(positions) < threshold:
#         return False
    
#     center_x = sum(p[0] for p in positions) / len(positions)
#     center_y = sum(p[1] for p in positions) / len(positions)
    
#     radii = [((p[0] - center_x) ** 2 + (p[1] - center_y) ** 2) ** 0.5 for p in positions]
#     avg_radius = sum(radii) / len(radii)
    
#     return all(abs(r - avg_radius) < avg_radius * 0.3 for r in radii)  # Increased tolerance
def detect_circular_motion(positions, threshold=5):
    if len(positions) < threshold:
        return False
    
    x_disp = abs(positions[-1][0] - positions[0][0])
    y_disp = abs(positions[-1][1] - positions[0][1])
    
    return x_disp > DISPLACEMENT_THRESHOLD and y_disp > DISPLACEMENT_THRESHOLD


def capture_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            frame_queue.get()  
        frame_queue.put(frame)

try:
    ser = serial.Serial('COM5', 9600, timeout=1)  
    time.sleep(2)  # Allow serial connection to establish
except serial.SerialException as e:
    print(f"Serial Error: {e}")
    ser = None  

def send_to_serial(message):
    
    if ser:
        ser.write((message + "").encode())  
        print(f"Sent to Br@y++: {message}")  

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
jump_threshold=0.05

def count_fingers(hand_landmarks, handedness):
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    open_fingers = 0

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    
    if (handedness == "Right" and thumb_tip.x < thumb_ip.x) or (handedness == "Left" and thumb_tip.x > thumb_ip.x):
        open_fingers += 1
        

    

    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            open_fingers += 1
    
    return open_fingers

def draw_threshold_lines(frame, height, width):
    cv2.line(frame, (0, int(CROUCH_THRESHOLD * height)), (width, int(CROUCH_THRESHOLD * height)), (0, 0, 255), 2)
    cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 0), 2)
def smooth_position(new_pos, prev_pos):
    return prev_pos * (1 - SMOOTHING_FACTOR) + new_pos * SMOOTHING_FACTOR if prev_pos else new_pos

def process_frame(frame, pose, hands, prev_hip_y):
 
   
    height, width, _ = frame.shape
    global left_shoulder_positions, right_shoulder_positions
    jump_ref_y = int(0.1 * height)
    global last_crouch_time, crouch_start_time, crouch_started
    global jump_count, crouch_count, t_pose_count, bend_left_count, bend_right_count
    height, width, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    hand_results = hands.process(rgb_frame)
    draw_threshold_lines(frame, height, width)
    cv2.line(frame, (0, jump_ref_y), (width, jump_ref_y), (255, 0, 0), 2)

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = pose_results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        smoothed_left_shoulder = (smooth_position(left_shoulder.x * width, left_shoulder_positions[-1][0] if left_shoulder_positions else None),
                                  smooth_position(left_shoulder.y * height, left_shoulder_positions[-1][1] if left_shoulder_positions else None))
        smoothed_right_shoulder = (smooth_position(right_shoulder.x * width, right_shoulder_positions[-1][0] if right_shoulder_positions else None),
                                   smooth_position(right_shoulder.y * height, right_shoulder_positions[-1][1] if right_shoulder_positions else None))
        

        left_shoulder_positions.append((left_shoulder.x * width, left_shoulder.y * height))
        right_shoulder_positions.append((right_shoulder.x * width, right_shoulder.y * height))
        
        if len(left_shoulder_positions) > 20:
            left_shoulder_positions.pop(0)
            print(f"Right Shoulder Positions: {right_shoulder_positions}")
        if len(right_shoulder_positions) > 20:
            right_shoulder_positions.pop(0)
            print(f"Left Shoulder Positions: {left_shoulder_positions}")
        
        

        if detect_circular_motion(left_shoulder_positions, threshold=3):
            print("Left Shoulder Circular Motion Detected")
            cv2.putText(frame, "Left Shoulder Circular Motion", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if detect_circular_motion(right_shoulder_positions, threshold=3):
            print("Right Shoulder Circular Motion Detected")
            cv2.putText(frame, "Right Shoulder Circular Motion", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
        
        cv2.putText(frame, f"Left Shoulder: X={left_shoulder.x:.2f}, Y={left_shoulder.y:.2f}",
            (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, f"Right Shoulder: X={right_shoulder.x:.2f}, Y={right_shoulder.y:.2f}",
            (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


        nose_x = int(landmarks[mp_pose.PoseLandmark.NOSE].x * width)
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

        upper_body_center = (nose_x + int(left_shoulder.x * width) + int(right_shoulder.x * width)) // 3
        lower_body_center = (int(left_hip.x * width) + int(right_hip.x * width)) // 2

        cv2.line(frame, (upper_body_center, 0), (lower_body_center, height), (0, 255, 255), 3)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks, classification in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            handedness = classification.classification[0].label  # "Right" or "Left"
            fingers_open = count_fingers(hand_landmarks, handedness)
            
            text_position = (50, 300) if handedness == "Right" else (50, 350)
            send_to_serial("H")
            cv2.putText(frame, f"{handedness} Hand: {fingers_open} fingers", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 10, 0), 2)

    if upper_body_center < lower_body_center - 20:
        send_to_serial("L")
        cv2.putText(frame, "Bending Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif upper_body_center > lower_body_center + 20:
        send_to_serial("R")
        cv2.putText(frame, "Bending Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        send_to_serial("C")
        cv2.putText(frame, "Standing Center", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        chest_y = int(avg_shoulder_y * height)
        cv2.line(frame, (0, chest_y), (width, chest_y), (0, 0, 255), 2)

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
        send_to_serial("s\n")

    cv2.putText(frame, f"Squats: {crouch_count}", (width - 250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        

        # Jump 
    if prev_hip_y is not None and avg_hip_y < prev_hip_y - jump_threshold:
        jump_count += 1
        cv2.putText(frame, f"Jumping: {jump_count}", (width - 250, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 20, 0), 2)

        # T-pose 
    arm_span = abs(left_wrist.x - right_wrist.x)  
    shoulder_span = abs(left_shoulder.x - right_shoulder.x)

    if arm_span > shoulder_span * 2:  
        cv2.putText(frame, "T-Pose ", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        send_to_serial("T\n")
        t_pose_count += 1
        cv2.putText(frame, f"T-Pose: {t_pose_count}", (width - 250, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    return frame

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global crouch_count  # Reset squat counter
    crouch_count = 0  
    
    await websocket.accept()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    prev_hip_y = None
    capture_thread = threading.Thread(target=capture_frames, args=(cap,), daemon=True)
    capture_thread.start()

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
                "crouch_count": crouch_count,  # Updated count
                "tpose_count": t_pose_count,
                "bend_left_count": bend_left_count,
                "bend_right_count": bend_right_count
            })

            await asyncio.sleep(0.04)
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        cap.release()

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
