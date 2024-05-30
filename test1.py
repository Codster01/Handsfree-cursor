import cv2
import time
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
pTime = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
pyautogui.FAILSAFE = False
while True:
    success, img = cap.read()
    if not success:
        break

    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime else 0
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    results = hands.process(imgRgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Check if it's a left hand or a right hand
            is_left_hand = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x

            # Focus on the detected hand
            hand = hand_landmarks

            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            # Check for a "click" gesture
            thumb_tip = hand.landmark[4]
            index_tip = hand.landmark[8]
            pinky_dip = hand.landmark[20]

            # Adjust the threshold based on your experimentation
            threshold = 0.013
            screen_width, screen_height = pyautogui.size()

            if is_left_hand:
                # Left hand controls mouse movement
                scaled_x = int((1 - pinky_dip.x) * screen_width)
                scaled_y = (pinky_dip.y * screen_height)
                pyautogui.moveTo(scaled_x, scaled_y, duration=0)
            else:
                # Right hand is responsible for clicking
                if abs(thumb_tip.x - index_tip.x) < threshold:
                    pyautogui.click()
                    print("Clicked!")
                    print(abs(thumb_tip.x - index_tip.x))
                else:
                    print(abs(thumb_tip.x - index_tip.x))
                    print("Not Clicked")

    cv2.imshow("Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
