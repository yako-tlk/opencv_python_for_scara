'''
FINGER TIP DETECTION WITH MEDIAPIPE
'''
# Library imports
import cv2
import mediapipe as mp
import math as m
import os

def setup_capture(source = 1, res = [854, 480]) -> any:
    '''
    CAMERA DEVICE SETUP
    '''
    capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    return capture

def draw_fingers(center, index_tip, thumb_tip, frame, workspace) -> None:
    '''
    DRAWING FUNCITON
    '''
    cx, cy = center[0], center[1]

    cv2.line(frame, (index_tip[0], index_tip[1]), (thumb_tip[0], thumb_tip[1]), (0,0,255), 2)
    cv2.circle(frame, (index_tip[0], index_tip[1]), 5, (0,0,255), cv2.FILLED)
    cv2.circle(frame, (thumb_tip[0], thumb_tip[1]), 5, (0,0,255), cv2.FILLED)
    cv2.circle(frame, (cx, cy), 5, (255,0,0), cv2.FILLED)

    new_center = transform_area(center, workspace)

    cv2.putText(
        frame,
        text=f'cx: {new_center[0]}cm, cy: {new_center[1]}cm',
        org=(25+70, 25+50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(0, 0, 0),
        thickness=2
        )
    
    if (get_distance(index_tip,thumb_tip) <= 50):
        state = True
        state_color = (255,0,0)
    else:
        state = False
        state_color = (0,0,0)
    
    cv2.putText(
        frame,
        text=f'Trigger: {state}',
        org=(25+70, 55+50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=state_color,
        thickness=2
        )

    pass

def get_distance(p1, p2) -> int:
    '''
    GET THE DISTANCE BETWEEN THE TWO TIPS
    '''
    distance = m.hypot(p2[0]-p1[0], p2[1]-p1[1])
    return int(distance)

def transform_area(center, workspace = [45, 25]) -> list:
    '''
    TRANSFORM DE CENTER COORDINATES BETWEEN SYSTEMS
    '''
    # Center -> (854, 480)
    # Mode 1: Safe zone -> (683, 384)px
    # Mode 2: Workspace -> (45, 25)cm
    x = workspace[0]
    y = workspace[1]
    new_center = [int(center[0]*(x/683)-5),int(center[1]*(y/384)-3)]
    if (new_center[0] >= x):
        new_center[0] = x
    elif(new_center[0] <= 0):
        new_center[0] = 0

    if (new_center[1] >= y):
        new_center[1] = y
    elif (new_center[1] <= 0):
        new_center[1] = 0

    return new_center

def info2send_bt(distance, center, frame, workspace) -> None:
    '''
    TRANSFORM AND SEND BLUETOOTH INFO
    '''
    if (distance <= 50):
        scara_trigger = 1
    else:
        scara_trigger = 0
    old_center = center
    center = transform_area(center, workspace)
    info2send = [center[0], center[1], scara_trigger]

    #safe_zone_resolution = [int(frame_resolution[0]*0.9)-int(frame_resolution[0]*0.1), int(frame_resolution[1]*0.9)-int(frame_resolution[1]*0.1)]
        #print(safe_zone_resolution)
                # Drawing the safe zone
    # Safe zone resolution -> (683, 384)
    start_point = (int(854*0.1), int(480*0.1))
    end_point = (int(854*0.9), int(480*0.9))

    # Change color for the safe zone
    if (old_center[0]>end_point[0] or old_center[0]<start_point[0] or old_center[1]>end_point[1] or old_center[1]<start_point[1]):
        zone_color = (0,0,255)
    else:
        zone_color = (0,255,0)
    
    # Draw the safe zone
    cv2.rectangle(frame, start_point, end_point, zone_color, 2)

    print(f'BT: {info2send}')

    '''
    Write down here the bluetooth code =)...
    '''
    pass

def analysis_results(results, frame, frame_resolution, drawing = True, workspace = [45, 25]) -> None:
    '''
    ANALYSIS & PROCESSING OF THE RESULTS & FRAME
    '''
    if results.multi_hand_landmarks:
        my_hand = results.multi_hand_landmarks[0]
        for id, lm in enumerate(my_hand.landmark):
            # Calculate de x & y coordinates (pixels) for every landmark
            x = int(lm.x*frame_resolution[0])
            y = int(lm.y*frame_resolution[1])

            #cv2.circle(frame, (x,y), 2, (0,0,255), cv2.FILLED)
            #landmark = [id, x, y]
            #print(landmark)
            
            if id == 4:
                index_tip = [x, y]
            if id == 8:
                thumb_tip = [x, y]
            
            try:
                cx = (index_tip[0] + thumb_tip[0])//2
                cy = (index_tip[1] + thumb_tip[1])//2
                centroid = [cx, cy]
                # centroid = transform_area(centroid, '1')
                draw_fingers(centroid, index_tip, thumb_tip, frame, workspace)
                distance_b2f = get_distance(index_tip, thumb_tip)

                info2send_bt(distance_b2f, centroid, frame, workspace)

            except:
                continue
    pass

def main() -> None:
    '''
    MAIN FUNCTION CODE HERE
    '''
    frame_resolution = [854,480]
    capture = setup_capture(0, frame_resolution)

    # Mediapipe config
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        model_complexity = 1,
        max_num_hands = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.7
    )

    Running = True
    while Running:
        # Capture the frame
        ret, frame = capture.read()


        # Flip the frame
        frame = cv2.flip(frame, 1)

        # Hands results
        recog_results = hands.process(frame)

        # Scara wokspace in cm
        workspace = [45, 25]

        # Get the landmarks, centroid & cv drawings
        # Get index_tip, thumb_tip, and centroid coordinates -> list...
        # ...as global variables =)
        analysis_results(recog_results, frame, frame_resolution, True, workspace)
        
        # Show the frame
        cv2.imshow('Frame', frame)


        # Define the quit key
        if cv2.waitKey(1) & 0xFF == 27:
            Running = False
    
    # Release & destroy
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()