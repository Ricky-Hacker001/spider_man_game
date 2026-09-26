import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from collections import deque

# ============================================================================
# 1. MEDIAPIPE SETUP
# ============================================================================

# Face Landmarker
try:
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    face_options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    face_detector = vision.FaceLandmarker.create_from_options(face_options)
except:
    print("⚠️ Face detector not available")
    face_detector = None

# Pose Landmarker
try:
    pose_base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
    pose_options = vision.PoseLandmarkerOptions(base_options=pose_base_options)
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
except:
    try:
        mp_pose = mp.solutions.pose
        pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except:
        pose_detector = None

# Hand Landmarker
try:
    hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    hand_options = vision.HandLandmarkerOptions(base_options=hand_base_options, num_hands=2)
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)
except:
    try:
        mp_hands = mp.solutions.hands
        hand_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    except:
        hand_detector = None

# ============================================================================
# 2. REALISTIC SPIDER-MAN SUIT
# ============================================================================

class RealisticSpiderManSuit:
    def __init__(self):
        self.suit_red = (0, 0, 200)  # Pure red
        self.suit_blue = (200, 100, 0)  # Blue accents
        self.suit_dark = (0, 0, 80)  # Dark shadows
        self.web_color = (150, 150, 150)  # Gray webs
        self.eye_white = (240, 240, 240)
        self.eye_black = (10, 10, 10)
        
        self.web_particles = deque(maxlen=1000)
        self.mask_img = cv2.imread('spiderman_mask.png', cv2.IMREAD_UNCHANGED)
    
    def draw_realistic_mask(self, frame, face_landmarks, w, h):
        if not face_landmarks:
            return

        # Use specific landmark indices
        left_eye = face_landmarks[133]
        right_eye = face_landmarks[362]
        
        # 1. Calculate center based on eyes
        center_x = int(((left_eye.x + right_eye.x) / 2) * w)
        center_y = int(((left_eye.y + right_eye.y) / 2) * h)
        
        # Increase the 3.5 to a higher number (e.g., 4.5 or 5.0) to make it wider
        face_w = int(abs(right_eye.x - left_eye.x) * w * 8.0) 
        
        # Keep the aspect ratio, but you can multiply by a higher number to make it taller
        face_h = int(face_w * 1)

        # 3. CRITICAL: Invert angle because of cv2.flip(frame, 1)
        dy = right_eye.y - left_eye.y
        dx = right_eye.x - left_eye.x
        angle = -math.degrees(math.atan2(dy, dx)) 

        # 4. Rotate the mask
        mask_resized = cv2.resize(self.mask_img, (face_w, face_h))
        M = cv2.getRotationMatrix2D((face_w // 2, face_h // 2), angle, 1.0)
        mask_rotated = cv2.warpAffine(mask_resized, M, (face_w, face_h), flags=cv2.INTER_LINEAR)

        # 5. Overlay with bounds checking
        x1, y1 = center_x - face_w // 2, center_y - face_h // 2 - int(face_h * 0.1)
        
        for i in range(face_h):
            for j in range(face_w):
                if 0 <= y1 + i < h and 0 <= x1 + j < w:
                    if mask_rotated[i, j, 3] > 0: # Check alpha
                        frame[y1 + i, x1 + j] = mask_rotated[i, j, :3]
    
    def draw_spiderman_eye(self, frame, pos, width, height):
        """Draw angular Spider-Man style eyes"""
        x, y = pos
        
        # Outer white part (angular teardrop shape)
        pts = np.array([
            [x - width // 2, y - height // 3],      # Top left
            [x + width // 2, y - height // 3],      # Top right
            [x + width // 2 + 5, y],                # Right point
            [x + width // 2, y + height // 2],      # Bottom right
            [x - width // 2, y + height // 2],      # Bottom left
            [x - width // 2 - 5, y],                # Left point
        ], np.int32)
        
        cv2.fillPoly(frame, [pts], self.eye_white)
        cv2.polylines(frame, [pts], True, self.suit_dark, 2)
        
        # Inner black pupil (smaller, offset)
        pupil_x = x + 3
        pupil_y = y + 2
        pupil_w = width // 4
        pupil_h = height // 3
        cv2.ellipse(frame, (pupil_x, pupil_y), (pupil_w, pupil_h), 
                   0, 0, 360, self.eye_black, -1)
        
        # Eye shine
        cv2.circle(frame, (pupil_x - 3, pupil_y - 2), 2, (255, 255, 255), -1)
    
    def draw_face_webbing(self, frame, nose_pos, center_x, center_y, face_height):
        """Draw intricate web patterns on the mask"""
        nx, ny = nose_pos
        
        # Vertical line down the center
        cv2.line(frame, nose_pos, (nx, ny + int(face_height * 0.4)), 
                self.web_color, 1)
        
        # Curved web lines from eyes to jaw
        for side_offset in [-1, 1]:
            # Create curved web lines
            points = []
            for i in range(20):
                progress = i / 19
                curve_x = int(center_x + side_offset * face_height * 0.3 * progress)
                curve_y = int(center_y - face_height * 0.2 + face_height * 0.5 * progress)
                points.append((curve_x, curve_y))
            
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], self.web_color, 1)
        
        # Diagonal web lines (creating diamond pattern)
        for angle in [30, -30, 150, 210]:
            rad = np.radians(angle)
            end_x = int(center_x + face_height * 0.4 * np.cos(rad))
            end_y = int(center_y + face_height * 0.4 * np.sin(rad))
            cv2.line(frame, (center_x, center_y), (end_x, end_y), 
                    self.web_color, 1)
    
    def draw_body_suit(self, frame, landmarks_list, w, h):
        """
        landmarks_list: The list of landmarks from results.pose_landmarks[0]
        """
        if not landmarks_list:
            return
        
        # Helper: Get pixel coordinates from a NormalizedLandmark
        def to_px(idx):
            lm = landmarks_list[idx]
            return (int(lm.x * w), int(lm.y * h))
        
        # Extract points using the correct indices for PoseLandmarker
        try:
            sl = to_px(11) # Left Shoulder
            sr = to_px(12) # Right Shoulder
            el = to_px(13) # Left Elbow
            er = to_px(14) # Right Elbow
            wl = to_px(15) # Left Wrist
            wr = to_px(16) # Right Wrist
            hl = to_px(23) # Left Hip
            hr = to_px(24) # Right Hip
            kl = to_px(25) # Left Knee
            kr = to_px(26) # Right Knee
            al = to_px(27) # Left Ankle
            ar = to_px(28) # Right Ankle
        except IndexError:
            return # Safety check

        # 1. Draw Torso
        torso_pts = np.array([sl, sr, hr, hl], np.int32)
        cv2.fillPoly(frame, [torso_pts], self.suit_red)
        
        # 2. Draw Limbs (Lines)
        # Arms
        cv2.line(frame, sl, el, self.suit_red, 20)
        cv2.line(frame, el, wl, self.suit_blue, 18)
        cv2.line(frame, sr, er, self.suit_red, 20)
        cv2.line(frame, er, wr, self.suit_blue, 18)
        
        # Legs
        cv2.line(frame, hl, kl, self.suit_red, 25)
        cv2.line(frame, kl, al, self.suit_blue, 22)
        cv2.line(frame, hr, kr, self.suit_red, 25)
        cv2.line(frame, kr, ar, self.suit_blue, 22)
    
    def draw_arm(self, frame, shoulder, elbow, wrist, w, h):
        """Draw arm with details"""
        # Upper arm
        cv2.line(frame, shoulder, elbow, self.suit_red, 28)
        cv2.circle(frame, elbow, 14, self.suit_red, -1)
        
        # Forearm
        cv2.line(frame, elbow, wrist, self.suit_blue, 24)
        cv2.circle(frame, wrist, 12, self.suit_blue, -1)
        
        # Muscle details
        mid = ((shoulder[0] + elbow[0]) // 2, (shoulder[1] + elbow[1]) // 2)
        cv2.circle(frame, mid, 10, self.suit_dark, 1)
    
    def draw_leg(self, frame, hip, knee, ankle, w, h):
        """Draw leg with details"""
        # Thigh
        cv2.line(frame, hip, knee, self.suit_red, 32)
        cv2.circle(frame, knee, 16, self.suit_red, -1)
        
        # Shin
        cv2.line(frame, knee, ankle, self.suit_blue, 28)
        cv2.circle(frame, ankle, 14, self.suit_blue, -1)
    
    def draw_torso_webbing(self, frame, sl, sr, hl, hr):
        """Draw web pattern on torso"""
        center_x = (sl[0] + sr[0] + hl[0] + hr[0]) // 4
        center_y = (sl[1] + sr[1] + hl[1] + hr[1]) // 4
        
        # Concentric circles
        max_radius = int(max(abs(sr[0] - sl[0]), abs(hr[1] - sr[1])) / 2)
        for radius in range(20, max_radius, 25):
            cv2.circle(frame, (center_x, center_y), radius, self.web_color, 1)
        
        # Radial lines
        for angle in range(0, 360, 30):
            rad = np.radians(angle)
            end_x = int(center_x + max_radius * np.cos(rad))
            end_y = int(center_y + max_radius * np.sin(rad))
            cv2.line(frame, (center_x, center_y), (end_x, end_y), 
                    self.web_color, 1)

# ============================================================================
# 3. WEB SHOOTER SYSTEM
# ============================================================================

class WebShooter:
    def __init__(self):
        self.particles = deque(maxlen=2000)
        self.max_speed = 18
    
    def create_web(self, start_pos, direction, power=1.0):
        """Create web particles"""
        num_particles = 50
        for i in range(num_particles):
            angle_offset = np.random.uniform(-0.4, 0.4)
            speed = self.max_speed * power * np.random.uniform(0.7, 1.3)
            
            vx = np.cos(direction + angle_offset) * speed
            vy = np.sin(direction + angle_offset) * speed
            
            particle = {
                'x': start_pos[0],
                'y': start_pos[1],
                'vx': vx,
                'vy': vy,
                'life': 120,
                'max_life': 120,
                'size': np.random.uniform(1, 3)
            }
            self.particles.append(particle)
    
    def update_particles(self):
        """Update particle physics"""
        to_remove = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.4  # Gravity
            p['vx'] *= 0.98  # Air resistance
            p['life'] -= 1
            
            if p['life'] <= 0:
                to_remove.append(p)
        
        for p in to_remove:
            self.particles.remove(p)
    
    def draw_particles(self, frame):
        """Draw web particles"""
        h, w, _ = frame.shape
        
        for p in self.particles:
            if 0 <= p['x'] < w and 0 <= p['y'] < h:
                alpha = p['life'] / p['max_life']
                intensity = int(200 * alpha)
                color = (intensity, intensity, intensity)
                
                size = int(p['size'] * alpha)
                if size > 0:
                    cv2.circle(frame, (int(p['x']), int(p['y'])), size, color, -1)

# ============================================================================
# 4. DETECTION FUNCTIONS
# ============================================================================

def detect_pose(frame, pose_detector):
    """Detect pose"""
    h, w, _ = frame.shape
    
    if hasattr(pose_detector, 'detect'):
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = pose_detector.detect(mp_image)
            return results.landmarks[0] if results.landmarks else None
        except:
            pass
    
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(frame_rgb)
        return results.pose_landmarks if results.pose_landmarks else None
    except:
        pass
    
    return None

def detect_hands(frame, hand_detector):
    """Detect hands"""
    if hand_detector is None:
        return [], []
    
    if hasattr(hand_detector, 'detect'):
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = hand_detector.detect(mp_image)
            return results.landmarks if results.landmarks else [], results.handedness if results.handedness else []
        except:
            pass
    
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(frame_rgb)
        return results.multi_hand_landmarks, results.multi_handedness
    except:
        pass
    
    return [], []

def detect_web_shoot(hand_landmarks):
    """Detect web shooting gesture"""
    if not hand_landmarks:
        return False, None
    
    wrist = hand_landmarks[0]
    middle = hand_landmarks[12]
    
    distance = math.sqrt((middle.x - wrist.x)**2 + (middle.y - wrist.y)**2)
    if distance > 0.15:
        direction = math.atan2(middle.y - wrist.y, middle.x - wrist.x)
        return True, (int(wrist.x * 1280), int(wrist.y * 720), direction)
    
    return False, None

# ============================================================================
# 5. MAIN APPLICATION
# ============================================================================

cap = cv2.VideoCapture(0)
cv2.namedWindow('🕷️ Spider-Man Suit AR', cv2.WINDOW_NORMAL)
cv2.resizeWindow('🕷️ Spider-Man Suit AR', 1280, 720)

suit = RealisticSpiderManSuit()
web_shooter = WebShooter()

frame_count = 0
fps_timer = cv2.getTickCount()
fps = 0

print("🕷️  Starting Spider-Man Suit AR...")
print("✨ Point fingers to shoot webs!")
print("Press Q to exit\n")

# 1. ADD THIS BEFORE YOUR WHILE LOOP
# Load the mask with alpha channel (if it has transparency)
mask_img = cv2.imread('spiderman_mask.png', cv2.IMREAD_UNCHANGED)

# 2. REPLACE YOUR DRAW_REALISTIC_MASK METHOD WITH THIS:
def draw_realistic_mask(self, frame, face_landmarks, w, h):
    global mask_img
    if not face_landmarks:
        return
    
    # Get landmarks for eyes/face bounding box
    forehead = face_landmarks[10]
    chin = face_landmarks[152]
    left_cheek = face_landmarks[234]
    right_cheek = face_landmarks[454]
    
    # Calculate box dimensions
    x1 = int(left_cheek.x * w)
    x2 = int(right_cheek.x * w)
    y1 = int(forehead.y * h)
    y2 = int(chin.y * h)
    
    # Calculate width and height of the face
    width = int(abs(x2 - x1) * 1.2) # Adding a bit of padding
    height = int((y2 - y1) * 1.3)
    
    # Resize the png mask to fit the face
    resized_mask = cv2.resize(mask_img, (width, height))
    
    # Calculate position (center the mask)
    x_offset = x1 - int(width * 0.1)
    y_offset = y1 - int(height * 0.2)
    
    # Overlay logic (Handling alpha channel for transparency)
    for i in range(height):
        for j in range(width):
            if y_offset + i < h and x_offset + j < w:
                # Get pixel from mask
                mask_pixel = resized_mask[i, j]
                # If mask has alpha (transparency)
                if mask_pixel[3] > 0:
                    frame[y_offset + i, x_offset + j] = mask_pixel[:3]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    frame_count += 1
    
    # ========== POSE DETECTION ==========
    try:
        landmarks = detect_pose(frame, pose_detector)
        if landmarks:
            suit.draw_body_suit(frame, landmarks, w, h)
    except Exception as e:
        pass
    
    # ========== FACE DETECTION ==========
    try:
        if face_detector is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face_results = face_detector.detect(mp_image)
            if face_results.face_landmarks:
                suit.draw_realistic_mask(frame, face_results.face_landmarks[0], w, h)
    except Exception as e:
        pass
    
    # ========== HAND & WEB DETECTION ==========
    try:
        hand_landmarks_list, _ = detect_hands(frame, hand_detector)
        
        for hand_landmarks in hand_landmarks_list:
            is_shooting, gesture_data = detect_web_shoot(hand_landmarks)
            
            if is_shooting and gesture_data:
                wrist_pos = (gesture_data[0], gesture_data[1])
                direction = gesture_data[2]
                
                # Create web
                web_shooter.create_web(wrist_pos, direction, power=1.5)
                
                # Muzzle flash
                for i in range(4):
                    radius = 8 + i * 4
                    alpha = 200 - (i * 50)
                    cv2.circle(frame, wrist_pos, radius, (100, 180, 255), 2)
    except Exception as e:
        pass
    
    # ========== UPDATE PARTICLES ==========
    web_shooter.update_particles()
    web_shooter.draw_particles(frame)
    
    # ========== FPS & INFO ==========
    if frame_count % 30 == 0:
        fps = 30 / ((cv2.getTickCount() - fps_timer) / cv2.getTickFrequency())
        fps_timer = cv2.getTickCount()
    
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, '🕷️  Point fingers to SHOOT WEB!', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, 'Press Q to exit', (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 2)
    
    cv2.imshow('🕷️ Spider-Man Suit AR', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n👋 Exiting Spider-Man mode...")
        break

cap.release()
cv2.destroyAllWindows()
print("Done! 🕷️")