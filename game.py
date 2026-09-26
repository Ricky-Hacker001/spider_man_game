import cv2
import pygame
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import math
import random
import time
import numpy as np
import os
import urllib.request

# --- CONFIGURATION CONSTANTS ---
FPS = 60
CAM_WIDTH = 320
CAM_HEIGHT = 240

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 30, 30)
BLUE = (30, 80, 200)
SPIDEY_BLUE = (20, 50, 150)
YELLOW = (255, 220, 0)
GRAY = (50, 50, 50)
LIGHT_BLUE = (100, 200, 255)

# --- COMPUTER VISION THREADED CAPTURE ---
class WebcamStream:
    """Handles webcam frame capture and MediaPipe Tasks hand tracking in a background thread."""
    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        if not self.stream.isOpened():
            self.running = False
            self.error = True
            return
        
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        
        self.running = True
        self.error = False
        self.frame = None
        self.hand_data = {"Left": None, "Right": None}
        
        # Auto-download the required Google tracking asset if missing
        self.model_path = "hand_landmarker.task"
        if not os.path.exists(self.model_path):
            print("Downloading MediaPipe Hand Landmarker model asset...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete!")

        # Configure the modern MediaPipe Tasks Detector
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        if not self.error:
            self.thread.start()
        return self

    def update(self):
        while self.running:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.running = False
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            results = self.detector.detect_for_video(mp_image, timestamp_ms)
            
            current_hand_data = {"Left": None, "Right": None}
            
            if results.hand_landmarks and results.handedness:
                for landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                    label = handedness[0].category_name # "Left" or "Right"
                    lm_list = [(lm.x, lm.y, lm.z) for lm in landmarks]
                    current_hand_data[label] = lm_list
            
            with self.lock:
                self.frame = rgb_frame 
                self.hand_data = current_hand_data
                
        self.stream.release()

    def get_data(self):
        with self.lock:
            return self.frame, self.hand_data

    def stop(self):
        self.running = False


# --- GESTURE PROCESSING HELPER ---
class GestureRecognizer:
    @staticmethod
    def get_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    @classmethod
    def recognize(cls, lm):
        if lm is None:
            return "None"
        
        wrist = lm[0]
        
        index_extended = cls.get_distance(lm[8], wrist) > cls.get_distance(lm[6], wrist)
        middle_extended = cls.get_distance(lm[12], wrist) > cls.get_distance(lm[10], wrist)
        ring_extended = cls.get_distance(lm[16], wrist) > cls.get_distance(lm[14], wrist)
        pinky_extended = cls.get_distance(lm[20], wrist) > cls.get_distance(lm[18], wrist)
        thumb_extended = cls.get_distance(lm[4], lm[5]) > cls.get_distance(lm[3], lm[5])

        if thumb_extended and index_extended and pinky_extended and not middle_extended and not ring_extended:
            return "Spidey"
        if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "Fist"
        if index_extended and middle_extended and ring_extended and pinky_extended:
            return "Palm"
            
        return "Unknown"


# --- GAME ENTITIES ---
class Player:
    def __init__(self, start_w, start_h):
        self.x = start_w // 2
        self.y = start_h - 200
        self.vx = 0
        self.vy = 0
        self.radius = 30
        self.hp = 100
        self.score = 0
        self.gravity = 0.5
        self.is_swinging = False
        self.web_anchor = None
        
    def update(self, current_w, current_h):
        if not self.is_swinging:
            self.vy += self.gravity
        else:
            dx = self.web_anchor[0] - self.x
            dy = self.web_anchor[1] - self.y
            dist = math.hypot(dx, dy)
            if dist > 10:
                self.vx += (dx / dist) * 0.8
                self.vy += (dy / dist) * 0.8
        
        self.vx *= 0.95
        self.vy *= 0.95
        self.x += self.vx
        self.y += self.vy
        
        # Keep player confined within boundaries even when window resizes
        if self.x < self.radius:
            self.x = self.radius
            self.vx = 0
        if self.x > current_w - self.radius:
            self.x = current_w - self.radius
            self.vx = 0
        if self.y < self.radius:
            self.y = self.radius
            self.vy = 0
        if self.y > current_h - 100: 
            self.y = current_h - 100
            self.vy = 0

    def draw(self, surface):
        if self.is_swinging and self.web_anchor:
            pygame.draw.line(surface, WHITE, (int(self.x), int(self.y)), self.web_anchor, 3)
            pygame.draw.circle(surface, LIGHT_BLUE, self.web_anchor, 6)

        pygame.draw.circle(surface, RED, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(surface, SPIDEY_BLUE, (int(self.x), int(self.y)), self.radius - 6)
        pygame.draw.polygon(surface, WHITE, [(self.x-12, self.y-5), (self.x-3, self.y-8), (self.x-6, self.y+4)])
        pygame.draw.polygon(surface, WHITE, [(self.x+12, self.y-5), (self.x+3, self.y-8), (self.x+6, self.y+4)])


class WebProjectile:
    def __init__(self, start_x, start_y, target_x, target_y):
        self.x = start_x
        self.y = start_y
        self.radius = 8
        self.speed = 20
        
        dx = target_x - start_x
        dy = target_y - start_y
        dist = math.hypot(dx, dy)
        
        if dist == 0:
            self.vx = self.speed
            self.vy = 0
        else:
            self.vx = (dx / dist) * self.speed
            self.vy = (dy / dist) * self.speed
            
    def update(self):
        self.x += self.vx
        self.y += self.vy
        
    def draw(self, surface):
        pygame.draw.circle(surface, WHITE, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(surface, LIGHT_BLUE, (int(self.x), int(self.y)), self.radius - 2, 1)


class Enemy:
    def __init__(self, current_w, current_h):
        self.x = random.choice([-50, current_w + 50])
        self.y = random.randint(100, current_h - 300)
        self.radius = 25
        self.speed = random.uniform(3, 6)
        
    def update(self, player_x, player_y):
        dx = player_x - self.x
        dy = player_y - self.y
        dist = math.hypot(dx, dy)
        if dist > 0:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed

    def draw(self, surface):
        pygame.draw.circle(surface, GRAY, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(surface, YELLOW, (int(self.x), int(self.y)), self.radius - 8)
        pygame.draw.rect(surface, RED, (int(self.x)-10, int(self.y)-3, 20, 6))


# --- MAIN ENGINE ---
class GameEngine:
    def __init__(self):
        pygame.init()
        
        # Fetch display info to calculate initially maximized sizes
        info = pygame.display.Info()
        self.width = info.current_w
        self.height = info.current_h
        
        # Initialize a window that features standard borders, dragging, and OS native window actions
        # Using SDL_VIDEO_WINDOW_POS to start it near the top left corner before filling the frame
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE | pygame.DOUBLEBUF | pygame.HWSURFACE)
        pygame.display.set_caption("Spider-Man POV: Gesture Web-Slinger")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 28, bold=True)
        self.large_font = pygame.font.SysFont("Arial", 64, bold=True)
        
        self.webcam = WebcamStream().start()
        self.player = Player(self.width, self.height)
        self.projectiles = []
        self.enemies = []
        
        self.enemy_spawn_timer = 0
        self.last_shot_time = 0
        self.last_punch_time = 0
        
        # Setup structural anchor configurations
        self.recalculate_anchors()

    def recalculate_anchors(self):
        """Dynamically adjusts sky anchors relative to the current width and height variables."""
        self.anchors = [
            (int(self.width * 0.2), int(self.height * 0.12)),
            (int(self.width * 0.4), int(self.height * 0.09)),
            (int(self.width * 0.6), int(self.height * 0.15)),
            (int(self.width * 0.8), int(self.height * 0.11))
        ]

    def process_gestures(self, hand_data):
        if hand_data is None:
            return
        
        # --- LEFT HAND: MOVEMENT ---
        left_lm = hand_data["Left"]
        if left_lm:
            left_gesture = GestureRecognizer.recognize(left_lm)
            lx = int(left_lm[9][0] * self.width)
            ly = int(left_lm[9][1] * self.height)
            
            if left_gesture == "Spidey":
                if not self.player.is_swinging:
                    closest_anchor = min(self.anchors, key=lambda a: math.hypot(a[0] - lx, a[1] - ly))
                    self.player.web_anchor = closest_anchor
                    self.player.is_swinging = True
            else:
                self.player.is_swinging = False
        else:
            self.player.is_swinging = False

        # --- RIGHT HAND: COMBAT ---
        right_lm = hand_data["Right"]
        if right_lm:
            right_gesture = GestureRecognizer.recognize(right_lm)
            rx = int(right_lm[9][0] * self.width)
            ry = int(right_lm[9][1] * self.height)
            
            current_time = time.time()
            
            if right_gesture == "Palm":
                if current_time - self.last_shot_time > 0.25:
                    self.projectiles.append(WebProjectile(self.player.x, self.player.y, rx, ry))
                    self.last_shot_time = current_time
            elif right_gesture == "Fist":
                if current_time - self.last_punch_time > 0.35:
                    self.last_punch_time = current_time
                    for enemy in list(self.enemies):
                        if math.hypot(enemy.x - self.player.x, enemy.y - self.player.y) < (self.player.radius + enemy.radius + 60):
                            if enemy in self.enemies:
                                self.enemies.remove(enemy)
                                self.player.score += 15

    def update_game_state(self):
        self.player.update(self.width, self.height)
        
        for proj in list(self.projectiles):
            proj.update()
            if proj.x < 0 or proj.x > self.width or proj.y < 0 or proj.y > self.height:
                self.projectiles.remove(proj)

        self.enemy_spawn_timer += 1
        if self.enemy_spawn_timer > 60:
            self.enemies.append(Enemy(self.width, self.height))
            self.enemy_spawn_timer = 0
            
        for enemy in list(self.enemies):
            enemy.update(self.player.x, self.player.y)
            
            for proj in list(self.projectiles):
                if math.hypot(enemy.x - proj.x, enemy.y - proj.y) < (enemy.radius + proj.radius):
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    if enemy in self.enemies: self.enemies.remove(enemy)
                    self.player.score += 10
                    break
            
            if math.hypot(enemy.x - self.player.x, enemy.y - self.player.y) < (enemy.radius + self.player.radius):
                self.player.hp -= 10
                if enemy in self.enemies:
                    self.enemies.remove(enemy)

    def draw_hud(self, cam_frame, hand_data):
        for anchor in self.anchors:
            pygame.draw.rect(self.screen, GRAY, (anchor[0]-30, 0, 60, anchor[1]))
            pygame.draw.circle(self.screen, YELLOW, anchor, 10)

        # UI bars
        pygame.draw.rect(self.screen, GRAY, (30, 30, 250, 30))
        hp_width = max(0, int(250 * (self.player.hp / 100)))
        pygame.draw.rect(self.screen, RED, (30, 30, hp_width, 30))
        hp_text = self.font.render(f"HP: {self.player.hp}", True, WHITE)
        self.screen.blit(hp_text, (40, 31))
        
        score_text = self.font.render(f"SCORE: {self.player.score}", True, WHITE)
        self.screen.blit(score_text, (310, 31))
        
        for hand_type, color in [("Left", BLUE), ("Right", RED)]:
            if hand_data and hand_data[hand_type]:
                lm = hand_data[hand_type]
                hx = int(lm[9][0] * self.width)
                hy = int(lm[9][1] * self.height)
                pygame.draw.circle(self.screen, color, (hx, hy), 16, 3)
                pygame.draw.circle(self.screen, color, (hx, hy), 3)

        if cam_frame is not None:
            frame_surface = pygame.surfarray.make_surface(np.rot90(cam_frame))
            frame_surface = pygame.transform.scale(frame_surface, (CAM_WIDTH, CAM_HEIGHT))
            preview_x = self.width - CAM_WIDTH - 30
            preview_y = 30
            self.screen.blit(frame_surface, (preview_x, preview_y))
            pygame.draw.rect(self.screen, WHITE, (preview_x, preview_y, CAM_WIDTH, CAM_HEIGHT), 2)
            
            gesture_l = GestureRecognizer.recognize(hand_data["Left"]) if hand_data else "None"
            gesture_r = GestureRecognizer.recognize(hand_data["Right"]) if hand_data else "None"
            lbl_l = self.font.render(f"L Hand: {gesture_l}", True, LIGHT_BLUE)
            lbl_r = self.font.render(f"R Hand: {gesture_r}", True, RED)
            self.screen.blit(lbl_l, (preview_x, preview_y + CAM_HEIGHT + 10))
            self.screen.blit(lbl_r, (preview_x, preview_y + CAM_HEIGHT + 40))

    def run(self):
        running = True
        
        if self.webcam.error:
            while running:
                self.screen.fill(BLACK)
                err_text = self.large_font.render("WEBCAM INITIALIZATION FAILED!", True, RED)
                self.screen.blit(err_text, (self.width // 2 - err_text.get_width() // 2, self.height // 2 - 40))
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False
                pygame.display.flip()
                self.clock.tick(15)
            pygame.quit()
            return

        while running and self.player.hp > 0:
            self.screen.fill(BLACK)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                # Dynamically catch OS window resize adjustments, maximize changes, or dragging events
                elif event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE | pygame.DOUBLEBUF | pygame.HWSURFACE)
                    self.recalculate_anchors()
            
            cam_frame, hand_data = self.webcam.get_data()
            
            self.process_gestures(hand_data)
            self.update_game_state()
            
            for proj in self.projectiles: proj.draw(self.screen)
            for enemy in self.enemies: enemy.draw(self.screen)
            self.player.draw(self.screen)
            
            self.draw_hud(cam_frame, hand_data)
            
            pygame.display.flip()
            self.clock.tick(FPS)
            
        if self.player.hp <= 0:
            go_running = True
            while go_running:
                self.screen.fill(BLACK)
                go_text = self.large_font.render("GAME OVER", True, RED)
                score_lbl = self.font.render(f"Final Score Record: {self.player.score}", True, WHITE)
                exit_lbl = self.font.render("Click the close [X] button to exit.", True, GRAY)
                
                self.screen.blit(go_text, (self.width//2 - go_text.get_width()//2, self.height//2 - 60))
                self.screen.blit(score_lbl, (self.width//2 - score_lbl.get_width()//2, self.height//2))
                self.screen.blit(exit_lbl, (self.width//2 - exit_lbl.get_width()//2, self.height//2 + 60))
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        go_running = False
                pygame.display.flip()
                self.clock.tick(15)

        self.webcam.stop()
        pygame.quit()

if __name__ == "__main__":
    game = GameEngine()
    game.run()