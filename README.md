# 🕷️ Spider-Man Gesture AR Game

A computer-vision Spider-Man experience built with **Python, OpenCV, MediaPipe, and Pygame**.

The project has two interactive modes:

- **Spider-Man Suit AR** — overlays a Spider-Man-style mask and suit onto the webcam feed and generates web particles from hand movement.
- **Spider-Man POV Game** — turns hand gestures into game controls for web shooting, swinging, and melee combat.

> ⚠️ This is an experimental computer-vision project. It uses a webcam and real-time hand/face/pose landmark detection.

## ✨ Features

### 🕷️ Suit AR Mode — `main.py`

- Real-time webcam processing with OpenCV
- Face landmark detection
- Pose landmark detection
- Hand landmark detection
- Spider-Man-style face mask overlay
- Red/blue suit body overlay
- Web-pattern visual effects
- Gesture-driven web shooting
- Web particle physics with gravity and air resistance
- FPS display and live instructions

### 🎮 Spider-Man POV Game — `game.py`

- Real-time hand tracking with MediaPipe
- Dedicated background thread for webcam processing
- Gesture recognition:
  - 🤟 **Spidey gesture** → attach/swing toward an anchor
  - ✋ **Open palm** → shoot web projectile
  - ✊ **Fist** → melee attack
- Player health and score system
- Enemy spawning and pursuit
- Web projectile collision detection
- Dynamic web-swing anchors
- Resizable game window
- Live webcam preview and gesture status
- Game-over screen

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Webcam capture and image processing |
| MediaPipe | Face, pose, and hand landmark detection |
| NumPy | Numerical and coordinate calculations |
| Pygame | Game rendering and game loop |
| Threading | Background webcam/hand-tracking processing |

## 📁 Project Structure

```text
spider_man_game/
├── main.py                  # Spider-Man Suit AR mode
├── game.py                  # Gesture-controlled Spider-Man game
├── spiderman_mask.png       # Mask overlay asset
├── face_landmarker.task     # MediaPipe face model
├── hand_landmarker.task     # MediaPipe hand model
└── README.md                # Project documentation
```

> `main.py` also attempts to use `pose_landmarker_full.task` when available and falls back to the legacy MediaPipe Pose API.

## ⚙️ Requirements

Recommended:

- Python 3.10+
- Webcam
- Working microphone is **not required**
- Linux, Windows, or macOS with a supported Python environment

Install the Python dependencies:

```bash
pip install opencv-python numpy mediapipe pygame
```

If you use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy mediapipe pygame
```

On Windows:

```powershell
.venv\Scripts\activate
pip install opencv-python numpy mediapipe pygame
```

## 🚀 Run the Project

### 1. Spider-Man Suit AR

Make sure the required model and mask assets are in the project directory, then run:

```bash
python main.py
```

Controls:

- **Q** → Exit

Point/move your fingers in front of the webcam to trigger the web particle effect.

### 2. Spider-Man POV Game

Run:

```bash
python game.py
```

The game automatically attempts to download `hand_landmarker.task` if it is missing.

### 🎮 Gesture Controls

| Gesture | Hand | Action |
|---|---|---|
| 🤟 Spidey | Left | Attach to the nearest web anchor and swing |
| ✋ Palm | Right | Shoot a web projectile |
| ✊ Fist | Right | Punch nearby enemies |
| No recognized gesture | Left | Stop swinging |

## 🧠 How It Works

### Computer Vision Pipeline

```text
Webcam
   │
   ▼
OpenCV Frame Capture
   │
   ▼
MediaPipe Landmark Detection
   │
   ├── Face Landmarks ──► Spider-Man Mask
   │
   ├── Pose Landmarks ──► Spider-Man Suit
   │
   └── Hand Landmarks ──► Gesture Recognition
                              │
                              ▼
                       Game / Web Actions
```

### Game Architecture

The game separates the main systems into reusable classes:

- `WebcamStream` — captures frames and performs hand tracking in a background thread.
- `GestureRecognizer` — converts hand landmarks into gestures.
- `Player` — handles movement, gravity, health, score, and swinging.
- `WebProjectile` — controls web projectile movement.
- `Enemy` — handles enemy movement and rendering.
- `GameEngine` — manages the game loop, collisions, rendering, input, and state.

## 🔧 Troubleshooting

### Webcam does not open

Check that your camera is available and not being used by another application.

On Linux, you can inspect video devices with:

```bash
ls /dev/video*
```

### MediaPipe model error

Make sure the model files are in the same directory from which you run the program:

```text
face_landmarker.task
hand_landmarker.task
```

For `main.py`, `pose_landmarker_full.task` is optional because the code contains a fallback to the legacy MediaPipe Pose implementation.

### Pygame window does not appear correctly

Try running the game from a normal desktop session with access to your display and webcam.

### Performance is low

The project performs real-time computer vision, which can be CPU intensive. Closing other camera-heavy applications and lowering the webcam resolution can help.

## ⚠️ Current Project Notes

- The project is designed as an experimental real-time CV/game prototype.
- Gesture recognition is heuristic-based, so lighting, camera angle, hand orientation, and distance from the camera can affect detection.
- `main.py` contains an additional mask-overlay implementation near the application startup; future cleanup could consolidate the mask-rendering logic.
- The repository currently contains large MediaPipe model assets, which can make Git history and cloning heavier.

## 🔮 Possible Improvements

- Add a proper requirements file
- Add configurable gesture sensitivity
- Add sound effects and background music
- Add multiple enemy types
- Add combo and level systems
- Add real web-swing physics
- Improve collision and animation systems
- Add a settings menu
- Add cross-platform camera/device selection
- Separate configuration, CV, game entities, and rendering into modules
- Add automated tests for gesture recognition
- Add GitHub Actions for linting and basic validation

## 📜 License

No license is currently specified for this repository.

If you plan to distribute or accept contributions, consider adding an appropriate open-source license.

## 👨‍💻 Author

**Ricky — Ricky-Hacker001**

Built as an experimental computer-vision and interactive gaming project.

---

⭐ If you found the project interesting, consider starring the repository and experimenting with the gesture system.
