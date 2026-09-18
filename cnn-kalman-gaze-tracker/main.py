import argparse
import csv
import math
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import pygame
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


DEFAULT_CSV = "lab_gaze_tracking_data.csv"

LEFT_EYE_LANDMARKS = (33, 133, 159, 145, 158, 153, 157, 173, 155, 246)
RIGHT_EYE_LANDMARKS = (362, 263, 386, 374, 385, 380, 387, 388, 382, 398)

STATE_IDLE = 0
STATE_CALIBRATE = 1
STATE_TRAIN = 2
STATE_TEST = 3


class CNNGazeModel:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.target_scaler = MinMaxScaler()

    def _build(self):
        inputs = keras.layers.Input(shape=(64, 64, 2), name="binocular_input")

        x = keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        outputs = keras.layers.Dense(2, activation="linear")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
        )
        return model

    def train(self, inputs, targets_x, targets_y):
        x_train = np.asarray(inputs, dtype=np.float32)
        if x_train.ndim != 4 or x_train.shape[1:] != (64, 64, 2):
            raise ValueError(
                f"Expected calibration input shape (N, 64, 64, 2), got {x_train.shape}"
            )

        targets = np.column_stack([targets_x, targets_y]).astype(np.float32)
        y_train = self.target_scaler.fit_transform(targets)

        print(f"\nStarting CNN training with {x_train.shape[0]} calibration samples...")

        self.model = self._build()

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        )

        self.model.fit(
            x_train,
            y_train,
            epochs=50,
            batch_size=16,
            verbose=1,
            callbacks=[early_stopping],
        )
        self.is_trained = True

    def predict(self, binocular_tensor):
        if not self.is_trained or self.model is None or binocular_tensor is None:
            return 0, 0

        model_input = np.expand_dims(
            np.asarray(binocular_tensor, dtype=np.float32),
            axis=0,
        )
        prediction_scaled = self.model(model_input, training=False).numpy()
        prediction = self.target_scaler.inverse_transform(prediction_scaled)[0]

        return int(prediction[0]), int(prediction[1])


class KalmanFilter:
    """Simple constant-velocity 2D Kalman filter based on OpenCV."""

    def __init__(self, process_noise=1e-1, measurement_noise=1e-4):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        )
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = (
            np.eye(4, dtype=np.float32) * float(process_noise)
        )
        self.kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * float(measurement_noise)
        )
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def update(self, x, y):
        measurement = np.array([[float(x)], [float(y)]], dtype=np.float32)

        if not self.initialized:
            initial_state = np.array(
                [[float(x)], [float(y)], [0.0], [0.0]],
                dtype=np.float32,
            )
            self.kf.statePre = initial_state.copy()
            self.kf.statePost = initial_state.copy()
            self.initialized = True
            return int(x), int(y)

        self.kf.predict()
        self.kf.correct(measurement)

        return int(self.kf.statePost[0][0]), int(self.kf.statePost[1][0])


class VisionWorker:
    """Capture webcam frames and extract normalized binocular eye crops."""

    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Camera {camera_id} could not be opened. Check the camera ID and permissions."
            )

        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.latest_frame = None
        self.latest_binocular = None
        self.latest_ear = 0.0
        self.latest_timestamp = 0.0

        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _eye_bbox(self, landmarks, indices):
        xs = [landmarks[i].x * self.width for i in indices]
        ys = [landmarks[i].y * self.height for i in indices]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        pad_w = max(8.0, 0.25 * (x_max - x_min))
        pad_h = max(8.0, 0.25 * (y_max - y_min))

        x0 = int(max(0, x_min - pad_w))
        y0 = int(max(0, y_min - pad_h))
        x1 = int(min(self.width, x_max + pad_w))
        y1 = int(min(self.height, y_max + pad_h))

        return x0, y0, x1, y1

    @staticmethod
    def _crop_grayscale_normalized(frame_bgr, box, output_size=64):
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return None

        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            gray,
            (output_size, output_size),
            interpolation=cv2.INTER_AREA,
        )

        return resized.astype(np.float32) / 255.0

    def build_binocular_tensor(self, frame_bgr, landmarks):
        left_box = self._eye_bbox(landmarks, LEFT_EYE_LANDMARKS)
        right_box = self._eye_bbox(landmarks, RIGHT_EYE_LANDMARKS)

        left_eye = self._crop_grayscale_normalized(frame_bgr, left_box)
        right_eye = self._crop_grayscale_normalized(frame_bgr, right_box)

        if left_eye is None or right_eye is None:
            return None

        return np.stack([left_eye, right_eye], axis=-1)

    def compute_ear(self, landmarks):
        def point(index):
            return np.array(
                [
                    landmarks[index].x * self.width,
                    landmarks[index].y * self.height,
                ]
            )

        left_inner, left_outer = point(133), point(33)
        left_top, left_bottom = point(159), point(145)
        right_inner, right_outer = point(362), point(263)
        right_top, right_bottom = point(386), point(374)

        left_width = np.linalg.norm(left_outer - left_inner)
        left_vertical = np.linalg.norm(left_top - left_bottom)
        right_width = np.linalg.norm(right_outer - right_inner)
        right_vertical = np.linalg.norm(right_top - right_bottom)

        ear = (
            left_vertical / (left_width + 1e-6)
            + right_vertical / (right_width + 1e-6)
        ) / 2.0

        return float(ear)

    def _update_loop(self):
        while self.running:
            capture_time = time.time()
            ok, frame = self.cap.read()

            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb_frame)

            ear = 0.0
            binocular = None

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                ear = self.compute_ear(landmarks)
                binocular = self.build_binocular_tensor(frame, landmarks)

            with self.lock:
                self.latest_frame = frame
                self.latest_ear = ear
                self.latest_binocular = binocular
                self.latest_timestamp = capture_time

            time.sleep(0.005)

    def get_data(self):
        with self.lock:
            return (
                self.latest_frame.copy()
                if self.latest_frame is not None
                else None,
                self.latest_ear,
                self.latest_binocular,
                self.latest_timestamp,
            )

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()
        self.face_mesh.close()


def draw_text_center(screen, text, font, offset_y=0, color=(255, 255, 255)):
    surface = font.render(text, True, color)
    rect = surface.get_rect(
        center=(
            screen.get_width() // 2,
            screen.get_height() // 2 + offset_y,
        )
    )
    screen.blit(surface, rect)


def nine_grid_waypoints(width, height, margin_ratio=0.08):
    margin = min(width, height) * margin_ratio
    xs = [int(margin), int(width // 2), int(width - margin)]
    ys = [int(margin), int(height // 2), int(height - margin)]

    return [(x, y) for y in ys for x in xs]


def main(camera_id=0, fullscreen=True, csv_path=DEFAULT_CSV, show_preview=False):
    pygame.init()

    display_info = pygame.display.Info()
    width, height = display_info.current_w, display_info.current_h

    flags = pygame.FULLSCREEN if fullscreen else 0
    screen = pygame.display.set_mode((width, height), flags)
    pygame.display.set_caption("Webcam Gaze Tracking Demo")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)
    big_font = pygame.font.SysFont("Arial", 40, bold=True)

    try:
        vision = VisionWorker(camera_id=camera_id)
    except RuntimeError as exc:
        print(exc)
        pygame.quit()
        return 1

    gaze_model = CNNGazeModel()
    kalman_filter = KalmanFilter(
        process_noise=1e-1,
        measurement_noise=1e-4,
    )

    current_state = STATE_IDLE
    running = True

    calibration_speed = 1.8
    ear_threshold = 0.14
    waypoints = nine_grid_waypoints(width, height)

    current_waypoint = 0
    current_x = float(waypoints[0][0])
    current_y = float(waypoints[0][1])

    dwell_seconds = 3.0
    waypoint_arrival_ticks = None

    calibration_inputs = []
    calibration_targets_x = []
    calibration_targets_y = []

    training_started = False
    training_completed = False
    training_error = None

    def training_task():
        nonlocal training_completed, training_error

        try:
            gaze_model.train(
                list(calibration_inputs),
                list(calibration_targets_x),
                list(calibration_targets_y),
            )
        except Exception as exc:
            training_error = exc
        finally:
            training_completed = True

    test_parameter = 0.0
    path_speed = 0.007
    path_cycles = 2
    max_parameter = path_cycles * (2 * math.pi)
    test_start_time = 0.0

    csv_file = None
    csv_writer = None
    rows_since_flush = 0

    last_frame_time = time.time()

    try:
        while running:
            current_time = time.time()
            dt = max(current_time - last_frame_time, 1e-3)
            last_frame_time = current_time

            screen.fill((20, 20, 20))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif (
                        event.key == pygame.K_RETURN
                        and current_state == STATE_IDLE
                    ):
                        current_state = STATE_CALIBRATE

            frame, ear, binocular, frame_timestamp = vision.get_data()

            if (
                show_preview
                and frame is not None
                and current_state != STATE_TEST
            ):
                preview = cv2.resize(frame, (320, 240))
                preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                preview_surface = pygame.image.frombuffer(
                    preview.tobytes(),
                    (320, 240),
                    "RGB",
                )
                screen.blit(preview_surface, (10, 10))

            if current_state == STATE_IDLE:
                draw_text_center(
                    screen,
                    "WEBCAM GAZE TRACKING (CNN + KALMAN)",
                    big_font,
                    -50,
                    (0, 255, 0),
                )
                draw_text_center(
                    screen,
                    "Press ENTER to start calibration",
                    font,
                    50,
                )

            elif current_state == STATE_CALIBRATE:
                target_x, target_y = waypoints[current_waypoint]
                distance = math.hypot(
                    target_x - current_x,
                    target_y - current_y,
                )

                if distance < calibration_speed:
                    current_x = float(target_x)
                    current_y = float(target_y)

                    if waypoint_arrival_ticks is None:
                        waypoint_arrival_ticks = pygame.time.get_ticks()

                    if binocular is not None and ear > ear_threshold:
                        calibration_inputs.append(np.copy(binocular))
                        calibration_targets_x.append(int(current_x))
                        calibration_targets_y.append(int(current_y))

                    elapsed = (
                        pygame.time.get_ticks() - waypoint_arrival_ticks
                    ) / 1000.0

                    remaining = max(0.0, dwell_seconds - elapsed)
                    draw_text_center(
                        screen,
                        f"Look at the target: {remaining:.1f}s",
                        font,
                        -150,
                    )

                    if elapsed >= dwell_seconds:
                        current_waypoint += 1
                        waypoint_arrival_ticks = None

                        if current_waypoint >= len(waypoints):
                            current_state = STATE_TRAIN
                else:
                    waypoint_arrival_ticks = None
                    current_x += (
                        (target_x - current_x) / distance
                    ) * calibration_speed
                    current_y += (
                        (target_y - current_y) / distance
                    ) * calibration_speed

                    draw_text_center(
                        screen,
                        "Follow the yellow target with your eyes...",
                        font,
                        -200,
                    )

                draw_x, draw_y = int(current_x), int(current_y)
                pygame.draw.circle(
                    screen,
                    (255, 255, 0),
                    (draw_x, draw_y),
                    15,
                )
                pygame.draw.circle(
                    screen,
                    (255, 0, 0),
                    (draw_x, draw_y),
                    4,
                )

            elif current_state == STATE_TRAIN:
                if not training_started:
                    if len(calibration_inputs) > 50:
                        threading.Thread(
                            target=training_task,
                            daemon=True,
                        ).start()
                        training_started = True
                    else:
                        print(
                            "Not enough calibration data were collected "
                            "(at least 50 frames are required)."
                        )
                        running = False

                dots = "." * (int(current_time * 3) % 4)
                draw_text_center(
                    screen,
                    f"Training CNN{dots}",
                    big_font,
                    -50,
                    (0, 255, 255),
                )
                draw_text_center(
                    screen,
                    f"Calibration samples: {len(calibration_inputs)}",
                    font,
                    50,
                )

                if training_completed:
                    if training_error is not None:
                        print(f"Training failed: {training_error}")
                        running = False
                    else:
                        current_state = STATE_TEST
                        test_start_time = time.time()

                        kalman_filter = KalmanFilter(
                            process_noise=1e-1,
                            measurement_noise=1e-4,
                        )

                        csv_file = open(
                            csv_path,
                            "w",
                            newline="",
                            encoding="utf-8",
                        )
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(
                            [
                                "Time_s",
                                "Frame_Age_ms",
                                "EAR",
                                "Face_Detected",
                                "Target_X",
                                "Target_Y",
                                "Gaze_X",
                                "Gaze_Y",
                                "Screen_W",
                                "Screen_H",
                            ]
                        )

            elif current_state == STATE_TEST:
                if test_parameter >= max_parameter:
                    running = False
                    continue

                scale = width / 3.5
                denominator = 1 + math.sin(test_parameter) ** 2

                target_x = int(
                    width / 2
                    + (scale * math.cos(test_parameter)) / denominator
                )
                target_y = int(
                    height / 2
                    + (
                        scale
                        * math.sin(test_parameter)
                        * math.cos(test_parameter)
                    )
                    / denominator
                )

                pygame.draw.circle(
                    screen,
                    (100, 100, 100),
                    (target_x, target_y),
                    30,
                )
                pygame.draw.circle(
                    screen,
                    (255, 0, 0),
                    (target_x, target_y),
                    5,
                )

                face_detected = int(binocular is not None)
                gaze_x = ""
                gaze_y = ""

                if binocular is not None and ear > ear_threshold:
                    raw_x, raw_y = gaze_model.predict(binocular)
                    filtered_x, filtered_y = kalman_filter.update(
                        raw_x,
                        raw_y,
                    )

                    gaze_x = filtered_x
                    gaze_y = filtered_y

                    pygame.draw.circle(
                        screen,
                        (0, 255, 0),
                        (filtered_x, filtered_y),
                        15,
                    )

                frame_age_ms = ""
                if frame_timestamp > 0:
                    frame_age_ms = round(
                        (current_time - frame_timestamp) * 1000,
                        2,
                    )

                if csv_writer is not None:
                    csv_writer.writerow(
                        [
                            round(current_time - test_start_time, 3),
                            frame_age_ms,
                            round(float(ear), 3),
                            face_detected,
                            target_x,
                            target_y,
                            gaze_x,
                            gaze_y,
                            width,
                            height,
                        ]
                    )

                    rows_since_flush += 1
                    if rows_since_flush >= 60:
                        csv_file.flush()
                        rows_since_flush = 0

                test_parameter += path_speed * dt * 60.0

                progress = int(
                    min(test_parameter / max_parameter, 1.0) * width
                )
                pygame.draw.rect(
                    screen,
                    (0, 200, 0),
                    (0, height - 10, progress, 10),
                )

            fps_surface = font.render(
                f"UI FPS: {int(clock.get_fps())}",
                True,
                (255, 255, 255),
            )
            screen.blit(
                fps_surface,
                (max(10, width - 150), 10),
            )

            pygame.display.flip()
            clock.tick(60)

    finally:
        vision.stop()

        if csv_file is not None:
            csv_file.flush()
            csv_file.close()

        pygame.quit()

    print("Gaze-tracking session finished.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Webcam gaze tracking with a shallow CNN and Kalman filtering"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera ID (default: 0)",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Run in a window instead of fullscreen mode",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV,
        help=f"Output CSV path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a small webcam preview during calibration",
    )

    args = parser.parse_args()

    exit_code = main(
        camera_id=args.camera,
        fullscreen=not args.windowed,
        csv_path=args.csv,
        show_preview=args.preview,
    )
    sys.exit(exit_code)
