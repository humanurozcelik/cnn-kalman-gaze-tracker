import cv2
import mediapipe as mp
import pygame
import threading
import time
import sys
import csv
import argparse
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras

DOSYA_ADI = "lab_gaze_tracking_data.csv"

# Goz kutusu icin MediaPipe yuz isaretleri (yalnizca kirpma / bbox)
_LEFT_EYE_LM = (33, 133, 159, 145, 158, 153, 157, 173, 155, 246)
_RIGHT_EYE_LM = (362, 263, 386, 374, 385, 380, 387, 388, 382, 398)

# --- DURUM (STATE) SABITLERI ---
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
        inp = keras.layers.Input(shape=(64, 64, 2), name="binocular_in")
        x = keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inp)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)

        x = keras.layers.Flatten()(x)

        x = keras.layers.Dense(64, activation="relu")(x)
        out = keras.layers.Dense(2, activation="linear")(x)

        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
        )
        return model

    def train(self, inputs, targets_x, targets_y, groups=None):
        X = np.asarray(inputs, dtype=np.float32)
        if X.ndim != 4 or X.shape[1:] != (64, 64, 2):
            raise ValueError(f"Beklenen X sekli (N,64,64,2), gelen: {X.shape}")

        raw_targets = np.column_stack([targets_x, targets_y]).astype(np.float32)
        y_scaled = self.target_scaler.fit_transform(raw_targets)

        print(f"\n--- CNN EGITIMI BASLADI ({X.shape[0]} ornek) ---")

        self.model = self._build()

        early_stop = keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        )

        self.model.fit(
            X,
            y_scaled,
            epochs=50,
            batch_size=16,
            verbose=1,
            callbacks=[early_stop],
        )
        self.is_trained = True

    def predict(self, tensor):
        if not self.is_trained or self.model is None or tensor is None:
            return 0, 0

        x = np.expand_dims(np.asarray(tensor, dtype=np.float32), axis=0)

        # HIZLANDIRMA: .predict() yerine dogrudan tensor cagrisi
        pred_scaled = self.model(x, training=False).numpy()

        pred_real = self.target_scaler.inverse_transform(pred_scaled)[0]

        return int(pred_real[0]), int(pred_real[1])


class KalmanFilter:
    """
    OpenCV cv2.KalmanFilter sarmalayici.
    process_noise (Q) yuksek: dinamikte gecikmeyi azaltir.
    measurement_noise (R): olcum guvenini ayarlar.
    """

    def __init__(self, process_noise=1e-1, measurement_noise=1e-4):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * float(process_noise)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(measurement_noise)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self.initialized = False

    def update(self, x, y):
        measured = np.array([[float(x)], [float(y)]], dtype=np.float32)
        if not self.initialized:
            z = np.array([[float(x)], [float(y)], [0.0], [0.0]], dtype=np.float32)
            self.kf.statePre = z.copy()
            self.kf.statePost = z.copy()
            self.initialized = True
            return int(x), int(y)

        self.kf.predict()
        self.kf.correct(measured)
        return int(self.kf.statePost[0][0]), int(self.kf.statePost[1][0])


class VisionWorker:
    """
    MediaPipe FaceMesh yalnizca yuz isaretlerinden sol/sag goz bbox kirpmasi ve EAR icin kullanilir.
    main(): get_data() -> frame, ear, binocular_tensor (64,64,2) veya None
    """

    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Kamera {camera_id} acilamadi. Lutfen dogru kamera ID'si secin.")
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.latest_frame = None
        self.latest_binocular = None
        self.latest_ear = 0
        self.latest_timestamp = 0

        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _eye_bbox(self, landmarks, indices):
        xs = [landmarks[i].x * self.w for i in indices]
        ys = [landmarks[i].y * self.h for i in indices]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        pad_w = max(8.0, 0.25 * (x_max - x_min))
        pad_h = max(8.0, 0.25 * (y_max - y_min))
        x0 = int(max(0, x_min - pad_w))
        y0 = int(max(0, y_min - pad_h))
        x1 = int(min(self.w, x_max + pad_w))
        y1 = int(min(self.h, y_max + pad_h))
        return x0, y0, x1, y1

    def _crop_grayscale_norm(self, frame_bgr, box, out_size=64):
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return None
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def build_binocular_tensor(self, frame_bgr, landmarks):
        """Sol goz kanal 0, sag goz kanal 1; sekil (64, 64, 2), float32 [0,1]."""
        lb = self._eye_bbox(landmarks, _LEFT_EYE_LM)
        rb = self._eye_bbox(landmarks, _RIGHT_EYE_LM)
        left = self._crop_grayscale_norm(frame_bgr, lb, out_size=64)
        right = self._crop_grayscale_norm(frame_bgr, rb, out_size=64)
        if left is None or right is None:
            return None
        return np.stack([left, right], axis=-1)

    def compute_ear(self, landmarks):
        p = lambda i: np.array([landmarks[i].x * self.w, landmarks[i].y * self.h])
        L_in, L_out, L_top, L_bot = p(133), p(33), p(159), p(145)
        R_in, R_out, R_top, R_bot = p(362), p(263), p(386), p(374)
        El_w, El_v = np.linalg.norm(L_out - L_in), np.linalg.norm(L_top - L_bot)
        Er_w, Er_v = np.linalg.norm(R_out - R_in), np.linalg.norm(R_top - R_bot)
        ear = (El_v / (El_w + 1e-6) + Er_v / (Er_w + 1e-6)) / 2.0
        return float(ear)

    def update(self):
        while self.running:
            capture_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.mp_face_mesh.process(rgb_frame)

            ear, binocular = 0.0, None
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                ear = self.compute_ear(lms)
                binocular = self.build_binocular_tensor(frame, lms)

            with self.lock:
                self.latest_frame = frame
                self.latest_ear = ear
                self.latest_binocular = binocular
                self.latest_timestamp = capture_time

            time.sleep(0.005)

    def get_data(self):
        with self.lock:
            return (
                self.latest_frame.copy() if self.latest_frame is not None else None,
                self.latest_ear,
                self.latest_binocular,
                self.latest_timestamp,
            )

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()


def draw_text_center(screen, txt, font, offset_y=0, color=(255, 255, 255)):
    s = font.render(txt, True, color)
    screen.blit(s, s.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 + offset_y)))


def _nine_grid_waypoints(W, H, margin_ratio=0.08):
    """Ekranda 3x3 izgara: 9 nokta (sol-orta-sag x ust-orta-alt)."""
    m = min(W, H) * margin_ratio
    xs = [int(m), int(W // 2), int(W - m)]
    ys = [int(m), int(H // 2), int(H - m)]
    waypoints = []
    for y in ys:
        for x in xs:
            waypoints.append((x, y))
    return waypoints


def main(camera_id=0, fullscreen=True, csv_path=DOSYA_ADI, show_preview=False):
    pygame.init()
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h
    flags = pygame.FULLSCREEN if fullscreen else 0
    screen = pygame.display.set_mode((W, H), flags)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)
    big_font = pygame.font.SysFont("Arial", 40, bold=True)

    try:
        vision = VisionWorker(camera_id=camera_id)
    except RuntimeError as e:
        print(str(e))
        pygame.quit()
        sys.exit(1)

    gaze_model = CNNGazeModel()
    kalman_filter = KalmanFilter(process_noise=1e-1, measurement_noise=1e-4)

    current_state = STATE_IDLE
    running = True

    CALIB_SPEED = 1.8
    EAR_THRESHOLD = 0.14
    waypoints = _nine_grid_waypoints(W, H, margin_ratio=0.08)

    current_wp = 0
    curr_x, curr_y = float(waypoints[0][0]), float(waypoints[0][1])

    DWELL_SECONDS = 3.0
    waypoint_arrival_ticks = None

    calib_inputs, calib_targets_x, calib_targets_y = [], [], []
    calib_groups = []

    training_started = False
    training_completed = False

    def training_task():
        nonlocal training_completed
        inputs = list(calib_inputs)
        tx = list(calib_targets_x)
        ty = list(calib_targets_y)
        groups = list(calib_groups)

        if len(inputs) <= 10:
            print("UYARI: Kalibrasyon verisi yetersiz.")
        else:
            # Filtreleme iptal edildi, ham veriler dogrudan egitilir
            gaze_model.train(inputs, tx, ty, groups=np.array(groups))

        training_completed = True

    test_t = 0
    HIZ_FAKTORU = 0.007
    TUR_SAYISI = 2
    MAX_TIME = TUR_SAYISI * (2 * math.pi)
    start_test_time = 0

    csv_file = None
    csv_writer = None

    last_frame_time = time.time()

    while running:
        current_time = time.time()
        dt = current_time - last_frame_time
        if dt <= 0:
            dt = 1e-3
        last_frame_time = current_time
        screen.fill((20, 20, 20))

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                if current_state == STATE_IDLE:
                    current_state = STATE_CALIBRATE

        frame, ear, binocular, frame_timestamp = vision.get_data()

        if show_preview and frame is not None and current_state != STATE_TEST:
            frame_rgb = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2RGB)
            screen.blit(pygame.image.frombuffer(frame_rgb.tobytes(), (320, 240), "RGB"), (10, 10))

        if current_state == STATE_IDLE:
            draw_text_center(screen, "LAB-GRADE GOZ TAKIP SISTEMI (CNN + Kalman)", big_font, -50, (0, 255, 0))
            draw_text_center(screen, "Kalibrasyona baslamak icin ENTER'a basin", font, 50)

        elif current_state == STATE_CALIBRATE:
            tx, ty = waypoints[current_wp]
            dist = math.hypot(tx - curr_x, ty - curr_y)

            if dist < CALIB_SPEED:
                curr_x, curr_y = float(tx), float(ty)

                if waypoint_arrival_ticks is None:
                    waypoint_arrival_ticks = pygame.time.get_ticks()

                if binocular is not None and ear > EAR_THRESHOLD:
                    calib_inputs.append(np.copy(binocular))
                    calib_targets_x.append(int(curr_x))
                    calib_targets_y.append(int(curr_y))
                    calib_groups.append(current_wp)

                elapsed_sec = (pygame.time.get_ticks() - waypoint_arrival_ticks) / 1000.0
                kalan_sure = max(0, DWELL_SECONDS - elapsed_sec)
                draw_text_center(screen, f"Lutfen noktaya bakin: {kalan_sure:.1f}s", font, -150)

                if elapsed_sec >= DWELL_SECONDS:
                    current_wp += 1
                    waypoint_arrival_ticks = None
                    if current_wp >= len(waypoints):
                        current_state = STATE_TRAIN
            else:
                waypoint_arrival_ticks = None
                curr_x += ((tx - curr_x) / dist) * CALIB_SPEED
                curr_y += ((ty - curr_y) / dist) * CALIB_SPEED
                draw_text_center(screen, "Gozlerinizle sari hedefi takip edin...", font, -200)

            draw_x, draw_y = int(curr_x), int(curr_y)
            pygame.draw.circle(screen, (255, 255, 0), (draw_x, draw_y), 15)
            pygame.draw.circle(screen, (255, 0, 0), (draw_x, draw_y), 4)

        elif current_state == STATE_TRAIN:
            if not training_started:
                if len(calib_inputs) > 50:
                    threading.Thread(target=training_task, daemon=True).start()
                    training_started = True
                else:
                    print("Kalibrasyon icin yeterli veri toplanamadi (en az 50 frame gerekli). Denemeyi yeniden baslatin.")
                    running = False

            noktalar = "." * (int(current_time * 3) % 4)
            draw_text_center(screen, f"CNN Egitiliyor{noktalar}", big_font, -50, (0, 255, 255))
            draw_text_center(screen, f"Toplanan Sabit Veri: {len(calib_inputs)}", font, 50)

            if training_completed:
                current_state = STATE_TEST
                start_test_time = time.time()

                kalman_filter = KalmanFilter(process_noise=1e-1, measurement_noise=1e-4)

                csv_file = open(csv_path, "w", newline="", encoding="utf-8")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(
                    ["Time_s", "Latency_ms", "EAR", "Face_Detected", "Target_X", "Target_Y", "Gaze_X", "Gaze_Y"]
                )

        elif current_state == STATE_TEST:
            if test_t >= MAX_TIME:
                running = False
                continue

            scale, denom = W / 3.5, 1 + math.sin(test_t) ** 2
            target_x = int(W / 2 + (scale * math.cos(test_t)) / denom)
            target_y = int(H / 2 + (scale * math.sin(test_t) * math.cos(test_t)) / denom)

            pygame.draw.circle(screen, (100, 100, 100), (target_x, target_y), 30)
            pygame.draw.circle(screen, (255, 0, 0), (target_x, target_y), 5)

            if binocular is not None and ear > EAR_THRESHOLD:
                raw_x, raw_y = gaze_model.predict(binocular)
                filtered_x, filtered_y = kalman_filter.update(raw_x, raw_y)

                pygame.draw.circle(screen, (0, 255, 0), (filtered_x, filtered_y), 15)

                latency_ms = round((current_time - frame_timestamp) * 1000, 2)
                yuz_var_mi = 1
                anlik_ear = round(ear, 3) if ear is not None else 0.0

                csv_writer.writerow(
                    [
                        round(current_time - start_test_time, 3),
                        latency_ms,
                        anlik_ear,
                        yuz_var_mi,
                        target_x,
                        target_y,
                        filtered_x,
                        filtered_y,
                    ]
                )
                csv_writer.flush() if hasattr(csv_writer, "flush") else None
                csv_file.flush()

            test_t += HIZ_FAKTORU * dt * 60.0
            pygame.draw.rect(screen, (0, 200, 0), (0, H - 10, int((test_t / MAX_TIME) * W), 10))

        fps_text = font.render(f"UI FPS: {int(clock.get_fps())}", True, (255, 255, 255))
        screen.blit(fps_text, (W - 150, 10))

        pygame.display.flip()
        clock.tick(60)

    vision.stop()
    pygame.quit()
    if csv_file:
        csv_file.close()
    print("Sistem basariyla kapatildi.")
    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab-grade goz takip sistemi (CNN + Kalman)")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Kullanilacak kamera ID'si (varsayilan: 0)",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Tam ekran yerine pencere modunda calistir",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DOSYA_ADI,
        help=f"Cikti CSV dosya adi/yolu (varsayilan: {DOSYA_ADI})",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Kalibrasyon esnasinda kucuk kamera goruntusu goster (varsayilan: gizli)",
    )
    args = parser.parse_args()

    fullscreen_flag = not args.windowed
    main(camera_id=args.camera, fullscreen=fullscreen_flag, csv_path=args.csv, show_preview=args.preview)
