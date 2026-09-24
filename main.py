"""Hand Drive Pro - Advanced 2D car dodging game.

Professional hand-controlled racing game with hybrid motion detection,
auto-calibration, particle effects, and progressive difficulty.

Driver's-eye view: obstacles appear far ahead at the horizon and grow as
they approach, exactly as the driver sees them — so the car visibly
dodges them at the last moment. The hand-control system is untouched.
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pygame
import pygame.sndarray

# --- شکل‌دهی صحیح متن فارسی/عربی (اتصال حروف + راست‌به‌چپ) ---
# pygame به‌تنهایی حروف فارسی را جدا از هم و بدون رعایت جهت راست‌به‌چپ
# رسم می‌کند. با این دو کتابخانه، متن قبل از رندر اصلاح می‌شود.
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _FARSI_SHAPING_AVAILABLE = True
except Exception:
    _FARSI_SHAPING_AVAILABLE = False


def shape_farsi(text: str) -> str:
    """متن فارسی/عربی را برای نمایش صحیح در pygame آماده می‌کند."""
    if not _FARSI_SHAPING_AVAILABLE:
        return text
    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def render_text(font: "pygame.font.Font", text: str, color) -> "pygame.Surface":
    """رندر متن با پشتیبانی خودکار از شکل‌دهی فارسی/عربی."""
    return font.render(shape_farsi(text), True, color)


WIDTH, HEIGHT = 1000, 750
ROAD_LEFT, ROAD_RIGHT = 180, 720
CAR_W, CAR_H = 60, 100
CAMERA_W, CAMERA_H = 280, 210
FPS = 60
HIGHSCORE_FILE = "highscore.txt"
MAX_LIVES = 3              # تعداد جان‌ها (قلب‌ها)
INVINCIBLE_TIME = 1.6      # زمان بی‌آسیبی بعد از هر برخورد (ثانیه)

# ---- نمای شبه‌سه‌بعدی (دید راننده) ----
HORIZON_Y = 300            # خط افق
VIEW_H = HEIGHT - HORIZON_Y
ROAD_HALF = (ROAD_RIGHT - ROAD_LEFT) / 2   # نصف عرض جاده در نزدیک‌ترین نقطه
NEAR = 120.0               # قدرت پرسپکتیو
SPAWN_Z = 650.0            # فاصله اسپاون موانع (جلوتر از افق مه‌آلود)
PLAYER_Z = 30.0            # موقعیت ماشین بازیکن روی جاده
DASH_PERIOD = 90.0
LANES = (-0.62, 0.0, 0.62)

SAMPLE_RATE = 44100


# ==================== PROCEDURAL SOUND ENGINE ====================
# همه صداهای بازی به‌صورت برنامه‌نویسی (بدون فایل صوتی خارجی) ساخته می‌شوند.

def _tone(freq: float, duration: float, volume: float = 0.5,
          wave: str = "sine", fade: float = 0.015) -> np.ndarray:
    n = max(1, int(SAMPLE_RATE * duration))
    t = np.linspace(0.0, duration, n, endpoint=False)
    if wave == "square":
        wf = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave == "sweep_down":
        inst_freq = freq * (1.0 - 0.6 * (t / max(duration, 1e-6)))
        wf = np.sin(2 * np.pi * np.cumsum(inst_freq) / SAMPLE_RATE)
    else:
        wf = np.sin(2 * np.pi * freq * t)
    n_fade = min(n // 2, int(SAMPLE_RATE * fade))
    if n_fade > 0:
        env = np.ones(n)
        env[:n_fade] = np.linspace(0.0, 1.0, n_fade)
        env[-n_fade:] = np.linspace(1.0, 0.0, n_fade)
        wf = wf * env
    return wf * volume


def _noise(duration: float, volume: float = 0.4, fade: float = 0.01) -> np.ndarray:
    n = max(1, int(SAMPLE_RATE * duration))
    wf = np.random.uniform(-1.0, 1.0, n)
    n_fade = min(n // 2, int(SAMPLE_RATE * fade))
    if n_fade > 0:
        env = np.ones(n)
        env[:n_fade] = np.linspace(0.0, 1.0, n_fade)
        env[-n_fade:] = np.linspace(1.0, 0.0, n_fade)
        wf = wf * env
    return wf * volume


def _to_sound(wf_mono: np.ndarray) -> "pygame.mixer.Sound":
    stereo = np.column_stack([wf_mono, wf_mono])
    arr = np.ascontiguousarray((np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16))
    return pygame.sndarray.make_sound(arr)


class SoundBank:
    """صداهای بازی: سکه، برخورد، تصادف نهایی، پایان بازی و صدای موتور.

    همه با موج‌سازی ریاضی (numpy) تولید می‌شوند، بنابراین هیچ فایل صوتی
    اضافه‌ای لازم نیست. اگر کارت صدا در دسترس نباشد، بازی بی‌صدا اما
    بدون خطا اجرا می‌شود.
    """

    def __init__(self) -> None:
        self.available = False
        self.muted = False
        self.master_volume = 0.65
        self._engine_fraction = 0.0
        self.engine_low_channel = None
        self.engine_high_channel = None

        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
            self._build_sounds()
            self.engine_low_channel = self.engine_low.play(loops=-1)
            self.engine_high_channel = self.engine_high.play(loops=-1)
            if self.engine_low_channel:
                self.engine_low_channel.set_volume(0.0)
            if self.engine_high_channel:
                self.engine_high_channel.set_volume(0.0)
            self.available = True
        except Exception:
            self.available = False

    def _build_sounds(self) -> None:
        # سکه: دو نتِ کوتاه صعودی
        coin_wf = np.concatenate([
            _tone(988.0, 0.06, volume=0.5),
            _tone(1318.0, 0.10, volume=0.5),
        ])
        self.coin = _to_sound(coin_wf)

        # برخورد غیرمرگبار: ضربه کوتاه + نویز خفیف
        n = max(len(t1 := _tone(120.0, 0.12, volume=0.55)),
                len(t2 := _noise(0.09, volume=0.3)))
        t1 = np.pad(t1, (0, n - len(t1)))
        t2 = np.pad(t2, (0, n - len(t2)))
        self.hit = _to_sound(t1 + t2)

        # تصادف نهایی: نویز بلندتر + صدای فروریزش (سوییپ نزولی)
        n = max(len(c1 := _tone(260.0, 0.55, volume=0.45, wave="sweep_down")),
                len(c2 := _noise(0.45, volume=0.55)))
        c1 = np.pad(c1, (0, n - len(c1)))
        c2 = np.pad(c2, (0, n - len(c2)))
        self.crash = _to_sound(c1 + c2)

        # پایان بازی: آرپژ نزولی چهار نتی
        parts = []
        for i, freq in enumerate((523.0, 440.0, 349.0, 262.0)):
            parts.append(_tone(freq, 0.16, volume=0.42,
                                wave="square" if i % 2 == 0 else "sine"))
            parts.append(np.zeros(int(SAMPLE_RATE * 0.02)))
        self.gameover = _to_sound(np.concatenate(parts))

        # سکه جدید (رکورد): زنگ درخشان کوتاه
        self.highscore_jingle = _to_sound(np.concatenate([
            _tone(784.0, 0.08, volume=0.4),
            _tone(988.0, 0.08, volume=0.4),
            _tone(1318.0, 0.16, volume=0.45),
        ]))

        # صدای موتور: هارمونیک‌های یک فرکانس پایه، طوری که حلقه بی‌درز باشد
        self.engine_low = _to_sound(self._engine_wave(70.0, 0.5,
                                                        ((1, 1.0), (2, 0.5), (3, 0.25))))
        self.engine_high = _to_sound(self._engine_wave(150.0, 0.5,
                                                         ((1, 1.0), (2, 0.35))))

    @staticmethod
    def _engine_wave(fundamental: float, duration: float,
                      harmonics: tuple) -> np.ndarray:
        n = int(SAMPLE_RATE * duration)
        t = np.linspace(0.0, duration, n, endpoint=False)
        wf = np.zeros(n)
        for mult, amp in harmonics:
            wf += amp * np.sin(2 * np.pi * fundamental * mult * t)
        peak = np.max(np.abs(wf))
        if peak > 1e-6:
            wf /= peak
        return wf * 0.35

    def _vol(self, base: float) -> float:
        return 0.0 if self.muted else base * self.master_volume

    def update_engine(self, speed_fraction: float) -> None:
        if not self.available:
            return
        self._engine_fraction = max(0.0, min(1.0, speed_fraction))
        if self.engine_low_channel:
            self.engine_low_channel.set_volume(self._vol(0.16 + 0.24 * self._engine_fraction))
        if self.engine_high_channel:
            swell = max(0.0, self._engine_fraction - 0.35) / 0.65
            self.engine_high_channel.set_volume(self._vol(0.22 * swell))

    def stop_engine(self) -> None:
        if not self.available:
            return
        if self.engine_low_channel:
            self.engine_low_channel.set_volume(0.0)
        if self.engine_high_channel:
            self.engine_high_channel.set_volume(0.0)

    def play_coin(self) -> None:
        if self.available and not self.muted:
            self.coin.set_volume(self.master_volume)
            self.coin.play()

    def play_hit(self) -> None:
        if self.available and not self.muted:
            self.hit.set_volume(self.master_volume)
            self.hit.play()

    def play_crash(self) -> None:
        if self.available and not self.muted:
            self.crash.set_volume(self.master_volume)
            self.crash.play()

    def play_gameover(self) -> None:
        if self.available and not self.muted:
            self.gameover.set_volume(self.master_volume)
            self.gameover.play()

    def play_highscore(self) -> None:
        if self.available and not self.muted:
            self.highscore_jingle.set_volume(self.master_volume)
            self.highscore_jingle.play()

    def toggle_mute(self) -> None:
        self.muted = not self.muted
        self.update_engine(self._engine_fraction)


# ==================== HAND STEERING ENGINE ====================

class HandSteering:
    """Hybrid hand tracker: Motion + Skin + Zone-based control."""

    def __init__(self) -> None:
        self.capture: Optional[cv2.VideoCapture] = None
        self.last_frame: Optional[pygame.Surface] = None
        self.value = 0.0
        self.available = False
        self.calibration_frames = 0
        self.calibration_target = 45  # 1.5 ثانیه برای یادگیری پس‌زمینه
        self.calibrated = False
        self.hand_confidence = 0.0
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=25, detectShadows=False
        )

        for camera_index in (0, 1, 2):
            try:
                cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(camera_index)

                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self.capture = cap
                        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.available = True
                        break
                    else:
                        cap.release()
            except Exception:
                continue

    def update(self) -> Optional[float]:
        if not self.available or self.capture is None:
            return None

        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # حذف کامل بالای تصویر (منطقه صورت)
        roi_top = int(h * 0.30)
        roi = frame[roi_top:, :]

        # ==== مرحله ۱: تشخیص حرکت (حذف دیوار) ====
        fg_mask = self.bg_subtractor.apply(roi, learningRate=0.015)

        # ==== مرحله ۲: تشخیص رنگ پوست ====
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(
            ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127])
        )

        # ==== مرحله ۳: ترکیب دو ماسک (AND منطقی) ====
        combined = cv2.bitwise_and(fg_mask, skin_mask)

        # پاکسازی حرفه‌ای
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.dilate(combined, kernel, iterations=3)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # وضعیت کالیبراسیون
        if self.calibration_frames < self.calibration_target:
            self.calibration_frames += 1
            progress = int(self.calibration_frames / self.calibration_target * 100)
            
            # نمایش نوار کالیبراسیون
            cv2.rectangle(frame, (10, h - 40), (w - 10, h - 20), (40, 40, 40), -1)
            bar_w = int((w - 20) * (self.calibration_frames / self.calibration_target))
            cv2.rectangle(frame, (10, h - 40), (10 + bar_w, h - 20), (0, 200, 255), -1)
            cv2.putText(frame, f"Calibrating background... {progress}%", 
                        (15, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            self._render_preview(frame)
            return None
        
        self.calibrated = True

        # پیدا کردن دست
        best_contour = None
        max_score = 0
        
        for c in contours:
            area = cv2.contourArea(c)
            if 1200 < area < 40000:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = bw / bh if bh > 0 else 0
                
                if 0.3 <= aspect <= 2.8:
                    # امتیازدهی بر اساس اندازه و مرکزیت عمودی
                    score = area
                    if score > max_score:
                        max_score = score
                        best_contour = (c, bx, by, bw, bh)

        # منطقه‌های کنترل
        left_bound = int(w * 0.38)
        right_bound = int(w * 0.62)
        
        # رسم مناطق کنترل
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, roi_top), (left_bound, h), (255, 100, 0), -1)
        cv2.rectangle(overlay, (right_bound, roi_top), (w, h), (0, 255, 100), -1)
        cv2.rectangle(overlay, (left_bound, roi_top), (right_bound, h), (100, 100, 100), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        
        # خطوط راهنما
        cv2.line(frame, (left_bound, roi_top), (left_bound, h), (255, 150, 0), 2)
        cv2.line(frame, (right_bound, roi_top), (right_bound, h), (0, 255, 100), 2)
        cv2.line(frame, (0, roi_top), (w, roi_top), (200, 100, 100), 2)
        cv2.putText(frame, "FACE ZONE (Ignored)", (10, roi_top - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 100), 1)
        cv2.putText(frame, "LEFT", (left_bound // 2 - 20, roi_top + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2)
        cv2.putText(frame, "RIGHT", (right_bound + 30, roi_top + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 150), 2)

        target = 0.0
        if best_contour is not None:
            _, bx, by, bw, bh = best_contour
            center_x = bx + bw / 2.0
            actual_y = by + roi_top
            
            self.hand_confidence = min(1.0, self.hand_confidence + 0.15)

            if center_x < left_bound:
                ratio = (left_bound - center_x) / left_bound
                target = -min(1.0, ratio * 1.6)
            elif center_x > right_bound:
                ratio = (center_x - right_bound) / (w - right_bound)
                target = min(1.0, ratio * 1.6)
            else:
                target = 0.0

            # رادار روی دست
            cv2.rectangle(frame, (bx, actual_y), (bx + bw, actual_y + bh), (0, 255, 255), 3)
            cv2.circle(frame, (int(center_x), int(actual_y + bh / 2)), 10, (0, 0, 255), -1)
            cv2.circle(frame, (int(center_x), int(actual_y + bh / 2)), 14, (0, 255, 255), 2)
            cv2.putText(frame, f"HAND LOCKED", (bx, actual_y - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            self.hand_confidence = max(0.0, self.hand_confidence - 0.05)

        self.value += (target - self.value) * 0.28
        steering = max(-1.0, min(1.0, self.value))

        # نوار فرمان زیر تصویر
        cv2.rectangle(frame, (int(w * 0.15), h - 20), (int(w * 0.85), h - 8), (40, 40, 40), -1)
        bar_center = int(w * 0.5 + (self.value * (w * 0.34)))
        color = (0, 255, 0) if abs(self.value) < 0.3 else (0, 255, 255) if abs(self.value) < 0.7 else (0, 100, 255)
        cv2.circle(frame, (bar_center, h - 14), 8, color, -1)
        cv2.circle(frame, (int(w * 0.5), h - 14), 3, (200, 200, 200), -1)

        self._render_preview(frame)
        return steering if best_contour is not None else 0.0

    def _render_preview(self, frame) -> None:
        preview = cv2.resize(frame, (CAMERA_W, CAMERA_H))
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        self.last_frame = pygame.surfarray.make_surface(preview.swapaxes(0, 1))

    def close(self) -> None:
        self.available = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None


# ==================== GAME ENTITIES ====================

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: tuple[int, int, int]
    size: float


@dataclass
class Coin:
    x: float          # موقعیت جانبی روی جاده (-۱ تا ۱)
    z: float          # فاصله جلوی راننده
    rotation: float = 0.0


@dataclass
class Obstacle:
    x: float          # مرکز جانبی روی جاده (-۱ تا ۱)
    z: float          # فاصله جلوی راننده
    width: float = 58
    height: float = 84
    color: tuple[int, int, int] = (216, 67, 67)
    kind: str = "car"  # "car" or "truck"
    passed: bool = False

    @property
    def half_width_norm(self) -> float:
        return (self.width / 2) / ROAD_HALF


@dataclass
class RoadsideProp:
    x: float          # خارج از جاده (|x| > 1)
    z: float
    kind: str = "tree"  # "tree" | "post" | "sign"


# ==================== MAIN GAME CLASS ====================

class CarGame:
    def __init__(self) -> None:
        pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
        pygame.init()
        pygame.display.set_caption("🏁 Hand Drive PRO - Ultimate Racing")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        
        font_name = pygame.font.match_font("tahoma") or pygame.font.match_font("arial")
        self.font = pygame.font.Font(font_name, 26)
        self.big_font = pygame.font.Font(font_name, 64)
        self.medium_font = pygame.font.Font(font_name, 36)
        self.small_font = pygame.font.Font(font_name, 18)
        self.tiny_font = pygame.font.Font(font_name, 14)
        
        self.steering = HandSteering()
        self.sound = SoundBank()
        self.running = True
        self.shake_amount = 0.0
        self.highscore = self._load_highscore()

        # پیش‌رندر آسمان، چمن و مه افق
        self.sky_surface = self._build_sky()
        self.grass_surface = self._build_grass()
        self.fog_surface = self._build_fog()
        self.clouds = [
            (random.uniform(0, WIDTH), random.uniform(30, HORIZON_Y - 110),
             random.uniform(90, 170))
            for _ in range(4)
        ]

        self.reset()

    # ---------- پیش‌رندر محیط ----------

    def _mountain_points(self, base_h: int, amp: int, seed_offset: int) -> list[tuple[int, int]]:
        rng = random.Random(seed_offset)
        pts: list[tuple[int, int]] = [(-20, HORIZON_Y + 4)]
        x = -20
        peak = True
        while x < WIDTH + 40:
            x += rng.randint(90, 200)
            h = base_h + rng.randint(0, amp) if peak else base_h // 2
            peak = not peak
            pts.append((min(x, WIDTH + 20), HORIZON_Y - h))
        pts.append((WIDTH + 20, HORIZON_Y + 4))
        return pts

    def _build_sky(self) -> pygame.Surface:
        sky = pygame.Surface((WIDTH, HORIZON_Y))
        top, bottom = (36, 46, 96), (242, 168, 118)
        for i in range(HORIZON_Y):
            t = i / HORIZON_Y
            color = tuple(int(a + (b - a) * t) for a, b in zip(top, bottom))
            pygame.draw.line(sky, color, (0, i), (WIDTH, i))
        # خورشید
        pygame.draw.circle(sky, (255, 214, 140), (int(WIDTH * 0.72), HORIZON_Y - 70), 52)
        pygame.draw.circle(sky, (255, 236, 180), (int(WIDTH * 0.72), HORIZON_Y - 70), 38)
        # کوه‌های دوردست
        pygame.draw.polygon(sky, (94, 84, 128), self._mountain_points(55, 130, 11))
        pygame.draw.polygon(sky, (72, 66, 108), self._mountain_points(90, 100, 77))
        return sky

    def _build_grass(self) -> pygame.Surface:
        grass = pygame.Surface((WIDTH, VIEW_H))
        top, bottom = (150, 155, 100), (46, 125, 55)
        for i in range(VIEW_H):
            t = i / VIEW_H
            color = tuple(int(a + (b - a) * t) for a, b in zip(top, bottom))
            pygame.draw.line(grass, color, (0, i), (WIDTH, i))
        return grass

    def _build_fog(self) -> pygame.Surface:
        fog = pygame.Surface((WIDTH, 130), pygame.SRCALPHA)
        for i in range(130):
            alpha = int(150 * (1.0 - i / 130.0) ** 1.4)
            pygame.draw.line(fog, (215, 178, 150, alpha), (0, i), (WIDTH, i))
        return fog

    # ---------- ذخیره رکورد ----------

    def _load_highscore(self) -> int:
        try:
            if os.path.exists(HIGHSCORE_FILE):
                with open(HIGHSCORE_FILE, "r") as f:
                    return int(f.read().strip())
        except Exception:
            pass
        return 0

    def _save_highscore(self) -> None:
        try:
            with open(HIGHSCORE_FILE, "w") as f:
                f.write(str(self.highscore))
        except Exception:
            pass

    # ---------- پرسپکتیو ----------

    def _project(self, x_norm: float, z: float) -> tuple[float, float, float]:
        """تبدیل مختصات دنیا (جانبی نرمال‌شده، فاصله z) به صفحه نمایش."""
        z = max(z, -40.0)
        scale = NEAR / (NEAR + z)
        y = HORIZON_Y + VIEW_H * scale
        x = WIDTH / 2 + x_norm * ROAD_HALF * scale
        return x, y, scale

    def _player_px(self) -> float:
        return ((self.car_x + CAR_W / 2) - (ROAD_LEFT + ROAD_RIGHT) / 2) / ROAD_HALF

    def _car_ground(self) -> tuple[float, float]:
        sx, gy, _ = self._project(self._player_px(), PLAYER_Z)
        return sx, gy

    def _car_screen_center(self) -> tuple[float, float]:
        sx, gy = self._car_ground()
        return sx, gy - CAR_H * 0.55

    # ---------- وضعیت بازی ----------

    def reset(self) -> None:
        self.car_x = (ROAD_LEFT + ROAD_RIGHT - CAR_W) / 2
        self.car_tilt = 0.0
        self.obstacles: list[Obstacle] = []
        self.coins: list[Coin] = []
        self.particles: list[Particle] = []
        self.props: list[RoadsideProp] = self._make_props()
        self.road_offset = 0.0
        self.spawn_timer = 1.0
        self.coin_timer = 2.0
        self.elapsed = 0.0
        self.score = 0
        self.coins_collected = 0
        self.game_over = False
        self.keyboard_direction = 0
        self.speed = 240.0
        self.health = 100
        self.explosion_timer = 0.0
        self.lives = MAX_LIVES
        self.invincible_timer = 0.0
        self.hit_flash = 0.0
        self.gameover_jingle_timer = 0.5
        self.gameover_jingle_played = False
        self.new_highscore = False
        if hasattr(self, "sound"):
            self.sound.stop_engine()

    def _make_props(self) -> list[RoadsideProp]:
        kinds = ["tree", "tree", "post", "tree", "sign", "tree"]
        return [
            RoadsideProp(
                x=random.choice((-1.0, 1.0)) * random.uniform(1.12, 1.85),
                z=40.0 + i * 46.0 + random.uniform(0.0, 24.0),
                kind=kinds[i % len(kinds)],
            )
            for i in range(18)
        ]

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_m:
                    self.sound.toggle_mute()

        keys = pygame.key.get_pressed()
        self.keyboard_direction = int(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - int(
            keys[pygame.K_LEFT] or keys[pygame.K_a]
        )

    # ---------- اسپاون ----------

    def spawn_obstacle(self) -> None:
        kind = random.choices(["car", "truck"], weights=[75, 25])[0]
        if kind == "truck":
            width, height = 72, 112
            colors = [(150, 80, 40), (80, 80, 100), (100, 50, 50)]
        else:
            width, height = 58, 84
            colors = [(216, 67, 67), (244, 143, 46), (160, 82, 190), (67, 180, 216), (240, 200, 60)]

        x = random.choice(LANES) + random.uniform(-0.07, 0.07)

        # انصاف: همیشه حداقل یک لاین باز بماند
        far = [o for o in self.obstacles if o.z > 430]
        blocked = sum(
            1 for lane in LANES if any(abs(o.x - lane) < 0.42 for o in far)
        )
        if blocked >= 2:
            return
        if any(abs(o.x - x) < 0.45 and o.z > SPAWN_Z - 200 for o in self.obstacles):
            return

        self.obstacles.append(Obstacle(
            x=x, z=SPAWN_Z, width=width, height=height,
            color=random.choice(colors), kind=kind
        ))

    def spawn_coin(self) -> None:
        self.coins.append(Coin(x=random.uniform(-0.8, 0.8), z=SPAWN_Z - 40))

    def spawn_particle(self, x: float, y: float, color: tuple, count: int = 5, spread: float = 3.0) -> None:
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(50, 200) * spread / 3.0
            life = random.uniform(0.3, 0.9)
            self.particles.append(Particle(
                x=x, y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                life=life, max_life=life,
                color=color, size=random.uniform(3, 7)
            ))

    # ---------- بروزرسانی ----------

    def update(self, dt: float) -> None:
        camera_steering = self.steering.update()
        
        if self.game_over:
            self.explosion_timer += dt
            if not self.gameover_jingle_played:
                self.gameover_jingle_timer -= dt
                if self.gameover_jingle_timer <= 0:
                    if self.new_highscore:
                        self.sound.play_highscore()
                    else:
                        self.sound.play_gameover()
                    self.gameover_jingle_played = True
            if self.explosion_timer < 0.3:
                car_center_x, car_center_y = self._car_screen_center()
                self.spawn_particle(car_center_x, car_center_y, 
                                    random.choice([(255, 100, 0), (255, 200, 0), (255, 50, 0)]), 
                                    count=3, spread=5.0)
            self._update_particles(dt)
            self.shake_amount = max(0, self.shake_amount - dt * 30)
            return

        self.elapsed += dt
        self.speed = min(500.0, 240.0 + self.elapsed * 8.0)
        self.road_offset = (self.road_offset + self.speed * dt) % DASH_PERIOD
        self.sound.update_engine((self.speed - 240.0) / (500.0 - 240.0))

        # کنترل ماشین
        if camera_steering is not None and abs(camera_steering) > 0.03:
            direction = camera_steering
        else:
            direction = float(self.keyboard_direction) * 0.9

        move = direction * (430.0 if camera_steering is not None else 340.0) * dt
        self.car_x += move
        self.car_x = max(ROAD_LEFT + 5, min(ROAD_RIGHT - CAR_W - 5, self.car_x))
        
        # افکت چرخش ماشین
        self.car_tilt += (direction * 15 - self.car_tilt) * 0.15

        # افکت دود
        if random.random() < 0.7:
            back_x, back_y = self._car_ground()
            self.spawn_particle(back_x + random.uniform(-12, 12), back_y - 6,
                                (200, 200, 200), count=1, spread=1.5)

        # اسپاون موانع
        self.spawn_timer -= dt
        if self.spawn_timer <= 0:
            self.spawn_obstacle()
            self.spawn_timer = max(0.45, 1.1 - self.elapsed * 0.009)

        # اسپاون سکه
        self.coin_timer -= dt
        if self.coin_timer <= 0:
            self.spawn_coin()
            self.coin_timer = random.uniform(1.5, 3.5)

        # پیشروی مناظر کنار جاده
        for prop in self.props:
            prop.z -= self.speed * dt
            if prop.z < -25:
                prop.z += 800.0 + random.uniform(0.0, 140.0)
                prop.x = random.choice((-1.0, 1.0)) * random.uniform(1.12, 1.85)

        # شمارش معکوس زمان بی‌آسیبی و افکت چشمک برخورد
        if self.invincible_timer > 0:
            self.invincible_timer = max(0.0, self.invincible_timer - dt)
        if self.hit_flash > 0:
            self.hit_flash = max(0.0, self.hit_flash - dt)

        # بروزرسانی موانع (موانع از افق به سمت راننده نزدیک می‌شوند)
        car_px = self._player_px()
        remaining_obs = []
        for obs in self.obstacles:
            obs.z -= self.speed * dt
            if obs.z <= 0 and not obs.passed:
                obs.passed = True
                self.score += 10
            if obs.z > -70:
                in_window = -2.0 <= obs.z <= 30.0
                lateral = abs(obs.x - car_px) < obs.half_width_norm + (CAR_W / 2) / ROAD_HALF - 0.02
                if in_window and lateral and self.invincible_timer <= 0:
                    self.lives -= 1
                    self.hit_flash = 0.6
                    self.shake_amount = 22

                    car_center_x, car_center_y = self._car_screen_center()
                    if self.lives > 0:
                        # ضربه — نه نابودی؛ جرقه و دود سفید/زرد + بی‌آسیبی موقت
                        self.invincible_timer = INVINCIBLE_TIME
                        self.sound.play_hit()
                        self.spawn_particle(car_center_x, car_center_y,
                                            random.choice([(255, 230, 120), (255, 255, 255), (255, 160, 60)]),
                                            count=16, spread=4.0)
                    else:
                        self.invincible_timer = 0.0
                        self.sound.play_crash()
                        self.sound.stop_engine()
                        # آخرین قلب مصرف شد → پایان بازی
                        self.game_over = True
                        self.health = 0
                        self.shake_amount = 30
                        self.spawn_particle(car_center_x, car_center_y,
                                            random.choice([(255, 100, 0), (255, 200, 0), (255, 50, 0)]),
                                            count=24, spread=5.0)
                        if self.score + self.coins_collected * 25 > self.highscore:
                            self.highscore = self.score + self.coins_collected * 25
                            self._save_highscore()
                            self.new_highscore = True

                    # مانع برخوردکننده حذف می‌شود تا بلافاصله دوباره ضربه نزند
                    continue
                remaining_obs.append(obs)
        self.obstacles = remaining_obs

        # بروزرسانی سکه‌ها
        remaining_coins = []
        for coin in self.coins:
            coin.z -= self.speed * dt
            coin.rotation += dt * 5
            if coin.z < -50:
                continue
            if -2.0 <= coin.z <= 26.0 and abs(coin.x - car_px) < 0.16:
                self.coins_collected += 1
                self.sound.play_coin()
                sx, sy, _ = self._project(coin.x, max(coin.z, 0.0))
                self.spawn_particle(sx, sy - 10, (255, 220, 60), count=15, spread=2.0)
                continue
            remaining_coins.append(coin)
        self.coins = remaining_coins

        self._update_particles(dt)
        self.shake_amount = max(0, self.shake_amount - dt * 30)

    def _update_particles(self, dt: float) -> None:
        for p in self.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vy += 200 * dt
            p.life -= dt
        self.particles = [p for p in self.particles if p.life > 0]

    # ---------- رسم ----------

    def _road_quad(self, x1: float, x2: float, z1: float, z2: float,
                   color: tuple, offset: tuple) -> None:
        ox, oy = offset
        ax, ay, _ = self._project(x1, z1)
        bx, by, _ = self._project(x2, z1)
        cx, cy, _ = self._project(x2, z2)
        dx, dy, _ = self._project(x1, z2)
        pygame.draw.polygon(self.screen, color, [
            (ax + ox, ay + oy), (bx + ox, by + oy),
            (cx + ox, cy + oy), (dx + ox, dy + oy),
        ])

    def _draw_prop(self, prop: RoadsideProp, x: float, y: float, sc: float) -> None:
        if prop.kind == "tree":
            trunk_w = max(2, int(10 * sc))
            trunk_h = max(4, int(34 * sc))
            pygame.draw.rect(self.screen, (110, 75, 45),
                             (round(x - trunk_w / 2), round(y - trunk_h), trunk_w, trunk_h))
            r1 = max(3, int(26 * sc))
            pygame.draw.circle(self.screen, (35, 105, 45),
                               (round(x), round(y - trunk_h - r1 * 0.6)), r1)
            r2 = max(2, int(17 * sc))
            pygame.draw.circle(self.screen, (52, 128, 56),
                               (round(x - r1 * 0.45), round(y - trunk_h - r1 * 0.25)), r2)
        elif prop.kind == "post":
            pw, ph = max(1, int(5 * sc)), max(4, int(20 * sc))
            pygame.draw.rect(self.screen, (235, 235, 235),
                             (round(x - pw / 2), round(y - ph), pw, ph))
            pygame.draw.rect(self.screen, (200, 40, 40),
                             (round(x - pw / 2), round(y - ph), pw, max(2, ph // 3)))
        else:  # sign
            pole_w, pole_h = max(1, int(4 * sc)), max(5, int(26 * sc))
            pygame.draw.rect(self.screen, (120, 120, 125),
                             (round(x - pole_w / 2), round(y - pole_h), pole_w, pole_h))
            bw, bh = max(4, int(30 * sc)), max(3, int(18 * sc))
            board = pygame.Rect(round(x - bw / 2), round(y - pole_h - bh), bw, bh)
            pygame.draw.rect(self.screen, (30, 90, 190), board)
            pygame.draw.rect(self.screen, (240, 240, 240), board, max(1, int(2 * sc)))

    def draw_car(self, offset: tuple = (0, 0)) -> None:
        ox, oy = offset
        sx, gy = self._car_ground()
        x, y = round(sx) + ox, round(gy) + oy

        # افکت چشمک‌زدن در زمان بی‌آسیبی موقت (بعد از برخورد با یک قلب باقی‌مانده)
        if self.invincible_timer > 0:
            blink_on = int(self.invincible_timer * 14) % 2 == 0
            if not blink_on:
                return

        # خطوط سرعت پشت ماشین (حس شتاب)
        if self.speed > 260:
            streak_alpha = min(140, int((self.speed - 240) * 0.5))
            streaks = pygame.Surface((CAR_W + 90, 46), pygame.SRCALPHA)
            for i, sxk in enumerate((-1, -0.55, 0.55, 1)):
                lx = streaks.get_width() / 2 + sxk * (CAR_W / 2 + 24)
                pygame.draw.line(streaks, (255, 255, 255, streak_alpha),
                                  (lx, 8), (lx + random.uniform(-4, 4), 40), 3)
            self.screen.blit(streaks, (x - streaks.get_width() // 2, y - 6))

        # سایه ماشین (نرم‌تر با گرادیان چگالی)
        shadow = pygame.Surface((CAR_W + 30, 30), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 95), (3, 6, CAR_W + 24, 20))
        pygame.draw.ellipse(shadow, (0, 0, 0, 55), (0, 0, CAR_W + 30, 30))
        self.screen.blit(shadow, (x - (CAR_W + 30) // 2, y - 20))

        # نمای پشت ماشین (ماشین ما) — با گرادیان نور، آینه‌های جانبی و اسپویلر
        pad = 18
        car_surface = pygame.Surface((CAR_W + pad * 2, CAR_H + pad * 2), pygame.SRCALPHA)
        cx, cy = pad, pad

        # رنگ بدنه: در حالت آسیب‌دیدگی موقت به سفید/قرمز می‌زند
        base_color = (200, 40, 40)
        if self.hit_flash > 0:
            flick = int(self.hit_flash * 20) % 2 == 0
            base_color = (255, 255, 255) if flick else (255, 90, 60)

        body_rect = pygame.Rect(cx, cy + 6, CAR_W, CAR_H - 6)

        # بدنهٔ اصلی با گرادیان عمودی (روشن بالا، تیره پایین) برای حس حجم و نور
        body_surf = pygame.Surface(body_rect.size, pygame.SRCALPHA)
        top_c = tuple(min(255, c + 55) for c in base_color)
        bottom_c = tuple(max(0, c - 55) for c in base_color)
        bh = body_rect.height
        for i in range(bh):
            t = i / max(1, bh - 1)
            col = tuple(int(a + (b - a) * t) for a, b in zip(top_c, bottom_c))
            pygame.draw.line(body_surf, col, (0, i), (body_rect.width, i))
        mask = pygame.Surface(body_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=14)
        body_surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        car_surface.blit(body_surf, body_rect.topleft)
        pygame.draw.rect(car_surface, tuple(max(0, c - 70) for c in base_color), body_rect, 2, border_radius=14)

        # خط برجستهٔ نور روی بدنه (highlight ظریف)
        pygame.draw.line(car_surface, (255, 255, 255, 90),
                          (cx + 8, cy + 14), (cx + CAR_W - 8, cy + 14), 2)

        # سقف
        roof_top = tuple(min(255, c + 30) for c in base_color)
        pygame.draw.rect(car_surface, roof_top, (cx + 4, cy + 6, CAR_W - 8, 16), border_radius=8)
        pygame.draw.rect(car_surface, tuple(max(0, c - 60) for c in base_color),
                          (cx + 4, cy + 6, CAR_W - 8, 16), 1, border_radius=8)

        # آینه‌های جانبی
        pygame.draw.rect(car_surface, tuple(max(0, c - 30) for c in base_color),
                          (cx - 6, cy + 20, 6, 10), border_radius=2)
        pygame.draw.rect(car_surface, tuple(max(0, c - 30) for c in base_color),
                          (cx + CAR_W, cy + 20, 6, 10), border_radius=2)

        # شیشه عقب با انعکاس گرادیانی آسمان
        glass_rect = pygame.Rect(cx + 8, cy + 26, CAR_W - 16, 24)
        glass_surf = pygame.Surface(glass_rect.size, pygame.SRCALPHA)
        glass_top, glass_bottom = (110, 150, 190), (35, 55, 85)
        gh = glass_rect.height
        for i in range(gh):
            t = i / max(1, gh - 1)
            col = tuple(int(a + (b - a) * t) for a, b in zip(glass_top, glass_bottom))
            pygame.draw.line(glass_surf, col, (0, i), (glass_rect.width, i))
        gmask = pygame.Surface(glass_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(gmask, (255, 255, 255, 255), gmask.get_rect(), border_radius=6)
        glass_surf.blit(gmask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        car_surface.blit(glass_surf, glass_rect.topleft)
        pygame.draw.rect(car_surface, (25, 45, 70), glass_rect, 2, border_radius=6)
        pygame.draw.line(car_surface, (220, 235, 250, 140),
                          (glass_rect.x + 4, glass_rect.y + 3),
                          (glass_rect.x + glass_rect.width - 10, glass_rect.y + 3), 2)

        # اسپویلر روی سقف صندوق
        spoiler_y = cy + CAR_H - 34
        pygame.draw.rect(car_surface, (40, 40, 40), (cx - 2, spoiler_y, 5, 10), border_radius=2)
        pygame.draw.rect(car_surface, (40, 40, 40), (cx + CAR_W - 3, spoiler_y, 5, 10), border_radius=2)
        pygame.draw.rect(car_surface, (30, 30, 30), (cx - 3, spoiler_y - 4, CAR_W + 6, 5), border_radius=3)

        # چراغ‌های عقب با هالهٔ درخشان (glow)
        for lx in (cx + 5, cx + CAR_W - 20):
            glow = pygame.Surface((34, 26), pygame.SRCALPHA)
            pygame.draw.ellipse(glow, (255, 50, 30, 70), glow.get_rect())
            car_surface.blit(glow, (lx - 10, cy + CAR_H - 40), special_flags=pygame.BLEND_RGBA_ADD)
            pygame.draw.rect(car_surface, (255, 40, 30), (lx, cy + CAR_H - 30, 15, 8), border_radius=3)
            pygame.draw.rect(car_surface, (255, 160, 140), (lx + 2, cy + CAR_H - 29, 11, 3), border_radius=2)

        # سپر و پلاک
        pygame.draw.rect(car_surface, (70, 70, 80), (cx + 3, cy + CAR_H - 18, CAR_W - 6, 9), border_radius=3)
        pygame.draw.rect(car_surface, (235, 235, 225), (cx + CAR_W // 2 - 9, cy + CAR_H - 16, 18, 6), border_radius=1)

        # لاستیک‌ها با رینگ نقره‌ای
        for wx in (cx - 4, cx + CAR_W - 5):
            for wy in (cy + 22, cy + CAR_H - 40):
                pygame.draw.rect(car_surface, (20, 20, 24), (wx, wy, 9, 22), border_radius=3)
                pygame.draw.circle(car_surface, (185, 185, 190), (wx + 4, wy + 11), 3)

        if abs(self.car_tilt) > 0.5:
            car_surface = pygame.transform.rotate(car_surface, -self.car_tilt)
        rect = car_surface.get_rect(center=(x, y - CAR_H // 2 - 6))
        self.screen.blit(car_surface, rect)

    def draw_obstacle(self, obs: Obstacle, offset: tuple) -> None:
        ox, oy = offset
        sx, gy, sc = self._project(obs.x, obs.z)
        w = max(6.0, obs.width * sc)
        h = max(8.0, obs.height * sc)
        rect = pygame.Rect(round(sx - w / 2 + ox), round(gy - h + oy), round(w), round(h))

        # سایه
        shadow = pygame.Surface((int(w + 12 * sc), max(4, int(10 * sc))), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 90), shadow.get_rect())
        self.screen.blit(shadow, (rect.x - int(6 * sc), rect.y + rect.height - max(2, int(4 * sc))))

        # بدنه (ماشین‌ها از پشت دیده می‌شوند چون ما دنبالشان می‌رویم)
        radius = max(4, int(10 * sc))
        pygame.draw.rect(self.screen, obs.color, rect, border_radius=radius)
        darker = tuple(max(0, c - 40) for c in obs.color)
        lighter = tuple(min(255, c + 30) for c in obs.color)
        pygame.draw.rect(self.screen, darker, rect, 2, border_radius=radius)
        # سقف روشن‌تر
        pygame.draw.rect(self.screen, lighter,
                         (rect.x + int(3 * sc), rect.y + int(2 * sc),
                          rect.width - int(6 * sc), max(3, int(10 * sc))),
                         border_radius=max(3, int(6 * sc)))

        if obs.kind == "truck":
            # خطوط کارگو
            for i in range(1, 3):
                lx = rect.x + int(rect.width * i / 3)
                pygame.draw.line(self.screen, darker,
                                 (lx, rect.y + int(6 * sc)),
                                 (lx, rect.bottom - int(8 * sc)), max(1, int(2 * sc)))
        else:
            # شیشه عقب
            win = pygame.Rect(round(rect.x + w * 0.16), round(rect.y + int(12 * sc)),
                              round(w * 0.68), max(3, int(15 * sc)))
            pygame.draw.rect(self.screen, (140, 170, 195), win, border_radius=max(2, int(4 * sc)))

        # چراغ‌های عقب
        light_r = max(2, int(4 * sc))
        pygame.draw.circle(self.screen, (255, 45, 35),
                           (rect.x + int(w * 0.2), rect.y + rect.height - int(9 * sc)), light_r)
        pygame.draw.circle(self.screen, (255, 45, 35),
                           (rect.x + int(w * 0.8), rect.y + rect.height - int(9 * sc)), light_r)

        # لاستیک‌ها
        tw, th = max(2, int(6 * sc)), max(3, int(14 * sc))
        pygame.draw.rect(self.screen, (25, 25, 30),
                         (rect.x - int(3 * sc), rect.y + rect.height - int(24 * sc), tw, th), border_radius=2)
        pygame.draw.rect(self.screen, (25, 25, 30),
                         (rect.right - tw + int(3 * sc), rect.y + rect.height - int(24 * sc), tw, th), border_radius=2)

    def draw_coin(self, coin: Coin, offset: tuple) -> None:
        ox, oy = offset
        sx, sy, sc = self._project(coin.x, coin.z)
        width_factor = abs(math.cos(coin.rotation))
        cw = max(3, int(26 * sc * width_factor))
        ch = max(3, int(26 * sc))
        rect = pygame.Rect(round(sx - cw / 2 + ox), round(sy - ch - int(8 * sc) + oy), cw, ch)

        pygame.draw.ellipse(self.screen, (200, 160, 20), rect)
        pygame.draw.ellipse(self.screen, (255, 220, 60),
                            rect.inflate(-max(1, int(2 * sc)), -max(1, int(2 * sc))))
        pygame.draw.ellipse(self.screen, (255, 240, 120),
                            rect.inflate(-max(2, int(6 * sc)), -max(2, int(6 * sc))))

    def draw_road(self, offset: tuple) -> None:
        ox, oy = offset

        # آسمان، خورشید و کوه‌ها
        self.screen.blit(self.sky_surface, (0, 0))

        # ابرهای متحرک
        for base_x, base_y, cw in self.clouds:
            cx = (base_x - self.elapsed * 9.0) % (WIDTH + 240) - 120
            pygame.draw.ellipse(self.screen, (235, 238, 245),
                                (int(cx) + ox, int(base_y) + oy, int(cw), int(cw * 0.35)))

        # زمین چمن
        self.screen.blit(self.grass_surface, (0, HORIZON_Y))

        # نوارهای چمن متحرک (حس سرعت)
        stripe_period = 180.0
        phase = self.road_offset % stripe_period
        for base in range(-180, 1200, 180):
            z1 = base + phase
            z2 = z1 + 90.0
            if z2 <= 0:
                continue
            _, y1, _ = self._project(0.0, max(z1, 0.0))
            _, y2, _ = self._project(0.0, z2)
            y1 = min(max(y1, HORIZON_Y), HEIGHT)
            y2 = min(max(y2, HORIZON_Y), HEIGHT)
            if y1 - y2 < 1:
                continue
            pygame.draw.rect(self.screen, (48, 122, 54),
                             (0, round(y2 + oy), WIDTH, round(y1 - y2)))

        # شانه خاکی
        self._road_quad(-1.10, -1.0, 0.0, 900.0, (200, 180, 130), offset)
        self._road_quad(1.0, 1.10, 0.0, 900.0, (200, 180, 130), offset)

        # آسفالت جاده (همگرا به نقطه گریز)
        self._road_quad(-1.0, 1.0, 0.0, 900.0, (45, 45, 55), offset)

        # خطوط زرد کناره
        self._road_quad(-0.985, -0.965, 0.0, 900.0, (240, 210, 90), offset)
        self._road_quad(0.965, 0.985, 0.0, 900.0, (240, 210, 90), offset)

        # خط‌چین لاین‌ها
        for lane_x in (-1.0 / 3.0, 1.0 / 3.0):
            phase = self.road_offset % DASH_PERIOD
            for base in range(0, 1000, int(DASH_PERIOD)):
                z1 = base - phase
                z2 = z1 + 35.0
                if z2 <= 0:
                    continue
                self._road_quad(lane_x - 0.011, lane_x + 0.011,
                                max(z1, 0.0), z2, (225, 225, 225), offset)

        # مناظر کنار جاده
        for prop in self.props:
            sx, sy, sc = self._project(prop.x, prop.z)
            if sc <= 0 or sx < -60 or sx > WIDTH + 60 or sy > HEIGHT + 40:
                continue
            self._draw_prop(prop, sx + ox, sy + oy, sc)

    def draw_particles(self, offset: tuple) -> None:
        ox, oy = offset
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            size = p.size * (p.life / p.max_life)
            if size < 1:
                continue
            surf = pygame.Surface((int(size * 2), int(size * 2)), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*p.color, alpha), (int(size), int(size)), int(size))
            self.screen.blit(surf, (p.x - size + ox, p.y - size + oy))

    def _draw_heart(self, cx: float, cy: float, size: float, filled: bool) -> None:
        """رسم یک آیکون قلب. filled=True یعنی جان باقی‌مانده، در غیر این صورت خالی/خاکستری."""
        color = (235, 45, 65) if filled else (70, 70, 78)
        outline = (140, 15, 30) if filled else (30, 30, 35)
        r = size * 0.32
        pygame.draw.circle(self.screen, color, (round(cx - r), round(cy - r * 0.55)), round(r))
        pygame.draw.circle(self.screen, color, (round(cx + r), round(cy - r * 0.55)), round(r))
        points = [
            (cx - size * 0.62, cy - r * 0.35),
            (cx, cy + size * 0.55),
            (cx + size * 0.62, cy - r * 0.35),
        ]
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.circle(self.screen, outline, (round(cx - r), round(cy - r * 0.55)), round(r), 1)
        pygame.draw.circle(self.screen, outline, (round(cx + r), round(cy - r * 0.55)), round(r), 1)
        pygame.draw.polygon(self.screen, outline, points, 1)
        if filled:
            pygame.draw.circle(self.screen, (255, 200, 205), (round(cx - r * 1.2), round(cy - r * 0.9)), max(1, round(r * 0.28)))

    def draw_hud(self) -> None:
        # پنل چپ بالا
        panel_h = 158
        panel = pygame.Surface((240, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 140))
        pygame.draw.rect(panel, (255, 255, 255, 80), (0, 0, 240, panel_h), 2, border_radius=10)
        self.screen.blit(panel, (10, 10))

        # قلب‌های جان (سه قلب — واکنش‌گر به self.lives)
        for i in range(MAX_LIVES):
            self._draw_heart(38 + i * 34, 32, 26, filled=i < self.lives)

        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (25, 50))
        
        coin_text = self.font.render(f"Coins: {self.coins_collected}", True, (255, 220, 60))
        self.screen.blit(coin_text, (25, 80))
        pygame.draw.circle(self.screen, (255, 220, 60), (200, 90), 10)
        pygame.draw.circle(self.screen, (200, 160, 20), (200, 90), 10, 2)
        
        # سرعت
        speed_kmh = int(self.speed * 0.5)
        speed_text = self.small_font.render(f"Speed: {speed_kmh} km/h", True, (100, 255, 200))
        self.screen.blit(speed_text, (25, 113))
        
        # رکورد
        hs_text = self.small_font.render(f"High Score: {self.highscore}", True, (255, 180, 100))
        self.screen.blit(hs_text, (25, 138))
        
        # وضعیت دوربین
        if self.steering.available:
            if self.steering.calibrated:
                status = "🎥 HAND CONTROL ACTIVE"
                color = (100, 255, 100)
            else:
                status = "⏳ Calibrating..."
                color = (255, 200, 100)
        else:
            status = "⌨️ KEYBOARD MODE (A/D)"
            color = (255, 150, 150)
        
        status_bg = pygame.Surface((320, 32), pygame.SRCALPHA)
        status_bg.fill((0, 0, 0, 150))
        self.screen.blit(status_bg, (10, HEIGHT - 42))
        status_text = self.small_font.render(status, True, color)
        self.screen.blit(status_text, (20, HEIGHT - 35))

        # نشانگر وضعیت صدا (M برای خاموش/روشن کردن)
        sound_label = "🔇 Muted (M)" if self.sound.muted else "🔊 Sound (M)"
        sound_color = (170, 170, 170) if self.sound.muted else (120, 220, 255)
        sound_bg = pygame.Surface((150, 32), pygame.SRCALPHA)
        sound_bg.fill((0, 0, 0, 150))
        self.screen.blit(sound_bg, (WIDTH - 160, HEIGHT - 42))
        sound_text = self.small_font.render(sound_label, True, sound_color)
        self.screen.blit(sound_text, (WIDTH - 150, HEIGHT - 35))
        
        # نشانگر نوار قدرت دست
        if self.steering.available and self.steering.calibrated:
            conf_x, conf_y = 340, HEIGHT - 35
            pygame.draw.rect(self.screen, (60, 60, 60), (conf_x, conf_y, 150, 18), border_radius=9)
            conf_w = int(150 * self.steering.hand_confidence)
            conf_color = (0, 255, 100) if self.steering.hand_confidence > 0.5 else (255, 200, 0)
            pygame.draw.rect(self.screen, conf_color, (conf_x, conf_y, conf_w, 18), border_radius=9)
            pygame.draw.rect(self.screen, (255, 255, 255), (conf_x, conf_y, 150, 18), 1, border_radius=9)

    def draw_camera_preview(self) -> None:
        if self.steering.last_frame is None:
            return
        
        px, py = WIDTH - CAMERA_W - 20, 20
        
        # سایه پشت کادر
        shadow = pygame.Surface((CAMERA_W + 20, CAMERA_H + 20), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 100), (0, 0, CAMERA_W + 20, CAMERA_H + 20), border_radius=12)
        self.screen.blit(shadow, (px - 5, py - 5))
        
        self.screen.blit(self.steering.last_frame, (px, py))
        pygame.draw.rect(self.screen, (0, 255, 200), (px, py, CAMERA_W, CAMERA_H), 3, border_radius=4)
        
        label = self.small_font.render("🎯 HAND RADAR", True, (0, 255, 200))
        self.screen.blit(label, (px, py + CAMERA_H + 5))

    def draw(self) -> None:
        # لرزش صفحه
        shake_x = random.uniform(-self.shake_amount, self.shake_amount) if self.shake_amount > 0 else 0
        shake_y = random.uniform(-self.shake_amount, self.shake_amount) if self.shake_amount > 0 else 0
        offset = (shake_x, shake_y)
        
        self.draw_road(offset)
        
        # سکه‌ها و موانع — دورترها اول رسم می‌شوند تا جلوی هم درست دیده شوند
        for kind, ent in sorted(
            [("o", o) for o in self.obstacles] + [("c", c) for c in self.coins],
            key=lambda t: t[1].z,
            reverse=True,
        ):
            if kind == "o":
                self.draw_obstacle(ent, offset)
            else:
                self.draw_coin(ent, offset)
        
        # مه افق — موانع دور را محو می‌کند
        self.screen.blit(self.fog_surface, (0, HORIZON_Y - 26))
        
        # ماشین بازیکن (اگر منفجر نشده)
        if not self.game_over or self.explosion_timer < 0.5:
            self.draw_car(offset)
        
        # ذرات
        self.draw_particles(offset)
        
        # HUD
        self.draw_hud()
        self.draw_camera_preview()
        
        # صفحه پایان بازی
        if self.game_over and self.explosion_timer > 0.5:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            # پنل مرکزی
            panel_w, panel_h = 500, 320
            panel_x = (WIDTH - panel_w) // 2
            panel_y = (HEIGHT - panel_h) // 2
            
            pygame.draw.rect(self.screen, (30, 30, 40), (panel_x, panel_y, panel_w, panel_h), border_radius=20)
            pygame.draw.rect(self.screen, (200, 50, 50), (panel_x, panel_y, panel_w, panel_h), 4, border_radius=20)

            # هر سه قلب مصرف شده — نمایش صریح کلمهٔ GAME OVER
            title = self.big_font.render("GAME OVER", True, (255, 60, 60))
            title_shadow = self.big_font.render("GAME OVER", True, (90, 0, 0))
            title_center = (WIDTH // 2, panel_y + 55)
            self.screen.blit(title_shadow, title_shadow.get_rect(center=(title_center[0] + 3, title_center[1] + 3)))
            self.screen.blit(title, title.get_rect(center=title_center))

            subtitle = render_text(self.font, "💥 هر سه قلب از دست رفت", (255, 150, 150))
            self.screen.blit(subtitle, subtitle.get_rect(center=(WIDTH // 2, panel_y + 100)))

            # سه قلب خالی برای تاکید بصری روی پایان بازی
            for i in range(MAX_LIVES):
                self._draw_heart(WIDTH // 2 - 34 + i * 34, panel_y + 128, 24, filled=False)

            final_score = self.score + self.coins_collected * 25
            score_txt = self.medium_font.render(f"Final Score: {final_score}", True, (255, 255, 255))
            self.screen.blit(score_txt, score_txt.get_rect(center=(WIDTH // 2, panel_y + 168)))
            
            coin_txt = self.font.render(f"Coins: {self.coins_collected} × 25 = {self.coins_collected * 25}", True, (255, 220, 60))
            self.screen.blit(coin_txt, coin_txt.get_rect(center=(WIDTH // 2, panel_y + 202)))
            
            if self.new_highscore:
                hs_txt = self.font.render("🏆 NEW HIGH SCORE!", True, (255, 220, 60))
                self.screen.blit(hs_txt, hs_txt.get_rect(center=(WIDTH // 2, panel_y + 234)))
            
            prompt = self.font.render("Press R to Restart • Q to Quit", True, (200, 200, 200))
            self.screen.blit(prompt, prompt.get_rect(center=(WIDTH // 2, panel_y + 272)))

        pygame.display.flip()

    def run(self) -> None:
        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            self.handle_events()
            self.update(dt)
            self.draw()
        self.steering.close()
        pygame.quit()
        sys.exit()


def main() -> None:
    CarGame().run()


if __name__ == "__main__":
    main()
