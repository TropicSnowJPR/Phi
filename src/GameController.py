from pynput.mouse import Controller as MouseController
from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Button
import time


class GameController:
    def __init__(self):
        self.mouse = MouseController()
        self.keyboard = KeyboardController()

    # ========================
    # KEY PRESS
    # ========================
    def press_key(self, key, duration_ms):
        if key == "space":
            key = Key.space
        elif key == "shift":
            key = Key.shift
        elif key == "ctrl":
            key = Key.ctrl
        
        self.keyboard.press(key)
        time.sleep(duration_ms / 1000)
        self.keyboard.release(key)

    def click(self, button="left"):
        if button == "left":
            self.mouse.click(Button.left)
        else:
            self.mouse.click(Button.right)

    # ========================
    # SMOOTH MOUSE MOVE
    # ========================
    def move_smooth(self, dx, dy, steps=10, delay=1/60):
        """
        dx, dy = total movement
        steps = how smooth (more = smoother)
        """

        for i in range(steps):
            t = i / (steps - 1)  # 0 → 1

            # ease-in-out curve (smooth acceleration + deceleration)
            smooth = 4 * t * (1 - t)

            self.mouse.move(dx * smooth, dy * smooth)
            time.sleep(delay)