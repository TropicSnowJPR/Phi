import platform
import subprocess
import shutil
import time

from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key

class GameController:
    def __init__(self):
        self.os = platform.system().lower()

        # Linux: check if xdotool exists
        self.use_xdotool = False
        if self.os == "linux":
            self.use_xdotool = shutil.which("xdotool") is not None

        # fallback controllers
        self.mouse = MouseController()
        self.keyboard = KeyboardController()

    def press_key(self, key, duration_ms=50):
        if self.use_xdotool:
            self._xdotool_key(key, duration_ms)
            return

        key = self._map_key(key)

        self.keyboard.press(key)
        time.sleep(duration_ms / 1000)
        self.keyboard.release(key)


    def click(self, button="left"):
        if self.use_xdotool:
            cmd = ["xdotool", "click", "1" if button == "left" else "3"]
            subprocess.run(cmd)
            return

        if button == "left":
            self.mouse.click(Button.left)
        else:
            self.mouse.click(Button.right)


    def move_smooth(self, dx, dy, steps=10, delay=1/60):
        if self.use_xdotool:
            if steps < 2:
                subprocess.run([
                    "xdotool",
                    "mousemove_relative",
                    "--",
                    str(dx),
                    str(dy)
                ])
                return

            for i in range(steps):
                t = i / (steps - 1)
                smooth = 4 * t * (1 - t)

                sx = int(dx * smooth)
                sy = int(dy * smooth)

                subprocess.run([
                    "xdotool",
                    "mousemove_relative",
                    "--",
                    str(sx),
                    str(sy)
                ])
                time.sleep(delay)
            return

        for i in range(steps):
            t = i / (steps - 1)
            smooth = 4 * t * (1 - t)

            self.mouse.move(dx * smooth, dy * smooth)
            time.sleep(delay)


    def _map_key(self, key):
        mapping = {
            # basic controls
            "space": Key.space,
            "enter": Key.enter,
            "tab": Key.tab,
            "esc": Key.esc,
            "escape": Key.esc,
            "backspace": Key.backspace,
            "delete": Key.delete,

            # modifiers
            "shift": Key.shift,
            "shift_l": Key.shift_l,
            "shift_r": Key.shift_r,
            "ctrl": Key.ctrl,
            "ctrl_l": Key.ctrl_l,
            "ctrl_r": Key.ctrl_r,
            "alt": Key.alt,
            "alt_l": Key.alt_l,
            "alt_r": Key.alt_r,
            "cmd": Key.cmd,
            "super": Key.cmd,

            # navigation
            "up": Key.up,
            "down": Key.down,
            "left": Key.left,
            "right": Key.right,
            "home": Key.home,
            "end": Key.end,
            "page_up": Key.page_up,
            "page_down": Key.page_down,

            # function keys
            "f1": Key.f1,
            "f2": Key.f2,
            "f3": Key.f3,
            "f4": Key.f4,
            "f5": Key.f5,
            "f6": Key.f6,
            "f7": Key.f7,
            "f8": Key.f8,
            "f9": Key.f9,
            "f10": Key.f10,
            "f11": Key.f11,
            "f12": Key.f12,

            # locks
            "caps_lock": Key.caps_lock,
            "num_lock": Key.num_lock,
            "scroll_lock": Key.scroll_lock,

            # insert
            "insert": Key.insert,
            "print_screen": Key.print_screen,
            "pause": Key.pause,
        }

        return mapping.get(key.lower(), key)


    def _xdotool_key(self, key, duration_ms):
        mapping = {
            # basic
            "space": "space",
            "enter": "Return",
            "tab": "Tab",
            "esc": "Escape",
            "escape": "Escape",
            "backspace": "BackSpace",
            "delete": "Delete",

            # modifiers
            "shift": "Shift_L",
            "shift_l": "Shift_L",
            "shift_r": "Shift_R",
            "ctrl": "Control_L",
            "ctrl_l": "Control_L",
            "ctrl_r": "Control_R",
            "alt": "Alt_L",
            "alt_l": "Alt_L",
            "alt_r": "Alt_R",
            "cmd": "Super_L",
            "super": "Super_L",

            # navigation
            "up": "Up",
            "down": "Down",
            "left": "Left",
            "right": "Right",
            "home": "Home",
            "end": "End",
            "page_up": "Page_Up",
            "page_down": "Page_Down",

            # function keys
            "f1": "F1",
            "f2": "F2",
            "f3": "F3",
            "f4": "F4",
            "f5": "F5",
            "f6": "F6",
            "f7": "F7",
            "f8": "F8",
            "f9": "F9",
            "f10": "F10",
            "f11": "F11",
            "f12": "F12",

            # locks / misc
            "caps_lock": "Caps_Lock",
            "num_lock": "Num_Lock",
            "scroll_lock": "Scroll_Lock",
            "insert": "Insert",
            "print_screen": "Print",
            "pause": "Pause",
        }

        linux_key = mapping.get(key.lower(), key)

        subprocess.run(["xdotool", "keydown", linux_key])
        time.sleep(duration_ms / 1000)
        subprocess.run(["xdotool", "keyup", linux_key])