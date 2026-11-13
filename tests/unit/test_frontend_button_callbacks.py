import unittest

from Front.ui_components import AnimatedButton


class ButtonDouble:
    def __init__(self):
        self.state = "normal"
        self._syncing = False
        self._hover_fg = "#hover"
        self._press_fg = "#pressed"
        self._original_fg = "#original"
        self.configurations = []
        self.scheduled = []

    def cget(self, name):
        if name == "state":
            return self.state
        raise ValueError(name)

    def configure(self, **kwargs):
        self.configurations.append(kwargs)

    def after(self, delay, callback):
        self.scheduled.append((delay, callback))

    def _restore_original_color(self):
        AnimatedButton._restore_original_color(self)


class AnimatedButtonCallbackTests(unittest.TestCase):
    def test_mouse_release_accepts_tk_event_and_restores_color(self):
        button = ButtonDouble()

        AnimatedButton._handle_mouse_release(button, object())

        self.assertEqual(button.configurations, [{"fg_color": "#hover"}])
        self.assertEqual(len(button.scheduled), 1)
        delay, restore = button.scheduled[0]
        self.assertEqual(delay, 120)
        restore()
        self.assertEqual(
            button.configurations,
            [{"fg_color": "#hover"}, {"fg_color": "#original"}],
        )

    def test_mouse_press_accepts_tk_event(self):
        button = ButtonDouble()

        AnimatedButton._handle_mouse_press(button, object())

        self.assertEqual(button.configurations, [{"fg_color": "#pressed"}])

    def test_disabled_button_does_not_schedule_color_changes(self):
        button = ButtonDouble()
        button.state = "disabled"

        AnimatedButton._handle_mouse_release(button, object())

        self.assertEqual(button.configurations, [])
        self.assertEqual(button.scheduled, [])


if __name__ == "__main__":
    unittest.main()
