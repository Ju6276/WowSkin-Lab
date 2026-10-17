"""Linux SpaceMouse focus and held keys, without a display or a device."""
import inspect
import sys
import unittest

from wowskin_lab.ui.keyboard import ESC, ViewerKeyboard
from wowskin_lab.ui.x11_input import held_keys, keymap_bits, keysyms_for, prepare_x11_threads, title_matches


class FakeX11:
    def __init__(self):
        self.focused = False
        self.pressed = set()

    def state(self):
        return self.focused, set(self.pressed)


class X11DecodeChecks(unittest.TestCase):
    def test_keysyms_cover_letters_and_specials(self):
        self.assertEqual(keysyms_for(ord('W')), (ord('W'), ord('w')))
        self.assertEqual(keysyms_for(ord('-')), (ord('-'),))
        self.assertEqual(keysyms_for(32), (0x20,))
        self.assertEqual(keysyms_for(ESC), (0xff1b,))
        self.assertEqual(keysyms_for(257)[:1], (0xff0d,))

    def test_keymap_bits_and_modifiers(self):
        keymap = bytearray(32)
        keymap[10] |= 1 << 3  # keycode 83
        self.assertEqual(keymap_bits(keymap), {83})
        self.assertEqual(held_keys({83}, {83: ord('W')}, set()), {ord('W')})
        self.assertEqual(held_keys({83, 37}, {83: ord('W')}, {37}), set())

    def test_title_match_uses_the_same_substring_as_macos(self):
        self.assertTrue(title_matches(['WowSkin-Lab | PandaPickCube: WowSkin'], 'PandaPickCube:'))
        self.assertFalse(title_matches(['Firefox'], 'PandaPickCube:'))
        self.assertFalse(title_matches(['PandaPickCube: WowSkin'], ''))

    def test_prepare_does_not_claim_x11_off_linux(self):
        if sys.platform.startswith('linux'):
            self.skipTest('DISPLAY may really be available')
        self.assertFalse(prepare_x11_threads())


class FocusPollChecks(unittest.TestCase):
    def keyboard(self):
        keyboard = ViewerKeyboard({ord('W'), ord('P')}, 'PandaPickCube:')
        keyboard._x11 = FakeX11()
        keyboard.continuous = True
        keyboard.focus_backend = 'x11'
        return keyboard

    def test_focus_gain_does_not_replay_a_key_already_down(self):
        keyboard = self.keyboard()
        keyboard._x11.focused = True
        keyboard._x11.pressed = {ord('W')}
        snap = keyboard.sample()
        self.assertTrue(keyboard.focused)
        self.assertEqual(snap.commands, ())
        self.assertNotIn(ord('W'), snap.held)

    def test_new_press_is_held_until_focus_leaves(self):
        keyboard = self.keyboard()
        keyboard._x11.focused = True
        keyboard.sample()
        keyboard._x11.pressed = {ord('W'), ord('P')}
        snap = keyboard.sample()
        self.assertEqual(set(snap.commands), {ord('W'), ord('P')})
        self.assertEqual(snap.held, frozenset({ord('W'), ord('P')}))
        snap = keyboard.sample()
        self.assertEqual(snap.commands, ())
        self.assertIn(ord('W'), snap.held)
        keyboard._x11.focused = False
        snap = keyboard.sample()
        self.assertFalse(keyboard.focused)
        self.assertEqual(snap.commands, ())
        self.assertEqual(snap.held, frozenset())
        keyboard.native_callback(ord('W'))
        self.assertEqual(keyboard.sample().commands, ())

    def test_single_step_callback_still_fires_once_without_x11(self):
        keyboard = ViewerKeyboard({ord('W')}, 'PandaPickCube:')
        keyboard.continuous = False
        keyboard.native_callback(ord('W'))
        snap = keyboard.sample()
        self.assertEqual(snap.commands, (ord('W'),))
        self.assertEqual(snap.held, frozenset())

    def test_teleop_no_longer_refuses_linux_before_opening_a_window(self):
        from wowskin_lab.envs import pick_cube
        source = inspect.getsource(pick_cube.viewer)
        self.assertNotIn('validated only on macOS', source)
        self.assertIn('x11_ready', source)


if __name__ == '__main__':
    unittest.main()
