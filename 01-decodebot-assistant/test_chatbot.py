import unittest

from chatbot import get_response, is_exit_command, normalize_input


class NormalizeInputTests(unittest.TestCase):
    def test_removes_surrounding_spaces_and_uses_lowercase(self):
        self.assertEqual(normalize_input("  HeLLo  "), "hello")

    def test_normalizes_whitespace_only_input(self):
        self.assertEqual(normalize_input("   "), "")


class ExitCommandTests(unittest.TestCase):
    def test_recognizes_exit_variants(self):
        for command in ("exit", "QUIT", " bye ", "goodbye"):
            with self.subTest(command=command):
                self.assertTrue(is_exit_command(command))

    def test_rejects_non_exit_input(self):
        self.assertFalse(is_exit_command("hello"))


class GetResponseTests(unittest.TestCase):
    def test_handles_each_supported_intent(self):
        expected_fragments = {
            "hello": "DecodeBot",
            "help": "Available commands",
            "project 1": "rule-based chatbot",
            "requirements": "continuous loop",
            "skills": "control flow",
            "exit": "Goodbye",
        }

        for user_input, expected_fragment in expected_fragments.items():
            with self.subTest(user_input=user_input):
                self.assertIn(expected_fragment, get_response(user_input))

    def test_handles_empty_input(self):
        self.assertIn("Please enter a message", get_response("  "))

    def test_returns_fallback_for_unknown_input(self):
        self.assertIn("don't understand", get_response("tell me a joke"))


if __name__ == "__main__":
    unittest.main()