import unittest

from chatbot import (
    INTENT_RULES,
    PHRASE_TO_INTENT,
    get_reply,
    get_response,
    is_exit_command,
    normalize_input,
    previous_supported_intent,
    resolve_intent,
    supported_phrases,
)


class NormalizeInputTests(unittest.TestCase):
    def test_normalizes_case_accents_punctuation_and_whitespace(self):
        self.assertEqual(normalize_input("  ¿CÓMO   está construido?!  "), "como esta construido")

    def test_normalizes_whitespace_only_input(self):
        self.assertEqual(normalize_input("   "), "")


class KnowledgeBaseTests(unittest.TestCase):
    def test_exposes_at_least_five_intents(self):
        self.assertGreaterEqual(len(INTENT_RULES), 5)

    def test_every_supported_phrase_has_a_direct_lookup(self):
        expected_count = sum(
            len(rule.phrases_es) + len(rule.phrases_en)
            for rule in INTENT_RULES.values()
        )
        self.assertEqual(len(PHRASE_TO_INTENT), expected_count)
        for phrase, match in PHRASE_TO_INTENT.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(resolve_intent(phrase), match)

    def test_supported_phrases_are_available_for_the_interface(self):
        catalog = supported_phrases("es")
        self.assertEqual(len(catalog), len(INTENT_RULES))
        self.assertIn("que es el proyecto 1", catalog["Proyecto"])


class ExitCommandTests(unittest.TestCase):
    def test_recognizes_english_and_spanish_exit_variants(self):
        for command in ("exit", "QUIT", " bye ", "adiós", "hasta luego"):
            with self.subTest(command=command):
                self.assertTrue(is_exit_command(command))

    def test_rejects_non_exit_input(self):
        self.assertFalse(is_exit_command("hello"))


class GetReplyTests(unittest.TestCase):
    def test_handles_each_supported_intent_in_english(self):
        expected_fragments = {
            "hello": "DecodeBot",
            "help": "Ask me",
            "project 1": "rule-based chatbot",
            "requirements": "run continuously",
            "skills": "control flow",
            "architecture": "SQLite",
            "memory": "stored locally",
            "exit": "Goodbye",
        }
        for user_input, expected_fragment in expected_fragments.items():
            with self.subTest(user_input=user_input):
                self.assertIn(expected_fragment, get_response(user_input))

    def test_handles_spanish_phrase_with_accents_and_punctuation(self):
        reply = get_reply("¿Cuáles son los requisitos?")
        self.assertEqual(reply.intent, "requirements")
        self.assertEqual(reply.language, "es")
        self.assertTrue(reply.matched)

    def test_uses_previous_intent_for_a_follow_up(self):
        reply = get_reply("Cuéntame más", previous_intent="architecture")
        self.assertEqual(reply.intent, "architecture")
        self.assertTrue(reply.contextual)
        self.assertIn("O(1)", reply.content)

    def test_recovers_previous_intent_from_persisted_messages(self):
        messages = [
            {"role": "user", "content": "¿Qué es el proyecto 1?"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "Cuéntame más"},
        ]
        self.assertEqual(previous_supported_intent(messages), "project")

    def test_handles_empty_input(self):
        reply = get_reply("  ")
        self.assertEqual(reply.intent, "empty")
        self.assertFalse(reply.matched)

    def test_returns_controlled_fallback_for_unknown_input(self):
        reply = get_reply("cuéntame un chiste")
        self.assertEqual(reply.intent, "fallback")
        self.assertFalse(reply.matched)
        self.assertIn("regla exacta", reply.content)


if __name__ == "__main__":
    unittest.main()
