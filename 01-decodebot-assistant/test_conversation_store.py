import unittest
from pathlib import Path
from uuid import uuid4

from conversation_store import ConversationStore


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.database_path = Path(__file__).parent / ".test-data" / f"{uuid4().hex}.db"
        self.store = ConversationStore(self.database_path)

    def tearDown(self):
        self.database_path.unlink(missing_ok=True)

    def test_creates_conversation_and_persists_messages(self):
        conversation_id = self.store.create_conversation("Test chat")
        self.store.add_message(conversation_id, "user", "hello")
        self.store.add_message(conversation_id, "assistant", "Hello!")

        conversations = self.store.list_conversations()
        self.assertEqual(conversations[0].title, "Test chat")
        self.assertEqual(
            self.store.get_messages(conversation_id),
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hello!"},
            ],
        )

    def test_renames_and_deletes_conversation(self):
        conversation_id = self.store.create_conversation()
        self.store.rename_conversation(conversation_id, "Project questions")
        self.assertEqual(self.store.list_conversations()[0].title, "Project questions")

        self.store.delete_conversation(conversation_id)
        self.assertEqual(self.store.list_conversations(), [])


if __name__ == "__main__":
    unittest.main()
