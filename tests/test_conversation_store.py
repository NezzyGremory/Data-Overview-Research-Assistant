from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.ai.conversation_store import ConversationStore, ConversationStoreError


class TestConversationStore(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tanya-conversations-")
        self.addCleanup(self.temporary.cleanup)
        self.storage = Path(self.temporary.name)
        self.store = ConversationStore(self.storage)

    def test_messages_persist_in_order_with_source_metadata(self):
        conversation = self.store.create_conversation()
        user = self.store.append_message(
            conversation.conversation_id,
            "user",
            "Apa itu median?",
        )
        assistant = self.store.append_message(
            conversation.conversation_id,
            "assistant",
            "Median adalah nilai tengah. [S1]",
            model="qwen2.5:7b",
            sources=({"source_id": "S1", "citation": "Statistics Guide, 2026"},),
        )

        restarted = ConversationStore(self.storage)
        messages = restarted.get_messages(conversation.conversation_id)
        saved_conversation = restarted.get_conversation(conversation.conversation_id)

        self.assertEqual([item.message_id for item in messages], [user.message_id, assistant.message_id])
        self.assertEqual([item.role for item in messages], ["user", "assistant"])
        self.assertEqual(messages[1].sources[0]["source_id"], "S1")
        self.assertEqual(messages[1].model, "qwen2.5:7b")
        self.assertEqual(saved_conversation.title, "Apa itu median?")

    def test_rename_delete_and_clear_apply_only_to_conversation_store(self):
        conversation = self.store.create_conversation("Percakapan awal")
        self.store.append_message(conversation.conversation_id, "user", "Teks lokal")

        renamed = self.store.rename_conversation(conversation.conversation_id, "Riset statistik")
        self.assertEqual(renamed.title, "Riset statistik")
        self.assertTrue(self.store.delete_conversation(conversation.conversation_id))
        self.assertEqual(self.store.get_messages(conversation.conversation_id), ())
        self.assertFalse(self.store.delete_conversation(conversation.conversation_id))

        second = self.store.create_conversation("Kedua")
        self.store.create_conversation("Ketiga")
        self.assertEqual(self.store.clear(), 2)
        self.assertIsNone(self.store.get_conversation(second.conversation_id))

    def test_invalid_message_and_title_are_rejected(self):
        with self.assertRaises(ValueError):
            self.store.create_conversation("  ")
        conversation = self.store.create_conversation()
        with self.assertRaises(ValueError):
            self.store.append_message(conversation.conversation_id, "assistant", "  ")
        with self.assertRaises(ValueError):
            self.store.append_message(conversation.conversation_id, "tool", "Unsupported role")
        with self.assertRaises(ConversationStoreError):
            self.store.append_message("missing", "user", "No conversation")


if __name__ == "__main__":
    unittest.main()
