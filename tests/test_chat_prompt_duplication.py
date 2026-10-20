import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ["GROQ_API_KEY"] = "gsk_mock_test_key_for_prompt_tests"

from streamlit.testing.v1 import AppTest
from langchain_core.documents import Document

from app import build_chat_prompt


class TestChatPromptDuplication(unittest.TestCase):
    """
    Regression test suite for preventing duplicate current questions in chat prompts.
    Tests A through F cover prompt construction, history preservation, and UI rendering.
    """

    def setUp(self):
        self.app_path = str(SRC_DIR / "app.py")

    # -------------------------------------------------------------------------
    # Test A — Current question appears exactly once
    # -------------------------------------------------------------------------
    def test_a_current_question_appears_exactly_once(self):
        """
        Verify that the current question appears exactly once in the constructed
        prompt and is not duplicated in both Conversation History and Current Question.
        """
        question = "What is demand paging in operating systems?"
        context = "Demand paging is a memory management scheme where pages are loaded on demand."

        # Case 1: First question with no prior history
        prompt_first = build_chat_prompt(question, context, history_messages=[])
        self.assertEqual(
            prompt_first.count(question),
            1,
            f"Expected current question to appear exactly once, but appeared {prompt_first.count(question)} times.",
        )
        self.assertIn(f"Current Question:\n{question}", prompt_first)

        # Ensure Conversation History section does not contain the current question
        history_part = prompt_first.split("Conversation History:")[1].split("Document Context:")[0]
        self.assertNotIn(question, history_part)

        # Case 2: Subsequent turn with prior conversation
        prior_messages = [
            {"role": "user", "content": "What is virtual memory?"},
            {"role": "assistant", "content": "Virtual memory maps virtual to physical addresses."},
        ]
        prompt_subsequent = build_chat_prompt(question, context, history_messages=prior_messages)
        self.assertEqual(
            prompt_subsequent.count(question),
            1,
            f"Expected current question to appear exactly once with history, but appeared {prompt_subsequent.count(question)} times.",
        )
        history_part_subsequent = prompt_subsequent.split("Conversation History:")[1].split("Document Context:")[0]
        self.assertNotIn(question, history_part_subsequent)

    # -------------------------------------------------------------------------
    # Test B — Previous messages are preserved in chronological order
    # -------------------------------------------------------------------------
    def test_b_previous_messages_are_preserved(self):
        """
        Verify that earlier user and assistant messages remain available to the model
        and retain their expected chronological order.
        """
        prior_history = [
            {"role": "user", "content": "Question 1: Explain kernel mode."},
            {"role": "assistant", "content": "Answer 1: Kernel mode grants privileged hardware access."},
            {"role": "user", "content": "Question 2: Explain user mode."},
            {"role": "assistant", "content": "Answer 2: User mode restricts direct hardware access."},
        ]

        current_question = "Question 3: How does switching between modes happen?"
        context = "Mode switches happen via system calls and interrupts."

        prompt = build_chat_prompt(current_question, context, history_messages=prior_history)

        # All prior messages are present in the prompt
        for msg in prior_history:
            self.assertIn(f"{msg['role']}: {msg['content']}", prompt)

        # Chronological ordering preserved
        idx_q1 = prompt.find("Question 1")
        idx_a1 = prompt.find("Answer 1")
        idx_q2 = prompt.find("Question 2")
        idx_a2 = prompt.find("Answer 2")
        idx_q3 = prompt.find("Question 3")

        self.assertTrue(
            idx_q1 < idx_a1 < idx_q2 < idx_a2 < idx_q3,
            "Chronological ordering of conversation turns was violated.",
        )

        # Current question is only in Current Question section
        self.assertEqual(prompt.count(current_question), 1)

    # -------------------------------------------------------------------------
    # Test C — Different current question across turns
    # -------------------------------------------------------------------------
    def test_c_different_current_question(self):
        """
        Send a second question after an assistant response. Verify that the new
        question appears once and the earlier conversation remains available.
        """
        # Turn 1
        q1 = "Define thrashing."
        context1 = "Thrashing occurs when the system spends more time paging than executing."
        prompt1 = build_chat_prompt(q1, context1, history_messages=[])
        self.assertEqual(prompt1.count(q1), 1)

        # Turn 2
        history_after_turn1 = [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": "Thrashing is excessive paging activity."},
        ]
        q2 = "How can thrashing be prevented?"
        context2 = "Thrashing can be prevented by using the working set model."
        prompt2 = build_chat_prompt(q2, context2, history_messages=history_after_turn1)

        # Q2 appears exactly once
        self.assertEqual(prompt2.count(q2), 1)
        self.assertIn(f"Current Question:\n{q2}", prompt2)

        # Q1 and its answer appear in Conversation History
        self.assertEqual(prompt2.count(q1), 1)
        self.assertIn("user: Define thrashing.", prompt2)
        self.assertIn("assistant: Thrashing is excessive paging activity.", prompt2)

    # -------------------------------------------------------------------------
    # Test D — Retrieved context is preserved
    # -------------------------------------------------------------------------
    def test_d_retrieved_context_is_preserved(self):
        """
        Verify that retrieved chunks and document context are faithfully preserved
        in the Document Context section without truncation or modification.
        """
        context = "Section 4.1: Semaphore wait() decrements semaphore value.\nSection 4.2: Semaphore signal() increments."
        question = "What does wait() do?"

        prompt = build_chat_prompt(question, context, history_messages=[])

        self.assertIn("Document Context:\n" + context, prompt)
        self.assertIn("Section 4.1: Semaphore wait()", prompt)
        self.assertIn("Section 4.2: Semaphore signal()", prompt)

    # -------------------------------------------------------------------------
    # Test E — Visible chat history in Streamlit UI
    # -------------------------------------------------------------------------
    def test_e_visible_chat_history(self):
        """
        Verify using AppTest that the user question appears once in visible chat UI,
        the assistant response renders normally with sources, and history persists on rerun.
        """
        test_docs = [
            Document(page_content="Deadlock requires mutual exclusion, hold and wait, no preemption, circular wait.", metadata={"source": "os.pdf", "page": 88}),
        ]

        with patch("langchain_groq.ChatGroq.invoke") as mock_invoke, \
             patch("retriever.retrieve_with_metadata", return_value=test_docs):
            mock_invoke.return_value = MagicMock(content="Deadlock requires four necessary conditions.")

            at = AppTest.from_file(self.app_path, default_timeout=10)
            at.session_state["document_ready"] = True
            at.session_state["processed_file_id"] = "os.pdf:deadlock123"
            at.session_state["current_doc_name"] = "os.pdf"
            at.run()

            # Submit question
            at.chat_input[0].set_value("What are the 4 conditions for deadlock?").run()

            # Verify prompt passed to mock_invoke had question exactly once
            self.assertTrue(mock_invoke.called)
            invoked_prompt = mock_invoke.call_args[0][0]
            self.assertEqual(
                invoked_prompt.count("What are the 4 conditions for deadlock?"),
                1,
                "Current question was duplicated in LLM prompt during AppTest execution.",
            )

            # Verify session state messages
            messages = at.session_state["messages"]
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]["role"], "user")
            self.assertEqual(messages[0]["content"], "What are the 4 conditions for deadlock?")
            self.assertEqual(messages[1]["role"], "assistant")
            self.assertEqual(messages[1]["content"], "Deadlock requires four necessary conditions.")

            # Verify user and assistant messages in chat UI
            chat_messages = at.chat_message
            user_chat_msgs = [m for m in chat_messages if getattr(m, "avatar", None) == "👤"]
            self.assertEqual(len(user_chat_msgs), 1)

            # Verify rerun keeps messages visible
            at.run()
            self.assertEqual(len(at.session_state["messages"]), 2)

    # -------------------------------------------------------------------------
    # Test F — No unnecessary prompt behavior changes
    # -------------------------------------------------------------------------
    def test_f_no_unnecessary_prompt_behavior_changes(self):
        """
        Verify that system instructions, required fallback phrases, and prompt
        templates match the established design.
        """
        question = "What is a spinlock?"
        context = "A spinlock is a lock where a thread waits in a loop."
        prompt = build_chat_prompt(question, context, history_messages=[])

        # Essential prompt guardrails must remain intact
        self.assertIn("You are a helpful study assistant.", prompt)
        self.assertIn("Use the conversation history and document context", prompt)
        self.assertIn("to answer the user's question.", prompt)
        self.assertIn("Conversation History:", prompt)
        self.assertIn("Document Context:", prompt)
        self.assertIn("Current Question:", prompt)
        self.assertIn("If the answer is not present in the document context,", prompt)
        self.assertIn("I could not find that information in the document.", prompt)


if __name__ == "__main__":
    unittest.main()
