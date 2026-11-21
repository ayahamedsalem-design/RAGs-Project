# Chat memory unified onto chat_history.json (shared with hybrid_rag and other_rags).
# Preserves Ext RAG's follow-up question rewriting (rewrite_question) and background
# prompt injection (add_history_to_prompt) while eliminating the separate SQLite DB.

import re
import json
import logging
from datetime import datetime
from pathlib import Path

from hybrid_rag.chat_history import (
    load_file_history,
    load_all_history,
    persist_chat_history,
)
from ext_rag.rag_common import chat

logger = logging.getLogger("rag.memory")

HISTORY_FILE = "chat_history.json"
MEMORY_TURNS = 3         # previous question/answer pairs used for follow-ups
MAX_ANSWER_CHARS = 800   # long earlier answers are shortened in the prompt

REWRITE_PROMPT = """You rewrite follow-up questions for a document search system.
You receive the CONVERSATION SO FAR and a FOLLOW-UP question.
Rewrite the follow-up so it can be understood on its own: replace words like "it", "they",
"that", "the second one" with what they refer to, and add the topic if it is missing.
Keep the meaning and keep it short. If the follow-up is already clear on its own, or starts
a new topic, return it unchanged. Reply with the rewritten question only, no other text."""


# ---------- Storage (backed by chat_history.json) ----------
def recent_turns(session_name, n=MEMORY_TURNS, filepath=None):
    """The last n (question, answer) pairs for session_name, oldest first."""
    target = filepath or HISTORY_FILE
    items = load_file_history(session_name, filepath=target)
    tail = items[-n:] if n > 0 else []
    turns = [(item.get("query", ""), item.get("answer", "")) for item in tail]
    logger.debug("Using %d earlier turns from %s for session %r", len(turns), target, session_name)
    return turns


def save_turn(session_name, question, standalone, answer, sources, system, cost_usd=None, filepath=None):
    """Persist one turn into the unified chat_history.json file."""
    target = filepath or HISTORY_FILE
    if isinstance(sources, list):
        context_text = "\n\n---\n\n".join(str(s) for s in sources)
    else:
        context_text = str(sources or "")
    persist_chat_history(session_name, question, answer, context_text, filepath=target)
    logger.info("Saved %s answer to unified %s under session %r", system, target, session_name)


def forget_session(session_name, filepath=None):
    """Remove all turns for session_name from chat_history.json. Returns True if any were removed."""
    target = filepath or HISTORY_FILE
    path = Path(target)
    if not path.exists():
        return False
    all_items = load_all_history(filepath=target)
    remaining = [item for item in all_items if item.get("file_name") != session_name]
    if len(remaining) == len(all_items):
        return False
    path.write_text(json.dumps(remaining, indent=4, ensure_ascii=False), encoding="utf-8")
    logger.info('Deleted conversation "%s" from %s', session_name, target)
    return True


# ---------- Using the history ----------
def strip_citations(text):
    """Remove [Source N] labels: they refer to an earlier search, not the current one."""
    text = re.sub(r"\s*\[Sources? \d+(?:\s*(?:,|and|&)\s*(?:Source )?\d+)*\]", "", text)
    return text if len(text) <= MAX_ANSWER_CHARS else text[:MAX_ANSWER_CHARS] + "..."


def format_history(turns):
    return "\n\n".join(f"User: {q}\nAssistant: {strip_citations(a)}" for q, a in turns)


def rewrite_question(question, turns):
    """Turn a follow-up into a stand-alone question. No history means nothing to rewrite."""
    if not turns:
        return question
    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {
            "role": "user",
            "content": f"CONVERSATION SO FAR:\n\n{format_history(turns)}\n\nFOLLOW-UP: {question}",
        },
    ]
    reply = chat(messages, temperature=0.0, step="rewrite").strip().strip('"').strip()
    if not reply or len(reply) > 4 * len(question) + 200 or "\n\n" in reply:
        logger.warning("Rewrite reply unusable, searching the original question: %r", reply[:300])
        return question
    logger.info("Follow-up %r rewritten as %r", question, reply.splitlines()[0])
    return reply.splitlines()[0]


def add_history_to_prompt(messages, turns):
    """Put the conversation so far in front of the sources, clearly marked as background."""
    if turns:
        messages[1]["content"] = (
            "CONVERSATION SO FAR (background only: use it to understand the question, "
            "but take every fact from the SOURCES and cite only them):\n\n"
            f"{format_history(turns)}\n\n" + messages[1]["content"]
        )
    return messages


# ---------- CLI commands for looking at previous chats ----------
def add_memory_args(parser):
    g = parser.add_argument_group("chat memory")
    g.add_argument("--chat", action="store_true",
                   help="interactive chat that remembers earlier questions")
    g.add_argument("--session", metavar="NAME",
                   help="conversation to continue (default for --chat: 'default')")
    g.add_argument("--sessions", action="store_true", help="list saved conversations")
    g.add_argument("--show", metavar="NAME", help="print a saved conversation")
    g.add_argument("--forget", metavar="NAME", help="delete a saved conversation")


def handle_memory_commands(args, filepath=None):
    """Run --sessions / --show / --forget. Returns True if one of them was given."""
    target = filepath or HISTORY_FILE
    if not (args.sessions or args.show or args.forget):
        return False
    if not Path(target).exists():
        print("No saved conversations yet. Start one with --chat.")
        return True
    if args.sessions:
        list_sessions(filepath=target)
    if args.show:
        show_session(args.show, filepath=target)
    if args.forget:
        if forget_session(args.forget, filepath=target):
            print(f'Deleted conversation "{args.forget}".')
        else:
            print(f'No conversation called "{args.forget}".')
    return True


def list_sessions(filepath=None):
    target = filepath or HISTORY_FILE
    all_items = load_all_history(filepath=target)
    counts = {}
    for item in all_items:
        name = item.get("file_name") or "default"
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        print("No saved conversations yet. Start one with --chat.")
        return
    print(f"{'NAME':30} {'QUESTIONS':>9}")
    for name, n in sorted(counts.items()):
        print(f"{name:30} {n:9}")


def show_session(name, last=None, filepath=None):
    target = filepath or HISTORY_FILE
    rows = load_file_history(name, filepath=target)
    if last:
        rows = rows[-last:]
    if not rows:
        print(f'No conversation called "{name}". See the list with --sessions.')
        return
    for i, item in enumerate(rows, 1):
        print(f"--- {i}. {name} ---")
        print(f"You: {item.get('query', '')}\n")
        print(f"{item.get('answer', '')}\n")


def run_chat(answer_fn, system, session_name, filepath=None):
    """Interactive loop. answer_fn(question, turns) -> (answer, sources, standalone, cost_usd)."""
    target = filepath or HISTORY_FILE
    count = len(load_file_history(session_name, filepath=target))
    print(f'Chat ({system} RAG, conversation "{session_name}", {count} earlier questions).')
    print("Commands: /history  show this conversation   /new  start fresh   /exit  quit\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("/exit", "/quit", "exit", "quit"):
            break
        if question.lower() == "/history":
            show_session(session_name, filepath=target)
            continue
        if question.lower() == "/new":
            session_name = datetime.now().strftime("chat-%Y%m%d-%H%M%S")
            print(f'Started a new conversation "{session_name}".\n')
            continue
        turns = recent_turns(session_name, filepath=target)
        answer, sources, standalone, cost_usd = answer_fn(question, turns)
        save_turn(session_name, question, standalone, answer, sources, system, cost_usd, filepath=target)
        print()
    print(f'Saved. Continue later with:  --chat --session "{session_name}"')
