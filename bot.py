# telegram bot: /ask, /image, /help, /summarize

import logging
import os
import subprocess
import time
from collections import defaultdict, deque

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

import rag
import vision

load_dotenv()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_TOKEN")

# per-user history: deque of {"role": "user"|"bot", "text": str}
_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=6))  # 3 pairs


def _add_history(user_id: int, role: str, text: str):
    _history[user_id].append({"role": role, "text": text})


def _fmt_sources(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    seen = {}
    for c in chunks:
        src = c["source"]
        if src not in seen:
            seen[src] = c["content"][:120].replace("\n", " ")
    lines = [f"\n\n📚 *Sources:*"]
    for src, snippet in seen.items():
        lines.append(f"• `{src}`: _{snippet}…_")
    return "\n".join(lines)


# handlers

async def cmd_help(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *GenAI Bot - Commands*\n\n"
        "/ask `<question>` - Query the knowledge base (RAG)\n"
        "/image - Send an image to get a caption + tags\n"
        "/summarize - Summarize your last 3 interactions\n"
        "/help - Show this message",
        parse_mode="Markdown",
    )


async def cmd_ask(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = " ".join(ctx.args).strip()
    if not query:
        await update.message.reply_text("Usage: /ask <your question>")
        return

    user_id = update.effective_user.id
    _add_history(user_id, "user", query)

    await update.message.reply_chat_action("typing")
    try:
        answer_text, chunks = rag.answer(query)
    except Exception as e:
        logger.exception("RAG error")
        await update.message.reply_text(f"⚠️ Error: {e}")
        return

    _add_history(user_id, "bot", answer_text)
    reply = answer_text + _fmt_sources(chunks)
    await update.message.reply_text(reply, parse_mode="Markdown")


async def cmd_image(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📷 Send me an image (as a photo or file) and I'll describe it!"
    )


async def handle_photo(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.reply_chat_action("upload_photo")

    # prefer document (uncompressed) over photo
    if update.message.document and update.message.document.mime_type.startswith("image"):
        file = await update.message.document.get_file()
    elif update.message.photo:
        file = await update.message.photo[-1].get_file()  # highest resolution
    else:
        await update.message.reply_text("Please send an image file.")
        return

    image_bytes = await file.download_as_bytearray()

    await update.message.reply_chat_action("typing")
    try:
        result = vision.describe_image(bytes(image_bytes))
    except Exception as e:
        logger.exception("Vision error")
        await update.message.reply_text(f"⚠️ Vision error: {e}")
        return

    caption = result["caption"]
    tags = result["tags"]
    tag_str = " • ".join(f"`{t}`" for t in tags) if tags else "—"

    reply = f"🖼️ *Caption:* {caption}\n\n🏷️ *Tags:* {tag_str}"
    _add_history(user_id, "user", "[image]")
    _add_history(user_id, "bot", caption)
    await update.message.reply_text(reply, parse_mode="Markdown")


async def cmd_summarize(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history = list(_history[user_id])
    if not history:
        await update.message.reply_text("No history yet. Use /ask or send an image first.")
        return

    lines = "\n".join(f"{h['role'].upper()}: {h['text']}" for h in history)
    prompt = f"Summarize this conversation in 2-3 sentences:\n\n{lines}"

    await update.message.reply_chat_action("typing")
    try:
        summary = rag.call_llm(prompt)
    except Exception as e:
        logger.exception("Summarize error")
        await update.message.reply_text(f"⚠️ Error: {e}")
        return

    await update.message.reply_text(f"📝 *Summary:*\n{summary}", parse_mode="Markdown")


# main

def start_ollama():
    """Start ollama.exe in the background if not already running."""
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:11434", timeout=2)
        logger.info("Ollama already running.")
        return
    except Exception:
        pass

    ollama_exe = os.path.join(os.path.dirname(__file__), "ollama.exe")
    ollama_models = os.path.join(os.path.dirname(__file__), "ollama_models")
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = ollama_models
    subprocess.Popen(
        [ollama_exe, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info("Waiting for Ollama to be ready...")
    for _ in range(20):
        time.sleep(1)
        try:
            urllib.request.urlopen("http://127.0.0.1:11434", timeout=2)
            logger.info("Ollama is ready.")
            return
        except Exception:
            pass
    raise RuntimeError("Ollama did not start in time. Check ollama.exe is in the project folder.")


def main():
    if not TOKEN:
        raise ValueError("TELEGRAM_TOKEN not set in .env")

    start_ollama()

    logger.info("Initializing RAG knowledge base…")
    rag.init_db()
    rag.load_docs()
    logger.info("RAG ready. Starting bot…")

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(CommandHandler("image", cmd_image))
    app.add_handler(CommandHandler("summarize", cmd_summarize))
    app.add_handler(
        MessageHandler(
            filters.PHOTO | (filters.Document.IMAGE),
            handle_photo,
        )
    )

    logger.info("Bot polling…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
