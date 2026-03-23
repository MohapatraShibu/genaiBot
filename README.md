# GenAI Hybrid Telegram Bot

A lightweight Telegram bot supporting **Mini-RAG** (text Q&A from documents) and **Vision** (image captioning), running fully locally and completely free.

---

## Features

| Command | Description |
|---|---|
| `/ask <question>` | Query the knowledge base using RAG |
| `/image` | Prompt to send an image for description |
| _(send photo)_ | Auto-describes any uploaded image |
| `/summarize` | Summarize your last 3 interactions |
| `/help` | Show usage instructions |

---

## System Design

<img width="268" height="314" alt="Image" src="https://github.com/user-attachments/assets/40a2e9f9-942a-4e91-b383-cff9cb2997d7" />

---

## Models Used

| Component | Model | Reason |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` | Fast, 80MB, strong semantic similarity |
| LLM | Ollama `llama3.2` | Fully local, no API cost, good quality |
| Vision | `Salesforce/blip-image-captioning-base` | Lightweight (~900MB), runs on CPU |

---

## Prerequisites

- Python 3.11+
- A Telegram account (free): [telegram.org](https://telegram.org)
- A Telegram bot token (free): from [@BotFather](https://t.me/BotFather)

---

## Setup (One Time Only)

### 1. Get a Telegram Bot Token
1. Open Telegram -> search **@BotFather** (blue checkmark ✅)
2. Send `/newbot`
3. Choose a display name (e.g. `GenAI Assistant`)
4. Choose a username ending in `bot` (e.g. `genai_assistant_bot`)
5. Copy the token BotFather gives you

### 2. Configure `.env`
```
copy .env.example .env
```
Open `.env` and set your token:
```env
TELEGRAM_TOKEN=123:ABCdef...
```
Everything else in `.env` can stay as-is.

### 3. Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Download Ollama (Portable)
- Go to [github.com/ollama/ollama/releases/latest](https://github.com/ollama/ollama/releases/latest)
- Download `ollama-windows-amd64.zip`
- Extract and copy `ollama.exe` into `\ProjectDirectory\genaiBot\`

### 5. Create Models Folder
```powershell
mkdir c:\ProjectDirectory\genaiBot\ollama_models
```

### 6. Pull the LLM Model (One Time)
```powershell
$env:OLLAMA_MODELS="c:\ProjectDirectory\genaiBot\ollama_models"
.\ollama.exe pull llama3.2
```
This downloads ~2GB. Wait for `success`.

---

## Running the Bot

After setup, you only need **one command** from the `genaiBot` folder:

```powershell
python bot.py
```

This automatically:
1. Starts the Ollama server in the background
2. Waits for it to be ready
3. Initializes the RAG knowledge base
4. Starts the Telegram bot

You should see:
```
Waiting for Ollama to be ready...
Ollama is ready.
Initializing RAG knowledge base...
RAG ready. Starting bot...
Bot polling...
```

Once you see `Bot polling...`, open Telegram and start chatting with your bot!

---

## Project Structure

```
genaiBot/
├── ollama.exe              # portable Ollama binary
├── ollama_models/          # LLM models stored here
├── bot.py                  # telegram handlers
├── rag.py                  # chunking, embedding, retrieval, LLM
├── vision.py               # BLIP image captioning
├── docs/                   # knowledge base documents
│   ├── tech_faq.md
│   ├── company_policies.md
│   └── recipes.md
├── rag_store.db            # auto-created on first run
├── requirements.txt
├── .gitignore
├── .env                    # your config (not committed)
└── .env.example            # config template
```

---

## Adding Documents

Drop `.md` or `.txt` files into the `docs/` folder and restart the bot. New chunks are auto-indexed on startup (already-seen chunks are skipped via MD5 hash deduplication).

---

## Caching

- **Query cache**: identical queries return cached answers without re-calling the LLM.
- **Chunk deduplication**: documents are only re-embedded if their content changes.

---

## Demo

```
User:  /ask how many days of annual leave do I get?
Bot:   You receive 20 days of annual leave per year, accrued monthly.

       📚 Sources:
       • company_policies.md: Employees may work remotely up to 3 days...

User:  [sends photo of a dog]
Bot:   🖼️ Caption: a dog sitting on a wooden floor
       🏷️ Tags: `golden retriever` • `brown and white` • `hardwood floor`

User:  /summarize
Bot:   📝 Summary:
       The user asked about leave policy and received details about 20 annual
       leave days. They also uploaded a photo of a dog sitting on a wooden floor.
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `TELEGRAM_TOKEN not set` | Check your `.env` file has the token |
| Bot not responding | Make sure `https://api.telegram.org/...` is running |
| Slow first `/image` | BLIP downloads ~900MB on first use, wait for it |
| `ollama.exe` not found | Make sure `ollama.exe` is in `c:\Projects\genaiBot\` |
