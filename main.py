# main.py
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import noise_cancellation, google
from prompts import AGENT_INSTRUCTION, SESSION_INSTRUCTION
from tools import get_weather, search_web, send_email
import asyncio

load_dotenv()

# ----------------------
# FastAPI server for mobile
# ----------------------
app = FastAPI()

@app.get("/")
async def home():
    return HTMLResponse("""
    <html>
        <head><title>AI Personal Assistant</title></head>
        <body>
            <h1>AI Personal Assistant is running!</h1>
            <p>Use your LiveKit client to connect and speak with Jarvis.</p>
        </body>
    </html>
    """)

# ----------------------
# LiveKit Assistant
# ----------------------
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=AGENT_INSTRUCTION,
            llm=google.beta.realtime.RealtimeModel(
                voice="charon",
                temperature=0.2,
            ),
            tools=[get_weather, search_web, send_email]
        )

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    session = AgentSession(
        stt="deepgram/nova-3:multi",
        turn_detection="multilingual",
    )
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            video_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await session.generate_reply(
        instructions=SESSION_INSTRUCTION,
    )

# ----------------------
# Run both FastAPI and LiveKit agent
# ----------------------
if __name__ == "__main__":
    import uvicorn
    # Run LiveKit agent in background
    loop = asyncio.get_event_loop()
    loop.create_task(agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint)))
    # Run FastAPI HTTP server
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))