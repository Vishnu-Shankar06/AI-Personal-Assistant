import os
import asyncio
from dotenv import load_dotenv

from tools import send_email

load_dotenv()

async def run_test():
    print("--- STARTING EMAIL TEST ---")
    
    target_email = "YOUR_EMAIL@gmail.com"
    
    print(f"Attempting to log into Gmail as: {os.getenv('GMAIL_USER')}")
    
    result = await send_email(
        context=None, 
        to_email=target_email, 
        subject="Direct System Test", 
        message="If you receive this, the Python email tool and Gmail passwords work perfectly."
    )
    
    print(f"\nRESULT: {result}")
    print("--- TEST FINISHED ---")

if __name__ == "__main__":
    asyncio.run(run_test())