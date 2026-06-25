"""
Test if .env file is loading correctly
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Check if API key is loaded
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    # Show first 10 characters for verification
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    print(f"✅ Model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
    print(f"✅ Temperature: {os.getenv('OPENAI_TEMPERATURE', '0')}")
else:
    print("❌ No API key found! Please check your .env file.")
    print("Expected location:", project_root / ".env")