"""Module configuring environment variables for OpenAI, VoyageAI and Deepgram API keys"""

import os
from dotenv import load_dotenv


# Specify path .env file ## CHANGE THIS TO YOUR OWN PATH
DOTENV_PATH = '/Users/marcocaporaletti/Desktop/MA Econ/RA David/Voices_code/.env'
# Load the .env file
load_dotenv(DOTENV_PATH)

# Set the OpenAI and Deepgram API key
os.environ['OPENAI_API_KEY']= os.getenv("OPENAI_API_KEY")
os.environ['DEEPGRAM_API_KEY']= os.getenv("DEEPGRAM_API_KEY")
os.environ['VOYAGE_API_KEY']= os.getenv("VOYAGE_API_KEY")
