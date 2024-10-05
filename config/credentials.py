"""Load the private credentials from .env files"""
from os import getenv
from dotenv import load_dotenv

load_dotenv()
ROBOFLOW_API_KEY = getenv("ROBOFLOW_API_KEY")
