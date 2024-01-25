from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

env_variables = {key: os.getenv(key) for key in os.environ.keys()}
print(env_variables)

