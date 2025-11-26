from dotenv import load_dotenv
import os

class Config:
    """
    Configuration class to manage environment settings and model configurations.
    Loads environment variables from a .env file and stores static configuration variables.
    """
    
    def __init__(self, env_file: str = '.env'):

        self.load_environment_variables(env_file)

        # llm configurations
        self.forecasting_model = "moonshotai/kimi-k2-instruct-0905"
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        # vector db configurations
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = "tcs-financial-forecast"

        # chunking model for tokenization
        self.chunking_model = "gpt-4"

    @staticmethod
    def load_environment_variables(env_file: str):
        """
        Load environment variables from a specified file.

        Args:
            env_file (str): The path to the environment file.
        """
        try:
            load_dotenv(env_file)
            print(f"Environment variables loaded from {env_file}")
        except Exception as e:
            print(f"Warning: Could not load environment file {env_file}: {e}")
            print("Continuing without .env file - make sure environment variables are set manually")

config = Config()