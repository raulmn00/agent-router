from dotenv import load_dotenv

load_dotenv()

from .intents import ID2LABEL, INTENTS, LABEL2ID, Intent

__all__ = ["INTENTS", "LABEL2ID", "ID2LABEL", "Intent"]
