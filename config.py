import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    openai_api_key: str | None
    tavily_api_key: str | None
    user_agent: str
    default_anthropic_model: str
    default_openai_model: str

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key)


settings = Settings(
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
    user_agent=os.getenv("USER_AGENT", "zorya-polunochnaya/0.1"),
    default_anthropic_model=os.getenv("DEFAULT_ANTHROPIC_MODEL", "claude-sonnet-4-5"),
    default_openai_model=os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini"),
)
