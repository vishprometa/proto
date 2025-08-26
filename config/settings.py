"""
Configuration management for ClickHouse AI Agent
"""

import os
from pathlib import Path
from typing import Optional

import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.prompt import Prompt, Confirm
from dotenv import load_dotenv

console = Console()

class ClickHouseConfig(BaseModel):
    """ClickHouse connection configuration"""
    
    host: str = Field(default="localhost", description="ClickHouse host")
    port: int = Field(default=8123, description="ClickHouse HTTP port")
    username: str = Field(default="default", description="ClickHouse username")
    password: str = Field(default="", description="ClickHouse password")
    database: str = Field(default="default", description="ClickHouse database")
    secure: bool = Field(default=False, description="Use HTTPS/secure connection")

    # Provider selection
    provider: str = Field(default="local", description="LLM provider: local only")
    
    # Local LLM (llama.cpp / llamafile / llama-cpp-python server) configuration
    local_llm_base_url: str = Field(default="http://127.0.0.1:8000/v1", description="Local LLM server base URL")
    local_llm_model: str = Field(default="qwen3-1.7b", description="Local LLM model name")
    
    # Legacy OpenRouter configuration (kept for backward compatibility)
    openrouter_api_key: str = Field(default="", description="Legacy OpenRouter API key")
    openrouter_model: str = Field(default="openai/gpt-4o-mini", description="Legacy OpenRouter model")
    openrouter_provider_only: str = Field(default="openai", description="Legacy OpenRouter provider preference")
    openrouter_data_collection: str = Field(default="deny", description="Legacy OpenRouter data collection setting")
    
    # Agent configuration
    max_tool_calls: int = Field(default=35, description="Maximum tool calls per conversation")
    temperature: float = Field(default=0.1, description="LLM temperature")
    max_tokens: int = Field(default=4000, description="Maximum tokens per response")

def load_config(
    config_file: Optional[Path] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None
) -> ClickHouseConfig:
    """Load configuration from file and command line arguments"""
    
    # Load environment variables from .env file
    load_dotenv()
    
    config_data = {}
    
    # Determine default config file if not provided
    default_config_file = Path.home() / ".config" / "proto" / "proto-config.json"
    legacy_config_file = Path("proto-config.json")
    candidate_config = None
    if config_file and Path(config_file).exists():
        candidate_config = Path(config_file)
    elif default_config_file.exists():
        candidate_config = default_config_file
    elif legacy_config_file.exists():
        candidate_config = legacy_config_file

    # Load from config file if found
    if candidate_config and candidate_config.exists():
        import json
        with open(candidate_config) as f:
            config_data = json.load(f)

    # Normalize keys from onboarding style (clickhouse_*) to model fields
    key_mapping = {
        "clickhouse_host": "host",
        "clickhouse_port": "port",
        "clickhouse_username": "username",
        "clickhouse_password": "password",
        "clickhouse_database": "database",
        "clickhouse_secure": "secure",
    }
    for old_key, new_key in key_mapping.items():
        if old_key in config_data and new_key not in config_data:
            config_data[new_key] = config_data[old_key]
    
    # Load from environment variables
    config_data.update({
        k.lower().replace("clickhouse_", ""): v 
        for k, v in os.environ.items() 
        if k.startswith("CLICKHOUSE_")
    })
    
    # OpenRouter configuration from environment
    if "OPENROUTER_API_KEY" in os.environ:
        config_data["openrouter_api_key"] = os.environ["OPENROUTER_API_KEY"]
    
    if "OPENROUTER_MODEL" in os.environ:
        config_data["openrouter_model"] = os.environ["OPENROUTER_MODEL"]
    
    if "OPENROUTER_PROVIDER_ONLY" in os.environ:
        config_data["openrouter_provider_only"] = os.environ["OPENROUTER_PROVIDER_ONLY"]
    
    if "OPENROUTER_DATA_COLLECTION" in os.environ:
        config_data["openrouter_data_collection"] = os.environ["OPENROUTER_DATA_COLLECTION"]
    
    # Override with command line arguments
    if host:
        config_data["host"] = host
    if port:
        config_data["port"] = port
    if username:
        config_data["username"] = username
    if password:
        config_data["password"] = password
    if database:
        config_data["database"] = database
    
    config = ClickHouseConfig(**config_data)
    
    # No interactive configuration needed for local provider
    if config.provider != "local":
        console.print("[yellow]⚠️  Only local provider is supported[/yellow]")
        console.print("[blue]ℹ️  Switching to local provider automatically[/blue]")
        config.provider = "local"
    
    return config

def save_env_config(config: ClickHouseConfig):
    """Save configuration to .env file"""
    env_path = Path(".env")
    
    env_content = f"""# ClickHouse Configuration
CLICKHOUSE_HOST={config.host}
CLICKHOUSE_PORT={config.port}
CLICKHOUSE_USERNAME={config.username}
CLICKHOUSE_PASSWORD={config.password}
CLICKHOUSE_DATABASE={config.database}
CLICKHOUSE_SECURE={config.secure}

# OpenRouter Configuration
OPENROUTER_API_KEY={config.openrouter_api_key}
OPENROUTER_MODEL={config.openrouter_model}
OPENROUTER_PROVIDER_ONLY={config.openrouter_provider_only}
OPENROUTER_DATA_COLLECTION={config.openrouter_data_collection}
"""
    
    with open(env_path, "w") as f:
        f.write(env_content)
    
    console.print(f"[green]✓[/green] Configuration saved to {env_path}")

def create_sample_config():
    """Create a sample configuration file"""
    config_data = {
        "host": "localhost",
        "port": 8123,
        "username": "default",
        "password": "",
        "database": "default",
        "secure": False,
        "openrouter_api_key": "your-api-key-here",
        "openrouter_model": "openai/gpt-4o-mini",
        "openrouter_provider_only": "openai",
        "openrouter_data_collection": "deny",
        "max_tool_calls": 35,
        "temperature": 0.1,
        "max_tokens": 4000
    }
    
    config_path = Path("proto-config.json")
    
    import json
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    
    console.print(f"[green]✓[/green] Sample configuration created at {config_path}")