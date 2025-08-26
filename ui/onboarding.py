"""
Beautiful onboarding flow for first-time users
Handles provider selection, API key setup, and initial configuration
"""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.table import Table
from rich.box import ROUNDED
from pathlib import Path
import json
from typing import Dict, Any, Optional
from ui.minimal_interface import ui

console = Console()

class OnboardingFlow:
    """Beautiful onboarding flow for new users"""
    
    def __init__(self):
        self.config = {}
        # Store config in user config directory (e.g., ~/.config/proto/proto-config.json)
        config_dir = Path.home() / ".config" / "proto"
        config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = config_dir / "proto-config.json"
        
    def show_welcome(self):
        """Show welcome message for first-time users"""
        welcome_text = Text()
        welcome_text.append("🎉 ", style="bold bright_yellow")
        welcome_text.append("Welcome to Proto!", style="bold bright_cyan")
        welcome_text.append(" 🎉", style="bold bright_yellow")
        
        subtitle = Text("Let's get you set up with your ClickHouse AI Agent", style="italic bright_white")
        
        welcome_panel = Panel(
            Align.center(
                Text.assemble(
                    welcome_text, "\n\n",
                    subtitle, "\n\n",
                    ("This will only take a minute! ⚡", "bold green")
                )
            ),
            box=ROUNDED,
            border_style="bright_cyan",
            padding=(2, 4),
            title="[bold bright_yellow]First Time Setup[/bold bright_yellow]",
            title_align="center"
        )
        
        console.print()
        console.print(welcome_panel)
        console.print()
        
    # Removed choose_ai_provider method - no longer needed
    
    def setup_openrouter(self) -> Dict[str, Any]:
        """Setup OpenRouter configuration"""
        console.print()
        info_panel = Panel(
            Text.assemble(
                ("🌐 OpenRouter Setup", "bold bright_cyan"), "\n\n",
                ("OpenRouter gives you access to multiple AI models including:", "bright_white"), "\n",
                ("• GPT-4o, GPT-4o-mini", "green"), "\n",
                ("• Claude 3.5 Sonnet", "green"), "\n", 
                ("• Llama, Mistral, and more", "green"), "\n\n",
                ("You'll need an API key from: ", "bright_white"),
                ("https://openrouter.ai", "bright_blue underline")
            ),
            border_style="bright_cyan",
            padding=(1, 2)
        )
        console.print(info_panel)
        console.print()
        
        # Get API key
        api_key = Prompt.ask(
            "[bold bright_cyan]Enter your OpenRouter API key[/bold bright_cyan]",
            password=True
        )
        
        # Choose model
        model_table = Table(show_header=False, box=ROUNDED, padding=(0, 1))
        model_table.add_column("Option", style="bold bright_cyan")
        model_table.add_column("Model", style="bright_white")
        model_table.add_column("Cost", style="bright_green")
        
        model_table.add_row("1", "GPT-4o-mini", "Cheapest, fast")
        model_table.add_row("2", "GPT-4o", "Most capable")
        model_table.add_row("3", "Claude 3.5 Sonnet", "Great for analysis")
        
        model_panel = Panel(
            model_table,
            title="[bold bright_green]Choose Model[/bold bright_green]",
            border_style="bright_green"
        )
        console.print(model_panel)
        
        model_choice = Prompt.ask(
            "[bold bright_green]Select model[/bold bright_green]",
            choices=["1", "2", "3"],
            default="1"
        )
        
        models = {
            "1": "openai/gpt-4o-mini",
            "2": "openai/gpt-4o", 
            "3": "anthropic/claude-3.5-sonnet"
        }
        
        return {
            "provider": "openrouter",
            "openrouter_api_key": api_key,
            "openrouter_model": models[model_choice],
            "openrouter_provider_only": "openai" if model_choice in ["1", "2"] else "anthropic",
            "openrouter_data_collection": "deny"
        }
    
    def setup_local(self) -> Dict[str, Any]:
        """Setup local ClickHouse AI Agent configuration"""
        console.print()
        info_panel = Panel(
            Text.assemble(
                ("🤖 ClickHouse AI Agent Setup", "bold bright_green"), "\n\n",
                ("Your ClickHouse AI Agent will be automatically downloaded and started.", "bright_white"), "\n",
                ("No API keys needed - everything is handled automatically.", "green"), "\n\n",
                ("The model will be downloaded on first run (~3.5GB).", "dim")
            ),
            border_style="bright_green",
            padding=(1, 2)
        )
        console.print(info_panel)
        console.print()
        
        return {
            "provider": "local",
            "local_llm_base_url": "http://127.0.0.1:8000/v1",
            "local_llm_model": "vishprometa/clickhouse-qwen3-1.7b-gguf"
        }
    
    def setup_lmstudio(self) -> Dict[str, Any]:
        """Setup LM Studio configuration"""
        console.print()
        info_panel = Panel(
            Text.assemble(
                ("🏠 LM Studio Setup", "bold bright_blue"), "\n\n",
                ("LM Studio provides a local server for AI models.", "bright_white"), "\n",
                ("Benefits: Free, private, easy GUI", "green"), "\n\n",
                ("Make sure LM Studio is running with local server enabled:", "bright_white"), "\n",
                ("• Download from: ", "bright_white"), ("https://lmstudio.ai", "bright_blue underline"), "\n",
                ("• Start local server on port 1234", "bright_yellow")
            ),
            border_style="bright_blue",
            padding=(1, 2)
        )
        console.print(info_panel)
        console.print()
        
        base_url = Prompt.ask(
            "[bold bright_blue]LM Studio base URL[/bold bright_blue]",
            default="http://localhost:1234"
        )
        
        return {
            "provider": "lmstudio", 
            "lmstudio_base_url": base_url
        }
    
    def setup_openai(self) -> Dict[str, Any]:
        """Setup direct OpenAI configuration"""
        console.print()
        info_panel = Panel(
            Text.assemble(
                ("☁️  OpenAI Direct Setup", "bold bright_red"), "\n\n",
                ("Connect directly to OpenAI's API.", "bright_white"), "\n",
                ("You'll need an API key from: ", "bright_white"),
                ("https://platform.openai.com", "bright_blue underline")
            ),
            border_style="bright_red",
            padding=(1, 2)
        )
        console.print(info_panel)
        console.print()
        
        api_key = Prompt.ask(
            "[bold bright_red]Enter your OpenAI API key[/bold bright_red]",
            password=True
        )
        
        model_choice = Prompt.ask(
            "[bold bright_red]Choose model[/bold bright_red]",
            choices=["gpt-4o-mini", "gpt-4o", "gpt-4"],
            default="gpt-4o-mini"
        )
        
        return {
            "provider": "openai",
            "openai_api_key": api_key,
            "openai_model": model_choice
        }
    
    def setup_clickhouse(self) -> Dict[str, Any]:
        """Setup ClickHouse connection"""
        console.print()
        
        # Ask if they want local or cloud
        connection_type = Prompt.ask(
            "[bold bright_cyan]ClickHouse connection type[/bold bright_cyan]",
            choices=["local", "cloud"],
            default="local"
        )
        
        if connection_type == "local":
            return {
                "clickhouse_host": "localhost",
                "clickhouse_port": 8123,
                "clickhouse_username": "default", 
                "clickhouse_password": "",
                "clickhouse_database": "default",
                "clickhouse_secure": False
            }
        else:
            console.print()
            cloud_panel = Panel(
                Text.assemble(
                    ("☁️  ClickHouse Cloud Setup", "bold bright_cyan"), "\n\n",
                    ("Enter your ClickHouse Cloud connection details:", "bright_white"), "\n\n",
                    ("💡 Common ports:", "blue"), "\n",
                    ("• 8123 (HTTP)", "dim"), "\n",
                    ("• 8443 (HTTPS)", "dim")
                ),
                border_style="bright_cyan",
                padding=(1, 2)
            )
            console.print(cloud_panel)
            
            host = Prompt.ask("[bold bright_cyan]Host[/bold bright_cyan]")
            port = int(Prompt.ask("[bold bright_cyan]Port[/bold bright_cyan]", default="8123"))
            username = Prompt.ask("[bold bright_cyan]Username[/bold bright_cyan]", default="default")
            password = Prompt.ask("[bold bright_cyan]Password[/bold bright_cyan]", password=True)
            database = Prompt.ask("[bold bright_cyan]Database[/bold bright_cyan]", default="default")
            
            # Ask about secure connection
            secure = Confirm.ask(
                "[bold bright_cyan]Use secure connection (HTTPS)?[/bold bright_cyan]",
                default=True if port == 8443 else False
            )
            
            return {
                "clickhouse_host": host,
                "clickhouse_port": port,
                "clickhouse_username": username,
                "clickhouse_password": password,
                "clickhouse_database": database,
                "clickhouse_secure": secure
            }
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        ui.show_success(f"Configuration saved to {self.config_file}")
        
    def run_onboarding(self) -> Dict[str, Any]:
        """Run the complete onboarding flow"""
        self.show_welcome()
        
        # Automatically use local ClickHouse AI Agent (no provider choice)
        ai_config = self.setup_local()
        
        # Setup ClickHouse
        clickhouse_config = self.setup_clickhouse()
        
        # Combine configs
        final_config = {
            **ai_config,
            **clickhouse_config,
            "temperature": 0.1,
            "max_tokens": 4000,
            "max_tool_calls": 35
        }
        
        # Save config
        self.save_config(final_config)
        
        # Show completion
        completion_panel = Panel(
            Align.center(
                Text.assemble(
                    ("🎉 Setup Complete! 🎉", "bold bright_green"), "\n\n",
                    ("Your Proto ClickHouse AI Agent is ready to use!", "bright_white"), "\n",
                    ("You can change these settings anytime with: ", "dim"),
                    ("proto settings", "bright_cyan")
                )
            ),
            border_style="bright_green",
            box=ROUNDED,
            padding=(2, 4)
        )
        console.print()
        console.print(completion_panel)
        console.print()
        
        return final_config

def needs_onboarding() -> bool:
    """Check if user needs onboarding"""
    # Preferred config location
    config_file = Path.home() / ".config" / "proto" / "proto-config.json"
    # Legacy location fallback (cwd)
    legacy_config_file = Path("proto-config.json")
    env_file = Path(".env")
    
    # If neither config file exists, needs onboarding
    if not (config_file.exists() or legacy_config_file.exists() or env_file.exists()):
        return True
    
    # If config exists but is empty or invalid
    active_config_file = config_file if config_file.exists() else legacy_config_file if legacy_config_file.exists() else None
    if active_config_file and active_config_file.exists():
        try:
            with open(active_config_file, 'r') as f:
                config = json.load(f)
                # Check if essential keys exist
                has_provider = any(
                    k in config for k in [
                        'local_llm_base_url',
                        'ollama_base_url', 
                        'lmstudio_base_url',
                        'openai_api_key',
                        # local provider keys
                        'local_llm_base_url',
                        'local_llm_model',
                    ]
                ) or config.get('provider') == 'local'  # Also check if provider is explicitly set to local
                has_clickhouse = 'clickhouse_host' in config
                return not (has_provider and has_clickhouse)
        except (json.JSONDecodeError, KeyError):
            return True
    
    return False