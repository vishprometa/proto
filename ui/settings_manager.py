"""
Settings management system for Proto ClickHouse AI Agent
Allows users to view, edit, and switch between configurations
"""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.table import Table
from rich.box import ROUNDED
from pathlib import Path
import json
from typing import Dict, Any, Optional
from ui.minimal_interface import ui
from ui.onboarding import OnboardingFlow

console = Console()

class SettingsManager:
    """Manage user settings and configurations"""
    
    def __init__(self):
        self.config_file = Path("proto-config.json")
        self.env_file = Path(".env")
        
    def load_current_config(self) -> Dict[str, Any]:
        """Load current configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    def show_current_settings(self):
        """Display current settings in a beautiful format"""
        config = self.load_current_config()
        
        if not config:
            ui.show_warning("No configuration found. Run onboarding first!")
            return
        
        # AI Provider Section
        ai_table = Table(show_header=False, box=None, padding=(0, 2))
        ai_table.add_column("Setting", style="bright_white")
        ai_table.add_column("Value", style="bright_cyan")
        
        provider = config.get("provider", "unknown")
        ai_table.add_row("🤖 Provider:", provider.title())
        
        if provider == "local":
            ai_table.add_row("🌐 Base URL:", config.get("local_llm_base_url", "Not set"))
            ai_table.add_row("🧠 Model:", config.get("local_llm_model", "Not set"))
        elif provider == "ollama":
            ai_table.add_row("🌐 Base URL:", config.get("ollama_base_url", "Not set"))
            ai_table.add_row("🧠 Model:", config.get("ollama_model", "Not set"))
        elif provider == "lmstudio":
            ai_table.add_row("🌐 Base URL:", config.get("lmstudio_base_url", "Not set"))
        elif provider == "openai":
            ai_table.add_row("🔑 API Key:", "••••••••" if config.get("openai_api_key") else "Not set")
            ai_table.add_row("🧠 Model:", config.get("openai_model", "Not set"))
        
        ai_panel = Panel(
            ai_table,
            title="[bold bright_magenta]AI Provider Settings[/bold bright_magenta]",
            border_style="bright_magenta",
            padding=(1, 2)
        )
        
        # ClickHouse Section
        ch_table = Table(show_header=False, box=None, padding=(0, 2))
        ch_table.add_column("Setting", style="bright_white")
        ch_table.add_column("Value", style="bright_cyan")
        
        ch_table.add_row("🏠 Host:", config.get("clickhouse_host", "Not set"))
        ch_table.add_row("🔌 Port:", str(config.get("clickhouse_port", "Not set")))
        ch_table.add_row("👤 Username:", config.get("clickhouse_username", "Not set"))
        ch_table.add_row("🔒 Password:", "••••••••" if config.get("clickhouse_password") else "Not set")
        ch_table.add_row("🗄️  Database:", config.get("clickhouse_database", "Not set"))
        ch_table.add_row("🔐 Secure:", "Yes" if config.get("clickhouse_secure") else "No")
        
        ch_panel = Panel(
            ch_table,
            title="[bold bright_cyan]ClickHouse Settings[/bold bright_cyan]",
            border_style="bright_cyan",
            padding=(1, 2)
        )
        
        # Agent Settings
        agent_table = Table(show_header=False, box=None, padding=(0, 2))
        agent_table.add_column("Setting", style="bright_white")
        agent_table.add_column("Value", style="bright_cyan")
        
        agent_table.add_row("🌡️  Temperature:", str(config.get("temperature", "Not set")))
        agent_table.add_row("📝 Max Tokens:", str(config.get("max_tokens", "Not set")))
        agent_table.add_row("🔧 Max Tool Calls:", str(config.get("max_tool_calls", "Not set")))
        
        agent_panel = Panel(
            agent_table,
            title="[bold bright_green]Agent Settings[/bold bright_green]",
            border_style="bright_green",
            padding=(1, 2)
        )
        
        console.print()
        console.print(ai_panel)
        console.print(ch_panel) 
        console.print(agent_panel)
        console.print()
    
    def edit_ai_provider(self):
        """Edit AI provider settings"""
        config = self.load_current_config()
        
        console.print()
        ui.show_info("ClickHouse AI Agent uses your own local model. No provider changes needed.")
        console.print()
    
    def edit_clickhouse(self):
        """Edit ClickHouse connection settings"""
        config = self.load_current_config()
        
        console.print()
        if Confirm.ask("[bold bright_cyan]Update ClickHouse settings?[/bold bright_cyan]"):
            onboarding = OnboardingFlow()
            ch_config = onboarding.setup_clickhouse()
            config.update(ch_config)
            self.save_config(config)
            ui.show_success("ClickHouse settings updated!")
    
    def edit_agent_settings(self):
        """Edit agent behavior settings"""
        config = self.load_current_config()
        
        console.print()
        if Confirm.ask("[bold bright_green]Update agent settings?[/bold bright_green]"):
            temp = float(Prompt.ask(
                "[bold bright_green]Temperature (0.0-2.0)[/bold bright_green]",
                default=str(config.get("temperature", 0.1))
            ))
            
            max_tokens = int(Prompt.ask(
                "[bold bright_green]Max tokens per response[/bold bright_green]",
                default=str(config.get("max_tokens", 4000))
            ))
            
            max_tool_calls = int(Prompt.ask(
                "[bold bright_green]Max tool calls per conversation[/bold bright_green]",
                default=str(config.get("max_tool_calls", 35))
            ))
            
            config.update({
                "temperature": temp,
                "max_tokens": max_tokens,
                "max_tool_calls": max_tool_calls
            })
            
            self.save_config(config)
            ui.show_success("Agent settings updated!")
    
    def reset_config(self):
        """Reset configuration and run onboarding"""
        console.print()
        if Confirm.ask("[bold red]⚠️  Reset all settings and run setup again?[/bold red]"):
            if self.config_file.exists():
                self.config_file.unlink()
            if self.env_file.exists():
                self.env_file.unlink()
            
            ui.show_success("Configuration reset! Running onboarding...")
            onboarding = OnboardingFlow()
            return onboarding.run_onboarding()
        return None
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def run_settings_menu(self):
        """Run interactive settings menu"""
        while True:
            console.clear()
            
            # Show title
            title_panel = Panel(
                Align.center(
                    Text.assemble(
                        ("⚙️  ", "bold bright_yellow"),
                        ("Proto Settings", "bold bright_cyan"),
                        (" ⚙️", "bold bright_yellow")
                    )
                ),
                border_style="bright_cyan",
                box=ROUNDED,
                padding=(1, 2)
            )
            console.print()
            console.print(title_panel)
            
            # Show current settings
            self.show_current_settings()
            
            # Show menu options
            menu_table = Table(show_header=False, box=ROUNDED, padding=(1, 2))
            menu_table.add_column("Option", style="bold bright_cyan")
            menu_table.add_column("Action", style="bright_white")
            
            menu_table.add_row("1", "🤖 AI Provider Info")
            menu_table.add_row("2", "🗄️  Update ClickHouse Connection") 
            menu_table.add_row("3", "⚙️  Adjust Agent Settings")
            menu_table.add_row("4", "🔄 Reset All Settings")
            menu_table.add_row("5", "❌ Exit Settings")
            
            menu_panel = Panel(
                menu_table,
                title="[bold bright_yellow]Settings Menu[/bold bright_yellow]",
                border_style="bright_yellow"
            )
            console.print(menu_panel)
            console.print()
            
            choice = Prompt.ask(
                "[bold bright_cyan]What would you like to do?[/bold bright_cyan]",
                choices=["1", "2", "3", "4", "5"],
                default="5"
            )
            
            if choice == "1":
                self.edit_ai_provider()
            elif choice == "2":
                self.edit_clickhouse()
            elif choice == "3":
                self.edit_agent_settings()
            elif choice == "4":
                new_config = self.reset_config()
                if new_config:
                    break
            elif choice == "5":
                ui.show_success("Settings saved! 👋")
                break
            
            if choice != "5":
                console.print()
                Prompt.ask("[dim]Press Enter to continue...[/dim]", default="")

from rich.align import Align