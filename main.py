#!/usr/bin/env python3
"""
Proto - ClickHouse AI Agent
A beautiful CLI AI agent for ClickHouse database analysis and operations.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agent import ClickHouseAgent
from config.settings import load_config
from utils.logging import setup_logging
from ui.minimal_interface import ui
from ui.onboarding import needs_onboarding, OnboardingFlow
from ui.settings_manager import SettingsManager

app = typer.Typer(
    name="proto",
    help="🚀 ClickHouse AI Agent - Intelligent database analysis and operations",
    rich_markup_mode="rich",
    no_args_is_help=False,
    add_completion=False,
)

console = Console()

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Default entry: start interactive chat (with onboarding on first run)."""
    if ctx.invoked_subcommand is not None:
        return
    # Explicitly pass defaults so Typer doesn't inject OptionInfo objects
    ctx.invoke(
        chat,
        config_file=None,
        host=None,
        port=None,
        username=None,
        password=None,
        database=None,
        verbose=False,
    )


@app.command()
def chat(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c", 
        help="Path to configuration file"
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        "-h",
        help="ClickHouse host"
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port", 
        "-p",
        help="ClickHouse port"
    ),
    username: Optional[str] = typer.Option(
        None,
        "--username",
        "-u",
        help="ClickHouse username"
    ),
    password: Optional[str] = typer.Option(
        None,
        "--password",
        help="ClickHouse password"
    ),
    database: Optional[str] = typer.Option(
        None,
        "--database",
        "-d",
        help="ClickHouse database"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """Start interactive chat with ClickHouse AI Agent"""
    
    # Setup logging (quiet mode for beautiful UI unless verbose)
    setup_logging(verbose=verbose, quiet_mode=not verbose)
    
    # Check if onboarding is needed
    if needs_onboarding():
        onboarding = OnboardingFlow()
        onboarding.run_onboarding()
        # After onboarding, rely on default config discovery
        config_file = None
    
    # Load configuration
    config = load_config(
        config_file=config_file,
        host=host,
        port=port,
        username=username,
        password=password,
        database=database
    )
    
    # Display beautiful welcome screen
    ui.show_welcome_screen()
    
    # Show connection status
    ui.show_connection_status(
        host=config.host,
        port=config.port,
        database=config.database,
        connected=True
    )
    
    # Start the agent
    try:
        agent = ClickHouseAgent(config)
        asyncio.run(agent.start_interactive_session())
    except KeyboardInterrupt:
        ui.show_goodbye()
        sys.exit(0)
    except Exception as e:
        ui.show_error(str(e))
        sys.exit(1)

@app.command()
def query(
    sql: str = typer.Argument(..., help="SQL query to execute"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv"),
    save_to: Optional[Path] = typer.Option(None, "--save", "-s", help="Save results to file")
):
    """Execute a single SQL query"""
    
    config = load_config(config_file=config_file)
    
    try:
        agent = ClickHouseAgent(config)
        asyncio.run(agent.execute_single_query(sql, output_format, save_to))
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)

@app.command()
def analyze(
    table: str = typer.Argument(..., help="Table name to analyze"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c"),
    deep: bool = typer.Option(False, "--deep", help="Perform deep analysis")
):
    """Analyze a specific table"""
    
    config = load_config(config_file=config_file)
    
    try:
        agent = ClickHouseAgent(config)
        asyncio.run(agent.analyze_table(table, deep=deep))
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)

@app.command()
def load_data(
    file_path: Path = typer.Argument(..., help="Path to data file"),
    table: str = typer.Argument(..., help="Target table name"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c"),
    create_table: bool = typer.Option(True, "--create-table", help="Create table if it doesn't exist"),
    batch_size: int = typer.Option(10000, "--batch-size", help="Batch size for data loading")
):
    """Load data from file into ClickHouse"""
    
    config = load_config(config_file=config_file)
    
    try:
        agent = ClickHouseAgent(config)
        asyncio.run(agent.load_data_from_file(file_path, table, create_table, batch_size))
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)

@app.command()
def settings():
    """Manage Proto settings and configuration"""
    settings_manager = SettingsManager()
    settings_manager.run_settings_menu()

@app.command()
def clear():
    """Clear all configuration and start fresh"""
    from pathlib import Path
    from rich.prompt import Confirm
    
    # Config file locations
    config_files = [
        Path.home() / ".config" / "proto" / "proto-config.json",
        Path("proto-config.json"),
        Path(".env")
    ]
    
    existing_files = [f for f in config_files if f.exists()]
    
    if not existing_files:
        console.print("[yellow]No configuration files found to clear.[/yellow]")
        return
    
    console.print("[bold red]⚠️  This will delete all Proto configuration:[/bold red]")
    for file in existing_files:
        console.print(f"  • {file}")
    console.print()
    
    if Confirm.ask("[bold red]Are you sure you want to clear all configuration?[/bold red]"):
        for file in existing_files:
            try:
                file.unlink()
                console.print(f"[green]✓[/green] Deleted {file}")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to delete {file}: {e}")
        
        console.print()
        console.print("[green]🎉 Configuration cleared! Run 'proto' to start fresh onboarding.[/green]")
    else:
        console.print("[blue]Configuration clearing cancelled.[/blue]")

@app.command()
def refresh_template():
    """Refresh chat template from Hugging Face repository"""
    from providers.local_llm import LocalLLMProvider
    
    console.print("[bold cyan]🔄 Refreshing Chat Template[/bold cyan]")
    console.print("Fetching latest chat template from Hugging Face repository...")
    console.print()
    
    try:
        # Create a minimal provider instance just for template refresh
        provider = LocalLLMProvider.__new__(LocalLLMProvider)
        provider.model_name = "vishprometa/clickhouse-qwen3-1.7b-gguf"
        provider.chat_template_file = "chat_template.jinja"
        provider.chat_template_url = f"https://huggingface.co/{provider.model_name}/resolve/main/{provider.chat_template_file}"
        provider.cache_dir = os.path.expanduser("~/.cache/llama.cpp")
        provider.chat_template_path = os.path.join(provider.cache_dir, f"{provider.model_name.replace('/', '_')}_{provider.chat_template_file}")
        
        # Add the announce method for user feedback
        def _announce(message: str):
            console.print(f"[cyan]{message}[/cyan]")
        provider._announce = _announce
        
        # Refresh the template
        success = provider.refresh_chat_template()
        
        if success:
            console.print()
            console.print("[green]✅ Chat template refreshed successfully![/green]")
            console.print(f"[dim]Template location: {provider.chat_template_path}[/dim]")
            
            # Show template info
            if os.path.exists(provider.chat_template_path):
                with open(provider.chat_template_path, 'r') as f:
                    content = f.read()
                    console.print(f"[dim]Template size: {len(content)} characters[/dim]")
                    
                    # Show a preview of the template
                    preview = content[:200] + "..." if len(content) > 200 else content
                    console.print()
                    console.print("[bold]Template Preview:[/bold]")
                    console.print(f"[dim]{preview}[/dim]")
        else:
            console.print()
            console.print("[red]❌ Failed to refresh chat template[/red]")
            console.print("[yellow]The agent will use the default template if available[/yellow]")
            sys.exit(1)
            
    except Exception as e:
        console.print()
        console.print(f"[red]❌ Error refreshing template: {e}[/red]")
        sys.exit(1)

@app.command()
def version():
    """Show version information"""
    console.print("[bold cyan]Proto ClickHouse AI Agent[/bold cyan]")
    console.print("Version: 1.0.0")
    console.print("Built with ❤️  for ClickHouse analysis")

if __name__ == "__main__":
    app()