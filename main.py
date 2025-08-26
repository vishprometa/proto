#!/usr/bin/env python3
"""
Moja - ClickHouse AI Agent
A beautiful CLI AI agent for ClickHouse database analysis and operations.
"""

import asyncio
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
    name="moja",
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
    """Manage Moja settings and configuration"""
    settings_manager = SettingsManager()
    settings_manager.run_settings_menu()

@app.command()
def version():
    """Show version information"""
    console.print("[bold cyan]Moja ClickHouse AI Agent[/bold cyan]")
    console.print("Version: 1.0.4")
    console.print("Built with ❤️  for ClickHouse analysis")

if __name__ == "__main__":
    app()