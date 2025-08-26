"""
Beautiful, user-friendly CLI interface for ClickHouse AI Agent
Designed to feel like a modern app with smooth animations and clean design
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.columns import Columns
from rich.box import ROUNDED, SIMPLE, MINIMAL, DOUBLE
from rich.padding import Padding
from rich.rule import Rule
from rich.emoji import Emoji
from rich.markdown import Markdown
import time
from typing import Optional, List, Dict, Any
import asyncio

console = Console()

class BeautifulInterface:
    """Beautiful, animated CLI interface for ClickHouse AI Agent"""
    
    def __init__(self):
        self.console = Console()
        self.current_step = None
        self.progress = None
        
    def show_welcome_screen(self):
        """Display beautiful welcome screen with animations"""
        self.console.clear()
        
        # Create gradient-like title
        title = Text()
        title.append("✨ ", style="bold bright_yellow")
        title.append("Moja", style="bold bright_blue")
        title.append(" ClickHouse AI Agent", style="bold bright_cyan")
        title.append(" ✨", style="bold bright_yellow")
        
        subtitle = Text("Intelligent Database Analysis & Operations", style="italic bright_white")
        
        welcome_panel = Panel(
            Align.center(
                Text.assemble(
                    title, "\n\n",
                    subtitle, "\n\n",
                    Text("🚀 Ready to explore your data!", style="bold green")
                )
            ),
            box=ROUNDED,
            border_style="bright_cyan",
            padding=(2, 4),
            title="[bold bright_yellow]Welcome[/bold bright_yellow]",
            title_align="center"
        )
        
        self.console.print()
        self.console.print(welcome_panel)
        self.console.print()
        
    def show_connection_status(self, host: str, port: int, database: str, connected: bool = True):
        """Show database connection status with beautiful formatting"""
        
        if connected:
            status_icon = "✅"
            status_text = "Connected"
            status_style = "bold green"
            border_style = "green"
        else:
            status_icon = "❌"
            status_text = "Disconnected"
            status_style = "bold red"
            border_style = "red"
            
        connection_info = Table(show_header=False, box=None, padding=(0, 1))
        connection_info.add_column("Field", style="bright_white")
        connection_info.add_column("Value", style="bright_cyan")
        
        connection_info.add_row("🏠 Host:", f"{host}:{port}")
        connection_info.add_row("🗄️  Database:", database)
        connection_info.add_row("📊 Status:", f"{status_icon} {status_text}")
        
        connection_panel = Panel(
            connection_info,
            title=f"[{status_style}]Database Connection[/{status_style}]",
            border_style=border_style,
            box=ROUNDED,
            padding=(1, 2)
        )
        
        self.console.print(connection_panel)
        
    def show_thinking_animation(self, message: str = "Thinking..."):
        """Show minimal, elegant thinking animation"""
        return Live(
            Align.center(
                Text.from_markup(f"[dim bright_cyan]⠋[/dim bright_cyan] [bright_white]{message}[/bright_white]")
            ),
            refresh_per_second=8
        )
        
    def show_tool_execution(self, tool_name: str, description: str = ""):
        """Show minimal tool execution animation"""
        clean_name = tool_name.replace('_', ' ').title()
        return Live(
            Align.center(
                Text.from_markup(f"[dim bright_yellow]⠸[/dim bright_yellow] [bright_white]{clean_name}[/bright_white]")
            ),
            refresh_per_second=8
        )
        
    def show_success(self, message: str):
        """Show minimal success message"""
        self.console.print(f"[dim bright_green]✓[/dim bright_green] [bright_white]{message}[/bright_white]")
        
    def show_error(self, message: str):
        """Show minimal error message"""
        self.console.print(f"[dim red]✗[/dim red] [bright_white]{message}[/bright_white]")
        
    def show_warning(self, message: str):
        """Show minimal warning message"""
        self.console.print(f"[dim yellow]⚠[/dim yellow] [bright_white]{message}[/bright_white]")
        
    def show_data_table(self, data: List[Dict], title: str = "Results", max_rows: int = 10):
        """Display data in a beautiful table format"""
        if not data:
            self.console.print(Panel(
                Text("No data to display", style="italic bright_white"),
                title=title,
                border_style="bright_blue",
                box=ROUNDED
            ))
            return
            
        # Create table
        table = Table(box=ROUNDED, show_lines=True, header_style="bold bright_cyan")
        
        # Add columns
        if data:
            for key in data[0].keys():
                table.add_column(str(key).replace('_', ' ').title(), style="bright_white")
        
        # Add rows (limit to max_rows)
        for i, row in enumerate(data[:max_rows]):
            table.add_row(*[str(value) for value in row.values()])
            
        if len(data) > max_rows:
            table.add_row(*["..." for _ in data[0].keys()], style="dim")
            
        # Show in panel
        data_panel = Panel(
            table,
            title=f"[bold bright_cyan]{title}[/bold bright_cyan]",
            border_style="bright_cyan",
            box=ROUNDED,
            padding=(1, 2)
        )
        
        self.console.print(data_panel)
        
        if len(data) > max_rows:
            self.console.print(f"[dim]Showing {max_rows} of {len(data)} rows[/dim]")
            
    def show_query_result(self, query: str, result_data: List[Dict], execution_time: float = None):
        """Show query and its results in a beautiful format"""
        
        # Show the query
        query_panel = Panel(
            Markdown(f"```sql\n{query}\n```"),
            title="[bold bright_magenta]Query[/bold bright_magenta]",
            border_style="bright_magenta",
            box=ROUNDED,
            padding=(1, 2)
        )
        self.console.print(query_panel)
        
        # Show execution time if provided
        if execution_time is not None:
            self.console.print(f"[dim]⏱️  Executed in {execution_time:.2f}s[/dim]")
            
        # Show results
        self.show_data_table(result_data, "Query Results")
        
    def show_agent_response(self, message: str):
        """Show minimal agent response"""
        self.console.print(f"[dim bright_blue]→[/dim bright_blue] [bright_white]{message}[/bright_white]")
        
    def show_user_input_prompt(self):
        """Show beautiful input prompt"""
        self.console.print()
        rule = Rule(style="dim")
        self.console.print(rule)
        
        prompt_text = Text.assemble(
            ("💬 ", "bright_cyan"),
            ("You: ", "bold bright_cyan")
        )
        self.console.print(prompt_text, end="")
        
    def show_goodbye(self):
        """Show beautiful goodbye message"""
        goodbye_panel = Panel(
            Align.center(
                Text.assemble(
                    ("👋 ", "bold bright_yellow"),
                    ("Thank you for using Moja ClickHouse AI Agent!", "bold bright_white"),
                    ("\n\n", ""),
                    ("Hope to see you again soon! ✨", "italic bright_cyan")
                )
            ),
            border_style="bright_yellow",
            box=ROUNDED,
            padding=(2, 4)
        )
        self.console.print()
        self.console.print(goodbye_panel)
        self.console.print()
        
    def show_statistics(self, stats: Dict[str, Any]):
        """Show session statistics in a beautiful format"""
        stats_table = Table(show_header=False, box=None, padding=(0, 2))
        stats_table.add_column("Metric", style="bright_white")
        stats_table.add_column("Value", style="bright_cyan")
        
        for key, value in stats.items():
            display_key = key.replace('_', ' ').title()
            stats_table.add_row(f"📊 {display_key}:", str(value))
            
        stats_panel = Panel(
            stats_table,
            title="[bold bright_green]Session Statistics[/bold bright_green]",
            border_style="bright_green",
            box=ROUNDED,
            padding=(1, 2)
        )
        
        self.console.print(stats_panel)
        
    def create_progress_bar(self, description: str, total: int = 100):
        """Create a beautiful progress bar"""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            console=self.console
        )
        
        task = progress.add_task(description, total=total)
        return progress, task
        
    def animate_typing(self, text: str, delay: float = 0.03):
        """Animate typing effect for text"""
        for char in text:
            self.console.print(char, end="", style="bright_white")
            time.sleep(delay)
        self.console.print()  # New line at the end

# Global instance
ui = BeautifulInterface()