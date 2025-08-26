"""
Minimal, elegant CLI interface inspired by Claude Code
Clean animations, creative loading messages, no boxy layouts
"""

from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.table import Table
from rich.rule import Rule
from rich.box import SIMPLE
from rich.spinner import Spinner
from rich.markdown import Markdown
import time
import random
from typing import Optional, List, Dict, Any
import asyncio

console = Console()

# Creative loading messages inspired by Claude Code
THINKING_MESSAGES = [
    "Analyzing your request...",
    "Synthesizing insights...",
    "Forging connections...",
    "Crunching data...",
    "Sparkling through possibilities...",
    "Weaving patterns...",
    "Distilling knowledge...",
    "Crafting response...",
    "Brewing intelligence...",
    "Conjuring solutions...",
]

TOOL_MESSAGES = {
    "show_tables": "Exploring database structure...",
    "execute_query": "Running SQL magic...",
    "describe_table": "Examining table schema...",
    "analyze_table": "Diving deep into data...",
    "create_visualization": "Painting data stories...",
    "load_data_from_file": "Ingesting fresh data...",
    "export_query_results": "Packaging results...",
    "optimize_table": "Tuning performance...",
    "get_system_info": "Gathering system intel...",
}

# Spinning characters for smooth animation
SPINNERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

class MinimalInterface:
    """Claude Code inspired minimal interface"""
    
    def __init__(self):
        self.console = Console()
        self.spinner_index = 0
        
    def _get_spinner(self) -> str:
        """Get next spinner character"""
        char = SPINNERS[self.spinner_index % len(SPINNERS)]
        self.spinner_index += 1
        return char
        
    def show_welcome_screen(self):
        """Show minimal welcome - no boxes"""
        self.console.clear()
        self.console.print()
        self.console.print()
        
        # Simple centered title with better spacing
        title = Text("✨ Proto", style="bold bright_cyan")
        subtitle = Text("ClickHouse AI Agent", style="dim bright_white")
        
        self.console.print(Align.center(title))
        self.console.print(Align.center(subtitle))
        self.console.print()
        self.console.print()
        
    def show_connection_status(self, host: str, port: int, database: str, connected: bool = True):
        """Minimal connection status with better spacing"""
        if connected:
            status = f"[dim bright_green]●[/dim bright_green] [bright_white]Connected to {host}:{port}[/bright_white]"
            db_info = f"[dim]Database: {database}[/dim]"
        else:
            status = f"[dim red]●[/dim red] [bright_white]Disconnected[/bright_white]"
            db_info = ""
            
        self.console.print(status)
        if db_info:
            self.console.print(db_info)
        self.console.print()
        self.console.print()
        
    def show_thinking_animation(self, message: str = None):
        """Smooth thinking animation like Claude Code"""
        if not message:
            message = random.choice(THINKING_MESSAGES)
            
        # Create simple spinner with message using Rich's built-in spinner
        spinner_with_text = Spinner("dots", text=message, style="dim bright_cyan")
        
        return Live(spinner_with_text, refresh_per_second=10, transient=True)
        
    def show_tool_execution(self, tool_name: str, description: str = "", arguments: dict = None):
        """Smooth tool execution animation with arguments display"""
        message = TOOL_MESSAGES.get(tool_name, f"Running {tool_name.replace('_', ' ')}...")
        
        # Show tool call with arguments before starting animation
        self.console.print(f"[dim bright_yellow]🔧[/dim bright_yellow] [bright_white]Executing tool: {tool_name}[/bright_white]")
        if arguments:
            # Show key arguments (truncate if too long)
            for key, value in arguments.items():
                display_value = str(value)
                if len(display_value) > 100:
                    display_value = display_value[:97] + "..."
                self.console.print(f"[dim]{key}>[/dim] [dim bright_white]{display_value}[/dim bright_white]")
        
        # Create simple spinner with message using Rich's built-in spinner
        spinner_with_text = Spinner("dots", text=message, style="dim bright_yellow")
        
        return Live(spinner_with_text, refresh_per_second=10, transient=True)
        
    def show_success(self, message: str):
        """Minimal success message with spacing"""
        self.console.print(f"[dim bright_green]✓[/dim bright_green] [bright_white]{message}[/bright_white]")
        
    def show_error(self, message: str):
        """Minimal error message with spacing"""
        self.console.print(f"[dim red]✗[/dim red] [bright_white]{message}[/bright_white]")
        
    def show_warning(self, message: str):
        """Minimal warning message with spacing"""
        self.console.print(f"[dim yellow]⚠[/dim yellow] [bright_white]{message}[/bright_white]")
        
    def show_agent_response(self, message: str):
        """Clean agent response with spacing"""
        # Add a subtle bullet point like Claude Code
        self.console.print(f"[dim bright_blue]●[/dim bright_blue] [bright_white]{message}[/bright_white]")
        self.console.print()
        
    def show_markdown(self, markdown_text: str):
        """Render markdown text beautifully in the terminal"""
        try:
            markdown = Markdown(markdown_text)
            self.console.print(markdown)
            self.console.print()
        except Exception as e:
            # Fallback to plain text if markdown parsing fails
            self.console.print(f"[bright_white]{markdown_text}[/bright_white]")
            self.console.print()
            
    def show_agent_response_markdown(self, message: str):
        """Show agent response with markdown rendering if it contains markdown"""        
        # Check if the message contains markdown-like syntax
        if any(marker in message for marker in ['##', '**', '|', '```', '- ', '1. ', '*']):
            # Add a subtle bullet point and render as markdown
            self.console.print(f"[dim bright_blue]●[/dim bright_blue] [dim]Response:[/dim]")
            self.console.print()
            self.show_markdown(message)
        else:
            # Fall back to regular response
            self.show_agent_response(message)
            
    def show_reasoning(self, reasoning_text: str):
        """Show reasoning in muted quote style like Notion"""
        if not reasoning_text:
            return
            
        # Split reasoning into lines for proper quote formatting
        lines = reasoning_text.split('\n')
        
        self.console.print(f"[dim bright_yellow]💭[/dim bright_yellow] [dim italic]Thinking:[/dim italic]")
        
        for line in lines:
            if line.strip():  # Only show non-empty lines
                # Quote-style formatting with muted colors and smaller appearance
                self.console.print(f"[dim]│[/dim] [dim italic]{line.strip()}[/dim italic]")
        
        self.console.print()  # Add spacing after reasoning
        
    def show_query_execution(self, query: str):
        """Show the actual SQL query being executed"""
        self.console.print(f"[dim bright_magenta]SQL>[/dim bright_magenta] [dim bright_white]{query}[/dim bright_white]")
        self.console.print()
        
    def show_data_table(self, data: List[Dict], title: str = "Results", max_rows: int = 10, total_rows: int = None):
        """Smart data table display with large dataset handling"""
        if not data:
            self.console.print(f"[dim]No data to display[/dim]")
            return
            
        total_rows = total_rows or len(data)
        
        # Handle large datasets with warnings and tips
        if total_rows > 1000:
            self.console.print()
            self.console.print(f"[dim bright_yellow]📊[/dim bright_yellow] [bright_white]Large dataset: {total_rows:,} total rows[/bright_white]")
            self.console.print(f"[dim]Showing first {min(max_rows, len(data))} rows. Use LIMIT for specific ranges.[/dim]")
        
        # Handle wide tables (many columns)
        columns = list(data[0].keys()) if data else []
        display_columns = columns
        
        if len(columns) > 8:
            self.console.print()
            self.console.print(f"[dim bright_blue]📋[/dim bright_blue] [bright_white]Wide table: {len(columns)} columns[/bright_white]")
            self.console.print(f"[dim]Showing first 8 columns. Use SELECT col1, col2... for specific columns.[/dim]")
            display_columns = columns[:8]
        
        # Create minimal table with smart title
        rows_text = f"{min(len(data), max_rows):,} of {total_rows:,}" if total_rows != len(data) else f"{len(data):,}"
        table_title = f"{title} ({rows_text} rows)"
        
        table = Table(title=table_title, box=SIMPLE, show_header=True, header_style="dim bright_cyan")
        
        # Add columns with smart truncation
        for key in display_columns:
            display_key = str(key).replace('_', ' ').title()
            if len(display_key) > 15:
                display_key = display_key[:12] + "..."
            table.add_column(display_key, style="bright_white")
        
        # Add rows with smart content truncation
        displayed_rows = min(max_rows, len(data))
        for i, row in enumerate(data[:displayed_rows]):
            row_values = []
            for key in display_columns:
                value = str(row.get(key, ''))
                
                # Smart truncation based on content
                if len(value) > 50:
                    if value.replace('.', '').replace('-', '').replace(',', '').isdigit():
                        # Numeric - show with ellipsis
                        row_values.append(value[:20] + "...")
                    elif len(value) > 100:
                        # Very long text - show start and end
                        row_values.append(value[:25] + "..." + value[-15:])
                    else:
                        # Regular text - truncate with ellipsis
                        row_values.append(value[:45] + "...")
                else:
                    row_values.append(value)
                    
            table.add_row(*row_values)
        
        # Show pagination indicator
        if len(data) > displayed_rows or total_rows > len(data):
            remaining = total_rows - displayed_rows
            if remaining > 0:
                pagination_text = f"... and {remaining:,} more rows"
                table.add_row(*[f"[dim]{pagination_text}[/dim]" if i == 0 else "[dim]...[/dim]" 
                               for i in range(len(display_columns))])
        
        self.console.print()
        self.console.print(table)
        
        # Show helpful tips for large datasets
        if total_rows > 100:
            self.console.print()
            self.console.print(f"[dim bright_cyan]💡[/dim bright_cyan] [dim]Tips: Use LIMIT, WHERE clauses, or ask to export results[/dim]")
        elif len(columns) > 8:
            self.console.print()
            self.console.print(f"[dim bright_cyan]💡[/dim bright_cyan] [dim]Tip: Use SELECT specific_columns FROM table for focused results[/dim]")
            
        self.console.print()
        
    def show_query_result(self, query: str, result_data: List[Dict], execution_time: float = None):
        """Show query and results cleanly"""
        # Show execution time if provided
        if execution_time is not None:
            self.console.print(f"[dim]Query executed in {execution_time:.2f}s[/dim]")
            
        # Show results
        self.show_data_table(result_data, "Results")
        
    def show_user_input_prompt(self):
        """Minimal input prompt like Claude Code"""
        self.console.print()
        # Subtle separator
        rule = Rule(style="dim")
        self.console.print(rule)
        self.console.print()
        
        # Simple prompt
        prompt_text = Text.assemble(
            ("> ", "dim bright_cyan"),
        )
        self.console.print(prompt_text, end="")
        
    def show_goodbye(self):
        """Minimal goodbye with spacing"""
        self.console.print()
        self.console.print()
        self.console.print("[dim bright_cyan]✨[/dim bright_cyan] [bright_white]Thanks for using Proto![/bright_white]")
        self.console.print()
        
    def show_statistics(self, stats: Dict[str, Any]):
        """Minimal stats display"""
        for key, value in stats.items():
            display_key = key.replace('_', ' ').title()
            self.console.print(f"[dim]{display_key}:[/dim] [bright_white]{value}[/bright_white]")
            
    def animate_typing(self, text: str, delay: float = 0.02):
        """Smooth typing animation"""
        for char in text:
            self.console.print(char, end="", style="bright_white")
            time.sleep(delay)
        self.console.print()

# Global instance
ui = MinimalInterface()