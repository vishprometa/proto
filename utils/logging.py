"""
Logging configuration for ClickHouse AI Agent
"""

import sys
from pathlib import Path
from typing import Any, Dict

import structlog
from rich.console import Console
from rich.logging import RichHandler

console = Console()

def setup_logging(verbose: bool = False, quiet_mode: bool = False) -> None:
    """Setup structured logging with rich formatting"""
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Setup Python logging
    import logging
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(logs_dir / "moja.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Rich handler for console output (only if not in quiet mode)
    rich_handler = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        rich_tracebacks=True
    )
    rich_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    # Only add console handler if not in quiet mode
    if not quiet_mode:
        root_logger.addHandler(rich_handler)
    else:
        # In quiet mode, set console logging to ERROR level only
        root_logger.setLevel(logging.ERROR)
    
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("clickhouse_connect").setLevel(logging.INFO)

def get_logger(name: str) -> Any:
    """Get a structured logger instance"""
    return structlog.get_logger(name)

class ToolExecutionLogger:
    """Logger for tool execution with rich formatting"""
    
    def __init__(self, console: Console):
        self.console = console
        self.logger = get_logger("tool_execution")
    
    def log_tool_start(self, tool_name: str, arguments: Dict[str, Any]):
        """Log tool execution start"""
        self.console.print(f"[blue]🔧 Executing tool:[/blue] [bold]{tool_name}[/bold]")
        self.logger.info("tool_execution_start", tool=tool_name, arguments=arguments)
    
    def log_tool_success(self, tool_name: str, result: str):
        """Log successful tool execution"""
        self.console.print(f"[green]✓ Tool completed:[/green] [bold]{tool_name}[/bold]")
        self.logger.info("tool_execution_success", tool=tool_name, result_length=len(result))
    
    def log_tool_error(self, tool_name: str, error: str):
        """Log tool execution error"""
        self.console.print(f"[red]❌ Tool failed:[/red] [bold]{tool_name}[/bold] - {error}")
        self.logger.error("tool_execution_error", tool=tool_name, error=error)
    
    def log_query_execution(self, query: str, duration: float, rows: int):
        """Log ClickHouse query execution"""
        self.console.print(f"[cyan]📊 Query executed:[/cyan] {duration:.2f}s, {rows} rows")
        self.logger.info("query_execution", query=query[:100], duration=duration, rows=rows)