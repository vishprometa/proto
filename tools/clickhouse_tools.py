"""
ClickHouse-specific tools for the AI agent
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import clickhouse_connect
from clickhouse_connect.driver import Client
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from config.settings import ClickHouseConfig
from utils.logging import get_logger, ToolExecutionLogger

logger = get_logger(__name__)
console = Console()
tool_logger = ToolExecutionLogger(console)

class ClickHouseConnection:
    """ClickHouse database connection manager"""
    
    def __init__(self, config: ClickHouseConfig):
        self.config = config
        self.client: Optional[Client] = None
    
    async def connect(self) -> Client:
        """Establish connection to ClickHouse"""
        if not self.client:
            try:
                self.client = clickhouse_connect.get_client(
                    host=self.config.host,
                    port=self.config.port,
                    username=self.config.username,
                    password=self.config.password,
                    database=self.config.database,
                    secure=self.config.secure
                )
                
                # Test connection
                result = self.client.query("SELECT 1 as test")
                logger.info("ClickHouse connection established successfully")
                
                return self.client
            except Exception as e:
                logger.error(f"Failed to connect to ClickHouse: {e}")
                raise
        
        return self.client
    
    def close(self):
        """Close ClickHouse connection"""
        if self.client:
            self.client.close()
            self.client = None

# Simplified tool definitions - just the essentials for data analysis
CLICKHOUSE_TOOLS = [
    {
        "name": "list_databases",
        "description": "List all available databases in ClickHouse, returns JSON with database names",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "switch_database",
        "description": "Switch to a different database in ClickHouse, returns confirmation with new database name",
        "input_schema": {
            "type": "object",
            "properties": {
                "database_name": {
                    "type": "string",
                    "description": "Name of the database to switch to"
                }
            },
            "required": ["database_name"]
        }
    },
    {
        "name": "execute_clickhouse_query",
        "description": "Execute SQL queries for SMALL to MEDIUM datasets (≤100 rows) or when immediate analysis is needed. Returns structured data for analysis. Use for: quick queries, aggregations, counts, schema exploration, small data previews.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query to execute"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "list_tables",
        "description": "List all tables in the current database, returns JSON with table names and count",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_table_schema",
        "description": "Get the schema/structure of a specific table, returns JSON with column details",
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Name of the table to get schema for"
                }
            },
            "required": ["table_name"]
        }
    },
    {
        "name": "search_table",
        "description": "Search and preview data in a table with optional filtering, returns JSON with row data",
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Name of the table to search"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of rows to return (default: 100)",
                    "default": 100
                },
                "where_clause": {
                    "type": "string",
                    "description": "Optional WHERE clause for filtering"
                }
            },
            "required": ["table_name"]
        }
    },
    {
        "name": "export_data_to_csv",
        "description": "Export LARGE datasets (>100 rows) to CSV file with truncated analysis preview. Use when: user requests data export, result set is large, need full dataset for external analysis, generating reports. Shows limited rows for analysis while saving complete data to CSV.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query to execute and export"
                },
                "filename": {
                    "type": "string",
                    "description": "Optional filename for the CSV export (without extension)"
                },
                "analysis_limit": {
                    "type": "integer",
                    "description": "Number of rows to show for analysis (default: 50)",
                    "default": 50
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "stop_agent",
        "description": "Stop the agent and end the conversation with a summary",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Summary of what was accomplished"
                }
            },
            "required": ["summary"]
        }
    }
]

class ClickHouseToolExecutor:
    """Executes ClickHouse-specific tools"""
    
    def __init__(self, connection: ClickHouseConnection):
        self.connection = connection
        self.client = None
    
    async def initialize(self):
        """Initialize the connection"""
        self.client = await self.connection.connect()
    
    async def execute_clickhouse_query(self, query: str) -> str:
        """Execute any SQL query and return formatted results"""
        
        tool_logger.log_tool_start("execute_clickhouse_query", {"query": query[:100]})
        
        try:
            # Show the actual SQL query being executed
            from ui.minimal_interface import ui
            ui.show_query_execution(query)
            
            # Execute query
            import time
            start_time = time.time()
            result = self.client.query(query)
            duration = time.time() - start_time
            
            # Convert to DataFrame for easier manipulation
            import pandas as pd
            df = pd.DataFrame(result.result_rows, columns=result.column_names)
            
            tool_logger.log_query_execution(query, duration, len(df))
            
            # Always show results in table format for better analysis
            from ui.minimal_interface import ui
            
            if len(df) > 0:
                # Convert to list of dicts for the UI
                data_list = df.to_dict('records')
                
                # Show with smart large dataset handling
                display_limit = 100 if len(df) > 100 else len(df)
                
                ui.show_data_table(
                    data_list[:display_limit], 
                    title="Query Results", 
                    max_rows=display_limit,
                    total_rows=len(df)
                )
                
                # For queries that return structured data, provide JSON
                # AGGRESSIVE data limiting to prevent context overflow
                if len(df) <= 10:  # Only return full JSON for very small results
                    result_data = {
                        "query": query,
                        "rows": data_list,
                        "row_count": len(df),
                        "columns": list(df.columns),
                        "summary": f"Query returned {len(df)} rows"
                    }
                    output = json.dumps(result_data, indent=2, default=str)
                elif len(df) <= 50:  # Return very limited rows for small results
                    limited_data = {
                        "query": query,
                        "rows": data_list[:5],  # Only first 5 rows for context
                        "row_count": len(df),
                        "columns": list(df.columns),
                        "summary": f"Query returned {len(df)} rows (showing first 5 for analysis). Full results displayed above."
                    }
                    output = json.dumps(limited_data, indent=2, default=str)
                else:
                    # For any larger results, just return summary to avoid context overflow
                    output = f"SUCCESS: Query returned {len(df):,} rows with {len(df.columns)} columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}. Results displayed in table above. Use export_data_to_csv for large dataset analysis."
                    
            else:
                ui.console.print("[dim yellow]No results found[/dim yellow]")
                result_data = {
                    "query": query,
                    "rows": [],
                    "row_count": 0,
                    "columns": [],
                    "summary": "No results found"
                }
                output = json.dumps(result_data, indent=2)
            
            tool_logger.log_tool_success("execute_clickhouse_query", f"Returned {len(df)} rows")
            return output
            
        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"
            tool_logger.log_tool_error("execute_clickhouse_query", error_msg)
            return error_msg
    
    async def list_tables(self) -> str:
        """List all tables in the database"""
        
        tool_logger.log_tool_start("list_tables", {})
        
        try:
            query = "SHOW TABLES"
            result = self.client.query(query)
            df = pd.DataFrame(result.result_rows, columns=result.column_names)
            
            if len(df) > 0:
                # Get table names as a list
                table_names = [str(row.iloc[0]) for _, row in df.iterrows()]
                
                # Display the table nicely
                table = Table(title=f"Tables in database ({len(df)} tables)")
                table.add_column("Table Name")
                
                for table_name in table_names:
                    table.add_row(table_name)
                
                console.print(table)
                
                # Return JSON data
                result_data = {
                    "tables": table_names,
                    "count": len(table_names)
                }
                output = json.dumps(result_data, indent=2)
            else:
                result_data = {
                    "tables": [],
                    "count": 0
                }
                output = json.dumps(result_data, indent=2)
            
            tool_logger.log_tool_success("list_tables", f"Found {len(table_names) if len(df) > 0 else 0} tables")
            return output
            
        except Exception as e:
            error_msg = f"Failed to list tables: {str(e)}"
            tool_logger.log_tool_error("list_tables", error_msg)
            return json.dumps({"error": error_msg})
    
    async def get_table_schema(self, table_name: str) -> str:
        """Get the schema/structure of a table"""
        
        tool_logger.log_tool_start("get_table_schema", {"table_name": table_name})
        
        try:
            # Get table structure
            describe_query = f"DESCRIBE TABLE {table_name}"
            result = self.client.query(describe_query)
            df = pd.DataFrame(result.result_rows, columns=result.column_names)
            
            # Display table structure
            table = Table(title=f"Schema: {table_name}")
            table.add_column("Column")
            table.add_column("Type")
            table.add_column("Default Type")
            table.add_column("Default Expression")
            
            # Build JSON schema data
            columns = []
            for _, row in df.iterrows():
                column_info = {
                    "name": str(row['name']),
                    "type": str(row['type']),
                    "default_type": str(row.get('default_type', '')),
                    "default_expression": str(row.get('default_expression', ''))
                }
                columns.append(column_info)
                
                # Add to display table
                table.add_row(
                    column_info["name"],
                    column_info["type"],
                    column_info["default_type"],
                    column_info["default_expression"]
                )
            
            console.print(table)
            
            # Return JSON data
            result_data = {
                "table_name": table_name,
                "columns": columns,
                "column_count": len(columns)
            }
            output = json.dumps(result_data, indent=2)
            
            tool_logger.log_tool_success("get_table_schema", f"Retrieved schema for {table_name} with {len(columns)} columns")
            return output
            
        except Exception as e:
            error_msg = f"Failed to get schema for table {table_name}: {str(e)}"
            tool_logger.log_tool_error("get_table_schema", error_msg)
            return json.dumps({"error": error_msg, "table_name": table_name})
    
    async def search_table(self, table_name: str, limit: int = 100, where_clause: str = None) -> str:
        """Search and preview data in a table"""
        
        tool_logger.log_tool_start("search_table", {"table_name": table_name, "limit": limit, "where_clause": where_clause})
        
        try:
            # Build query
            query = f"SELECT * FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
            query += f" LIMIT {limit}"
            
            # Execute query directly to get structured data
            result = self.client.query(query)
            df = pd.DataFrame(result.result_rows, columns=result.column_names)
            
            # Display the results
            from ui.minimal_interface import ui
            if len(df) > 0:
                data_list = df.to_dict('records')
                ui.show_data_table(
                    data_list, 
                    title=f"Search Results: {table_name}", 
                    max_rows=min(50, len(df)),
                    total_rows=len(df)
                )
                
                # Return JSON data
                result_data = {
                    "table_name": table_name,
                    "query": query,
                    "rows": data_list,
                    "row_count": len(df),
                    "columns": list(df.columns)
                }
                output = json.dumps(result_data, indent=2, default=str)  # default=str handles dates/timestamps
            else:
                ui.console.print("[dim yellow]No results found[/dim yellow]")
                result_data = {
                    "table_name": table_name,
                    "query": query,
                    "rows": [],
                    "row_count": 0,
                    "columns": []
                }
                output = json.dumps(result_data, indent=2)
            
            tool_logger.log_tool_success("search_table", f"Searched table {table_name}, found {len(df)} rows")
            return output
            
        except Exception as e:
            error_msg = f"Failed to search table {table_name}: {str(e)}"
            tool_logger.log_tool_error("search_table", error_msg)
            return json.dumps({"error": error_msg, "table_name": table_name})
    
    async def export_data_to_csv(
        self, 
        query: str, 
        filename: str = None, 
        analysis_limit: int = 50
    ) -> str:
        """Export query results to CSV with truncated analysis display"""
        
        tool_logger.log_tool_start("export_data_to_csv", {
            "query": query[:100], 
            "filename": filename,
            "analysis_limit": analysis_limit
        })
        
        try:
            from ui.minimal_interface import ui
            import os
            from datetime import datetime
            
            # Show the query being executed
            ui.show_query_execution(query)
            
            # Execute query
            import time
            start_time = time.time()
            result = self.client.query(query)
            duration = time.time() - start_time
            
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(result.result_rows, columns=result.column_names)
            
            if len(df) == 0:
                return "No data returned from query"
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data_export_{timestamp}"
            
            # Ensure exports directory exists
            os.makedirs("exports", exist_ok=True)
            filepath = os.path.join("exports", f"{filename}.csv")
            
            # Export full data to CSV
            df.to_csv(filepath, index=False)
            
            file_size = os.path.getsize(filepath)
            file_size_mb = file_size / (1024 * 1024)
            
            # Show truncated data for analysis
            display_rows = min(analysis_limit, len(df))
            truncated_df = df.head(display_rows)
            data_list = truncated_df.to_dict('records')
            
            # Display the truncated results
            ui.show_data_table(
                data_list, 
                title=f"Analysis Preview ({display_rows} of {len(df):,} rows)", 
                max_rows=display_rows,
                total_rows=len(df)
            )
            
            # Show export information
            ui.console.print()
            ui.console.print(f"[dim bright_green]📁[/dim bright_green] [bright_white]Data Export Complete![/bright_white]")
            ui.console.print(f"[dim]• Full dataset: {filepath}[/dim]")
            ui.console.print(f"[dim]• File size: {file_size_mb:.2f} MB[/dim]")
            ui.console.print(f"[dim]• Total rows: {len(df):,}[/dim]")
            ui.console.print(f"[dim]• Analysis shown: {display_rows} rows[/dim]")
            ui.console.print(f"[yellow]💡 Open the CSV file for complete data analysis[/yellow]")
            ui.console.print()
            
            tool_logger.log_tool_success("export_data_to_csv", f"Exported {len(df)} rows to {filepath}")
            
            # Return structured data for the LLM
            result_data = {
                "query": query,
                "total_rows": len(df),
                "analysis_rows": truncated_df.to_dict('records'),
                "export_file": filepath,
                "file_size_mb": round(file_size_mb, 2),
                "columns": list(df.columns),
                "summary": f"Exported {len(df):,} rows to CSV. Analysis shows {display_rows} sample rows.",
                "message": f"Full dataset with {len(df):,} rows exported to {filepath}. Analysis limited to {display_rows} rows for display. Open CSV file for complete data."
            }
            
            return json.dumps(result_data, indent=2, default=str)
            
        except Exception as e:
            error_msg = f"CSV export failed: {str(e)}"
            tool_logger.log_tool_error("export_data_to_csv", error_msg)
            return error_msg
    
    async def export_query_results(
        self, 
        query: str, 
        filename: str = None, 
        format: str = "csv",
        limit: int = None
    ) -> str:
        """Export large query results to file for datasets too big to display"""
        
        tool_logger.log_tool_start("export_query_results", {
            "query": query[:100], 
            "format": format, 
            "limit": limit
        })
        
        try:
            from ui.minimal_interface import ui
            import os
            from datetime import datetime
            
            # Add limit if specified
            if limit and "LIMIT" not in query.upper():
                query = f"{query.rstrip(';')} LIMIT {limit}"
            
            # Show the query being executed
            ui.show_query_execution(query)
            
            # Execute query
            import time
            start_time = time.time()
            result = self.client.query(query)
            duration = time.time() - start_time
            
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(result.result_rows, columns=result.column_names)
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"clickhouse_export_{timestamp}.{format}"
            
            # Ensure exports directory exists
            os.makedirs("exports", exist_ok=True)
            filepath = os.path.join("exports", filename)
            
            # Export based on format
            if format.lower() == "csv":
                df.to_csv(filepath, index=False)
            elif format.lower() == "json":
                df.to_json(filepath, orient="records", indent=2)
            elif format.lower() == "excel":
                df.to_excel(filepath, index=False)
            else:
                return f"Unsupported export format: {format}. Use csv, json, or excel."
            
            file_size = os.path.getsize(filepath)
            file_size_mb = file_size / (1024 * 1024)
            
            ui.console.print()
            ui.console.print(f"[dim bright_green]📁[/dim bright_green] [bright_white]Export completed![/bright_white]")
            ui.console.print(f"[dim]File: {filepath}[/dim]")
            ui.console.print(f"[dim]Size: {file_size_mb:.2f} MB ({len(df):,} rows)[/dim]")
            ui.console.print()
            
            tool_logger.log_tool_success("export_query_results", f"Exported {len(df)} rows to {filepath}")
            
            return f"Successfully exported {len(df):,} rows to {filepath} ({file_size_mb:.2f} MB) in {duration:.2f}s"
            
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            tool_logger.log_tool_error("export_query_results", error_msg)
            return error_msg
    



# Simplified OpenAI format tools
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_clickhouse_query",
            "description": "Execute SQL queries for SMALL to MEDIUM datasets (≤100 rows) or when immediate analysis is needed. Returns structured data for analysis. Use for: quick queries, aggregations, counts, schema exploration, small data previews.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List all tables in the current database, returns JSON with table names and count",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_schema",
            "description": "Get the schema/structure of a specific table, returns JSON with column details",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to get schema for"
                    }
                },
                "required": ["table_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_table",
            "description": "Search and preview data in a table with optional filtering, returns JSON with row data",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to search"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of rows to return (default: 100)",
                        "default": 100
                    },
                    "where_clause": {
                        "type": "string",
                        "description": "Optional WHERE clause for filtering"
                    }
                },
                "required": ["table_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "export_data_to_csv",
            "description": "Export LARGE datasets (>100 rows) to CSV file with truncated analysis preview. Use when: user requests data export, result set is large, need full dataset for external analysis, generating reports. Shows limited rows for analysis while saving complete data to CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute and export"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional filename for the CSV export (without extension)"
                    },
                    "analysis_limit": {
                        "type": "integer",
                        "description": "Number of rows to show for analysis (default: 50)",
                        "default": 50
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stop_agent",
            "description": "Stop the agent and end the conversation with a summary CALL THIS TOOL WHEN TASKS ARE DONE AND COMPLETE.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of what was accomplished"
                    }
                },
                "required": ["summary"]
            }
        }
    }
]