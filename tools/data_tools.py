"""
Data loading and analysis tools for ClickHouse AI Agent
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import plotly.graph_objects as go
import plotly.express as px
from rich.console import Console
from rich.progress import Progress, TaskID

from utils.logging import get_logger, ToolExecutionLogger

logger = get_logger(__name__)
console = Console()
tool_logger = ToolExecutionLogger(console)

class DataLoader:
    """Data loading utilities for ClickHouse"""
    
    def __init__(self, client):
        self.client = client
    
    async def load_from_csv(
        self,
        file_path: str,
        table_name: str,
        create_table: bool = True,
        batch_size: int = 10000
    ) -> str:
        """Load data from CSV file into ClickHouse table"""
        
        tool_logger.log_tool_start("load_from_csv", {
            "file_path": file_path,
            "table_name": table_name,
            "create_table": create_table,
            "batch_size": batch_size
        })
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            if len(df) == 0:
                return "CSV file is empty"
            
            # Create table if requested
            if create_table:
                create_sql = self._generate_create_table_sql(df, table_name)
                console.print(f"[cyan]Creating table with SQL:[/cyan]")
                console.print(create_sql)
                
                self.client.command(create_sql)
                console.print(f"[green]✓ Table {table_name} created successfully[/green]")
            
            # Insert data in batches
            total_rows = len(df)
            inserted_rows = 0
            
            with Progress() as progress:
                task = progress.add_task(f"Loading data into {table_name}", total=total_rows)
                
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch_df = df.iloc[start_idx:end_idx]
                    
                    # Insert batch
                    self.client.insert_df(table_name, batch_df)
                    
                    inserted_rows += len(batch_df)
                    progress.update(task, completed=inserted_rows)
            
            result = f"Successfully loaded {inserted_rows:,} rows from {file_path.name} into table {table_name}"
            tool_logger.log_tool_success("load_from_csv", result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to load CSV data: {str(e)}"
            tool_logger.log_tool_error("load_from_csv", error_msg)
            return error_msg
    
    async def load_from_json(
        self,
        file_path: str,
        table_name: str,
        create_table: bool = True,
        batch_size: int = 10000
    ) -> str:
        """Load data from JSON file into ClickHouse table"""
        
        tool_logger.log_tool_start("load_from_json", {
            "file_path": file_path,
            "table_name": table_name,
            "create_table": create_table,
            "batch_size": batch_size
        })
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("JSON file must contain a list of objects or a single object")
            
            if len(df) == 0:
                return "JSON file contains no data"
            
            # Create table if requested
            if create_table:
                create_sql = self._generate_create_table_sql(df, table_name)
                console.print(f"[cyan]Creating table with SQL:[/cyan]")
                console.print(create_sql)
                
                self.client.command(create_sql)
                console.print(f"[green]✓ Table {table_name} created successfully[/green]")
            
            # Insert data in batches
            total_rows = len(df)
            inserted_rows = 0
            
            with Progress() as progress:
                task = progress.add_task(f"Loading data into {table_name}", total=total_rows)
                
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch_df = df.iloc[start_idx:end_idx]
                    
                    # Insert batch
                    self.client.insert_df(table_name, batch_df)
                    
                    inserted_rows += len(batch_df)
                    progress.update(task, completed=inserted_rows)
            
            result = f"Successfully loaded {inserted_rows:,} rows from {file_path.name} into table {table_name}"
            tool_logger.log_tool_success("load_from_json", result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to load JSON data: {str(e)}"
            tool_logger.log_tool_error("load_from_json", error_msg)
            return error_msg
    
    def _generate_create_table_sql(self, df: pd.DataFrame, table_name: str) -> str:
        """Generate CREATE TABLE SQL from DataFrame schema"""
        
        column_definitions = []
        
        for column_name, dtype in df.dtypes.items():
            # Map pandas dtypes to ClickHouse types
            if pd.api.types.is_integer_dtype(dtype):
                if df[column_name].min() >= 0:
                    ch_type = "UInt64"
                else:
                    ch_type = "Int64"
            elif pd.api.types.is_float_dtype(dtype):
                ch_type = "Float64"
            elif pd.api.types.is_bool_dtype(dtype):
                ch_type = "Bool"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                ch_type = "DateTime"
            else:
                ch_type = "String"
            
            column_definitions.append(f"`{column_name}` {ch_type}")
        
        create_sql = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    {',\n    '.join(column_definitions)}
) ENGINE = MergeTree()
ORDER BY tuple()
"""
        
        return create_sql

class DataVisualizer:
    """Data visualization utilities"""
    
    def __init__(self, client):
        self.client = client
    
    async def create_chart(
        self,
        query: str,
        chart_type: str,
        x_column: str,
        y_column: str,
        title: str,
        save_path: Optional[str] = None
    ) -> str:
        """Create a chart from query results"""
        
        tool_logger.log_tool_start("create_chart", {
            "query": query[:100],
            "chart_type": chart_type,
            "x_column": x_column,
            "y_column": y_column,
            "title": title
        })
        
        try:
            # Execute query
            result = self.client.query(query)
            df = result.result_as_dataframe()
            
            if len(df) == 0:
                return "No data returned from query"
            
            if x_column not in df.columns:
                return f"Column '{x_column}' not found in query results"
            
            if y_column not in df.columns:
                return f"Column '{y_column}' not found in query results"
            
            # Create chart based on type
            if chart_type == "line":
                fig = px.line(df, x=x_column, y=y_column, title=title)
            elif chart_type == "bar":
                fig = px.bar(df, x=x_column, y=y_column, title=title)
            elif chart_type == "scatter":
                fig = px.scatter(df, x=x_column, y=y_column, title=title)
            elif chart_type == "pie":
                fig = px.pie(df, names=x_column, values=y_column, title=title)
            elif chart_type == "histogram":
                fig = px.histogram(df, x=x_column, title=title)
            elif chart_type == "box":
                fig = px.box(df, y=y_column, title=title)
            else:
                return f"Unsupported chart type: {chart_type}"
            
            # Save chart if path specified
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                if save_path.suffix.lower() == '.html':
                    fig.write_html(str(save_path))
                elif save_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    fig.write_image(str(save_path))
                else:
                    # Default to HTML
                    save_path = save_path.with_suffix('.html')
                    fig.write_html(str(save_path))
                
                console.print(f"[green]✓ Chart saved to {save_path}[/green]")
                result = f"Created {chart_type} chart with {len(df)} data points and saved to {save_path}"
            else:
                # Show chart in browser
                fig.show()
                result = f"Created {chart_type} chart with {len(df)} data points and displayed in browser"
            
            tool_logger.log_tool_success("create_chart", result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to create chart: {str(e)}"
            tool_logger.log_tool_error("create_chart", error_msg)
            return error_msg

class DataExporter:
    """Data export utilities"""
    
    def __init__(self, client):
        self.client = client
    
    async def export_to_file(
        self,
        query: str,
        output_path: str,
        format: str = "csv"
    ) -> str:
        """Export query results to file"""
        
        tool_logger.log_tool_start("export_to_file", {
            "query": query[:100],
            "output_path": output_path,
            "format": format
        })
        
        try:
            # Execute query
            result = self.client.query(query)
            df = result.result_as_dataframe()
            
            if len(df) == 0:
                return "No data to export"
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export based on format
            if format == "csv":
                df.to_csv(output_path, index=False)
            elif format == "json":
                df.to_json(output_path, orient="records", indent=2)
            elif format == "parquet":
                df.to_parquet(output_path, index=False)
            elif format == "excel":
                df.to_excel(output_path, index=False)
            else:
                return f"Unsupported export format: {format}"
            
            result = f"Exported {len(df):,} rows to {output_path} in {format} format"
            console.print(f"[green]✓ {result}[/green]")
            
            tool_logger.log_tool_success("export_to_file", result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to export data: {str(e)}"
            tool_logger.log_tool_error("export_to_file", error_msg)
            return error_msg