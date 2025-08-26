"""
OpenRouter provider for ClickHouse AI Agent
Based on the architecture from erpai-agent/single/utils/openrouter-wrapper.ts
"""

import json
from typing import Dict, List, Any, Optional, AsyncGenerator
import httpx
import os
from datetime import datetime
from utils.logging import get_logger

logger = get_logger(__name__)

class OpenRouterProvider:
    """OpenRouter API wrapper using OpenAI format"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "qwen/qwen-3-coder",
        base_url: str = "https://openrouter.ai/api/v1",
        provider_only: str = "groq",
        data_collection: str = "deny"
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.provider_config = {
            "only": [provider_only] if provider_only else [],
            "data_collection": data_collection
        }
        
        # Create logs directory if it doesn't exist
        self.logs_dir = "logs/openrouter_requests"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.client = httpx.AsyncClient(
            timeout=60.0,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/vish/moja",
                "X-Title": "Moja ClickHouse AI Agent"
            }
        )
    
    def _log_request(self, request_data: Dict[str, Any], response_data: Dict[str, Any] = None):
        """Log OpenRouter request and response to local file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = os.path.join(self.logs_dir, f"openrouter_request_{timestamp}.json")
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request": request_data,
            "response": response_data if response_data else None
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        # Also log a summary for quick debugging
        summary_log = os.path.join(self.logs_dir, "requests_summary.log")
        with open(summary_log, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
            f.write(f"MODEL: {request_data.get('model', 'unknown')}\n")
            f.write(f"MESSAGES COUNT: {len(request_data.get('messages', []))}\n")
            f.write(f"TOOLS COUNT: {len(request_data.get('tools', []))}\n")
            f.write(f"TEMPERATURE: {request_data.get('temperature', 'N/A')}\n")
            f.write(f"MAX_TOKENS: {request_data.get('max_tokens', 'N/A')}\n")
            
            # Log messages summary
            messages = request_data.get('messages', [])
            for i, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content_preview = str(msg.get('content', ''))[:100] + ('...' if len(str(msg.get('content', ''))) > 100 else '')
                f.write(f"MESSAGE {i+1} ({role}): {content_preview}\n")
            
            if response_data:
                f.write(f"RESPONSE TOKENS: {response_data.get('usage', {}).get('total_tokens', 'N/A')}\n")
            f.write(f"{'='*80}\n")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 4000,
        live_animation=None
    ) -> Dict[str, Any]:
        """Create a chat completion using OpenRouter API"""
        
        
        request_body = {
            "model": self.model,
            "messages": messages,  
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tool_choice": "auto" if tools else None,
            "tools": tools,
            "provider": self.provider_config,
            "stream": False  # Disable streaming for now
        }
        
        logger.info(f"🚀 OpenRouter API call to {self.model} with {len(messages)} messages")
        
        # Log the complete request to file
        self._log_request(request_body)
        
        try:
            if request_body.get("stream", False):
                # Use streaming response
                async with self.client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=request_body
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(f"OpenRouter API error: {response.status_code} - {error_text}")
                        # Log error response too
                        self._log_request(request_body, {"error": error_text.decode(), "status_code": response.status_code})
                        raise Exception(f"OpenRouter API error: {response.status_code} - {error_text.decode()}")

                    # Process streaming response
                    response_data = await self._process_streaming_response(response, request_body, live_animation)
            else:
                # Use regular non-streaming response
                response = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=request_body
                )
                
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"OpenRouter API error: {response.status_code} - {error_text}")
                    # Log error response too
                    self._log_request(request_body, {"error": error_text, "status_code": response.status_code})
                    raise Exception(f"OpenRouter API error: {response.status_code} - {error_text}")
                
                response_data = response.json()
                
                # Log the complete response
                self._log_request(request_body, response_data)
            
            # Log usage information
            usage = response_data.get("usage", {})
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                logger.info(f"✅ {self.model} completed: {prompt_tokens}→{completion_tokens} tokens")
            
            return response_data
            
        except httpx.TimeoutException:
            logger.error("OpenRouter API request timed out")
            raise Exception("OpenRouter API request timed out")
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    def _format_markdown_for_rich(self, text: str) -> str:
        """Convert basic markdown to rich markup for streaming display"""
        # Convert **bold** to [bold]text[/bold]
        import re
        
        # Handle bold text
        text = re.sub(r'\*\*([^*]+)\*\*', r'[bold]\1[/bold]', text)
        
        # Handle italic text
        text = re.sub(r'\*([^*]+)\*', r'[italic]\1[/italic]', text)
        
        # Handle numbered lists (make them bright)
        text = re.sub(r'^(\d+\.\s)', r'[bright_cyan]\1[/bright_cyan]', text, flags=re.MULTILINE)
        
        return text
    
    async def _process_streaming_response(self, response, request_body, live_animation=None):
        """Process streaming response from OpenRouter"""
        import json
        from ui.minimal_interface import ui
        from rich.text import Text
        
        full_content = ""
        tool_calls = []
        current_tool_call = None
        usage_info = {}
        
        # If no live animation provided, create basic display
        if not live_animation:
            ui.console.print()
            ui.console.print(f"[dim bright_blue]●[/dim bright_blue] [bright_white]", end="")
        
        # Buffer for smooth display
        sentence_buffer = ""  # Accumulate until we have complete sentences
        
        try:
            async for chunk in response.aiter_lines():
                if not chunk or chunk == "data: [DONE]":
                    continue
                    
                if chunk.startswith("data: "):
                    try:
                        data = json.loads(chunk[6:])  # Remove "data: " prefix
                        
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            
                            # Handle text content streaming
                            if "content" in delta and delta["content"]:
                                content_chunk = delta["content"]
                                full_content += content_chunk
                                sentence_buffer += content_chunk
                                
                                # Update live animation if available, otherwise print directly
                                if live_animation:
                                    try:
                                        # Convert markdown and create display text
                                        formatted_text = self._format_markdown_for_rich(full_content)
                                        # Use Text.from_markup to properly handle Rich markup
                                        display_text = Text.from_markup(f"[dim bright_blue]●[/dim bright_blue] [bright_white]{formatted_text}[/bright_white]")
                                        live_animation.update(display_text)
                                    except Exception as e:
                                        # Fallback to simple text if Rich formatting fails
                                        simple_text = Text()
                                        simple_text.append("● ", style="dim bright_blue")
                                        simple_text.append(full_content, style="bright_white")
                                        live_animation.update(simple_text)
                                else:
                                    # Fallback: display when we have complete sentences or line breaks
                                    if any(end_char in content_chunk for end_char in [".", "!", "?", "\n"]) or len(sentence_buffer) > 100:
                                        formatted_text = self._format_markdown_for_rich(sentence_buffer)
                                        ui.console.print(formatted_text, end="")
                                        sentence_buffer = ""
                            
                            # Handle tool calls
                            if "tool_calls" in delta:
                                for tool_call in delta["tool_calls"]:
                                    if tool_call.get("function"):
                                        if current_tool_call is None:
                                            current_tool_call = {
                                                "id": tool_call.get("id", ""),
                                                "type": "function",
                                                "function": {
                                                    "name": tool_call["function"].get("name", ""),
                                                    "arguments": ""
                                                }
                                            }
                                        
                                        if "arguments" in tool_call["function"]:
                                            current_tool_call["function"]["arguments"] += tool_call["function"]["arguments"]
                        
                        # Handle usage info
                        if "usage" in data:
                            usage_info = data["usage"]
                            
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON chunks
                        
        except Exception as e:
            logger.error(f"Error processing streaming response: {e}")
            
        # Display any remaining buffer (only for non-live animation mode)
        if not live_animation and sentence_buffer:
            formatted_text = self._format_markdown_for_rich(sentence_buffer)
            ui.console.print(formatted_text, end="")
            
        # Finish the line (only for non-live animation mode)
        if not live_animation:
            ui.console.print()
            ui.console.print()
        
        # Add completed tool call if exists
        if current_tool_call and current_tool_call["function"]["arguments"]:
            tool_calls.append(current_tool_call)
        
        # Create response in expected format
        response_data = {
            "choices": [{
                "message": {
                    "content": full_content if full_content else None,
                    "tool_calls": tool_calls if tool_calls else None
                }
            }],
            "usage": usage_info
        }
        
        # Log the complete response
        self._log_request(request_body, response_data)
        
        # Log usage information
        if usage_info:
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
            logger.info(f"✅ {self.model} completed: {prompt_tokens}→{completion_tokens} tokens")
        
        return response_data

def convert_anthropic_tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Anthropic tool format to OpenAI function calling format"""
    openai_tools = []
    
    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool.get("input_schema", {})
            }
        }
        openai_tools.append(openai_tool)
    
    return openai_tools

def convert_openai_response_to_anthropic(response: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAI response format to Anthropic-like format for compatibility"""
    choice = response["choices"][0]
    message = choice["message"]
    
    content = []
    
    # Add text content if present
    if message.get("content"):
        content.append({
            "type": "text",
            "text": message["content"]
        })
    
    # Add tool calls if present
    if message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            content.append({
                "type": "tool_use",
                "id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "input": json.loads(tool_call["function"]["arguments"])
            })
    
    return {
        "content": content,
        "usage": response.get("usage", {})
    }

def convert_anthropic_messages_to_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Anthropic message format to OpenAI format for OpenRouter"""
    import json
    
    openai_messages = []
    
    for message in messages:
        if message["role"] in ["system", "user"]:
            # System and user messages can be passed through as-is if they have string content
            if isinstance(message.get("content"), str):
                openai_messages.append(message)
            elif isinstance(message.get("content"), list):
                # Handle list content for user messages (like tool results)
                content_text = ""
                for item in message["content"]:
                    if item.get("type") == "text":
                        content_text += item.get("text", "")
                    elif item.get("type") == "tool_result":
                        # This is a tool result - convert to OpenAI format
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": item.get("tool_use_id"),
                            "name": item.get("name", "unknown"),
                            "content": item.get("content", "")
                        })
                        continue
                
                if content_text:
                    openai_messages.append({
                        "role": message["role"],
                        "content": content_text
                    })
        
        elif message["role"] == "tool":
            # Tool messages are already in OpenAI format, pass through as-is
            openai_messages.append(message)
        
        elif message["role"] == "assistant":
            # Convert assistant messages with tool_use to OpenAI format
            content = message.get("content")
            
            if isinstance(content, str):
                # Simple text response
                openai_messages.append({
                    "role": "assistant",
                    "content": content
                })
            elif isinstance(content, list):
                # Complex content with potential tool calls
                text_content = ""
                tool_calls = []
                
                for item in content:
                    if item.get("type") == "text":
                        text_content += item.get("text", "")
                    elif item.get("type") == "tool_use":
                        # Convert Anthropic tool_use to OpenAI tool_calls
                        tool_calls.append({
                            "id": item.get("id"),
                            "type": "function",
                            "function": {
                                "name": item.get("name"),
                                "arguments": json.dumps(item.get("input", {}))
                            }
                        })
                
                openai_messages.append({
                    "role": "assistant",
                    "content": text_content if text_content else None,
                    "tool_calls": tool_calls if tool_calls else None
                })
    
    return openai_messages