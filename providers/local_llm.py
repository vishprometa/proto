"""
Simplified Local LLM provider that manages llama-server process
"""

import subprocess
import time
import signal
import os
import psutil
import shutil
import sys
from typing import Dict, List, Any, Optional
from utils.logging import get_logger

try:
    from rich.console import Console
    _console: Optional[Console] = Console()
except Exception:
    _console = None

logger = get_logger(__name__)


class LocalLLMProvider:
    """Simple provider that manages llama-server process and provides connection info"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", model: str = "vishprometa/clickhouse-qwen3-1.7b-gguf"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.host = "127.0.0.1"
        self.port = 8000
        self.server_process = None
        
        # Extract host and port from base_url
        try:
            if "://" in base_url:
                url_part = base_url.split("://")[1]
                if "/" in url_part:
                    host_port = url_part.split("/")[0]
                else:
                    host_port = url_part
                
                if ":" in host_port:
                    self.host, port_str = host_port.split(":")
                    self.port = int(port_str)
        except Exception as e:
            logger.warning(f"Could not parse base_url {base_url}, using defaults: {e}")

        # Auto-setup: install llama-server if needed and start the model
        self._auto_setup()

    def _auto_setup(self):
        """Automatically setup everything without user interaction"""
        try:
            # Ensure llama-server is installed
            self._ensure_llama_server_installed()
            
            # Check if server is already running
            if self._is_server_running():
                self._announce(f"✅ ClickHouse AI Agent ready on {self.host}:{self.port}")
                return
            
            # Start the server (will download model if needed)
            self._start_server()
            self._wait_for_server(timeout=600)  # 10 minutes for model download
            
        except Exception as e:
            logger.error(f"Auto-setup failed: {e}")
            raise RuntimeError(f"Failed to setup ClickHouse AI Agent: {e}")

    def _ensure_llama_server_installed(self):
        """Ensure llama-server is installed and available"""
        if shutil.which("llama-server"):
            logger.info("llama-server is already installed")
            return
        
        self._announce("📦 Installing llama.cpp for ClickHouse AI Agent...")
        
        try:
            # Detect platform and install accordingly
            if sys.platform == "darwin":  # macOS
                self._install_llama_server_macos()
            elif sys.platform.startswith("linux"):  # Linux
                self._install_llama_server_linux()
            else:
                raise RuntimeError(f"Unsupported platform: {sys.platform}")
                
        except Exception as e:
            logger.error(f"Failed to install llama-server: {e}")
            raise RuntimeError(
                f"Could not install llama-server automatically. "
                f"Please install llama.cpp manually: https://github.com/ggerganov/llama.cpp"
            )
    
    def _install_llama_server_macos(self):
        """Install llama-server on macOS using Homebrew"""
        try:
            # Check if Homebrew is available
            if not shutil.which("brew"):
                raise RuntimeError("Homebrew is required but not installed. Please install Homebrew first.")
            
            # Install llama.cpp
            subprocess.run(["brew", "install", "llama.cpp"], check=True)
            self._announce("✅ llama.cpp installed successfully via Homebrew")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to install llama.cpp via Homebrew: {e}")
    
    def _install_llama_server_linux(self):
        """Install llama-server on Linux"""
        try:
            # Try to install via package manager
            if shutil.which("apt-get"):
                # Ubuntu/Debian - try to build from source
                self._build_llama_cpp_from_source()
            elif shutil.which("yum") or shutil.which("dnf"):
                # RHEL/CentOS/Fedora - try to build from source
                self._build_llama_cpp_from_source()
            else:
                # Fallback to building from source
                self._build_llama_cpp_from_source()
                
        except Exception as e:
            raise RuntimeError(f"Failed to install llama.cpp on Linux: {e}")
    
    def _build_llama_cpp_from_source(self):
        """Build llama.cpp from source (Linux fallback)"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            self._announce("Building llama.cpp from source...")
            
            # Clone the repository
            subprocess.run([
                "git", "clone", "https://github.com/ggerganov/llama.cpp.git", 
                os.path.join(temp_dir, "llama.cpp")
            ], check=True)
            
            # Build
            build_dir = os.path.join(temp_dir, "llama.cpp")
            subprocess.run(["make", "-j", "4"], cwd=build_dir, check=True)
            
            # Install to /usr/local/bin
            llama_server_src = os.path.join(build_dir, "llama-server")
            llama_server_dst = "/usr/local/bin/llama-server"
            
            subprocess.run(["sudo", "cp", llama_server_src, llama_server_dst], check=True)
            subprocess.run(["sudo", "chmod", "+x", llama_server_dst], check=True)
            
            self._announce("✅ llama.cpp built and installed successfully")

    def _is_server_running(self) -> bool:
        """Check if llama-server is already running on our port"""
        try:
            import requests
            response = requests.get(f"http://{self.host}:{self.port}/v1/models", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _start_server(self):
        """Start llama-server process"""
        try:
            self._announce(f"🚀 Starting ClickHouse AI Agent on {self.host}:{self.port}")
            
            # Kill any existing server on this port
            self._kill_existing_server()
            
            # Use the exact command that works with fixed chat template
            template_path = os.path.join(os.path.dirname(__file__), "..", "fixed_chat_template.jinja")
            cmd = [
                "llama-server",
                "-hf", "vishprometa/clickhouse-qwen3-1.7b-gguf",  # Use SafeTensors version for now
                "--jinja",
                "--chat-template-file", template_path,
                "--reasoning-format", "deepseek",
                "-ngl", "99",
                "-fa", 
                "-sm", "row",
                "--temp", "0.6",
                "--top-k", "20",
                "--top-p", "0.95",
                "--min-p", "0",
                "-c", "40960",
                "-n", "32768",
                "--no-context-shift",
                "--host", self.host,
                "--port", str(self.port)
            ]
            
            # Start in background
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            logger.info(f"Started ClickHouse AI Agent with PID {self.server_process.pid}")
            
        except Exception as e:
            logger.error(f"Failed to start ClickHouse AI Agent: {e}")
            raise

    def _kill_existing_server(self):
        """Kill any existing llama-server processes on our port"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'llama-server' in proc.info['name'] or 'llama-server' in ' '.join(proc.info['cmdline'] or []):
                        # Check if it's using our port
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if f"--port {self.port}" in cmdline or f":{self.port}" in cmdline:
                            logger.info(f"Stopping existing ClickHouse AI Agent process {proc.info['pid']}")
                            proc.terminate()
                            try:
                                proc.wait(timeout=5)
                            except psutil.TimeoutExpired:
                                proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.warning(f"Error killing existing servers: {e}")

    def _wait_for_server(self, timeout: int = 600):
        """Wait for server to be ready with download progress"""
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=_console,
            transient=False,
        ) as progress:
            task = progress.add_task("🔄 ClickHouse AI Agent LLM download in progress...", total=100)
            
            start_time = time.time()
            last_check = 0
            download_progress = 0
            
            while time.time() - start_time < timeout:
                if self._is_server_running():
                    progress.update(task, description="✅ ClickHouse AI Agent ready!", completed=100)
                    return
                
                # Simulate download progress (since we can't get real progress from llama-server)
                elapsed = int(time.time() - start_time)
                if elapsed > 0:
                    # Estimate progress based on time (model is ~3.5GB, typical download speed ~10MB/s)
                    estimated_progress = min(95, int((elapsed * 10) / 35))  # 10MB/s = 350s for 3.5GB
                    if estimated_progress > download_progress:
                        download_progress = estimated_progress
                        progress.update(task, completed=download_progress)
                
                # Update description every 30 seconds
                if elapsed >= 30 and elapsed % 30 == 0 and elapsed != last_check:
                    progress.update(task, description=f"🔄 ClickHouse AI Agent LLM download in progress... ({elapsed}s elapsed)")
                    last_check = elapsed
                
                time.sleep(2)  # Check every 2 seconds
            
            progress.update(task, description="❌ Timeout waiting for ClickHouse AI Agent")
            raise TimeoutError(f"ClickHouse AI Agent did not start within {timeout} seconds")

    def _announce(self, message: str):
        """Announce status message"""
        logger.info(message)
        if _console:
            _console.print(f"[cyan]{message}[/cyan]")
        else:
            print(message)

    def get_openai_config(self) -> Dict[str, Any]:
        """Get configuration for OpenAI client"""
        return {
            "base_url": self.base_url,
            "api_key": "not-needed"  # llama-server doesn't require API key
        }

    async def chat_completion(self, *args, **kwargs):
        """Legacy method - should use OpenAI client directly instead"""
        raise NotImplementedError(
            "Use OpenAI client directly with get_openai_config(). "
            "This provider only manages the llama-server process."
        )

    async def close(self):
        """Clean up resources"""
        if self.server_process:
            try:
                # Terminate the process group
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                else:
                    self.server_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    if os.name != 'nt':
                        os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                    else:
                        self.server_process.kill()
                
                logger.info("ClickHouse AI Agent process terminated")
            except Exception as e:
                logger.error(f"Error terminating ClickHouse AI Agent: {e}")

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'server_process') and self.server_process:
            try:
                if self.server_process.poll() is None:  # Still running
                    self.server_process.terminate()
            except Exception:
                pass