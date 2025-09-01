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
import requests
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
        
        # Model download configuration
        self.model_name = "vishprometa/clickhouse-qwen3-1.7b-gguf"
        self.display_name = "clickhouse-qwen3-1.7b-gguf"
        self.model_file = "unsloth.F16.gguf"
        self.model_url = f"https://huggingface.co/{self.model_name}/resolve/main/{self.model_file}"
        self.cache_dir = os.path.expanduser("~/.cache/llama.cpp")
        self.model_path = os.path.join(self.cache_dir, f"{self.model_name.replace('/', '_')}_{self.model_file}")
        
        # Chat template configuration
        self.chat_template_file = "chat_template.jinja"
        self.chat_template_url = f"https://huggingface.co/{self.model_name}/resolve/main/{self.chat_template_file}"
        self.chat_template_path = os.path.join(self.cache_dir, f"{self.model_name.replace('/', '_')}_{self.chat_template_file}")
        
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
            
            # Download model and chat template together
            model_exists = os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 3_000_000_000
            template_exists = os.path.exists(self.chat_template_path)
            
            logger.info(f"Model status: exists={model_exists}, path={self.model_path}")
            logger.info(f"Template status: exists={template_exists}, path={self.chat_template_path}")
            
            # If model doesn't exist, download both model and template
            if not model_exists:
                logger.info("Model not found, downloading model and template...")
                if not self._download_model_with_progress():
                    raise RuntimeError("Model download failed")
                # Download template right after model download
                if not self._download_chat_template():
                    logger.warning("Chat template download failed, will try to start server without custom template")
                    self.chat_template_path = None
            
            # If model exists but template doesn't (existing users), download just the template
            elif not template_exists:
                logger.info("Model exists but template missing, downloading template...")
                self._announce("📥 Downloading missing chat template for existing model...")
                if not self._download_chat_template():
                    logger.warning("Chat template download failed, will try to start server without custom template")
                    self.chat_template_path = None
            else:
                logger.info("Both model and template exist, proceeding with startup...")
            
            # Check if server is already running
            if self._is_server_running():
                self._announce(f"✅ ClickHouse AI Agent ready on {self.host}:{self.port}")
                return
            
            # Start the server with local model file
            self._start_server()
            self._wait_for_server()  # No timeout for model download
            
        except Exception as e:
            logger.error(f"Auto-setup failed: {e}")
            raise RuntimeError(f"Failed to setup ClickHouse AI Agent: {e}")

    def _download_model_with_progress(self):
        """Download model with real progress tracking using curl"""
        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn, DownloadColumn, TransferSpeedColumn
            
            # Ensure cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Check if model already exists
            if os.path.exists(self.model_path):
                file_size = os.path.getsize(self.model_path)
                if file_size > 3_000_000_000:  # > 3GB, assume complete
                    self._announce(f"✅ Model already downloaded: {self.model_path}")
                    return True
            
            self._announce(f"📥 Downloading {self.display_name}...")
            
            # Get file size for progress tracking
            try:
                response = requests.head(self.model_url, allow_redirects=True)
                total_size = int(response.headers.get('content-length', 0))
            except Exception as e:
                logger.warning(f"Could not get file size: {e}")
                total_size = 3_447_349_440  # Known size ~3.2GB
            
            # Try simple download first (more reliable)
            if self._simple_download_with_progress(total_size):
                return True
            
            # Fallback to curl if simple download fails
            return self._curl_download_with_progress(total_size)
                    
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise RuntimeError(f"Model download failed: {e}")

    def _download_chat_template(self, quiet: bool = False) -> bool:
        """Download chat template from Hugging Face repository"""
        try:
            # Check if template already exists and is recent (less than 24 hours old)
            if os.path.exists(self.chat_template_path):
                file_age = time.time() - os.path.getmtime(self.chat_template_path)
                if file_age < 86400:  # 24 hours in seconds
                    if not quiet:
                        logger.info(f"Chat template already exists and is recent: {self.chat_template_path}")
                    return True
                else:
                    if not quiet:
                        self._announce("🔄 Updating chat template...")
            else:
                if not quiet:
                    self._announce("📥 Downloading chat template...")
            
            # Ensure cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Download the chat template
            logger.info(f"Downloading chat template from: {self.chat_template_url}")
            response = requests.get(self.chat_template_url, timeout=30)
            response.raise_for_status()
            
            # Write to file
            with open(self.chat_template_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Verify the file was written correctly
            file_size = os.path.getsize(self.chat_template_path)
            logger.info(f"Chat template saved: {self.chat_template_path} ({file_size} bytes)")
            
            if not quiet:
                self._announce(f"✅ Chat template downloaded: {self.chat_template_path}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to download chat template from {self.chat_template_url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to save chat template: {e}")
            return False

    def refresh_chat_template(self) -> bool:
        """Force refresh the chat template from Hugging Face repository"""
        try:
            # Remove existing template to force re-download
            if os.path.exists(self.chat_template_path):
                os.remove(self.chat_template_path)
                self._announce("🗑️ Removed old chat template")
            
            # Download fresh template
            return self._download_chat_template()
            
        except Exception as e:
            logger.error(f"Failed to refresh chat template: {e}")
            return False

    def _simple_download_with_progress(self, total_size: int) -> bool:
        """Simple download using requests with progress tracking"""
        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn, DownloadColumn, TransferSpeedColumn
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeElapsedColumn(),
                console=_console,
                transient=False,
            ) as progress:
                task = progress.add_task(
                    f"🔄 Downloading {self.display_name}...", 
                    total=total_size
                )
                
                # Download with requests and stream
                response = requests.get(self.model_url, stream=True, allow_redirects=True)
                response.raise_for_status()
                
                downloaded_size = 0
                with open(self.model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            progress.update(task, completed=downloaded_size)
                
                if downloaded_size > 3_000_000_000:  # > 3GB
                    progress.update(task, description="✅ Model download completed!", completed=total_size)
                    self._announce(f"✅ Model downloaded successfully: {self.model_path}")
                    
                    # Download chat template immediately after successful model download
                    progress.update(task, description="📥 Downloading chat template...")
                    if self._download_chat_template(quiet=True):
                        progress.update(task, description="✅ Model and template ready!")
                    else:
                        progress.update(task, description="✅ Model ready (template download failed)")
                    
                    return True
                else:
                    progress.update(task, description="❌ Download incomplete")
                    return False
                    
        except Exception as e:
            logger.warning(f"Simple download failed, trying curl: {e}")
            return False

    def _curl_download_with_progress(self, total_size: int) -> bool:
        """Download using curl with progress tracking"""
        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn, DownloadColumn, TransferSpeedColumn
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeElapsedColumn(),
                console=_console,
                transient=False,
            ) as progress:
                task = progress.add_task(
                    f"🔄 Downloading {self.display_name}...", 
                    total=total_size
                )
                
                # Use curl for reliable download with progress
                cmd = [
                    "curl", "-L", "-o", self.model_path,
                    "--progress-bar",
                    self.model_url
                ]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1
                )
                
                downloaded_size = 0
                last_update_time = time.time()
                
                # Simple progress tracking based on file size
                start_time = time.time()
                while process.poll() is None:
                    # Check if file exists and get its size
                    if os.path.exists(self.model_path):
                        current_size = os.path.getsize(self.model_path)
                        if current_size > downloaded_size:
                            downloaded_size = current_size
                            progress.update(task, completed=downloaded_size)
                            last_update_time = time.time()
                    
                    # Update description periodically
                    current_time = time.time()
                    if current_time - last_update_time > 2.0:  # Update every 2 seconds
                        if downloaded_size < total_size:
                            elapsed = current_time - start_time
                            if elapsed > 0 and downloaded_size > 0:
                                # Calculate estimated speed and remaining time
                                speed = downloaded_size / elapsed
                                remaining = (total_size - downloaded_size) / speed if speed > 0 else 0
                                progress.update(task, description=f"🔄 Downloading {self.display_name}... ({downloaded_size}/{total_size} bytes, {speed/1024/1024:.1f} MB/s)")
                        last_update_time = current_time
                    
                    time.sleep(0.5)  # Check every 0.5 seconds
                
                process.wait()
                
                if process.returncode == 0 and os.path.exists(self.model_path):
                    file_size = os.path.getsize(self.model_path)
                    if file_size > 3_000_000_000:  # > 3GB
                        progress.update(task, description="✅ Model download completed!", completed=total_size)
                        self._announce(f"✅ Model downloaded successfully: {self.model_path}")
                        
                        # Download chat template immediately after successful model download
                        progress.update(task, description="📥 Downloading chat template...")
                        if self._download_chat_template(quiet=True):
                            progress.update(task, description="✅ Model and template ready!")
                        else:
                            progress.update(task, description="✅ Model ready (template download failed)")
                        
                        return True
                    else:
                        progress.update(task, description="❌ Download incomplete")
                        raise RuntimeError(f"Downloaded file too small: {file_size} bytes")
                else:
                    progress.update(task, description="❌ Download failed")
                    raise RuntimeError(f"Download failed with return code: {process.returncode}")
                    
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise RuntimeError(f"Model download failed: {e}")

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
            
            # Use the downloaded chat template if available
            cmd = [
                "llama-server",
                "-m", self.model_path,  # Use local model file instead of Hugging Face URL
                "--jinja",
            ]
            
            # Add chat template file if available
            if self.chat_template_path and os.path.exists(self.chat_template_path):
                cmd.extend(["--chat-template-file", self.chat_template_path])
                logger.info(f"✅ Using custom chat template: {self.chat_template_path}")
                # Removed the announcement message as requested
            else:
                logger.warning("❌ No custom chat template found, using llama-server default")
                # Removed the announcement message as requested
            
            # Continue with other parameters
            cmd.extend([
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
            ])
            
            # Log the full command for debugging
            logger.info(f"Starting llama-server with command: {' '.join(cmd)}")
            
            # Start in background
            logger.info("Attempting to start llama-server process...")
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None,
                text=True
            )
            
            logger.info(f"Started ClickHouse AI Agent with PID {self.server_process.pid}")
            
            # Give the process a moment to start and check if it's still running
            import time
            time.sleep(2)
            
            if self.server_process.poll() is not None:
                # Process has already terminated
                stdout, stderr = self.server_process.communicate()
                logger.error(f"llama-server process terminated immediately!")
                logger.error(f"Exit code: {self.server_process.returncode}")
                if stdout:
                    logger.error(f"STDOUT: {stdout}")
                if stderr:
                    logger.error(f"STDERR: {stderr}")
                raise RuntimeError(f"llama-server failed to start. Exit code: {self.server_process.returncode}")
            else:
                logger.info("llama-server process is running, proceeding with startup...")
            
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

    def _wait_for_server(self, timeout: int = 60):
        """Wait for server to be ready with simple progress"""
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=_console,
            transient=False,
        ) as progress:
            task = progress.add_task("🔄 Starting ClickHouse AI Agent...", total=None)
            
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self._is_server_running():
                    progress.update(task, description="✅ ClickHouse AI Agent ready!")
                    return
                
                time.sleep(1)  # Check every second
            
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