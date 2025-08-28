# Proto - ClickHouse AI Agent

> Natural language interface for ClickHouse databases

Proto is an intelligent CLI agent that allows you to interact with ClickHouse databases using natural language. Ask questions about your data, generate SQL queries, and get insights without writing complex SQL.

## Features

- 🤖 **AI-Powered**: Natural language interface powered by local or cloud AI models
- 📊 **Smart Analysis**: Automatic table analysis and data insights
- 🔍 **Query Generation**: Convert questions to optimized SQL queries
- 📈 **Data Visualization**: Generate charts and visualizations from your data
- ⚡ **Fast Setup**: One-command installation, no Python knowledge required
- 🔒 **Privacy-First**: Option to run completely locally with local AI models
- 🚀 **Cross-Platform**: Works on macOS (Intel/Apple Silicon), Linux, and Windows
- 📦 **Easy Installation**: Install via pipx, pip, or one-liner script

## Quick Start

### Install Proto

```bash
curl -fsSL https://proto.dev/install.sh | sh
```

### Start Using Proto

```bash
proto
```

Follow the interactive onboarding to configure your ClickHouse connection and AI provider.

## Installation Options

### Using pipx (Recommended)
```bash
pipx install proto-clickhouse-agent
```

### Using pip
```bash
pip install proto-clickhouse-agent
```

### One-liner (Legacy)
```bash
curl -fsSL https://proto.dev/install.sh | sh
```

## Usage Examples

```bash
# Start interactive chat
proto

# Execute a single query
proto query "Show me the top 10 users by activity"

# Analyze a specific table
proto analyze users

# Load data from a file
proto load-data users.csv users
```

## Configuration

Proto supports multiple AI providers:

- **Local LLM**: Run completely offline with local models
- **Local LLM**: Built-in ClickHouse AI model (no API keys needed)
- **OpenAI**: Direct OpenAI API integration

Configuration is stored in `~/.config/proto/proto-config.json`.

## System Requirements

- macOS 10.15+ or Linux
- ClickHouse database (local or cloud)
- AI provider (Local LLM built-in)
- ~3.5GB free space for AI model (first run)

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/proto.git
cd proto

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Proto
python main.py
```

### Building Standalone Binaries

```bash
./build_installers.sh
```

This creates platform-specific binaries in the `builds/` directory.

## Architecture

```
proto/
├── agent/           # Core AI agent logic
├── config/          # Configuration management
├── providers/       # AI provider integrations
├── tools/           # Database and data tools
├── ui/              # User interface components
├── utils/           # Utility functions
└── main.py          # Entry point
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 [Documentation](https://proto.dev)
- 🐛 [Report Issues](https://github.com/yourusername/proto/issues)
- 💬 [Discussions](https://github.com/yourusername/proto/discussions)

## Roadmap

- [ ] Web interface
- [ ] More AI providers
- [ ] Advanced data visualization
- [ ] Query optimization suggestions
- [ ] Multi-database support