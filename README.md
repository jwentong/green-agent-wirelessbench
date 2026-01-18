# WCHW Green Agent - Wireless Communication Benchmark

A Green Agent for evaluating AI agents on the **WCHW (Wireless Communication Homework)** benchmark, built for the UC Berkeley RDI Foundation's [AgentBeats](https://agentbeats.dev) platform.

## 📡 Overview

This Green Agent evaluates **Purple Agents** (wireless communication problem solvers) using the [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) protocol. The WCHW benchmark covers advanced telecommunications topics including:

- **Information Theory**: Shannon capacity, entropy, channel coding
- **Channel Modeling**: Path loss, fading, MIMO, propagation
- **Signal Processing**: Modulation, detection, filtering
- **System Design**: Link budget, resource allocation

## 🏗️ Project Structure

```
├── src/
│   ├── server.py      # Server setup and agent card configuration
│   ├── executor.py    # A2A request handling
│   ├── agent.py       # WCHW evaluator implementation
│   └── messenger.py   # A2A messaging utilities
├── data/
│   ├── wchw_test.jsonl      # Test problems
│   └── wchw_validate.jsonl  # Validation problems
├── config/
│   └── agentbeats.json      # AgentBeats configuration
├── tests/
│   └── test_agent.py        # Agent tests
├── Dockerfile
└── pyproject.toml
```

## 🎯 Scoring System

The evaluator supports multiple answer types with intelligent scoring:

| Answer Type | Example | Scoring Method |
|-------------|---------|----------------|
| Numeric with units | `6.87 Mbps`, `240 m` | Relative error tolerance |
| Scientific notation | `5.42e-6`, `2.2×10^-8` | Numeric comparison |
| Mathematical formulas | `1/(2τ_0)`, `A^2 T` | Symbolic matching |
| Text/conceptual | Phase sequences | Keyword matching |

### Scoring Rules

| Accuracy | Score |
|----------|-------|
| Exact match (<1% error) | 1.0 |
| Close match (<5% error) | 0.9 |
| Acceptable (<10% error) | 0.7 |
| Unit conversion error | 0.5 |
| Factor of 2 error | 0.3 |

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Install dependencies
uv sync

# Run the server
uv run src/server.py
```

### Configuration

The agent listens on port 9009 by default. Override with:

```bash
uv run src/server.py --host 0.0.0.0 --port 8080
```

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t wchw-green-agent .

# Run the container
docker run -p 9009:9009 wchw-green-agent
```

## 📝 API Usage

### Request Format

Send an evaluation request with participants and config:

```json
{
  "participants": {
    "wireless_solver": "http://purple-agent:8080"
  },
  "config": {
    "num_problems": 10,
    "timeout": 60,
    "problem_indices": [0, 5, 10]  // optional
  }
}
```

### Response Format

The agent returns structured results:

```json
{
  "summary": {
    "total_problems": 100,
    "evaluated": 10,
    "average_score": 0.85,
    "max_score": 1.0,
    "min_score": 0.5,
    "by_answer_type": {
      "numeric": {"count": 8, "average": 0.9},
      "formula": {"count": 2, "average": 0.7}
    }
  },
  "results": [...]
}
```

## 🧪 Testing

```bash
# Install test dependencies
uv sync --extra test

# Start your agent
uv run src/server.py &

# Run tests
uv run pytest --agent-url http://localhost:9009
```

## 📊 Sample Problems

| Question | Expected Answer | Category |
|----------|-----------------|----------|
| Shannon capacity. B=50 MHz, SNR=0.1. Compute C (Mbps). | 6.87 Mbps | Information Theory |
| Two-ray critical distance. f_c=900 MHz, h_t=10 m, h_r=2 m. | 240 m | Channel Modeling |
| Convert 30 dBm to watts. | 1.00 W | Signal Processing |

## 🔗 Links

- [A2A Protocol Documentation](https://a2a-protocol.org/latest/)
- [AgentBeats Platform](https://agentbeats.dev)
- [UC Berkeley RDI Foundation](https://rdi.berkeley.edu/)

## 📄 License

MIT License - see LICENSE file for details.

## 👤 Author

**Jingwen Tong**

UC Berkeley RDI Foundation AgentBeats Competition
