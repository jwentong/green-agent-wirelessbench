import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the WCHW Green Agent - Wireless Communication Benchmark Evaluator")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    # WCHW Benchmark Evaluation Skill
    # This skill defines how this green agent evaluates wireless communication problem solvers
    skill = AgentSkill(
        id="wchw-benchmark-eval",
        name="WCHW Benchmark Evaluation",
        description="""Evaluates AI agents on the Wireless Communication Homework (WCHW) benchmark.
        
This benchmark tests agents on 100+ problems covering:
- Information Theory (Shannon capacity, entropy, channel coding)
- Channel Modeling (path loss, fading, MIMO, propagation)
- Signal Processing (modulation, detection, filtering)
- System Design (link budget, resource allocation)

The evaluator supports multiple answer types:
- Numeric with units (e.g., "16 kbit/s", "44.8 kHz")
- Scientific notation (e.g., "5.42e-6")
- Mathematical formulas (e.g., "1/(2τ_0)")
- Text/conceptual answers

Scoring uses relative error tolerance with partial credit for:
- Exact matches (<1% error): 1.0
- Close matches (<5% error): 0.9  
- Acceptable (<10% error): 0.7
- Unit conversion errors: 0.5
- Factor of 2 errors: 0.3""",
        tags=[
            "wireless-communication",
            "telecommunications",
            "benchmark",
            "evaluation",
            "information-theory",
            "signal-processing",
            "channel-modeling"
        ],
        examples=[
            "Evaluate an agent on 10 randomly selected WCHW problems",
            "Run full WCHW benchmark evaluation with 70 problems",
            "Test wireless communication problem-solving capabilities"
        ]
    )

    agent_card = AgentCard(
        name="WCHW Green Agent",
        description="""Green Agent for WCHW (Wireless Communication Homework) Benchmark Evaluation.

Part of UC Berkeley RDI Foundation's AgentBeats Competition.

This agent evaluates purple agents (wireless communication problem solvers) by:
1. Sending test problems from the WCHW dataset
2. Evaluating responses with type-aware scoring
3. Computing comprehensive benchmark metrics

The WCHW benchmark covers advanced telecommunications topics including:
- Channel capacity and Shannon limits
- Modulation and detection schemes
- Propagation and path loss models
- MIMO and diversity techniques
- Noise analysis and SNR calculations

Author: Jingwen Tong
Version: 1.0.0""",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text', 'data'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║             WCHW Green Agent - AgentBeats Competition            ║
║          Wireless Communication Homework Benchmark               ║
╠══════════════════════════════════════════════════════════════════╣
║  Author: Jingwen Tong                                            ║
║  UC Berkeley RDI Foundation                                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Server: http://{args.host}:{args.port}/                                       ║
║  Agent Card: http://{args.host}:{args.port}/.well-known/agent-card.json        ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
