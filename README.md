# RFQ Validation Agent

## Overview
RFQ Validation Agent is a neutral, methodical, precise, reliable RFQ Processing agent designed for text_to_text interactions.

## Features


## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the agent:
```bash
python agent.py
```

## Configuration

The agent uses the following environment variables:
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key (if using Anthropic)
- `GOOGLE_API_KEY`: Google API key (if using Google)

## Usage

```python
from agent import RFQ Validation AgentAgent

agent = RFQ Validation AgentAgent()
response = await agent.process_message("Hello!")
```

## Domain: RFQ Processing
## Personality: neutral, methodical, precise, reliable
## Modality: text_to_text