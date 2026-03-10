# llm-sentinel
llm_sentinel is a Python middleware library that wraps LLM API clients (OpenAI, Anthropic, etc.) to detect, log, and alert on prompt injection and jailbreak attempts in real time. It analyzes incoming prompts against a configurable rule engine combining regex patterns, heuristic scoring, and keyword blacklists. Detected threats are logged locally w
