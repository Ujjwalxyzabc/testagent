
# config.py

import os
from dotenv import load_dotenv

class ConfigError(Exception):
    pass

class Config:
    # 1. Environment variable loading
    load_dotenv()
    ENV = os.getenv("ENV", "production")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # 2. API key management
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMERSON_MODEL_API_URL = os.getenv("EMERSON_MODEL_API_URL")
    EMERSON_ATP_API_URL = os.getenv("EMERSON_ATP_API_URL")
    EMERSON_CLIENT_ID = os.getenv("EMERSON_CLIENT_ID")
    EMERSON_CLIENT_SECRET = os.getenv("EMERSON_CLIENT_SECRET")
    EMERSON_TOKEN_URL = os.getenv("EMERSON_TOKEN_URL")
    RFQ_API_RATE_LIMIT = int(os.getenv("RFQ_API_RATE_LIMIT", "50"))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    LOG_ENCRYPTION_KEY = os.getenv("LOG_ENCRYPTION_KEY", None)

    # 3. LLM configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    LLM_SYSTEM_PROMPT = (
        os.getenv("LLM_SYSTEM_PROMPT") or
        "You are an RFQ Validation Agent. Accept input in either wrapped or direct RFQ format, validate all mandatory fields, verify model strings and ATP via Emerson APIs, enrich the data, and return a structured validation report. Always provide clear error messages and detailed validation status."
    )
    LLM_USER_PROMPT_TEMPLATE = (
        os.getenv("LLM_USER_PROMPT_TEMPLATE") or
        "Please provide the RFQ data in either wrapped or direct format for validation."
    )
    LLM_FEW_SHOT_EXAMPLES = [
        '{ "success": true, "data": "{\\"header\\":{\\"rfq_number\\":\\"RFQ123\\",\\"customer\\":\\"Acme Corp\\",\\"date\\":\\"2024-06-01\\"},\\"items\\":[{\\"model_string\\":\\"EM-1001\\",\\"quantity\\":5,\\"unit_price\\":100.0}]}", "error": "", "filename": "rfq_acme_20240601.json" }',
        '{ "header": {"rfq_number": "", "customer": "Beta Ltd", "date": "2024-06-02"}, "items": [{"model_string": "INVALID", "quantity": 2, "unit_price": 200.0}] }'
    ]

    # 4. Domain-specific settings
    DOMAIN = "RFQ Processing"
    AGENT_NAME = "RFQ Validation Agent"
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24 hours
    LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "30"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "50000"))

    # 5. Validation and error handling
    ERROR_RESPONSES = {
        "missing_api_key": {
            "success": False,
            "error": "Missing required API key or secret.",
            "error_type": "ConfigError",
            "tips": "Check environment variables and secrets configuration.",
        },
        "missing_env": {
            "success": False,
            "error": "Missing required environment variable.",
            "error_type": "ConfigError",
            "tips": "Check .env file or deployment environment.",
        },
        "invalid_input": {
            "success": False,
            "error": "Invalid input data.",
            "error_type": "ValidationError",
            "tips": "Check input format and required fields.",
        },
        "external_api_error": {
            "success": False,
            "error": "External API call failed.",
            "error_type": "ExternalAPIError",
            "tips": "Try again later or contact support.",
        },
        "unknown_error": {
            "success": False,
            "error": "An unknown error occurred.",
            "error_type": "UnknownError",
            "tips": "Contact support with error details.",
        }
    }

    # 6. Default values and fallbacks
    @classmethod
    def get_llm_config(cls):
        return {
            "provider": cls.LLM_PROVIDER,
            "model": cls.LLM_MODEL,
            "temperature": cls.LLM_TEMPERATURE,
            "max_tokens": cls.LLM_MAX_TOKENS,
            "system_prompt": cls.LLM_SYSTEM_PROMPT,
            "user_prompt_template": cls.LLM_USER_PROMPT_TEMPLATE,
            "few_shot_examples": cls.LLM_FEW_SHOT_EXAMPLES
        }

    @classmethod
    def get_api_requirements(cls):
        return [
            {
                "name": "Emerson Model Validation API",
                "type": "external",
                "purpose": "Validate model_string and retrieve canonical model_name.",
                "authentication": "OAuth2",
                "rate_limits": "As per Emerson API agreement"
            },
            {
                "name": "Emerson ATP API",
                "type": "external",
                "purpose": "Check product availability and lead time for model_string and quantity.",
                "authentication": "OAuth2",
                "rate_limits": "As per Emerson API agreement"
            },
            {
                "name": "RFQ Validation API",
                "type": "internal",
                "purpose": "Expose validation endpoint for RFQ data.",
                "authentication": "OAuth2",
                "rate_limits": f"{cls.RFQ_API_RATE_LIMIT} concurrent requests"
            }
        ]

    @classmethod
    def check_required(cls):
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not cls.EMERSON_MODEL_API_URL:
            missing.append("EMERSON_MODEL_API_URL")
        if not cls.EMERSON_ATP_API_URL:
            missing.append("EMERSON_ATP_API_URL")
        if not cls.EMERSON_CLIENT_ID:
            missing.append("EMERSON_CLIENT_ID")
        if not cls.EMERSON_CLIENT_SECRET:
            missing.append("EMERSON_CLIENT_SECRET")
        if not cls.EMERSON_TOKEN_URL:
            missing.append("EMERSON_TOKEN_URL")
        if missing:
            raise ConfigError(f"Missing required API keys or URLs: {', '.join(missing)}")

    # 7. Error handling for missing API keys
    @classmethod
    def validate_or_raise(cls):
        try:
            cls.check_required()
        except ConfigError as e:
            raise

# Validate config at import time
try:
    Config.validate_or_raise()
except Exception as e:
    # Comment out the next line if you want to suppress import-time errors
    # raise
    print(f"Config error: {e}")

