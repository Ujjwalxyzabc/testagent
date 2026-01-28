
import os
import re
import json
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Tuple, Union

from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator, constr
from dotenv import load_dotenv
import openai
import httpx
import redis
from loguru import logger
from cryptography.fernet import Fernet
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# =========================
# Configuration Management
# =========================

class Config:
    """Configuration loader and validator."""
    load_dotenv()
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMERSON_MODEL_API_URL: str = os.getenv("EMERSON_MODEL_API_URL", "")
    EMERSON_ATP_API_URL: str = os.getenv("EMERSON_ATP_API_URL", "")
    EMERSON_CLIENT_ID: str = os.getenv("EMERSON_CLIENT_ID", "")
    EMERSON_CLIENT_SECRET: str = os.getenv("EMERSON_CLIENT_SECRET", "")
    EMERSON_TOKEN_URL: str = os.getenv("EMERSON_TOKEN_URL", "")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    LOG_ENCRYPTION_KEY: str = os.getenv("LOG_ENCRYPTION_KEY", Fernet.generate_key().decode())
    API_RATE_LIMIT: int = int(os.getenv("API_RATE_LIMIT", "50"))
    MAX_TEXT_LENGTH: int = 50000

    @classmethod
    def validate(cls):
        missing = []
        for attr in [
            "OPENAI_API_KEY", "EMERSON_MODEL_API_URL", "EMERSON_ATP_API_URL",
            "EMERSON_CLIENT_ID", "EMERSON_CLIENT_SECRET", "EMERSON_TOKEN_URL"
        ]:
            if not getattr(cls, attr):
                missing.append(attr)
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

Config.validate()

# =========================
# Logging Configuration
# =========================

logger.remove()
logger.add(
    "rfq_validation_agent.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO",
    format="{time} | {level} | {message}"
)

# =========================
# Security Utilities
# =========================

class SecurityUtils:
    """Utility for encryption and PII masking."""
    fernet = Fernet(Config.LOG_ENCRYPTION_KEY.encode())

    @staticmethod
    def encrypt(data: str) -> str:
        return SecurityUtils.fernet.encrypt(data.encode()).decode()

    @staticmethod
    def decrypt(token: str) -> str:
        return SecurityUtils.fernet.decrypt(token.encode()).decode()

    @staticmethod
    def mask_pii(data: Any) -> Any:
        """Mask PII fields in dicts/lists/strings."""
        if isinstance(data, dict):
            masked = {}
            for k, v in data.items():
                if re.search(r"(customer|email|phone|contact|address)", k, re.I):
                    masked[k] = "***REDACTED***"
                else:
                    masked[k] = SecurityUtils.mask_pii(v)
            return masked
        elif isinstance(data, list):
            return [SecurityUtils.mask_pii(i) for i in data]
        elif isinstance(data, str):
            # Mask emails and phone numbers in strings
            data = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "***REDACTED***", data)
            data = re.sub(r"\b\d{10,}\b", "***REDACTED***", data)
            return data
        else:
            return data

# =========================
# Pydantic Models & DTOs
# =========================

class RFQItem(BaseModel):
    model_string: constr(strip_whitespace=True, min_length=1, max_length=100)
    quantity: int = Field(..., ge=1, le=100000)
    unit_price: float = Field(..., ge=0.0, le=1e7)

class RFQHeader(BaseModel):
    rfq_number: constr(strip_whitespace=True, min_length=1, max_length=100)
    customer: constr(strip_whitespace=True, min_length=1, max_length=200)
    date: constr(strip_whitespace=True, min_length=8, max_length=20)

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except Exception:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

class RFQNormalized(BaseModel):
    header: RFQHeader
    items: List[RFQItem]

    @model_validator(mode="after")
    def check_items(self):
        if not self.items or len(self.items) == 0:
            raise ValueError("At least one item is required in RFQ")
        return self

class RFQValidationRequest(BaseModel):
    content: constr(strip_whitespace=True, min_length=1, max_length=Config.MAX_TEXT_LENGTH)

    @field_validator("content")
    @classmethod
    def clean_content(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Input content is empty")
        if len(v) > Config.MAX_TEXT_LENGTH:
            raise ValueError(f"Input exceeds {Config.MAX_TEXT_LENGTH} characters")
        return v

class RFQValidationResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    tips: Optional[str] = None
    metadata: Optional[dict] = None

# =========================
# Base Classes
# =========================

class BaseService:
    """Base class for all services."""
    def __init__(self):
        self.logger = logger

class BaseAPIClient(BaseService):
    """Base class for all external API clients."""
    def __init__(self):
        super().__init__()

# =========================
# Cache Manager
# =========================

class CacheManager(BaseService):
    """Redis-based cache for model_name and lead_time_atp."""
    def __init__(self):
        super().__init__()
        self.redis = redis.Redis.from_url(Config.REDIS_URL, decode_responses=True)

    def get(self, key: str) -> Optional[str]:
        try:
            return self.redis.get(key)
        except Exception as e:
            self.logger.warning(f"Cache get error: {e}")
            return None

    def set(self, key: str, value: str, ttl: int = 86400):
        try:
            self.redis.set(key, value, ex=ttl)
        except Exception as e:
            self.logger.warning(f"Cache set error: {e}")

# =========================
# Audit Logger
# =========================

class AuditLogger(BaseService):
    """Logs validation events, errors, and audit trails with PII masking and encryption."""
    def __init__(self):
        super().__init__()

    def log(self, event_type: str, data: Any):
        masked = SecurityUtils.mask_pii(data)
        encrypted = SecurityUtils.encrypt(json.dumps(masked))
        self.logger.info(f"AuditLog | {event_type} | {encrypted}")

# =========================
# Error Handler
# =========================

class ErrorHandler(BaseService):
    """Centralized error handling, retry logic, and fallback behaviors."""
    def __init__(self, audit_logger: AuditLogger):
        super().__init__()
        self.audit_logger = audit_logger

    def handle_error(self, error_type: str, context: dict) -> dict:
        error_map = {
            "FormatError": ("Malformed input or unrecognized RFQ format.", "Check JSON structure, quotes, and required fields."),
            "ValidationError": ("Missing or invalid mandatory fields.", "Ensure all required fields are present and correctly formatted."),
            "ExternalAPIError": ("External API call failed.", "Try again later or contact support."),
            "LLMError": ("LLM processing failed.", "Try again or contact support."),
            "ExtractionError": ("Failed to extract/normalize input.", "Ensure input is valid JSON or wrapped RFQ format."),
            "UnknownError": ("An unknown error occurred.", "Contact support with error details."),
        }
        msg, tip = error_map.get(error_type, ("Unknown error", "Contact support."))
        self.audit_logger.log("error", {"error_type": error_type, "context": context})
        return {
            "success": False,
            "error": msg,
            "error_type": error_type,
            "tips": tip,
            "metadata": {"timestamp": datetime.utcnow().isoformat()}
        }

# =========================
# Input Normalizer
# =========================

class InputNormalizer(BaseService):
    """Detects and normalizes input from wrapped or direct RFQ formats."""
    def normalize(self, input_data: str) -> dict:
        # Try direct JSON first
        try:
            data = json.loads(input_data)
            if "header" in data and "items" in data:
                return data
        except Exception:
            pass
        # Try to extract JSON from wrapped format
        try:
            match = re.search(r"\{.*\}", input_data, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if "header" in data and "items" in data:
                    return data
        except Exception:
            pass
        raise ValueError("FormatError: Unable to parse input as valid RFQ JSON.")

# =========================
# Mandatory Field Validator
# =========================

class MandatoryFieldValidator(BaseService):
    """Checks for presence and correctness of required RFQ fields."""
    def validate_fields(self, normalized_data: dict) -> Tuple[bool, List[str]]:
        errors = []
        try:
            RFQNormalized(**normalized_data)
        except ValidationError as ve:
            for err in ve.errors():
                errors.append(f"{err['loc']}: {err['msg']}")
        except Exception as e:
            errors.append(str(e))
        return (len(errors) == 0, errors)

# =========================
# Emerson Model Validation API Client
# =========================

class EmersonModelValidationAPIClient(BaseAPIClient):
    """Handles requests to Emerson Model Validation API."""
    def __init__(self, cache: CacheManager):
        super().__init__()
        self.cache = cache

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, min=1, max=4))
    async def validate_model(self, model_string: str) -> dict:
        cache_key = f"model_name:{model_string}"
        cached = self.cache.get(cache_key)
        if cached:
            return {"valid": True, "model_name": cached}
        # Get OAuth2 token
        token = await self._get_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"model_string": model_string}
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(Config.EMERSON_MODEL_API_URL, json=payload, headers=headers)
            if resp.status_code == 200:
                result = resp.json()
                if result.get("valid"):
                    model_name = result.get("model_name", model_string)
                    self.cache.set(cache_key, model_name)
                    return {"valid": True, "model_name": model_name}
                else:
                    return {"valid": False, "model_name": None}
            else:
                raise Exception(f"Model API error: {resp.status_code} {resp.text}")

    async def _get_token(self) -> str:
        cache_key = "emerson_token"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        data = {
            "grant_type": "client_credentials",
            "client_id": Config.EMERSON_CLIENT_ID,
            "client_secret": Config.EMERSON_CLIENT_SECRET,
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(Config.EMERSON_TOKEN_URL, data=data)
            if resp.status_code == 200:
                token = resp.json().get("access_token")
                self.cache.set(cache_key, token, ttl=3500)
                return token
            else:
                raise Exception(f"Token API error: {resp.status_code} {resp.text}")

# =========================
# Emerson ATP API Client
# =========================

class EmersonATPAPIClient(BaseAPIClient):
    """Handles requests to Emerson ATP API."""
    def __init__(self, cache: CacheManager):
        super().__init__()
        self.cache = cache

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, min=1, max=4))
    async def get_atp(self, model_string: str, quantity: int) -> dict:
        cache_key = f"atp:{model_string}:{quantity}"
        cached = self.cache.get(cache_key)
        if cached:
            return json.loads(cached)
        token = await self._get_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"model_string": model_string, "quantity": quantity}
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(Config.EMERSON_ATP_API_URL, json=payload, headers=headers)
            if resp.status_code == 200:
                result = resp.json()
                self.cache.set(cache_key, json.dumps(result))
                return result
            else:
                raise Exception(f"ATP API error: {resp.status_code} {resp.text}")

    async def _get_token(self) -> str:
        cache_key = "emerson_token"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        data = {
            "grant_type": "client_credentials",
            "client_id": Config.EMERSON_CLIENT_ID,
            "client_secret": Config.EMERSON_CLIENT_SECRET,
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(Config.EMERSON_TOKEN_URL, data=data)
            if resp.status_code == 200:
                token = resp.json().get("access_token")
                self.cache.set(cache_key, token, ttl=3500)
                return token
            else:
                raise Exception(f"Token API error: {resp.status_code} {resp.text}")

# =========================
# Model String Validator
# =========================

class ModelStringValidator(BaseService):
    """Validates model_string via Emerson Model Validation API and enriches with model_name."""
    def __init__(self, api_client: EmersonModelValidationAPIClient):
        super().__init__()
        self.api_client = api_client

    async def validate_model_string(self, model_string: str) -> dict:
        try:
            result = await self.api_client.validate_model(model_string)
            return result
        except RetryError as e:
            self.logger.error(f"Model validation retry failed: {e}")
            return {"valid": False, "model_name": None}
        except Exception as e:
            self.logger.error(f"Model validation error: {e}")
            return {"valid": False, "model_name": None}

# =========================
# ATP Validator
# =========================

class ATPValidator(BaseService):
    """Validates product availability and lead time via Emerson ATP API."""
    def __init__(self, api_client: EmersonATPAPIClient):
        super().__init__()
        self.api_client = api_client

    async def validate_atp(self, model_string: str, quantity: int) -> dict:
        try:
            result = await self.api_client.get_atp(model_string, quantity)
            return result
        except RetryError as e:
            self.logger.error(f"ATP validation retry failed: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"ATP validation error: {e}")
            return {}

# =========================
# Validation Report Builder
# =========================

class ValidationReportBuilder(BaseService):
    """Aggregates validation results, enriches data, and builds structured validation report."""
    def build_report(self, validation_results: dict, enriched_data: dict, errors: List[str]) -> dict:
        report = {
            "status": "SUCCESS" if not errors else "FAILED",
            "validation_results": validation_results,
            "enriched_data": enriched_data,
            "errors": errors,
            "timestamp": datetime.utcnow().isoformat()
        }
        return report

# =========================
# LLM Orchestrator
# =========================

class LLMOrchestrator(BaseService):
    """Handles LLM prompt construction, interaction, and fallback logic."""
    def __init__(self):
        super().__init__()
        self.client = openai.AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = "gpt-4o"
        self.fallback_model = "gpt-3.5-turbo"
        self.temperature = 0.2
        self.max_tokens = 2048
        self.system_prompt = (
            "You are an RFQ Validation Agent. Accept input in either wrapped or direct RFQ format, "
            "validate all mandatory fields, verify model strings and ATP via Emerson APIs, enrich the data, "
            "and return a structured validation report. Always provide clear error messages and detailed validation status."
        )
        self.user_prompt_template = "Please provide the RFQ data in either wrapped or direct format for validation."
        self.few_shot_examples = [
            '{ "success": true, "data": "{\\"header\\":{\\"rfq_number\\":\\"RFQ123\\",\\"customer\\":\\"Acme Corp\\",\\"date\\":\\"2024-06-01\\"},\\"items\\":[{\\"model_string\\":\\"EM-1001\\",\\"quantity\\":5,\\"unit_price\\":100.0}]}", "error": "", "filename": "rfq_acme_20240601.json" }',
            '{ "header": {"rfq_number": "", "customer": "Beta Ltd", "date": "2024-06-02"}, "items": [{"model_string": "INVALID", "quantity": 2, "unit_price": 200.0}] }'
        ]

    async def generate_response(self, context: dict) -> dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_template},
            {"role": "user", "content": json.dumps(context)}
        ]
        # Add few-shot examples as user messages
        for ex in self.few_shot_examples:
            messages.append({"role": "user", "content": ex})
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            content = response.choices[0].message.content
            try:
                return json.loads(content)
            except Exception:
                return {"llm_output": content}
        except Exception as e:
            self.logger.warning(f"LLM error: {e}, trying fallback model.")
            # Fallback to gpt-3.5-turbo
            try:
                response = await self.client.chat.completions.create(
                    model=self.fallback_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                content = response.choices[0].message.content
                try:
                    return json.loads(content)
                except Exception:
                    return {"llm_output": content}
            except Exception as e2:
                self.logger.error(f"LLM fallback error: {e2}")
                return {"llm_output": "LLM processing failed."}

# =========================
# Main Agent
# =========================

class RFQValidationAgent(BaseService):
    """Main agent orchestrating the RFQ validation workflow."""
    def __init__(self):
        super().__init__()
        self.cache = CacheManager()
        self.audit_logger = AuditLogger()
        self.error_handler = ErrorHandler(self.audit_logger)
        self.input_normalizer = InputNormalizer()
        self.mandatory_field_validator = MandatoryFieldValidator()
        self.model_api_client = EmersonModelValidationAPIClient(self.cache)
        self.atp_api_client = EmersonATPAPIClient(self.cache)
        self.model_string_validator = ModelStringValidator(self.model_api_client)
        self.atp_validator = ATPValidator(self.atp_api_client)
        self.report_builder = ValidationReportBuilder()
        self.llm_orchestrator = LLMOrchestrator()

    async def validate_rfq(self, rfq_data: str, request_context: dict) -> dict:
        self.audit_logger.log("request_received", {"rfq_data": rfq_data, "context": request_context})
        try:
            # Step 1: Normalize input
            try:
                normalized = self.input_normalizer.normalize(rfq_data)
            except Exception as e:
                return self.error_handler.handle_error("FormatError", {"error": str(e)})

            # Step 2: Validate mandatory fields
            valid_fields, field_errors = self.mandatory_field_validator.validate_fields(normalized)
            if not valid_fields:
                return self.error_handler.handle_error("ValidationError", {"errors": field_errors})

            # Step 3: Validate/enrich model_string and ATP for each item
            validation_results = []
            enriched_items = []
            errors = []
            for item in normalized["items"]:
                model_result = await self.model_string_validator.validate_model_string(item["model_string"])
                item_result = {"model_string": item["model_string"], "model_valid": model_result["valid"]}
                if model_result["valid"]:
                    item_result["model_name"] = model_result["model_name"]
                    # ATP validation
                    atp_result = await self.atp_validator.validate_atp(item["model_string"], item["quantity"])
                    item_result["atp"] = atp_result
                else:
                    item_result["model_name"] = None
                    item_result["atp"] = {}
                    errors.append(f"Invalid model_string: {item['model_string']}")
                validation_results.append(item_result)
                enriched_items.append({
                    **item,
                    "model_name": model_result.get("model_name"),
                    "atp": item_result["atp"]
                })

            # Step 4: Build report
            enriched_data = {
                "header": normalized["header"],
                "items": enriched_items
            }
            report = self.report_builder.build_report(validation_results, enriched_data, errors)
            self.audit_logger.log("validation_report", report)
            return {
                "success": report["status"] == "SUCCESS",
                "data": report,
                "error": None if report["status"] == "SUCCESS" else "; ".join(errors),
                "error_type": None if report["status"] == "SUCCESS" else "ValidationError",
                "tips": None,
                "metadata": {"timestamp": report["timestamp"]}
            }
        except Exception as e:
            self.logger.error(f"Unknown error in validate_rfq: {e}")
            return self.error_handler.handle_error("UnknownError", {"error": str(e)})

# =========================
# API Handler (FastAPI)
# =========================

app = FastAPI(
    title="RFQ Validation Agent",
    description="API for validating and enriching RFQ data.",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

agent = RFQValidationAgent()

# =========================
# Exception Handlers
# =========================

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Input validation failed.",
            "error_type": "ValidationError",
            "tips": "Check your JSON structure, required fields, and value types.",
            "metadata": {"timestamp": datetime.utcnow().isoformat()}
        }
    )

@app.exception_handler(json.JSONDecodeError)
async def json_decode_exception_handler(request: Request, exc: json.JSONDecodeError):
    logger.warning(f"Malformed JSON: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error": "Malformed JSON in request.",
            "error_type": "FormatError",
            "tips": "Ensure your JSON is properly formatted. Common issues: missing quotes, trailing commas, or unescaped characters.",
            "metadata": {"timestamp": datetime.utcnow().isoformat()}
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error.",
            "error_type": "UnknownError",
            "tips": "Contact support with error details.",
            "metadata": {"timestamp": datetime.utcnow().isoformat()}
        }
    )

# =========================
# Dependency: OAuth2 Auth
# =========================

async def get_current_token(token: str = Depends(oauth2_scheme)):
    # In production, validate token with OAuth2 provider
    if not token or len(token) < 10:
        raise HTTPException(status_code=401, detail="Invalid or missing OAuth2 token")
    return token

# =========================
# API Endpoints
# =========================

@app.post("/validate_rfq", response_model=RFQValidationResponse)
async def validate_rfq_endpoint(
    request: Request,
    rfq_req: RFQValidationRequest,
    token: str = Depends(get_current_token)
):
    """
    Validate and enrich RFQ data.
    """
    try:
        # Input validation and sanitization
        content = rfq_req.content.strip()
        if not content:
            return RFQValidationResponse(
                success=False,
                error="Input content is empty.",
                error_type="ValidationError",
                tips="Provide valid RFQ data in JSON or wrapped format.",
                metadata={"timestamp": datetime.utcnow().isoformat()}
            )
        if len(content) > Config.MAX_TEXT_LENGTH:
            return RFQValidationResponse(
                success=False,
                error=f"Input exceeds {Config.MAX_TEXT_LENGTH} characters.",
                error_type="ValidationError",
                tips=f"Reduce input size below {Config.MAX_TEXT_LENGTH} characters.",
                metadata={"timestamp": datetime.utcnow().isoformat()}
            )
        # Main validation workflow
        result = await agent.validate_rfq(content, {"ip": request.client.host, "token": token})
        return RFQValidationResponse(**result)
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve}")
        return RFQValidationResponse(
            success=False,
            error="Input validation failed.",
            error_type="ValidationError",
            tips="Check your JSON structure, required fields, and value types.",
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
    except json.JSONDecodeError as je:
        logger.warning(f"Malformed JSON: {je}")
        return RFQValidationResponse(
            success=False,
            error="Malformed JSON in request.",
            error_type="FormatError",
            tips="Ensure your JSON is properly formatted. Common issues: missing quotes, trailing commas, or unescaped characters.",
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
    except Exception as e:
        logger.error(f"Unhandled exception in /validate_rfq: {e}")
        return RFQValidationResponse(
            success=False,
            error="Internal server error.",
            error_type="UnknownError",
            tips="Contact support with error details.",
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"success": True, "status": "ok", "timestamp": datetime.utcnow().isoformat()}

# =========================
# Main Entry Point
# =========================

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting RFQ Validation Agent API...")
    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=False)
