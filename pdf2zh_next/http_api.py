"""HTTP API for PDF translation service using FastAPI."""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import chardet

from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pydantic import Field
from sse_starlette.sse import EventSourceResponse

from pdf2zh_next.high_level import do_translate_async_stream
from pdf2zh_next.models_config import SUPPORTED_MODELS, DEFAULT_MODEL, is_model_supported

logger = logging.getLogger(__name__)

# Configuration constants
TEMP_DIR_BASE = Path(tempfile.gettempdir()) / "pdf2zh_next_api"
MAX_TASK_AGE_HOURS = 2  # Auto-cleanup tasks older than this (aggressive: 2 hours)
MIN_TASK_AGE_MINUTES = 30  # Minimum age before task can be cleaned (allow download time)
CLEANUP_INTERVAL_SECONDS = 300  # Run cleanup every 5 minutes (aggressive)
MAX_STORAGE_GB = 2  # Maximum storage limit in GB (aggressive)
STORAGE_WARNING_THRESHOLD = 0.7  # Start aggressive cleanup at 70% capacity
MAX_TASK_SIZE_MB = 200  # Maximum size per task in MB

# In-memory storage for translation tasks
translation_tasks: dict[str, dict[str, Any]] = {}
# Store output files
translation_files: dict[str, dict[str, Path]] = {}
# Store task directories for cleanup
task_directories: dict[str, Path] = {}


class TranslationRequest(BaseModel):
    """Request model for translation."""

    # Model selection (required)
    model: str = Field(
        default=DEFAULT_MODEL, 
        description=f"Translation model name. Use GET /v1/models to see all available models. Default: {DEFAULT_MODEL}"
    )
    
    # Basic settings
    lang_in: str = Field(default="en", description="Source language code")
    lang_out: str = Field(default="zh", description="Target language code")
    pages: str | None = Field(
        default=None, description="Pages to translate (e.g. '1,2,1-,-3,3-5')"
    )
    no_dual: bool = Field(
        default=False, description="Do not output bilingual PDF files"
    )
    no_mono: bool = Field(
        default=False, description="Do not output monolingual PDF files"
    )
    
    # PDF processing options
    split_short_lines: bool = Field(
        default=False, description="Force split short lines into different paragraphs"
    )
    short_line_split_factor: float = Field(
        default=0.8, description="Split threshold factor for short lines"
    )
    skip_clean: bool = Field(
        default=False, description="Skip PDF cleaning step"
    )
    dual_translate_first: bool = Field(
        default=False, description="Put translated pages first in dual PDF mode"
    )
    disable_rich_text_translate: bool = Field(
        default=False, description="Disable rich text translation"
    )
    enhance_compatibility: bool = Field(
        default=False, description="Enable all compatibility enhancement options"
    )
    use_alternating_pages_dual: bool = Field(
        default=False, description="Use alternating pages mode for dual PDF"
    )
    watermark_output_mode: str = Field(
        default="no_watermark", 
        description="Control watermark output mode: 'watermarked' (default), 'no_watermark', or 'both'"
    )
    max_pages_per_part: int | None = Field(
        default=None, description="Maximum number of pages per part for split translation"
    )
    translate_table_text: bool = Field(
        default=True, description="Translate table text (experimental)"
    )
    formular_font_pattern: str | None = Field(
        default=None, description="Font pattern to identify formula text"
    )
    formular_char_pattern: str | None = Field(
        default=None, description="Character pattern to identify formula text"
    )
    skip_scanned_detection: bool = Field(
        default=False, description="Skip scanned document detection"
    )
    ocr_workaround: bool = Field(
        default=False, description="Use OCR workaround for scanned PDFs"
    )
    auto_enable_ocr_workaround: bool = Field(
        default=False, description="Enable automatic OCR workaround for scanned PDFs"
    )
    only_include_translated_page: bool = Field(
        default=False, description="Only include translated pages in the output PDF"
    )
    merge_alternating_line_numbers: bool = Field(
        default=True, description="Enable merging alternating line-number layouts"
    )
    remove_non_formula_lines: bool = Field(
        default=True, description="Remove non-formula lines from paragraph areas"
    )
    non_formula_line_iou_threshold: float = Field(
        default=0.9, description="IoU threshold for detecting paragraph overlap"
    )
    figure_table_protection_threshold: float = Field(
        default=0.9, description="IoU threshold for protecting lines in figure/table areas"
    )
    primary_font_family: str | None = Field(
        default=None, description="Override primary font family: 'serif', 'sans-serif', or 'script'"
    )
    
    # Translation service options
    min_text_length: int = Field(
        default=5, description="Minimum text length to translate"
    )
    qps: int = Field(
        default=4, description="QPS (Queries Per Second) limit for translation service"
    )
    ignore_cache: bool = Field(
        default=False, description="Ignore translation cache and force retranslation"
    )
    custom_system_prompt: str | None = Field(
        default=None, description="Custom system prompt for translation"
    )
    pool_max_workers: int | None = Field(
        default=None, description="Maximum number of worker threads for task processing"
    )
    
    # Glossary options
    no_auto_extract_glossary: bool = Field(
        default=True, description="Disable automatic term extraction"
    )
    save_auto_extracted_glossary: bool = Field(
        default=False, description="Save automatically extracted glossary"
    )
    
    # Advanced options
    rpc_doclayout: str | None = Field(
        default=None, description="RPC service host address for document layout analysis"
    )
    
    # OpenAI translator specific
    openai_model: str | None = Field(default=None, description="OpenAI model")
    openai_base_url: str | None = Field(default=None, description="OpenAI base URL")
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    # Azure translator specific
    azure_api_key: str | None = Field(default=None, description="Azure API key")
    azure_endpoint: str | None = Field(default=None, description="Azure endpoint")
    # DeepL translator specific
    deepl_auth_key: str | None = Field(default=None, description="DeepL auth key")
    # Ollama translator specific
    ollama_model: str | None = Field(default="gemma2", description="Ollama model")
    ollama_host: str | None = Field(default=None, description="Ollama host")
    # Tencent translator specific
    tencentcloud_secret_id: str | None = Field(
        default=None, description="Tencent secret ID"
    )
    tencentcloud_secret_key: str | None = Field(
        default=None, description="Tencent secret key"
    )
    # Xinference translator specific
    xinference_model: str | None = Field(
        default="gemma-2-it", description="Xinference model"
    )
    xinference_host: str | None = Field(default=None, description="Xinference host")


class TranslationStatusResponse(BaseModel):
    """Response model for translation status."""

    task_id: str
    status: str  # queued, processing, completed, error
    progress: dict[str, Any] | None = None
    error: str | None = None
    mono_file: str | None = None
    dual_file: str | None = None


def get_task_directory(task_id: str) -> Path:
    """Get the directory path for a specific task."""
    return TEMP_DIR_BASE / task_id


def get_current_storage_usage() -> int:
    """Get current storage usage in bytes."""
    if not TEMP_DIR_BASE.exists():
        return 0
    
    current_usage = 0
    for file_path in TEMP_DIR_BASE.rglob("*"):
        if file_path.is_file():
            try:
                current_usage += file_path.stat().st_size
            except Exception:
                pass
    return current_usage


def check_storage_available(required_size: int = 0) -> tuple[bool, str]:
    """Check if storage is available for new tasks.
    
    Args:
        required_size: Estimated size needed for new task (in bytes)
        
    Returns:
        Tuple of (is_available, error_message)
    """
    # Check single task size limit
    max_task_bytes = MAX_TASK_SIZE_MB * 1024 * 1024
    if required_size > max_task_bytes:
        return False, (
            f"File too large. Maximum allowed size: {MAX_TASK_SIZE_MB}MB. "
            f"Please use a smaller PDF file."
        )
    
    current_usage = get_current_storage_usage()
    max_bytes = MAX_STORAGE_GB * 1024 * 1024 * 1024
    
    if current_usage + required_size > max_bytes:
        current_mb = current_usage / (1024 ** 2)
        max_mb = MAX_STORAGE_GB * 1024
        return False, (
            f"Storage limit exceeded. Current usage: {current_mb:.1f}MB / {max_mb:.0f}MB. "
            f"Please wait for automatic cleanup or delete old tasks."
        )
    
    return True, ""


async def cleanup_old_tasks():
    """Clean up tasks based on age and storage pressure.
    
    Cleanup strategy (aggressive):
    1. Always delete tasks older than MAX_TASK_AGE_HOURS (2 hours)
    2. If storage > 70%, delete tasks older than MIN_TASK_AGE_MINUTES (30 min)
    3. If storage > 90%, delete oldest completed tasks regardless of age
    """
    current_time = time.time()
    max_age_seconds = MAX_TASK_AGE_HOURS * 3600
    min_age_seconds = MIN_TASK_AGE_MINUTES * 60
    
    # Check storage pressure
    current_usage = get_current_storage_usage()
    max_bytes = MAX_STORAGE_GB * 1024 * 1024 * 1024
    usage_ratio = current_usage / max_bytes if max_bytes > 0 else 0
    
    tasks_to_delete = []
    completed_tasks_by_age = []  # (task_id, completed_at)
    
    for task_id, task_data in translation_tasks.items():
        # Only cleanup completed or error tasks
        if task_data["status"] not in ["completed", "error", "cancelled"]:
            continue
        
        completed_at = task_data.get("completed_at", task_data.get("created_at", 0))
        task_age = current_time - completed_at
        
        # Strategy 1: Always delete old tasks
        if task_age > max_age_seconds:
            tasks_to_delete.append(task_id)
            continue
        
        # Strategy 2: Under storage pressure, delete tasks older than minimum age
        if usage_ratio > STORAGE_WARNING_THRESHOLD and task_age > min_age_seconds:
            tasks_to_delete.append(task_id)
            continue
        
        # Track for emergency cleanup
        completed_tasks_by_age.append((task_id, completed_at))
    
    # Strategy 3: Emergency cleanup - delete oldest tasks if storage critical (>90%)
    if usage_ratio > 0.9 and completed_tasks_by_age:
        completed_tasks_by_age.sort(key=lambda x: x[1])  # Sort by age, oldest first
        # Delete oldest 50% of remaining tasks
        emergency_delete_count = max(1, len(completed_tasks_by_age) // 2)
        for task_id, _ in completed_tasks_by_age[:emergency_delete_count]:
            if task_id not in tasks_to_delete:
                tasks_to_delete.append(task_id)
                logger.warning(f"Emergency cleanup: deleting task {task_id} due to storage pressure")
    
    # Delete tasks
    deleted_count = 0
    for task_id in tasks_to_delete:
        try:
            await delete_task_files(task_id)
            deleted_count += 1
            logger.info(f"Auto-cleaned task: {task_id}")
        except Exception as e:
            logger.error(f"Error auto-cleaning task {task_id}: {e}")
    
    if deleted_count > 0:
        logger.info(f"Cleanup completed: {deleted_count} tasks deleted, storage usage was {usage_ratio*100:.1f}%")


async def delete_task_files(task_id: str):
    """Delete all files associated with a task."""
    # Remove from memory
    if task_id in translation_files:
        del translation_files[task_id]
    
    if task_id in translation_tasks:
        del translation_tasks[task_id]
    
    # Delete task directory
    if task_id in task_directories:
        task_dir = task_directories[task_id]
        try:
            if task_dir.exists():
                import shutil
                shutil.rmtree(task_dir)
                logger.debug(f"Deleted task directory: {task_dir}")
        except Exception as e:
            logger.error(f"Error deleting task directory {task_dir}: {e}")
        del task_directories[task_id]


async def cleanup_task_loop():
    """Background task to periodically clean up old tasks."""
    # Run initial cleanup on startup
    await asyncio.sleep(5)  # Wait for server to fully start
    try:
        await cleanup_old_tasks()
        logger.info("Initial cleanup completed")
    except Exception as e:
        logger.error(f"Error in initial cleanup: {e}")
    
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            await cleanup_old_tasks()
        except asyncio.CancelledError:
            logger.info("Cleanup task loop cancelled")
            break
        except Exception as e:
            logger.error(f"Error in cleanup task loop: {e}")


import csv


class GlossaryProcessingError(Exception):
    """Exception raised when glossary file processing fails."""
    def __init__(self, filename: str, message: str):
        self.filename = filename
        self.message = message
        super().__init__(f"Glossary file '{filename}': {message}")


class GlossaryProcessingResult:
    """Result of processing glossary files."""
    def __init__(self):
        self.glossary_paths: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def glossaries_str(self) -> str | None:
        return ",".join(self.glossary_paths) if self.glossary_paths else None


def validate_glossary_csv(content: str, filename: str) -> tuple[bool, str | None]:
    """Validate that the CSV content has required columns.
    
    Args:
        content: CSV file content as string
        filename: Original filename for error messages
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try to parse CSV
        reader = csv.DictReader(io.StringIO(content))
        fieldnames = reader.fieldnames
        
        if not fieldnames:
            return False, "CSV file is empty or has no header row"
        
        # Check for required columns
        required_columns = {"source", "target"}
        available_columns = {col.lower().strip() for col in fieldnames}
        
        missing_columns = required_columns - available_columns
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}. CSV must have 'source' and 'target' columns"
        
        # Check if there's at least one data row
        first_row = next(reader, None)
        if first_row is None:
            return False, "CSV file has header but no data rows"
        
        return True, None
        
    except csv.Error as e:
        return False, f"Invalid CSV format: {e}"


def process_glossary_files(glossary_files: list[UploadFile] | None, task_dir: Path) -> GlossaryProcessingResult:
    """Process uploaded glossary files and save them to the task directory.
    
    Args:
        glossary_files: List of uploaded glossary CSV files
        task_dir: Task directory to save processed files
        
    Returns:
        GlossaryProcessingResult containing paths, warnings, and errors
    """
    result = GlossaryProcessingResult()
    
    if not glossary_files:
        return result
    
    for idx, file in enumerate(glossary_files):
        filename = file.filename or f"unknown_{idx}"
        
        # Check if file is empty
        if file.size == 0:
            result.warnings.append(f"'{filename}': File is empty, skipped")
            continue
        
        # Check file extension
        if not filename.lower().endswith(".csv"):
            result.errors.append(f"'{filename}': Only CSV files are supported. Please upload a .csv file")
            continue
            
        try:
            # Read file content
            content = file.file.read()
            file.file.seek(0)  # Reset file pointer
            
            if not content:
                result.warnings.append(f"'{filename}': File content is empty, skipped")
                continue
                
            # Detect encoding and decode
            detected = chardet.detect(content)
            encoding = detected.get("encoding", "utf-8") or "utf-8"
            
            try:
                text_content = content.decode(encoding)
            except UnicodeDecodeError as e:
                result.errors.append(f"'{filename}': Failed to decode file with detected encoding '{encoding}': {e}")
                continue
            
            # Validate CSV format
            is_valid, error_msg = validate_glossary_csv(text_content, filename)
            if not is_valid:
                result.errors.append(f"'{filename}': {error_msg}")
                continue
            
            # Save to task directory with unique name
            glossary_filename = f"glossary_{idx}_{filename}"
            glossary_path = task_dir / glossary_filename
            
            with open(glossary_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            
            result.glossary_paths.append(str(glossary_path))
            logger.debug(f"Processed glossary file: {filename} -> {glossary_path}")
            
        except IOError as e:
            result.errors.append(f"'{filename}': Failed to read file: {e}")
            continue
    
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI app."""
    # Startup
    logger.info("HTTP API server starting...")

    # Disable noisy loggers
    for logger_name in ["httpx", "openai", "httpcore", "http11"]:
        logging.getLogger(logger_name).setLevel("CRITICAL")
        logging.getLogger(logger_name).propagate = False

    # Start background cleanup task
    cleanup_task = asyncio.create_task(cleanup_task_loop())

    yield

    # Shutdown
    logger.info("HTTP API server shutting down...")
    
    # Cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    # Cleanup all task directories
    import shutil
    if TEMP_DIR_BASE.exists():
        try:
            shutil.rmtree(TEMP_DIR_BASE)
            logger.info(f"Cleaned up temp directory: {TEMP_DIR_BASE}")
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {e}")


app = FastAPI(
    title="PDF Translation API",
    description="API for translating PDF documents with support for multiple translation engines",
    version="2.6.4",
    lifespan=lifespan,
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


def create_settings_from_request(
    request: TranslationRequest, input_file: Path, output_dir: Path,
    glossaries: str | None = None
) -> Any:
    """Create settings from translation request.
    
    All translations use OpenAI Compatible API.
    Configuration is loaded from environment variables:
    - OPENAI_API_BASE: API endpoint (base URL)
    - OPENAI_API_KEY: API key
    
    Args:
        request: Translation request parameters
        input_file: Path to input PDF file
        output_dir: Directory for output files
        glossaries: Comma-separated list of glossary file paths
    """
    import os
    from pdf2zh_next.config import (
        BasicSettings,
        OpenAICompatibleSettings,
        PDFSettings,
        SettingsModel,
        TranslationSettings,
    )

    # Validate model
    if not is_model_supported(request.model):
        raise ValueError(
            f"Unsupported model: {request.model}. "
            f"Use GET /v1/models to see all available models."
        )

    # Read configuration from environment variables
    base_url = os.environ.get("OPENAI_API_BASE")
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not base_url:
        raise ValueError(
            "OPENAI_API_BASE environment variable is required. "
            "Please set it to your OpenAI-compatible API endpoint."
        )
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Please set it to your API key."
        )

    logger.info(f"Using model: {request.model}")

    # Create basic settings
    basic = BasicSettings(input_files={str(input_file)})

    # Create translation settings with output directory
    translation = TranslationSettings(
        lang_in=request.lang_in,
        lang_out=request.lang_out,
        min_text_length=request.min_text_length,
        qps=request.qps,
        ignore_cache=request.ignore_cache,
        custom_system_prompt=request.custom_system_prompt,
        pool_max_workers=request.pool_max_workers,
        no_auto_extract_glossary=request.no_auto_extract_glossary,
        save_auto_extracted_glossary=request.save_auto_extracted_glossary,
        primary_font_family=request.primary_font_family,
        rpc_doclayout=request.rpc_doclayout,
        output=str(output_dir),  # Set output directory
        glossaries=glossaries,  # Pass glossary file paths
    )

    # Create PDF settings
    pdf = PDFSettings(
        pages=request.pages,
        no_dual=request.no_dual,
        no_mono=request.no_mono,
        split_short_lines=request.split_short_lines,
        short_line_split_factor=request.short_line_split_factor,
        skip_clean=request.skip_clean,
        dual_translate_first=request.dual_translate_first,
        disable_rich_text_translate=request.disable_rich_text_translate,
        enhance_compatibility=request.enhance_compatibility,
        use_alternating_pages_dual=request.use_alternating_pages_dual,
        watermark_output_mode=request.watermark_output_mode,
        max_pages_per_part=request.max_pages_per_part,
        translate_table_text=request.translate_table_text,
        formular_font_pattern=request.formular_font_pattern,
        formular_char_pattern=request.formular_char_pattern,
        skip_scanned_detection=request.skip_scanned_detection,
        ocr_workaround=request.ocr_workaround,
        auto_enable_ocr_workaround=request.auto_enable_ocr_workaround,
        only_include_translated_page=request.only_include_translated_page,
        no_merge_alternating_line_numbers=not request.merge_alternating_line_numbers,
        no_remove_non_formula_lines=not request.remove_non_formula_lines,
        non_formula_line_iou_threshold=request.non_formula_line_iou_threshold,
        figure_table_protection_threshold=request.figure_table_protection_threshold,
    )

    # Create OpenAI Compatible settings for AIHubMix
    translate_engine_settings = OpenAICompatibleSettings(
        openai_compatible_model=request.model,
        openai_compatible_base_url=base_url,
        openai_compatible_api_key=api_key,
    )

    # Create settings model
    settings = SettingsModel(
        basic=basic,
        translation=translation,
        pdf=pdf,
        translate_engine_settings=translate_engine_settings,
    )

    return settings


async def process_translation_task(task_id: str, settings: Any, input_file: Path):
    """Process translation task in background."""
    try:
        translation_tasks[task_id]["status"] = "processing"
        translation_tasks[task_id]["started_at"] = time.time()

        async for event in do_translate_async_stream(settings, input_file):
            event_type = event.get("type", "unknown")
            
            if event_type in ("progress_start", "progress_update", "progress_end"):
                # Extract progress information from the event
                stage = event.get("stage", "Processing")
                overall_progress = event.get("overall_progress", 0)
                stage_current = event.get("stage_current", 0)
                stage_total = event.get("stage_total", 0)
                
                translation_tasks[task_id]["progress"] = {
                    "percentage": round(overall_progress, 2),
                    "message": f"{stage}: {stage_current}/{stage_total}",
                }
                
            elif event_type == "finish":
                translation_tasks[task_id]["status"] = "completed"
                translation_tasks[task_id]["completed_at"] = time.time()

                result = event["translate_result"]

                # Store file paths
                files = {}
                if result.mono_pdf_path:
                    files["mono"] = Path(result.mono_pdf_path)
                if result.dual_pdf_path:
                    files["dual"] = Path(result.dual_pdf_path)

                translation_files[task_id] = files
                translation_tasks[task_id]["result"] = {
                    "mono_file": str(result.mono_pdf_path) if result.mono_pdf_path else None,
                    "dual_file": str(result.dual_pdf_path) if result.dual_pdf_path else None,
                    "time_cost": result.total_seconds,
                }
                
                # Set progress to 100% when finished
                translation_tasks[task_id]["progress"] = {
                    "percentage": 100.0,
                    "message": "Translation completed",
                }

                logger.info(f"Translation task {task_id} completed successfully")
                break
            elif event["type"] == "error":
                raise RuntimeError(event.get("error", "Unknown error"))

    except asyncio.CancelledError:
        translation_tasks[task_id]["status"] = "cancelled"
        translation_tasks[task_id]["error"] = "Task was cancelled"
        logger.info(f"Translation task {task_id} was cancelled")
        raise
    except Exception as e:
        translation_tasks[task_id]["status"] = "error"
        translation_tasks[task_id]["error"] = str(e)
        translation_tasks[task_id]["completed_at"] = time.time()
        logger.error(f"Translation task {task_id} failed: {e}")


@app.post("/v1/translate", response_model=dict)
async def create_translation(
    file: UploadFile = File(...),
    glossary_files: list[UploadFile] | None = File(default=None, description="Glossary CSV files for term translation"),
    model: str = Form(DEFAULT_MODEL),
    lang_in: str = Form("en"),
    lang_out: str = Form("zh"),
    pages: str | None = Form(None),
    no_dual: bool = Form(False),
    no_mono: bool = Form(False),
    # PDF processing options
    split_short_lines: bool = Form(False),
    short_line_split_factor: float = Form(0.8),
    skip_clean: bool = Form(False),
    dual_translate_first: bool = Form(False),
    disable_rich_text_translate: bool = Form(False),
    enhance_compatibility: bool = Form(False),
    use_alternating_pages_dual: bool = Form(False),
    watermark_output_mode: str = Form("no_watermark"),
    max_pages_per_part: int | None = Form(None),
    translate_table_text: bool = Form(True),
    formular_font_pattern: str | None = Form(None),
    formular_char_pattern: str | None = Form(None),
    skip_scanned_detection: bool = Form(False),
    ocr_workaround: bool = Form(False),
    auto_enable_ocr_workaround: bool = Form(False),
    only_include_translated_page: bool = Form(False),
    merge_alternating_line_numbers: bool = Form(True),
    remove_non_formula_lines: bool = Form(True),
    non_formula_line_iou_threshold: float = Form(0.9),
    figure_table_protection_threshold: float = Form(0.9),
    primary_font_family: str | None = Form(None),
    # Translation service options
    min_text_length: int = Form(5),
    qps: int = Form(4),
    ignore_cache: bool = Form(False),
    custom_system_prompt: str | None = Form(None),
    pool_max_workers: int | None = Form(None),
    # Glossary options
    no_auto_extract_glossary: bool = Form(True),
    save_auto_extracted_glossary: bool = Form(False),
    # Advanced options
    rpc_doclayout: str | None = Form(None),
):
    """
    Create a new translation task.

    All translations use OpenAI Compatible API.
    Configuration must be set via environment variables:
    - OPENAI_API_BASE: OpenAI-compatible API endpoint (base URL)
    - OPENAI_API_KEY: API key

    Args:
        file: PDF file to translate
        glossary_files: Optional list of glossary CSV files for term translation.
            Each CSV file should contain term pairs (source term, target term).
        model: Model name (use GET /v1/models to see available models)
        lang_in: Source language code
        lang_out: Target language code
        ... (other PDF processing options)

    Returns:
        Task information with task_id for status checking
    """
    # Validate file
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Check storage availability
        content = await file.read()
        file_size = len(content)
        
        is_available, error_msg = check_storage_available(file_size * 3)  # Estimate 3x size for processing
        if not is_available:
            raise HTTPException(status_code=507, detail=error_msg)
        
        # Generate task ID
        task_id = str(uuid.uuid4())

        # Create task directory
        task_dir = get_task_directory(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        task_directories[task_id] = task_dir

        # Save uploaded file to task directory
        file_hash = hashlib.md5(content).hexdigest()[:8]
        input_file = task_dir / f"input_{file_hash}_{file.filename}"

        with open(input_file, "wb") as f:
            f.write(content)

        # Create translation request
        request = TranslationRequest(
            model=model,
            lang_in=lang_in,
            lang_out=lang_out,
            pages=pages,
            no_dual=no_dual,
            no_mono=no_mono,
            # PDF processing options
            split_short_lines=split_short_lines,
            short_line_split_factor=short_line_split_factor,
            skip_clean=skip_clean,
            dual_translate_first=dual_translate_first,
            disable_rich_text_translate=disable_rich_text_translate,
            enhance_compatibility=enhance_compatibility,
            use_alternating_pages_dual=use_alternating_pages_dual,
            watermark_output_mode=watermark_output_mode,
            max_pages_per_part=max_pages_per_part,
            translate_table_text=translate_table_text,
            formular_font_pattern=formular_font_pattern,
            formular_char_pattern=formular_char_pattern,
            skip_scanned_detection=skip_scanned_detection,
            ocr_workaround=ocr_workaround,
            auto_enable_ocr_workaround=auto_enable_ocr_workaround,
            only_include_translated_page=only_include_translated_page,
            merge_alternating_line_numbers=merge_alternating_line_numbers,
            remove_non_formula_lines=remove_non_formula_lines,
            non_formula_line_iou_threshold=non_formula_line_iou_threshold,
            figure_table_protection_threshold=figure_table_protection_threshold,
            primary_font_family=primary_font_family,
            # Translation service options
            min_text_length=min_text_length,
            qps=qps,
            ignore_cache=ignore_cache,
            custom_system_prompt=custom_system_prompt,
            pool_max_workers=pool_max_workers,
            # Glossary options
            no_auto_extract_glossary=no_auto_extract_glossary,
            save_auto_extracted_glossary=save_auto_extracted_glossary,
            # Advanced options
            rpc_doclayout=rpc_doclayout,
        )

        # Process glossary files
        glossary_result = process_glossary_files(glossary_files, task_dir)
        
        # Check for glossary processing errors
        if glossary_result.has_errors:
            # Clean up task directory on error
            import shutil
            shutil.rmtree(task_dir, ignore_errors=True)
            if task_id in task_directories:
                del task_directories[task_id]
            
            error_detail = {
                "message": "Failed to process glossary files",
                "errors": glossary_result.errors,
                "warnings": glossary_result.warnings if glossary_result.warnings else None,
            }
            raise HTTPException(status_code=400, detail=error_detail)
        
        # Log warnings if any
        if glossary_result.warnings:
            for warning in glossary_result.warnings:
                logger.warning(f"Task {task_id} glossary warning: {warning}")
        
        glossaries = glossary_result.glossaries_str
        if glossaries:
            logger.info(f"Task {task_id}: Loaded glossary files: {glossaries}")

        # Create settings with output directory and glossaries
        settings = create_settings_from_request(request, input_file, task_dir, glossaries)

        # Initialize task
        translation_tasks[task_id] = {
            "status": "queued",
            "created_at": time.time(),
            "filename": file.filename,
            "progress": None,
            "error": None,
            "glossary_warnings": glossary_result.warnings if glossary_result.warnings else None,
        }

        # Start translation task in background
        asyncio.create_task(process_translation_task(task_id, settings, input_file))

        logger.info(f"Created translation task {task_id} for file {file.filename}")

        response: dict[str, Any] = {"task_id": task_id, "status": "queued"}
        if glossary_result.warnings:
            response["glossary_warnings"] = glossary_result.warnings
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating translation task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/translate/{task_id}", response_model=TranslationStatusResponse)
async def get_translation_status(task_id: str):
    """
    Get the status of a translation task.
    """
    if task_id not in translation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = translation_tasks[task_id]

    response = TranslationStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress"),
        error=task.get("error"),
        mono_file=task.get("result", {}).get("mono_file") if task.get("result") else None,
        dual_file=task.get("result", {}).get("dual_file") if task.get("result") else None,
    )

    return response


@app.get("/v1/translate/{task_id}/stream")
async def stream_translation_progress(task_id: str):
    """
    Stream translation progress using Server-Sent Events (SSE).
    """
    if task_id not in translation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        """Generate SSE events for translation progress."""
        import json
        
        last_progress = None

        try:
            while True:
                if task_id not in translation_tasks:
                    yield {"data": json.dumps({"status": "error", "error": "Task not found", "progress": 0})}
                    break

                task = translation_tasks[task_id]
                current_status = task["status"]
                progress_data = task.get("progress") or {}

                # Build response data
                response_data = {
                    "status": current_status,
                    "progress": progress_data.get("percentage", 0),
                    "message": progress_data.get("message", ""),
                }
                
                if current_status == "error":
                    response_data["error"] = task.get("error", "Unknown error")
                elif current_status == "completed":
                    result = task.get("result") or {}
                    response_data["mono_file"] = result.get("mono_file")
                    response_data["dual_file"] = result.get("dual_file")
                    response_data["time_cost"] = result.get("time_cost")

                # Send update if progress changed or task finished
                if progress_data != last_progress or current_status in ["completed", "error", "cancelled"]:
                    last_progress = progress_data
                    yield {"data": json.dumps(response_data)}

                # Check if task is finished
                if current_status in ["completed", "error", "cancelled"]:
                    break

                await asyncio.sleep(0.5)
                
        except asyncio.CancelledError:
            logger.info(f"SSE stream for task {task_id} was cancelled by client")
        except Exception as e:
            logger.error(f"Error in SSE stream for task {task_id}: {e}")
            yield {"data": json.dumps({"status": "error", "error": str(e), "progress": 0})}

    return EventSourceResponse(event_generator())


@app.get("/v1/translate/{task_id}/mono")
async def download_mono_file(task_id: str):
    """
    Download the monolingual translated PDF file.
    """
    if task_id not in translation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = translation_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400, detail=f"Task is not completed (status: {task['status']})"
        )

    if task_id not in translation_files or "mono" not in translation_files[task_id]:
        raise HTTPException(status_code=404, detail="Monolingual file not found")

    file_path = translation_files[task_id]["mono"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=f"{task['filename'].rsplit('.', 1)[0]}_mono.pdf",
    )


@app.get("/v1/translate/{task_id}/dual")
async def download_dual_file(task_id: str):
    """
    Download the bilingual translated PDF file.
    """
    if task_id not in translation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = translation_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400, detail=f"Task is not completed (status: {task['status']})"
        )

    if task_id not in translation_files or "dual" not in translation_files[task_id]:
        raise HTTPException(status_code=404, detail="Bilingual file not found")

    file_path = translation_files[task_id]["dual"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=f"{task['filename'].rsplit('.', 1)[0]}_dual.pdf",
    )


@app.delete("/v1/translate/{task_id}")
async def delete_translation_task(task_id: str):
    """
    Cancel and delete a translation task.
    """
    if task_id not in translation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    await delete_task_files(task_id)
    
    logger.info(f"Deleted translation task {task_id}")

    return {"message": "Task deleted successfully"}


@app.get("/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "pdf2zh-next-api"}


@app.get("/v1/models")
async def list_models():
    """
    Get list of supported translation models.
    
    Returns all available models that can be used for translation.
    Models are grouped by provider (OpenAI, Anthropic, Google, etc.).
    
    Response includes:
    - id: Model identifier to use in translation requests
    - name: Human-readable model name
    - provider: Model provider (OpenAI, Anthropic, Google, DeepSeek, Alibaba)
    - description: Brief description of the model
    - context_length: Maximum context length in tokens
    """
    return {
        "models": SUPPORTED_MODELS,
        "total": len(SUPPORTED_MODELS),
        "default_model": DEFAULT_MODEL,
    }


@app.get("/v1/tasks")
async def list_tasks():
    """List all translation tasks."""
    tasks = []
    for task_id, task_data in translation_tasks.items():
        tasks.append(
            {
                "task_id": task_id,
                "status": task_data["status"],
                "filename": task_data.get("filename"),
                "created_at": task_data.get("created_at"),
                "started_at": task_data.get("started_at"),
                "completed_at": task_data.get("completed_at"),
            }
        )
    return {"tasks": tasks, "total": len(tasks)}


def run_server(host: str = "0.0.0.0", port: int = 11008, reload: bool = False):
    """Run the HTTP API server."""
    import uvicorn
    
    uvicorn.run(
        "pdf2zh_next.http_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True,
        log_config=None, # let uvicorn use root logger (Rich)
    )


if __name__ == "__main__":
    run_server()
