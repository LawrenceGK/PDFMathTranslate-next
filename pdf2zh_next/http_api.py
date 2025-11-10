"""HTTP API for PDF translation service using FastAPI."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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

logger = logging.getLogger(__name__)

# In-memory storage for translation tasks
translation_tasks: dict[str, dict[str, Any]] = {}
# Store output files
translation_files: dict[str, dict[str, Path]] = {}
# Store temp files for cleanup
temp_files: dict[str, list[Path]] = {}


class TranslationRequest(BaseModel):
    """Request model for translation."""

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
    translate_engine_type: str = Field(
        default="google", description="Translation engine type"
    )
    service: str | None = Field(
        default=None, description="Translation service (alias for translate_engine_type)"
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
        default="watermarked", 
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
        default=False, description="Disable automatic term extraction"
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI app."""
    # Startup
    logger.info("HTTP API server starting...")

    # Disable noisy loggers
    for logger_name in ["httpx", "openai", "httpcore", "http11"]:
        logging.getLogger(logger_name).setLevel("CRITICAL")
        logging.getLogger(logger_name).propagate = False

    yield

    # Shutdown
    logger.info("HTTP API server shutting down...")
    # Cleanup temp files
    for task_id, files in temp_files.items():
        for file in files:
            try:
                if file.exists():
                    file.unlink()
            except Exception as e:
                logger.error(f"Error cleaning up temp file {file}: {e}")


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
    request: TranslationRequest, input_file: Path
) -> Any:
    """Create settings from translation request."""
    from pdf2zh_next.config import (
        AzureSettings,
        BasicSettings,
        DeepLSettings,
        GoogleSettings,
        OllamaSettings,
        OpenAISettings,
        PDFSettings,
        SettingsModel,
        TencentSettings,
        TranslationSettings,
        XinferenceSettings,
    )

    # Use service field if provided, otherwise use translate_engine_type
    engine_type = request.service or request.translate_engine_type

    # Create basic settings
    basic = BasicSettings(input_files={str(input_file)})

    # Create translation settings
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

    # Create engine-specific settings
    translate_engine_settings = None
    engine_type_lower = engine_type.lower()

    # Map lowercase engine types to proper case format
    engine_type_map = {
        "google": "Google",
        "deepl": "DeepL",
        "openai": "OpenAI",
        "azure": "Azure",
        "azureopenai": "AzureOpenAI",
        "ollama": "Ollama",
        "tencent": "TencentMechineTranslation",
        "tencentmechinetranslation": "TencentMechineTranslation",
        "xinference": "Xinference",
        "bing": "Bing",
        "deepseek": "DeepSeek",
        "modelscope": "ModelScope",
        "zhipu": "Zhipu",
        "siliconflow": "SiliconFlow",
        "siliconflowfree": "SiliconFlowFree",
        "gemini": "Gemini",
        "anythingllm": "AnythingLLM",
        "dify": "Dify",
        "grok": "Grok",
        "groq": "Groq",
        "qwenmt": "QwenMt",
        "openaicompatible": "OpenAICompatible",
    }
    
    proper_engine_type = engine_type_map.get(engine_type_lower, "Google")

    logger.info(f"Creating translation settings with engine type: {proper_engine_type} (from input: {engine_type})")

    if engine_type_lower == "google":
        translate_engine_settings = GoogleSettings()
    elif engine_type_lower == "openai":
        # OpenAI requires model and api_key
        if not request.openai_model:
            request.openai_model = "gpt-4o-mini"
        translate_engine_settings = OpenAISettings(
            openai_model=request.openai_model,
            openai_base_url=request.openai_base_url,
            openai_api_key=request.openai_api_key,
        )
    elif engine_type_lower == "azure":
        translate_engine_settings = AzureSettings(
            azure_api_key=request.azure_api_key,
            azure_endpoint=request.azure_endpoint,
        )
    elif engine_type_lower == "deepl":
        translate_engine_settings = DeepLSettings(
            deepl_auth_key=request.deepl_auth_key,
        )
    elif engine_type_lower == "ollama":
        # Ollama requires model
        if not request.ollama_model:
            request.ollama_model = "gemma2"
        translate_engine_settings = OllamaSettings(
            ollama_model=request.ollama_model,
            ollama_host=request.ollama_host,
        )
    elif engine_type_lower == "tencent" or engine_type_lower == "tencentmechinetranslation":
        translate_engine_settings = TencentSettings(
            tencentcloud_secret_id=request.tencentcloud_secret_id,
            tencentcloud_secret_key=request.tencentcloud_secret_key,
        )
    elif engine_type_lower == "xinference":
        # Xinference requires model
        if not request.xinference_model:
            request.xinference_model = "gemma-2-it"
        translate_engine_settings = XinferenceSettings(
            xinference_model=request.xinference_model,
            xinference_host=request.xinference_host,
        )
    else:
        # Default to Google if unknown engine type
        translate_engine_settings = GoogleSettings()

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
    lang_in: str = Form("en"),
    lang_out: str = Form("zh"),
    pages: str | None = Form(None),
    no_dual: bool = Form(False),
    no_mono: bool = Form(False),
    translate_engine_type: str = Form("google"),
    service: str | None = Form(None),
    # PDF processing options
    split_short_lines: bool = Form(False),
    short_line_split_factor: float = Form(0.8),
    skip_clean: bool = Form(False),
    dual_translate_first: bool = Form(False),
    disable_rich_text_translate: bool = Form(False),
    enhance_compatibility: bool = Form(False),
    use_alternating_pages_dual: bool = Form(False),
    watermark_output_mode: str = Form("watermarked"),
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
    no_auto_extract_glossary: bool = Form(False),
    save_auto_extracted_glossary: bool = Form(False),
    # Advanced options
    rpc_doclayout: str | None = Form(None),
    # Engine-specific parameters
    openai_model: str | None = Form(None),
    openai_base_url: str | None = Form(None),
    openai_api_key: str | None = Form(None),
    azure_api_key: str | None = Form(None),
    azure_endpoint: str | None = Form(None),
    deepl_auth_key: str | None = Form(None),
    ollama_model: str | None = Form(None),
    ollama_host: str | None = Form(None),
    tencentcloud_secret_id: str | None = Form(None),
    tencentcloud_secret_key: str | None = Form(None),
    xinference_model: str | None = Form(None),
    xinference_host: str | None = Form(None),
):
    """
    Create a new translation task.

    Returns a task ID that can be used to check status and download results.
    """
    # Validate file
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Generate task ID
        task_id = str(uuid.uuid4())

        # Save uploaded file to temp location
        temp_dir = Path(tempfile.gettempdir()) / "pdf2zh_next_api"
        temp_dir.mkdir(exist_ok=True)

        # Create unique filename using hash to avoid collisions
        content = await file.read()
        file_hash = hashlib.md5(content).hexdigest()[:8]
        input_file = temp_dir / f"{task_id}_{file_hash}_{file.filename}"

        with open(input_file, "wb") as f:
            f.write(content)

        # Track temp file for cleanup
        if task_id not in temp_files:
            temp_files[task_id] = []
        temp_files[task_id].append(input_file)

        # Create translation request
        request = TranslationRequest(
            lang_in=lang_in,
            lang_out=lang_out,
            pages=pages,
            no_dual=no_dual,
            no_mono=no_mono,
            translate_engine_type=translate_engine_type,
            service=service,
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
            # Engine-specific parameters
            openai_model=openai_model,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            azure_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            deepl_auth_key=deepl_auth_key,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            tencentcloud_secret_id=tencentcloud_secret_id,
            tencentcloud_secret_key=tencentcloud_secret_key,
            xinference_model=xinference_model,
            xinference_host=xinference_host,
        )

        # Create settings
        settings = create_settings_from_request(request, input_file)

        # Initialize task
        translation_tasks[task_id] = {
            "status": "queued",
            "created_at": time.time(),
            "filename": file.filename,
            "progress": None,
            "error": None,
        }

        # Start translation task in background
        asyncio.create_task(process_translation_task(task_id, settings, input_file))

        logger.info(f"Created translation task {task_id} for file {file.filename}")

        return {"task_id": task_id, "status": "queued"}

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

    # Clean up files
    if task_id in translation_files:
        for file_path in translation_files[task_id].values():
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")
        del translation_files[task_id]

    # Clean up temp files
    if task_id in temp_files:
        for file_path in temp_files[task_id]:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting temp file {file_path}: {e}")
        del temp_files[task_id]

    # Remove task
    del translation_tasks[task_id]

    logger.info(f"Deleted translation task {task_id}")

    return {"message": "Task deleted successfully"}


@app.get("/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "pdf2zh-next-api"}


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
