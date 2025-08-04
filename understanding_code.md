

PART 1: COLUMN CATEGORIES AND EXPLANATIONS
| **Category** | **Explanation** |
|--------------|------------------|
| **File Name** | `rumiai_runner.py` — This is the **main entry point script** for executing RumiAI’s full video analysis pipeline via CLI. |
| **Directory** | `scripts/` — This file lives in the `scripts/` folder at the root level. It is intended to be run directly using `python scripts/rumiai_runner.py`. |
| **Description** | The orchestrator that runs the **entire RumiAI v2 pipeline** for TikTok video analysis. It downloads the video, runs all ML services, compiles a timeline, generates Claude prompts (v2), validates the structured outputs, and creates a PDF summary. |
| **Data In** | 1. TikTok video URL (from CLI)<br>2. Env variables: `USE_ML_PRECOMPUTE`, `USE_CLAUDE_SONNET`, `OUTPUT_FORMAT_VERSION`<br>3. Internal config settings for timeouts, prompt delays, model version, etc. |
| **Data Out** | 1. Claude prompt outputs → `/insights/[video_id]/[prompt_name]/` (.txt + .json)<br>2. Final analysis report → `/analysis_results/[video_id]_FullCreativeAnalysis.pdf`<br>3. Unified and temporal analysis files<br>4. Logs, metrics, and system info |
| **Output Size (est)** | Typically 10–80 MB per run, depending on video length, number of prompts, and size of the PDF report. |
| **Called By** | This file is called **directly by the CLI user**, usually via terminal command. May also be wrapped by another Python or bash script in bulk processing mode. |
| **How Often** | One run per video; called **manually** for single videos or **batched** in loops. Can be scaled via CLI automation. |
| **Risk** | **High** — this file governs the end-to-end flow. Failure anywhere (e.g. Claude timeout, ML output missing) can block report generation. Sensitive to env vars and format flags. |
| **Dep. Services** | 1. `video_downloader.py` (Apify client)<br>2. `metadata_extractor.py`<br>3. `ml_precompute.py`<br>4. `insight_engine.py` (timeline + markers)<br>5. `prompt_router.py` (v1 vs v2)<br>6. `pdf_generator.py` |
| **Dep. 3rd Party** | 1. **Claude API** (via `anthropic` SDK)<br>2. **YOLO (ultralytics)** for object detection<br>3. **Whisper** (speech transcription)<br>4. **MediaPipe** (pose/gesture)<br>5. **EasyOCR** (text overlays)<br>6. **OpenCV**, **PyMuPDF**, **Torch**, **psutil** for support |
| **Notes** | Heavily dependent on **feature flags**: Claude model (`Sonnet` vs `Haiku`), output format (`v2` vs `v1`), and use of ML precompute. If `USE_ML_PRECOMPUTE=false`, legacy prompt path is used. |


PART 2: EXAMPLE
| **File Name**     | **Directory** | **Description** | **Data In** | **Data Out** | **Output Size (est)** | **Called By** | **How Often** | **Risk** | **Dep. Services** | **Dep. 3rd Party** | **Notes** |
|-------------------|---------------|------------------|-------------|--------------|------------------------|----------------|----------------|----------|--------------------|---------------------|-----------|
| `rumiai_runner.py` | `scripts/` | Main runtime orchestrator for RumiAI. Executes the full pipeline from TikTok video input to final Claude + PDF output. | TikTok video URL (CLI arg); env vars (`USE_CLAUDE_SONNET`, `USE_ML_PRECOMPUTE`, `OUTPUT_FORMAT_VERSION`); config flags | `/insights/[video_id]/[v2 .txt files]`; `/analysis_results/[video_id]_FullCreativeAnalysis.pdf`; temp logs; intermediate outputs | 10–80 MB per run depending on video length, metadata density, and prompt richness | CLI directly (user runs: `python rumiai_runner.py [video_url]`) | Real-time, per-video; manually triggered or batched in CLI script | **High** — failure disables pipeline; full system outage | `video_downloader.py`, `metadata_extractor.py`, `ml_precompute.py`, `insight_engine.py`, `prompt_router.py`, `pdf_generator.py` | YOLO, Whisper, MediaPipe, EasyOCR, Claude API (Sonnet), PyMuPDF, OpenCV | Controlled via `.env`; `USE_ML_PRECOMPUTE` toggles Claude v2 pipeline; highly sensitive to prompt structure and memory limits |
