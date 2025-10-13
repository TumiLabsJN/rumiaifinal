# Foundation - Technical Implementation Document

> **Generated From**: FoundationCHILD.md (complete HLD)
> **Version**: 1.0
> **Date**: 2025-10-08
> **Status**: Implementation Ready

---

## 1. Document Metadata

```yaml
TI_Document: FoundationCHILDTI2.md
Parent_HLD: FoundationCHILD.md
Foundation_HLD: FoundationCHILD.md  # Self-reference (Foundation implements itself)
Covers_HLD_Sections:
  # From FoundationCHILD.md
  - Section 1: System Goals & Success Criteria
  - Section 2: Client Architecture & Directory Structure
  - Section 2.1: Directory Structure
  - Section 2.2: Path Generation Templates
  - Section 2.2.1: Path Sanitization Rules
  - Section 3: Configuration Dimensions
  - Section 3.1: Target Types
  - Section 3.2: Analysis Modes
  - Section 3.3: Selection Strategies
  - Section 3.4: Video Count (N)
  - Section 3.5: Date Filtering
  - Section 3.6: Report Types
  - Section 3.7: Report Audience
  - Section 4: CLI Command Structure
  - Section 4.1: CLI Parameters
  - Section 4.2: Default Value Logic
  - Section 5: Configuration Schemas
  - Section 5.1: config.json Schema
  - Section 5.2: Apify Video Metadata Schema
  - Section 5.3: Checkpoint Schema
  - Section 6: Bucket Definitions
  - Section 6.1: Bucket Assignment Logic
  - Section 7: References
  - Appendix A: Glossary
Related_TI_Docs:
  - Depends_On: []  # Foundation has no upstream dependencies (it IS the foundation)
  - Feeds_Into:
      - VideoDiscoveryTI.md (Stage 1)
      - VideoProcessingTI.md (Stage 2)
      - PipelineValidationTI.md (Stage 2.4)
      - FeatureAggregationTI.md (Stage 3)
      - FeatureTransformationTI.md (Stage 4)
      - MLModelTrainingTI.md (Stage 5)
      - MLAnalysisGenerationTI.md (Stage 6)
      - LLMReportGenerationTI.md (Stage 7)
Implementation_Priority: CRITICAL  # All stages depend on Foundation
```

**Special Note**: Foundation is unique - it implements the shared infrastructure that all other stages depend on. There is no "upstream" TI for Foundation because it is the first layer.

---

## 2. Stage Contract

### 2.1 Input Contract

```python
class FoundationInput:
    """
    Foundation receives CLI arguments as input.

    Source: FoundationCHILD.md Section 4.1: CLI Parameters
    """
    # CLI Arguments (from command line)
    client: str                    # Required, --client, Alphanumeric + underscore, Example: "acme_corp"
    analysis_type: str             # Required, --analysis-type, Enum: ["hashtag", "competitor", "creator"]
    target: str                    # Required, --target, Format depends on analysis_type (#nutrition, @rival_brand)
    analysis_mode: str             # Optional, --analysis-mode, Enum: ["top", "recent"], Default: depends on type
    selection_strategy: str        # Optional, --selection-strategy, Enum: ["contrastive", "top"], Default: depends on type
    video_count: int               # Optional, --video-count, Range: 10-500, Default: depends on strategy
    date_filter: str               # Optional, --date-filter, Format: last_N_days, Default: "last_90_days"
    report_type: str               # Optional, --report-type, Enum: ["single", "comparison"], Default: "single"
    report_audience: str           # Optional, --report-audience, Enum: ["client", "internal", "creator"], Default: depends on type
    auto_confirm: bool             # Optional, --auto-confirm, Boolean flag, Default: False

    # Validation rules from FoundationCHILD.md Section 4.1: CLI Parameters
    # - client: Regex ^[a-zA-Z0-9_]+$ (min 1 char)
    # - target: Must start with # (hashtag) or @ (competitor/creator), min 2 chars
    # - video_count: Integer range 10-500 (inclusive)
    # - date_filter: Regex ^last_\d+_days$ where \d+ is 1-365
```

### 2.2 Output Contract

```python
class FoundationOutput:
    """
    Foundation produces validated config and directory structure.

    Source: FoundationCHILD.md Section 2.1: Directory Structure, Section 5.1: config.json Schema
    """
    # Configuration File Created
    config_json_path: str          # {target_dir}/config.json
    config_object: Config          # Validated Pydantic model

    # Directory Structure Created (from Section 2.1, lines 86-230)
    client_base_path: str          # /data/clients/{client_id}/
    analysis_type_path: str        # {client_base}/hashtags/ or competitors/ or creators/
    target_path: str               # {analysis_type_path}/{sanitized_target}/
    mode_strategy_path: str        # {target_path}/{mode}_{strategy}/

    # Bucket Directories Created (8 buckets)
    bucket_paths: Dict[str, str]   # {"0-3s": "/data/.../bucket_0-3s/", ...}

    # Subdirectories per bucket (from Section 2.1: Directory Structure)
    # Each bucket contains:
    #   - videos/          # Raw MP4 files
    #   - analysis/        # RumiAI JSON outputs (temporal_windows_updated.json)
    #   - ml_analysis/     # Aggregated features, transformed data
    #   - models/          # Trained ML models (RF, KM)
    #   - reports/         # Generated PDF/markdown reports
    #   - checkpoints/     # Stage checkpoints for resumption
    #   - logs/            # Stage-specific logs

    # Path Utilities Available
    path_builder: PathBuilder      # Object for generating paths
    bucket_assigner: BucketAssigner # Object for assigning videos to buckets

    # Validation Passed
    cli_args_validated: bool       # True if all CLI args passed validation
    config_saved: bool             # True if config.json saved successfully
    directories_created: bool      # True if all directories created
```

---

## 3. Data Schemas

### 3.1 Foundation Schemas

**Source**: FoundationCHILD.md Section 5: Configuration Schemas

#### Config Schema

```python
# From FoundationCHILD.md Section 5.1: config.json Schema
ConfigSchema = {
    "client_id": str,              # Required, alphanumeric + underscore, Example: "acme_corp"
    "analysis_type": str,          # Required, ["hashtag", "competitor", "creator"], Example: "hashtag"
    "target": str,                 # Required, format depends on analysis_type, Example: "#nutrition"
    "analysis_mode": str,          # Required, ["top", "recent"], Example: "top"
    "selection_strategy": str,     # Required, ["contrastive", "top"], Example: "contrastive"
    "video_count": int,            # Required, Range: 10-500, Example: 100
    "date_filter": str,            # Required, "last_N_days", Example: "last_90_days"
    "report_type": str,            # Required, ["single", "comparison"], Example: "single"
    "report_audience": str,        # Required, ["client", "internal", "creator"], Example: "client"
    "auto_confirm": bool,          # Required, skip interactive prompts, Example: false
    "run_date": str,               # Required, ISO 8601 format, Example: "2025-01-28T10:30:00Z"
}
```

**Example** (from FoundationCHILD.md Section 5.1: config.json Schema):
```json
{
  "client_id": "acme_corp",
  "analysis_type": "hashtag",
  "target": "#nutrition",
  "analysis_mode": "top",
  "selection_strategy": "contrastive",
  "video_count": 100,
  "date_filter": "last_90_days",
  "report_type": "single",
  "report_audience": "client",
  "auto_confirm": false,
  "run_date": "2025-01-28T10:30:00Z"
}
```

#### Apify Video Metadata Schema

```python
# From FoundationCHILD.md Section 5.2: Apify Video Metadata Schema
ApifyVideoMetadataSchema = {
    "id": str,                     # Required, Unique video identifier, Example: "7428596413707144481"
    "createTime": int,             # Required, Unix timestamp in UTC, Example: 1706467200
    "duration": int,               # Required, Video length in seconds, Example: 18
    "playCount": int,              # Required, View count, Example: 1500000
    "shareCount": int,             # Required, Share count, Example: 5000
    "commentCount": int,           # Required, Comment count, Example: 1200
    "likeCount": int,              # Required, Like count, Example: 85000
    "webVideoUrl": str,            # Required, TikTok web URL, Example: "https://www.tiktok.com/@user/video/..."
}
```

**Note**: Apify returns 30+ fields total. This schema documents the 8 core fields required for Stage 1 processing. Additional fields (text, hashtags, musicMeta, mentions, covers, authorMeta) are available but optional. See VideoDiscoveryCHILD.md Section 5.2 for complete field list.

#### Checkpoint Schema

```python
# From FoundationCHILD.md Section 5.3: Checkpoint Schema
CheckpointSchema = {
    "stage": str,                  # Required, Stage name, Example: "video_processing"
    "bucket": str,                 # Required, Bucket name, Example: "18-33s"
    "total_videos": int,           # Required, Total videos to process, Example: 100
    "completed": int,              # Required, Successfully processed, Example: 45
    "failed": int,                 # Required, Failed with errors, Example: 2
    "remaining": int,              # Required, Not yet processed, Example: 53
    "last_checkpoint": str,        # Required, ISO timestamp, Example: "2025-01-28T14:32:15Z"
    "completed_video_ids": list[str],   # Required, List of processed video IDs
    "failed_video_ids": list[dict],     # Required, List of failure records
        # Nested schema for failed_video_ids items:
        # {
        #   "video_id": str,        # Required, Video ID that failed
        #   "error": str,           # Required, Error message/reason
        #   "timestamp": str,       # Optional, ISO timestamp of failure
        #   "stage": str            # Optional, Substage that failed (e.g., "FEAT", "Whisper")
        # }
}
```

**Example** (from FoundationCHILD.md Section 5.3: Checkpoint Schema):
```json
{
  "stage": "video_processing",
  "bucket": "18-33s",
  "total_videos": 100,
  "completed": 45,
  "failed": 2,
  "remaining": 53,
  "last_checkpoint": "2025-01-28T14:32:15Z",
  "completed_video_ids": ["123", "124", "125"],
  "failed_video_ids": [
    {
      "video_id": "321",
      "error": "FEAT timeout after 120s",
      "timestamp": "2025-01-28T14:28:10Z",
      "stage": "FEAT"
    }
  ]
}
```

### 3.2 Bucket Definitions Schema

```python
# From FoundationCHILD.md Section 6: Bucket Definitions
BUCKET_DEFINITIONS = {
    "0-3s": (0, 3),                # Lower: 0s, Upper: 3s (exclusive)
    "3-9s": (3, 9),                # Lower: 3s, Upper: 9s (exclusive)
    "9-13s": (9, 13),              # Lower: 9s, Upper: 13s (exclusive)
    "13-18s": (13, 18),            # Lower: 13s, Upper: 18s (exclusive)
    "18-33s": (18, 33),            # Lower: 18s, Upper: 33s (exclusive)
    "33-60s": (33, 60),            # Lower: 33s, Upper: 60s (exclusive)
    "60-90s": (60, 90),            # Lower: 60s, Upper: 90s (exclusive)
    "90-120s": (90, 120),          # Lower: 90s, Upper: 120s (INCLUSIVE - exception for final bucket)
}
```

**Boundary Rules** (from Section 6.1: Bucket Assignment Logic):
- All buckets use **inclusive lower bound**: `duration >= lower_bound`
- All buckets use **exclusive upper bound**: `duration < upper_bound`
- **Exception**: Final bucket "90-120s" uses **inclusive upper bound**: `duration <= 120`

**Edge Cases** (from Section 6.1: Bucket Assignment Logic):
- Video exactly 9.0s → assigns to "9-13s" bucket (inclusive lower bound)
- Video exactly 120.0s → assigns to "90-120s" bucket (inclusive upper bound)
- Video >120s → rejected (TikTok platform maximum is 120s)
- Video <3s → assigns to "0-3s" bucket (valid but rare)

---

## 4. Algorithmic Specifications

### 4.1 CLI Argument Parsing

**Function**: `parse_args()`

**Source**: FoundationCHILD.md Section 4.1: CLI Parameters

**Purpose**: Parse command-line arguments with validation and default application

**Algorithm**:
```python
def parse_args(argv: list[str] | None = None) -> CLIArgs:
    """
    Parse CLI arguments with validation and defaults.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        CLIArgs: Validated CLI arguments with defaults applied

    Raises:
        SystemExit: If validation fails (argparse handles exit)
        ValueError: If argument combinations are invalid
    """
    # Step 1: Create argument parser
    parser = argparse.ArgumentParser(
        prog="rumiai_ml_batch.py",
        description="RumiAI ML Pipeline - Batch video analysis"
    )

    # Step 2: Add required arguments
    parser.add_argument("--client", required=True, type=str,
                       help="Client identifier (alphanumeric + underscore)")
    parser.add_argument("--analysis-type", required=True,
                       choices=["hashtag", "competitor", "creator"],
                       help="Target type to analyze")
    parser.add_argument("--target", required=True, type=str,
                       help="Target identifier (#nutrition, @rival_brand, @creator_name)")

    # Step 3: Add optional arguments with defaults
    parser.add_argument("--analysis-mode", choices=["top", "recent"], default=None)
    parser.add_argument("--selection-strategy", choices=["contrastive", "top"], default=None)
    parser.add_argument("--video-count", type=int, default=None)
    parser.add_argument("--date-filter", type=str, default="last_90_days")
    parser.add_argument("--report-type", choices=["single", "comparison"], default="single")
    parser.add_argument("--report-audience", choices=["client", "internal", "creator"], default=None)
    parser.add_argument("--auto-confirm", action="store_true", default=False)

    # Step 4: Parse arguments
    parsed = parser.parse_args(argv)

    # Step 5: Apply defaults based on analysis_type (Section 4.2: Default Value Logic)
    args_with_defaults = apply_defaults(parsed, parsed.analysis_type)

    # Step 6: Validate argument combinations
    validate_cli_args(args_with_defaults)

    # Step 7: Return as dataclass
    return CLIArgs(**vars(args_with_defaults))
```

**Edge Cases**:
- Case 1: User provides no optional args → Defaults applied based on analysis_type (Section 4.2: Default Value Logic)
- Case 2: User overrides defaults → Use user-provided values (no default application)
- Case 3: Invalid client format → Validation fails with specific error message
- Case 4: Invalid target prefix → Validation fails (hashtag must start with #, competitor/creator with @)

**Validation Rules** (from Section 4.1: CLI Parameters):
```python
# client validation
assert re.match(r"^[a-zA-Z0-9_]+$", args.client), "Client must be alphanumeric + underscore"

# target validation based on analysis_type
if args.analysis_type == "hashtag":
    assert args.target.startswith("#") and len(args.target) >= 2, \
        "Hashtag target must start with # and have at least 2 characters"
elif args.analysis_type in ["competitor", "creator"]:
    assert args.target.startswith("@") and len(args.target) >= 2, \
        f"{args.analysis_type} target must start with @ and have at least 2 characters"

# video_count validation
assert 10 <= args.video_count <= 500, "video_count must be between 10 and 500"

# date_filter validation
assert re.match(r"^last_\d+_days$", args.date_filter), \
    "date_filter must match format 'last_N_days'"
days = int(args.date_filter.split("_")[1])
assert 1 <= days <= 365, "date_filter days must be between 1 and 365"
```

**Example Input**:
```bash
python rumiai_ml_batch.py \
  --client "acme_corp" \
  --analysis-type hashtag \
  --target "#nutrition"
```

**Example Output**:
```python
CLIArgs(
    client="acme_corp",
    analysis_type="hashtag",
    target="#nutrition",
    analysis_mode="top",              # Default applied
    selection_strategy="contrastive", # Default applied
    video_count=100,                  # Default applied
    date_filter="last_90_days",       # Default from parser
    report_type="single",             # Default from parser
    report_audience="client",         # Default applied
    auto_confirm=False                # Default from parser
)
```

### 4.2 Default Value Application

**Function**: `apply_defaults()`

**Source**: FoundationCHILD.md Section 4.2: Default Value Logic

**Purpose**: Apply type-specific defaults based on analysis_type

**Algorithm**:
```python
def apply_defaults(args: argparse.Namespace, analysis_type: str) -> argparse.Namespace:
    """
    Apply default values based on analysis_type.

    Source: FoundationCHILD.md Section 4.2: Default Value Logic

    Logic from Section 4.2: Default Value Logic:
    - Hashtag: mode=top, strategy=contrastive, video_count=100, audience=client
    - Competitor: mode=top, strategy=contrastive, video_count=100, audience=client
    - Creator: mode=recent, strategy=top, video_count=40, audience=creator
    """
    # Step 1: Apply analysis_mode default
    if args.analysis_mode is None:
        if analysis_type == "creator":
            args.analysis_mode = "recent"
        else:  # hashtag or competitor
            args.analysis_mode = "top"

    # Step 2: Apply selection_strategy default
    if args.selection_strategy is None:
        if analysis_type == "creator":
            args.selection_strategy = "top"
        else:  # hashtag or competitor
            args.selection_strategy = "contrastive"

    # Step 3: Apply video_count default
    if args.video_count is None:
        if analysis_type == "creator":
            args.video_count = 40
        else:  # hashtag or competitor
            args.video_count = 100

    # Step 4: Apply report_audience default
    if args.report_audience is None:
        if analysis_type == "creator":
            args.report_audience = "creator"
        else:  # hashtag or competitor
            args.report_audience = "client"

    return args
```

**Edge Cases**:
- Case 1: User overrides default (e.g., `--video-count 50`) → Use user value (no override)
- Case 2: Invalid analysis_type → Should never reach this (caught by argparse choices)
- Case 3: All args provided by user → No defaults applied (all None checks false)

**Example Trace**:
```
Input: argparse.Namespace(
    client="acme",
    analysis_type="hashtag",
    target="#nutrition",
    analysis_mode=None,
    selection_strategy=None,
    video_count=None,
    report_audience=None
)

Step 1: analysis_mode is None, analysis_type is "hashtag" → Set analysis_mode="top"
Step 2: selection_strategy is None, analysis_type is "hashtag" → Set selection_strategy="contrastive"
Step 3: video_count is None, analysis_type is "hashtag" → Set video_count=100
Step 4: report_audience is None, analysis_type is "hashtag" → Set report_audience="client"

Output: argparse.Namespace(
    client="acme",
    analysis_type="hashtag",
    target="#nutrition",
    analysis_mode="top",
    selection_strategy="contrastive",
    video_count=100,
    report_audience="client"
)
```

### 4.3 Path Sanitization

**Function**: `sanitize_target()`

**Source**: FoundationCHILD.md Section 2.2.1: Path Sanitization Rules

**Purpose**: Sanitize target for filesystem path usage

**Algorithm**:
```python
import re

def sanitize_target(target: str, analysis_type: str) -> str:
    """
    Sanitize target for filesystem path usage.

    Rules from FoundationCHILD.md Section 2.2.1: Path Sanitization Rules:
    1. Remove prefix (# for hashtag, @ for competitor/creator)
    2. Convert to lowercase
    3. Replace spaces with underscores
    4. Remove special characters (keep only alphanumeric, underscore, hyphen)
    5. Collapse multiple underscores to single underscore
    6. Strip leading/trailing underscores

    Args:
        target: Original target with prefix
        analysis_type: "hashtag", "competitor", or "creator"

    Returns:
        Sanitized target string
    """
    # Step 1: Remove prefix
    if analysis_type == "hashtag" and target.startswith("#"):
        sanitized = target[1:]
    elif analysis_type in ["competitor", "creator"] and target.startswith("@"):
        sanitized = target[1:]
    else:
        sanitized = target

    # Step 2: Lowercase
    sanitized = sanitized.lower()

    # Step 3: Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")

    # Step 4: Remove special characters (keep alphanumeric, underscore, hyphen)
    sanitized = re.sub(r'[^a-z0-9_-]', '', sanitized)

    # Step 5: Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)

    # Step 6: Strip leading/trailing underscores
    sanitized = sanitized.strip('_')

    return sanitized
```

**Edge Cases** (from Section 2.2.1: Path Sanitization Rules):
- Case 1: `#Fitness & Nutrition!` → `fitness_nutrition`
- Case 2: `@My Brand 2024` → `my_brand_2024`
- Case 3: `#nutrition` → `nutrition`
- Case 4: `@rival__brand` → `rival_brand` (double underscore collapsed)
- Case 5: `#Weight-Loss` → `weight-loss` (hyphen preserved)
- Case 6: `@_special_user_` → `special_user` (leading/trailing underscores stripped)

**Example Trace**:
```
Input: target="#Fitness & Nutrition!", analysis_type="hashtag"

Step 1: Remove # prefix → "Fitness & Nutrition!"
Step 2: Lowercase → "fitness & nutrition!"
Step 3: Replace spaces → "fitness_&_nutrition!"
Step 4: Remove special chars → "fitness__nutrition"
Step 5: Collapse underscores → "fitness_nutrition"
Step 6: Strip leading/trailing → "fitness_nutrition"

Output: "fitness_nutrition"
```

### 4.4 Bucket Assignment

**Function**: `assign_bucket()`

**Source**: FoundationCHILD.md Section 6.1: Bucket Assignment Logic: Bucket Assignment Logic

**Purpose**: Assign video to duration bucket based on video length

**Algorithm**:
```python
def assign_bucket(duration: float) -> str:
    """
    Assign video to duration bucket based on video length.

    Boundary Behavior (Section 6.1: Bucket Assignment Logic):
    - All buckets use inclusive lower bound: duration >= lower_bound
    - All buckets use exclusive upper bound: duration < upper_bound
    - Exception: Final bucket "90-120s" uses inclusive upper bound: duration <= 120

    Args:
        duration: Video duration in seconds

    Returns:
        Bucket name (e.g., "18-33s")

    Raises:
        ValueError: If duration > 120s (exceeds TikTok maximum)
    """
    if duration < 3:
        return "0-3s"
    elif duration < 9:
        return "3-9s"
    elif duration < 13:
        return "9-13s"
    elif duration < 18:
        return "13-18s"
    elif duration < 33:
        return "18-33s"
    elif duration < 60:
        return "33-60s"
    elif duration < 90:
        return "60-90s"
    elif duration <= 120:  # Note: inclusive upper bound for final bucket
        return "90-120s"
    else:
        raise ValueError(f"Video duration {duration}s exceeds TikTok maximum (120s)")
```

**Edge Cases** (from Section 6.1: Bucket Assignment Logic):
- Case 1: duration=2.5s → "0-3s" (inclusive lower, exclusive upper)
- Case 2: duration=9.0s → "9-13s" (exactly on boundary, uses next bucket's inclusive lower)
- Case 3: duration=120.0s → "90-120s" (final bucket has inclusive upper bound)
- Case 4: duration=125.0s → ValueError (exceeds TikTok max)
- Case 5: duration=0.5s → "0-3s" (valid but rare, very short video)

**Example Trace 1** (normal case):
```
Input: duration=18.5

Step 1: Check if < 3 → False
Step 2: Check if < 9 → False
Step 3: Check if < 13 → False
Step 4: Check if < 18 → False
Step 5: Check if < 33 → True

Output: "18-33s"
```

**Example Trace 2** (edge case - exactly 120s):
```
Input: duration=120.0

Step 1-7: All False
Step 8: Check if <= 120 → True (inclusive upper bound for final bucket)

Output: "90-120s"
```

**Example Trace 3** (error case):
```
Input: duration=125.0

Step 1-8: All False
Step 9: Else clause → Raise ValueError

Error: ValueError("Video duration 125.0s exceeds TikTok maximum (120s)")
```

### 4.5 Directory Structure Creation

**Function**: `create_directory_structure()`

**Source**: FoundationCHILD.md Section 2.1: Directory Structure

**Purpose**: Create complete directory structure for client/target/mode/strategy

**Algorithm**:
```python
from pathlib import Path

def create_directory_structure(
    client_id: str,
    analysis_type: str,
    target: str,
    analysis_mode: str,
    selection_strategy: str,
    base_path: Path = Path("/data")
) -> Dict[str, Path]:
    """
    Create directory structure for target.

    Structure from FoundationCHILD.md Section 2.1: Directory Structure:
    /data/clients/{client_id}/{analysis_type}s/{sanitized_target}/{mode}_{strategy}/
        ├── config.json
        ├── buckets/
        │   ├── bucket_0-3s/
        │   │   ├── videos/
        │   │   ├── analysis/
        │   │   ├── ml_analysis/
        │   │   ├── models/
        │   │   ├── reports/
        │   │   ├── checkpoints/
        │   │   └── logs/
        │   ├── bucket_3-9s/
        │   │   └── [same subdirectories]
        │   ... (8 buckets total)

    Returns:
        Dict mapping bucket names to their paths
    """
    # Step 1: Sanitize target for filesystem
    sanitized_target = sanitize_target(target, analysis_type)

    # Step 2: Build target directory path
    analysis_type_plural = f"{analysis_type}s"  # hashtag → hashtags
    mode_strategy = f"{analysis_mode}_{selection_strategy}"

    target_dir = (
        base_path
        / "clients"
        / client_id
        / analysis_type_plural
        / sanitized_target
        / mode_strategy
    )

    # Step 3: Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Step 4: Create buckets parent directory
    buckets_dir = target_dir / "buckets"
    buckets_dir.mkdir(exist_ok=True)

    # Step 5: Create each bucket subdirectory
    bucket_paths = {}
    for bucket_name in BUCKET_DEFINITIONS.keys():
        bucket_dir = buckets_dir / f"bucket_{bucket_name}"
        bucket_dir.mkdir(exist_ok=True)

        # Step 6: Create subdirectories within each bucket
        (bucket_dir / "videos").mkdir(exist_ok=True)
        (bucket_dir / "analysis").mkdir(exist_ok=True)
        (bucket_dir / "ml_analysis").mkdir(exist_ok=True)
        (bucket_dir / "models").mkdir(exist_ok=True)
        (bucket_dir / "reports").mkdir(exist_ok=True)
        (bucket_dir / "checkpoints").mkdir(exist_ok=True)
        (bucket_dir / "logs").mkdir(exist_ok=True)

        bucket_paths[bucket_name] = bucket_dir

    # Step 7: Return bucket paths for reference
    return bucket_paths
```

**Edge Cases**:
- Case 1: Directory already exists → `mkdir(exist_ok=True)` handles (no error)
- Case 2: No write permissions → Raises PermissionError
- Case 3: Parent path doesn't exist → `mkdir(parents=True)` creates full hierarchy
- Case 4: Target contains special characters → Sanitization handles before directory creation

**Example Trace**:
```
Input:
  client_id="acme_corp"
  analysis_type="hashtag"
  target="#nutrition"
  analysis_mode="top"
  selection_strategy="contrastive"
  base_path=Path("/data")

Step 1: Sanitize target → "nutrition"
Step 2: Build path components
  - analysis_type_plural = "hashtags"
  - mode_strategy = "top_contrastive"
  - target_dir = Path("/data/clients/acme_corp/hashtags/nutrition/top_contrastive")
Step 3: Create target_dir → Directory created
Step 4: Create buckets_dir → "/data/.../buckets/" created
Step 5-6: Loop through 8 buckets:
  - Create "bucket_0-3s" with 7 subdirectories
  - Create "bucket_3-9s" with 7 subdirectories
  - ... (8 total)

Output: {
  "0-3s": Path("/data/.../buckets/bucket_0-3s"),
  "3-9s": Path("/data/.../buckets/bucket_3-9s"),
  ... (8 entries)
}

Files Created:
  - 8 bucket directories
  - 56 subdirectories (7 per bucket × 8 buckets)
  - Total: 64 directories
```

### 4.6 Configuration Persistence

**Function**: `save_config()`

**Source**: FoundationCHILD.md Section 5.1: config.json Schema

**Purpose**: Save validated configuration to config.json

**Algorithm**:
```python
import json
from datetime import datetime
from pathlib import Path

def save_config(
    cli_args: CLIArgs,
    target_dir: Path
) -> Config:
    """
    Create Config from CLI args and save to config.json.

    Args:
        cli_args: Validated CLI arguments
        target_dir: Target directory path

    Returns:
        Config object (Pydantic model)

    Raises:
        IOError: If config.json cannot be written
    """
    # Step 1: Create Config object with current timestamp
    config = Config(
        client_id=cli_args.client,
        analysis_type=cli_args.analysis_type,
        target=cli_args.target,
        analysis_mode=cli_args.analysis_mode,
        selection_strategy=cli_args.selection_strategy,
        video_count=cli_args.video_count,
        date_filter=cli_args.date_filter,
        report_type=cli_args.report_type,
        report_audience=cli_args.report_audience,
        auto_confirm=cli_args.auto_confirm,
        run_date=datetime.utcnow().isoformat() + "Z"  # ISO 8601 with UTC
    )

    # Step 2: Validate Config (Pydantic handles this)
    # Pydantic will raise ValidationError if any field is invalid

    # Step 3: Convert to dict
    config_dict = config.dict()

    # Step 4: Write to config.json
    config_path = target_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Step 5: Return Config object for use by stages
    return config
```

**Edge Cases**:
- Case 1: target_dir doesn't exist → Raises FileNotFoundError
- Case 2: config.json already exists → Overwrites (no error)
- Case 3: No write permissions → Raises PermissionError
- Case 4: Invalid config values → Pydantic ValidationError raised

**Example Trace**:
```
Input:
  cli_args=CLIArgs(
    client="acme_corp",
    analysis_type="hashtag",
    target="#nutrition",
    analysis_mode="top",
    selection_strategy="contrastive",
    video_count=100,
    date_filter="last_90_days",
    report_type="single",
    report_audience="client",
    auto_confirm=False
  )
  target_dir=Path("/data/clients/acme_corp/hashtags/nutrition/top_contrastive")

Step 1: Create Config object
  - Extract all fields from cli_args
  - Generate run_date = "2025-10-08T15:30:00Z"
Step 2: Pydantic validation → Passed (all fields valid)
Step 3: Convert to dict → config_dict with 11 fields
Step 4: Write to file
  - config_path = "/data/.../config.json"
  - Write JSON with indent=2
Step 5: Return Config object

Output: Config(
  client_id="acme_corp",
  analysis_type="hashtag",
  target="#nutrition",
  analysis_mode="top",
  selection_strategy="contrastive",
  video_count=100,
  date_filter="last_90_days",
  report_type="single",
  report_audience="client",
  auto_confirm=False,
  run_date="2025-10-08T15:30:00Z"
)

File Created: /data/clients/acme_corp/hashtags/nutrition/top_contrastive/config.json
```

---

## 5. Validation Rules

### 5.1 CLI Argument Validation

**Source**: FoundationCHILD.md Section 4.1: CLI Parameters

```python
def validate_cli_args(args: argparse.Namespace) -> None:
    """
    Validate CLI argument combinations.

    Raises:
        ValueError: With specific error message guiding user to fix issue
    """
    # Validate client_id format
    if not re.match(r"^[a-zA-Z0-9_]+$", args.client):
        raise ValueError(
            f"Invalid --client '{args.client}'. "
            f"Must contain only alphanumeric characters and underscores."
        )

    # Validate target format based on analysis_type
    if args.analysis_type == "hashtag":
        if not args.target.startswith("#") or len(args.target) < 2:
            raise ValueError(
                f"Invalid --target '{args.target}' for analysis_type 'hashtag'. "
                f"Must start with # and have at least 2 characters (e.g., #nutrition)."
            )
    elif args.analysis_type in ["competitor", "creator"]:
        if not args.target.startswith("@") or len(args.target) < 2:
            raise ValueError(
                f"Invalid --target '{args.target}' for analysis_type '{args.analysis_type}'. "
                f"Must start with @ and have at least 2 characters (e.g., @rival_brand)."
            )

    # Validate video_count range
    if not (10 <= args.video_count <= 500):
        raise ValueError(
            f"Invalid --video-count {args.video_count}. "
            f"Must be between 10 and 500 (inclusive)."
        )

    # Validate date_filter format
    if not re.match(r"^last_\d+_days$", args.date_filter):
        raise ValueError(
            f"Invalid --date-filter '{args.date_filter}'. "
            f"Must match format 'last_N_days' where N is 1-365 (e.g., last_90_days)."
        )
    days = int(args.date_filter.split("_")[1])
    if not (1 <= days <= 365):
        raise ValueError(
            f"Invalid --date-filter '{args.date_filter}'. "
            f"Days must be between 1 and 365."
        )
```

### 5.2 Configuration Schema Validation

**Source**: FoundationCHILD.md Section 5.1: config.json Schema

Validation is handled by Pydantic `Config` model:

```python
from pydantic import BaseModel, Field, validator

class Config(BaseModel):
    """Configuration schema with built-in validation."""

    client_id: str = Field(..., regex=r"^[a-zA-Z0-9_]+$")
    analysis_type: str = Field(..., regex=r"^(hashtag|competitor|creator)$")
    target: str  # Custom validator below
    analysis_mode: str = Field(..., regex=r"^(top|recent)$")
    selection_strategy: str = Field(..., regex=r"^(contrastive|top)$")
    video_count: int = Field(..., ge=10, le=500)
    date_filter: str = Field(..., regex=r"^last_\d+_days$")
    report_type: str = Field(..., regex=r"^(single|comparison)$")
    report_audience: str = Field(..., regex=r"^(client|internal|creator)$")
    auto_confirm: bool
    run_date: str  # ISO 8601 format

    @validator("target")
    def validate_target_format(cls, v, values):
        """Validate target format based on analysis_type."""
        analysis_type = values.get("analysis_type")
        if analysis_type == "hashtag":
            if not v.startswith("#") or len(v) < 2:
                raise ValueError("Hashtag target must start with # and have at least 2 characters")
        elif analysis_type in ["competitor", "creator"]:
            if not v.startswith("@") or len(v) < 2:
                raise ValueError(f"{analysis_type.capitalize()} target must start with @ and have at least 2 characters")
        return v

    @validator("date_filter")
    def validate_date_filter_range(cls, v):
        """Validate date filter days are in range 1-365."""
        days = int(v.split("_")[1])
        if not (1 <= days <= 365):
            raise ValueError(f"Date filter days must be between 1 and 365, got {days}")
        return v

    class Config:
        frozen = True  # Immutable after creation
```

### 5.3 Minimum N Recommendations

**Source**: FoundationCHILD.md Section 3.4: Video Count (N)

```python
def check_minimum_n_recommendations(video_count: int, selection_strategy: str) -> None:
    """
    Check if video_count meets recommended minimums.

    Warning thresholds from Section 3.4: Video Count (N):
    - Contrastive: N < 50 → Warn (bottom 20% = only 10 videos)
    - Top: N < 20 → Warn (insufficient for 3-cluster K-Means)

    Raises warning but does NOT fail (allows N=10-49 with warning).
    """
    if selection_strategy == "contrastive" and video_count < 50:
        bottom_count = int(video_count * 0.2)
        logger.warning(
            f"Low bottom performer count ({bottom_count} videos, {video_count} × 0.2). "
            f"Recommend N ≥ 50 for robust classification. "
            f"Statistical validity may be limited."
        )
    elif selection_strategy == "top" and video_count < 20:
        logger.warning(
            f"Low sample size for clustering ({video_count} videos). "
            f"Recommend N ≥ 20 for pattern detection. "
            f"K-Means may produce unstable clusters."
        )

    # Absolute minimum check (hard limit)
    if video_count < 10:
        raise ValueError(
            f"Insufficient sample size. Minimum N=10 required, got {video_count}."
        )
```

---

## 6. Error Handling

**Source**: Not explicitly documented in FoundationCHILD.md (Foundation is infrastructure, errors handled at usage sites)

Foundation provides error types that stages will catch:

```python
ERROR_CONDITIONS = {
    "invalid_client_id": {
        "condition": "Client ID contains special characters or spaces",
        "error_type": "ValueError",
        "action": "Fail-fast (raise ValueError)",
        "retry_policy": "No retry",
        "user_message": "Invalid --client '{client_id}'. Must contain only alphanumeric characters and underscores."
    },

    "invalid_target_prefix": {
        "condition": "Target missing required prefix (# or @)",
        "error_type": "ValueError",
        "action": "Fail-fast (raise ValueError)",
        "retry_policy": "No retry",
        "user_message": "Invalid --target '{target}' for analysis_type '{analysis_type}'. Must start with {required_prefix}."
    },

    "video_count_out_of_range": {
        "condition": "video_count < 10 or video_count > 500",
        "error_type": "ValueError",
        "action": "Fail-fast (raise ValueError)",
        "retry_policy": "No retry",
        "user_message": "Invalid --video-count {video_count}. Must be between 10 and 500 (inclusive)."
    },

    "date_filter_invalid_format": {
        "condition": "date_filter doesn't match regex ^last_\\d+_days$",
        "error_type": "ValueError",
        "action": "Fail-fast (raise ValueError)",
        "retry_policy": "No retry",
        "user_message": "Invalid --date-filter '{date_filter}'. Must match format 'last_N_days' (e.g., last_90_days)."
    },

    "date_filter_out_of_range": {
        "condition": "Days in date_filter < 1 or > 365",
        "error_type": "ValueError",
        "action": "Fail-fast (raise ValueError)",
        "retry_policy": "No retry",
        "user_message": "Invalid --date-filter '{date_filter}'. Days must be between 1 and 365."
    },

    "duration_exceeds_maximum": {
        "condition": "Video duration > 120 seconds",
        "error_type": "ValueError",
        "action": "Fail-fast (raise ValueError)",
        "retry_policy": "No retry",
        "user_message": "Video duration {duration}s exceeds TikTok maximum (120s)."
    },

    "directory_creation_failed": {
        "condition": "mkdir() raises PermissionError or OSError",
        "error_type": "PermissionError | OSError",
        "action": "Fail-fast (raise exception)",
        "retry_policy": "No retry",
        "user_message": "Cannot create directory {path}. Check write permissions for /data/clients/."
    },

    "config_save_failed": {
        "condition": "Cannot write config.json (permissions, disk space)",
        "error_type": "IOError | PermissionError",
        "action": "Fail-fast (raise exception)",
        "retry_policy": "No retry",
        "user_message": "Cannot write config.json to {path}. Check write permissions and disk space."
    },

    "config_validation_failed": {
        "condition": "Pydantic validation fails for Config",
        "error_type": "ValidationError",
        "action": "Fail-fast (raise ValidationError)",
        "retry_policy": "No retry",
        "user_message": "Invalid configuration: {pydantic_error_details}"
    },
}
```

---

## 7. Complete Example Traces

### TRACE 1: Normal Processing (Happy Path)

**Source**: FoundationCHILD.md Section 2.1: Directory Structure, Section 4: CLI Command Structure, Section 5: Configuration Schemas

**Scenario**: User runs CLI with valid arguments for hashtag analysis

**Input**:
```bash
python rumiai_ml_batch.py \
  --client "acme_corp" \
  --analysis-type hashtag \
  --target "#nutrition" \
  --video-count 100
```

**Processing Steps**:

```
Step 1: Parse CLI arguments
  - parser.parse_args() called
  - Raw parsed: Namespace(
      client="acme_corp",
      analysis_type="hashtag",
      target="#nutrition",
      analysis_mode=None,
      selection_strategy=None,
      video_count=100,
      date_filter="last_90_days",
      report_type="single",
      report_audience=None,
      auto_confirm=False
    )

Step 2: Apply defaults based on analysis_type="hashtag"
  - analysis_mode: None → "top"
  - selection_strategy: None → "contrastive"
  - video_count: 100 (user provided, keep as-is)
  - report_audience: None → "client"
  - Result: All optional fields now have values

Step 3: Validate CLI arguments
  - Validate client_id "acme_corp" → Regex match: PASS
  - Validate target "#nutrition" for hashtag → Starts with #, length ≥ 2: PASS
  - Validate video_count 100 → In range [10, 500]: PASS
  - Validate date_filter "last_90_days" → Regex match, days=90 in [1, 365]: PASS
  - All validations: PASSED

Step 4: Sanitize target for filesystem
  - Input: "#nutrition"
  - Remove # prefix → "nutrition"
  - Lowercase → "nutrition" (already lowercase)
  - No spaces, no special chars → "nutrition"
  - Output: "nutrition"

Step 5: Build directory paths
  - client_base = "/data/clients/acme_corp/"
  - analysis_type_path = "/data/clients/acme_corp/hashtags/"
  - target_path = "/data/clients/acme_corp/hashtags/nutrition/"
  - mode_strategy_path = "/data/clients/acme_corp/hashtags/nutrition/top_contrastive/"

Step 6: Create directory structure
  - Create target_dir: /data/.../top_contrastive/
  - Create buckets/ directory
  - Loop through 8 buckets:
      - Create bucket_0-3s/ with 7 subdirectories (videos/, analysis/, ml_analysis/, models/, reports/, checkpoints/, logs/)
      - Create bucket_3-9s/ with 7 subdirectories
      - ... (6 more buckets)
  - Total directories created: 64 (8 buckets × 8 directories per bucket)

Step 7: Create Config object
  - Extract all fields from validated CLI args
  - Generate run_date: "2025-10-08T15:45:30Z" (current UTC time)
  - Pydantic validation: PASSED

Step 8: Save config.json
  - Path: /data/clients/acme_corp/hashtags/nutrition/top_contrastive/config.json
  - Write JSON with indent=2
  - File saved successfully
```

**Output**:

Directories Created:
```
/data/clients/acme_corp/hashtags/nutrition/top_contrastive/
├── config.json
├── buckets/
│   ├── bucket_0-3s/
│   │   ├── videos/
│   │   ├── analysis/
│   │   ├── ml_analysis/
│   │   ├── models/
│   │   ├── reports/
│   │   ├── checkpoints/
│   │   └── logs/
│   ├── bucket_3-9s/ [same structure]
│   ├── bucket_9-13s/ [same structure]
│   ├── bucket_13-18s/ [same structure]
│   ├── bucket_18-33s/ [same structure]
│   ├── bucket_33-60s/ [same structure]
│   ├── bucket_60-90s/ [same structure]
│   └── bucket_90-120s/ [same structure]
```

config.json Contents:
```json
{
  "client_id": "acme_corp",
  "analysis_type": "hashtag",
  "target": "#nutrition",
  "analysis_mode": "top",
  "selection_strategy": "contrastive",
  "video_count": 100,
  "date_filter": "last_90_days",
  "report_type": "single",
  "report_audience": "client",
  "auto_confirm": false,
  "run_date": "2025-10-08T15:45:30Z"
}
```

**Logs**:
```
INFO: Parsing CLI arguments
INFO: Applying defaults for analysis_type: hashtag
INFO: Validating CLI arguments
INFO: CLI validation passed
INFO: Sanitizing target: #nutrition → nutrition
INFO: Creating directory structure: /data/clients/acme_corp/hashtags/nutrition/top_contrastive/
INFO: Created 8 bucket directories with 7 subdirectories each (64 total)
INFO: Saving configuration to config.json
INFO: Foundation setup complete. Pipeline ready for Stage 1.
```

### TRACE 2: Edge Case - Creator with Custom Parameters

**Source**: FoundationCHILD.md Section 4.2: Default Value Logic (Creator defaults differ from Hashtag)

**Scenario**: User runs CLI for creator analysis with minimal args

**Input**:
```bash
python rumiai_ml_batch.py \
  --client "test_client" \
  --analysis-type creator \
  --target "@fitness_guru"
```

**Processing Steps**:

```
Step 1: Parse CLI arguments
  - Raw parsed: Namespace(
      client="test_client",
      analysis_type="creator",
      target="@fitness_guru",
      analysis_mode=None,
      selection_strategy=None,
      video_count=None,
      date_filter="last_90_days",
      report_type="single",
      report_audience=None,
      auto_confirm=False
    )

Step 2: Apply defaults based on analysis_type="creator"
  - analysis_mode: None → "recent" (DIFFERENT from hashtag's "top")
  - selection_strategy: None → "top" (DIFFERENT from hashtag's "contrastive")
  - video_count: None → 40 (DIFFERENT from hashtag's 100)
  - report_audience: None → "creator" (DIFFERENT from hashtag's "client")
  - Result: Creator-specific defaults applied

Step 3: Validate CLI arguments → PASSED

Step 4: Sanitize target for filesystem
  - Input: "@fitness_guru"
  - Remove @ prefix → "fitness_guru"
  - Already lowercase, no spaces, no special chars
  - Output: "fitness_guru"

Step 5-8: [Same as Trace 1, but with creator-specific values]
```

**Output**:

Directory: `/data/clients/test_client/creators/fitness_guru/recent_top/`

config.json:
```json
{
  "client_id": "test_client",
  "analysis_type": "creator",
  "target": "@fitness_guru",
  "analysis_mode": "recent",
  "selection_strategy": "top",
  "video_count": 40,
  "date_filter": "last_90_days",
  "report_type": "single",
  "report_audience": "creator",
  "auto_confirm": false,
  "run_date": "2025-10-08T15:50:00Z"
}
```

**Key Differences from Trace 1**:
- analysis_mode: "recent" (not "top")
- selection_strategy: "top" (not "contrastive")
- video_count: 40 (not 100)
- report_audience: "creator" (not "client")
- Directory path: creators/ (not hashtags/)

### TRACE 3: Error Case - Invalid Target Format

**Source**: FoundationCHILD.md Section 4.1: CLI Parameters validation rules

**Scenario**: User provides hashtag target without # prefix

**Input**:
```bash
python rumiai_ml_batch.py \
  --client "acme_corp" \
  --analysis-type hashtag \
  --target "nutrition"
```

**Processing Steps**:

```
Step 1: Parse CLI arguments → Success (argparse doesn't validate target format)

Step 2: Apply defaults → Success

Step 3: Validate CLI arguments
  - Validate client_id "acme_corp" → PASS
  - Validate target "nutrition" for analysis_type="hashtag"
      - Check if starts with # → FALSE
      - Validation FAILED

Validation Error Raised:
  ValueError: Invalid --target 'nutrition' for analysis_type 'hashtag'.
              Must start with # and have at least 2 characters (e.g., #nutrition).

Process Terminated: Exit code 2
```

**Output**: No directories created, no config.json

**Error**:
```
ValueError: Invalid --target 'nutrition' for analysis_type 'hashtag'.
            Must start with # and have at least 2 characters (e.g., #nutrition).
```

**Logs**:
```
INFO: Parsing CLI arguments
INFO: Applying defaults for analysis_type: hashtag
INFO: Validating CLI arguments
ERROR: CLI validation failed: Invalid --target 'nutrition' for analysis_type 'hashtag'
ERROR: Must start with # and have at least 2 characters (e.g., #nutrition)
ERROR: Process terminated with exit code 2
```

**User Guidance**: Error message clearly tells user to add # prefix

### TRACE 4: Bucket Assignment Edge Case

**Source**: FoundationCHILD.md Section 6.1: Bucket Assignment Logic

**Scenario**: Assign video with duration exactly 9.0 seconds

**Input**:
```python
duration = 9.0
bucket = assign_bucket(duration)
```

**Processing Steps**:

```
Step 1: Check if duration < 3 → 9.0 < 3? FALSE
Step 2: Check if duration < 9 → 9.0 < 9? FALSE (inclusive lower bound)
Step 3: Check if duration < 13 → 9.0 < 13? TRUE

Result: bucket = "9-13s"

Rationale: Video at exactly 9.0s goes to "9-13s" bucket because:
  - All buckets use INCLUSIVE lower bound (duration >= lower)
  - All buckets use EXCLUSIVE upper bound (duration < upper)
  - 9.0 is NOT less than 9 (boundary), so skip "3-9s"
  - 9.0 IS less than 13, so assign to "9-13s"
```

**Output**: `"9-13s"`

---

## 8. File Structure & Integration

### 8.1 Module Location

```python
# Foundation package structure
FILE_STRUCTURE = """
foundation/
├── __init__.py                 # Package init, exports main interfaces
├── cli.py                      # CLI argument parsing
├── config.py                   # Configuration management
├── paths.py                    # Directory structure and path utilities
├── schemas.py                  # Pydantic schemas (Config, Apify, Checkpoint)
├── buckets.py                  # Bucket assignment logic
└── constants.py                # Shared constants
"""

ENTRY_POINT = "/rumiai_v2/foundation/"
```

### 8.2 Imports

**Source**: FoundationCHILD.md doesn't specify external dependencies (foundation is pure Python + Pydantic)

```python
IMPORTS = [
    # Standard library
    "import argparse",              # CLI parsing
    "import json",                  # Config serialization
    "import re",                    # Regex validation
    "from pathlib import Path",     # Path manipulation
    "from dataclasses import dataclass",  # CLI args dataclass
    "from typing import Dict, List, Tuple, Optional",  # Type hints
    "from datetime import datetime",  # Timestamps

    # External dependencies
    "from pydantic import BaseModel, Field, validator",  # Schema validation, 2.0.0+
]
```

### 8.3 Integration Points

```python
CALLS_TO_EXTERNAL_SYSTEMS = {}  # Foundation has no external dependencies
```

### 8.4 Base Directory Structure

**Source**: FoundationCHILD.md Section 2.1 (lines 86-230)

```python
BASE_PATHS = {
    "data_root": "/data/",
    "clients_base": "/data/clients/",
    "client_base": "/data/clients/{client_id}/",
    "analysis_type_base": "{client_base}/{analysis_type}s/",  # Note: pluralized
    "target_base": "{analysis_type_base}/{sanitized_target}/",
    "mode_strategy_base": "{target_base}/{mode}_{strategy}/",
    "buckets_base": "{mode_strategy_base}/buckets/",
    "bucket_base": "{buckets_base}/bucket_{bucket}/",

    # Subdirectories within each bucket
    "videos": "{bucket_base}/videos/",
    "analysis": "{bucket_base}/analysis/",
    "ml_analysis": "{bucket_base}/ml_analysis/",
    "models": "{bucket_base}/models/",
    "reports": "{bucket_base}/reports/",
    "checkpoints": "{bucket_base}/checkpoints/",
    "logs": "{bucket_base}/logs/",
}

# Example: Full path construction
# /data/clients/acme_corp/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/videos/
```

### 8.5 Stage Output Paths

Foundation produces paths used by all stages:

```python
OUTPUT_PATHS = {
    # Configuration file
    "config_json": "{mode_strategy_base}/config.json",

    # Bucket directories (8 buckets)
    "bucket_dirs": [
        "{buckets_base}/bucket_0-3s/",
        "{buckets_base}/bucket_3-9s/",
        "{buckets_base}/bucket_9-13s/",
        "{buckets_base}/bucket_13-18s/",
        "{buckets_base}/bucket_18-33s/",
        "{buckets_base}/bucket_33-60s/",
        "{buckets_base}/bucket_60-90s/",
        "{buckets_base}/bucket_90-120s/",
    ],

    # Subdirectories per bucket (7 per bucket × 8 buckets = 56)
    "bucket_subdirs": [
        "videos/", "analysis/", "ml_analysis/", "models/",
        "reports/", "checkpoints/", "logs/"
    ],
}
```

---

## 9. Configuration & Environment

### 9.1 Environment Variables

**Source**: Not specified in FoundationCHILD.md (Foundation uses hardcoded /data/ base path)

```python
ENV_VARS = {
    "DATA_ROOT": {
        "required": False,
        "type": "str",
        "default": "/data",
        "example": "/data",
        "validation": "Must be absolute path with write access"
    },
}
```

### 9.2 CLI Parameters Configuration

**Source**: FoundationCHILD.md Section 4.1 (lines 569-584)

```python
CONFIG_SCHEMA = {
    "cli_params": {
        # Required parameters (no defaults)
        "client": None,                    # Required, no default
        "analysis_type": None,             # Required, no default
        "target": None,                    # Required, no default

        # Optional parameters with defaults (from parser)
        "date_filter": "last_90_days",
        "report_type": "single",
        "auto_confirm": False,

        # Optional parameters with type-specific defaults (from apply_defaults())
        "analysis_mode": {
            "hashtag": "top",
            "competitor": "top",
            "creator": "recent",
        },
        "selection_strategy": {
            "hashtag": "contrastive",
            "competitor": "contrastive",
            "creator": "top",
        },
        "video_count": {
            "hashtag": 100,
            "competitor": 100,
            "creator": 40,
        },
        "report_audience": {
            "hashtag": "client",
            "competitor": "client",
            "creator": "creator",
        },
    },
}
```

### 9.3 Constants

**Source**: FoundationCHILD.md Section 3: Configuration Dimensions, Section 6: Bucket Definitions

```python
CONSTANTS = {
    # Bucket Definitions (Section 6)
    "BUCKET_DEFINITIONS": {
        "0-3s": (0, 3),
        "3-9s": (3, 9),
        "9-13s": (9, 13),
        "13-18s": (13, 18),
        "18-33s": (18, 33),
        "33-60s": (33, 60),
        "60-90s": (60, 90),
        "90-120s": (90, 120),
    },

    # Valid Enum Values (Section 4.1)
    "VALID_ANALYSIS_TYPES": ["hashtag", "competitor", "creator"],
    "VALID_MODES": ["top", "recent"],
    "VALID_STRATEGIES": ["contrastive", "top"],
    "VALID_REPORT_TYPES": ["single", "comparison"],
    "VALID_REPORT_AUDIENCES": ["client", "internal", "creator"],

    # Default Values by Target Type (Section 4.2)
    "DEFAULT_VIDEO_COUNTS": {
        "hashtag": 100,
        "competitor": 100,
        "creator": 40,
    },
    "DEFAULT_ANALYSIS_MODES": {
        "hashtag": "top",
        "competitor": "top",
        "creator": "recent",
    },
    "DEFAULT_SELECTION_STRATEGIES": {
        "hashtag": "contrastive",
        "competitor": "contrastive",
        "creator": "top",
    },
    "DEFAULT_REPORT_AUDIENCES": {
        "hashtag": "client",
        "competitor": "client",
        "creator": "creator",
    },

    # Minimum Recommended N (Section 3.4)
    "MIN_RECOMMENDED_N": {
        "contrastive": 50,  # Ensures 10 bottom performers (20%)
        "top": 20,          # Minimum for K-Means clustering
    },

    # Engagement Score Formula (Section 3.2)
    "ENGAGEMENT_SHARE_WEIGHT": 10,  # 10x weight for shares

    # File Naming Patterns
    "CONFIG_FILENAME": "config.json",
    "CHECKPOINT_FILENAME": "checkpoint.json",
    "SELECTED_VIDEOS_FILENAME": "selected_videos.json",
}
```

---

## 10. Logging Specifications

**Source**: Not explicitly defined in FoundationCHILD.md (inferred from process steps)

```python
LOG_MESSAGES = {
    # CLI Parsing
    "cli_parsing_start": ("INFO", "Parsing CLI arguments"),
    "cli_parsing_complete": ("INFO", "CLI parsing complete: {arg_count} arguments processed"),

    # Default Application
    "applying_defaults": ("INFO", "Applying defaults for analysis_type: {analysis_type}"),
    "defaults_applied": ("INFO", "Defaults applied: mode={mode}, strategy={strategy}, video_count={count}, audience={audience}"),

    # Validation
    "validation_start": ("INFO", "Validating CLI arguments"),
    "validation_passed": ("INFO", "CLI validation passed"),
    "validation_failed": ("ERROR", "CLI validation failed: {error_message}"),

    # Path Sanitization
    "sanitizing_target": ("INFO", "Sanitizing target: {original} → {sanitized}"),
    "sanitization_complete": ("DEBUG", "Target sanitized: {sanitized}"),

    # Directory Creation
    "creating_directories": ("INFO", "Creating directory structure: {target_dir}"),
    "bucket_creation_start": ("DEBUG", "Creating bucket directories: {bucket_count} buckets"),
    "bucket_created": ("DEBUG", "Created bucket: {bucket_name} at {path}"),
    "directories_complete": ("INFO", "Created {dir_count} directories (8 buckets with 7 subdirectories each)"),

    # Configuration
    "saving_config": ("INFO", "Saving configuration to config.json"),
    "config_saved": ("INFO", "Configuration saved: {config_path}"),

    # Success
    "foundation_complete": ("INFO", "Foundation setup complete. Pipeline ready for Stage 1."),

    # Warnings
    "low_video_count_contrastive": ("WARNING", "Low bottom performer count ({bottom_count} videos, N × 0.2). Recommend N ≥ 50 for robust classification."),
    "low_video_count_top": ("WARNING", "Low sample size for clustering ({video_count} videos). Recommend N ≥ 20 for pattern detection."),

    # Errors
    "invalid_client_id": ("ERROR", "Invalid --client '{client_id}'. Must contain only alphanumeric characters and underscores."),
    "invalid_target_format": ("ERROR", "Invalid --target '{target}' for analysis_type '{analysis_type}'. {guidance}"),
    "video_count_out_of_range": ("ERROR", "Invalid --video-count {video_count}. Must be between 10 and 500."),
    "directory_creation_failed": ("ERROR", "Cannot create directory {path}. Check write permissions."),
    "config_save_failed": ("ERROR", "Cannot write config.json to {path}. {error_details}"),
}

METRICS = {
    "cli_parse_time_ms": "Time spent parsing CLI arguments",
    "validation_time_ms": "Time spent validating arguments",
    "directory_creation_time_ms": "Time spent creating directory structure",
    "directories_created_count": "Total number of directories created",
    "config_save_time_ms": "Time spent saving config.json",
    "total_foundation_time_ms": "Total time for foundation setup",
}
```

---

## 11. Dependencies & Prerequisites

### 11.1 External Dependencies

```python
EXTERNAL_DEPS = {
    "pydantic": {
        "version": ">=2.0.0",
        "purpose": "Schema validation with type safety for Config, Apify metadata, Checkpoint schemas",
        "pip_install": "pip install pydantic>=2.0.0"
    },
}
```

### 11.2 Upstream TI Requirements

```python
UPSTREAM_OUTPUTS_REQUIRED = {}  # Foundation has no upstream dependencies
```

Foundation is the first layer - all other stages depend on it, not the reverse.

### 11.3 System Prerequisites

```python
SYSTEM_REQUIREMENTS = {
    "disk_space": "100MB minimum (for directory structure, config files)",
    "memory": "50MB (minimal - Foundation is lightweight)",
    "permissions": "Write access to /data/clients/ directory",
    "python_version": "Python 3.10+ (for type hints and Pydantic 2.0)",
    "api_keys": [],  # None required
    "network": "Not required (pure filesystem operations)",
}
```

---

## 12. HLD Traceability Matrix

| HLD Section | TI Section | Implementation Status |
|-------------|------------|----------------------|
| Section 1: System Goals & Success Criteria | Section 1: Document Metadata | To Implement |
| Section 2: Client Architecture & Directory Structure | Section 2.2: Output Contract, Section 8.4: Base Directory Structure | To Implement |
| Section 2.1: Directory Structure | Section 4.5: create_directory_structure() | To Implement |
| Section 2.2: Path Generation Templates | Section 8.4: Base Directory Structure | To Implement |
| Section 2.2.1: Path Sanitization Rules | Section 4.3: sanitize_target() | To Implement |
| Section 3: Configuration Dimensions | Section 9.2: CLI Parameters Configuration | To Implement |
| Section 3.1: Target Types | Section 9.2: CLI Parameters (analysis_type validation) | To Implement |
| Section 3.2: Analysis Modes | Section 9.2: CLI Parameters (analysis_mode defaults) | To Implement |
| Section 3.3: Selection Strategies | Section 9.2: CLI Parameters (selection_strategy defaults) | To Implement |
| Section 3.4: Video Count (N) | Section 5.3: Minimum N Recommendations | To Implement |
| Section 3.5: Date Filtering | Section 5.1: CLI Argument Validation (date_filter) | To Implement |
| Section 3.6: Report Types | Section 9.2: CLI Parameters (report_type) | To Implement |
| Section 3.7: Report Audience | Section 9.2: CLI Parameters (report_audience) | To Implement |
| Section 4: CLI Command Structure | Section 4.1: parse_args() | To Implement |
| Section 4.1: CLI Parameters | Section 2.1: Input Contract, Section 5.1: Validation | To Implement |
| Section 4.2: Default Value Logic | Section 4.2: apply_defaults() | To Implement |
| Section 5: Configuration Schemas | Section 3: Data Schemas | To Implement |
| Section 5.1: config.json Schema | Section 3.1: Config Schema, Section 4.6: save_config() | To Implement |
| Section 5.2: Apify Video Metadata Schema | Section 3.1: Apify Metadata Schema | To Implement |
| Section 5.3: Checkpoint Schema | Section 3.1: Checkpoint Schema | To Implement |
| Section 6: Bucket Definitions | Section 3.2: Bucket Definitions Schema | To Implement |
| Section 6.1: Bucket Assignment Logic | Section 4.4: assign_bucket() | To Implement |
| Section 7: References | Section 1: Document Metadata (Related_TI_Docs) | To Implement |
| Appendix A: Glossary | Section 8: Integration (shared terminology) | To Implement |

---

## 13. Implementation Checklist

**Before marking Foundation as COMPLETE**:

- [ ] All 6 modules implemented (cli.py, config.py, paths.py, schemas.py, buckets.py, constants.py)
- [ ] CLI accepts all 11 parameters with argparse
- [ ] Defaults applied correctly per target type (hashtag/competitor/creator)
- [ ] All validation rules enforced (client format, target prefix, video_count range, date_filter format)
- [ ] Path sanitization removes # and @ prefixes, handles special characters correctly
- [ ] Bucket assignment handles edge cases (9.0s, 120.0s, >120s)
- [ ] Directory structure creation works for all 8 buckets with 7 subdirectories each
- [ ] Config round-trip works (CLI → Config → JSON → Config)
- [ ] Pydantic schemas validate all fields (Config, Apify, Checkpoint)
- [ ] Error messages guide user to fix issues (specific, actionable)
- [ ] Type hints on all public APIs
- [ ] 100% test coverage for validation logic
- [ ] Integration test passes: CLI input → directories created → config.json saved
- [ ] All constants centralized in constants.py
- [ ] Foundation package installable via `pip install -e foundation/`

---

**Version**: 1.0
**Status**: Implementation Ready
**Estimated Effort**: 5-7 days (1 developer)
**Next Steps**: Implement foundation package, then proceed to Stage 1 (VideoDiscoveryTI)
