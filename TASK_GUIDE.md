# TASK_GUIDE.md - RumiAI Configuration Simplification

## [Project Overview]

RumiAI is a TikTok video analysis pipeline that processes videos through ML services and generates insights using Claude AI. The system currently supports multiple configuration options through environment variables, but the production workflow only uses a specific configuration.

### Current State
- System has 30+ environment variables for flexibility
- Supports both v1 (legacy) and v2 (6-block) output formats
- Has conditional logic throughout for backward compatibility
- Documentation covers all possible configurations

### Goal
Simplify the codebase to support ONLY the production configuration:
- Always use ML precompute functions
- Always use v2 (6-block) output format
- Always enable temporal markers
- Always enable cost tracking
- Remove all v1 legacy code and compatibility layers

## [Tasks]

### Task 1: Remove v1 Output Format Code

#### 1.1 Delete output_adapter.py
```bash
rm rumiai_v2/processors/output_adapter.py
```

#### 1.2 Update rumiai_runner.py
- Find: `def __init__(self, legacy_mode: bool = False):`
  Replace with: `def __init__(self):`
  
- Find and remove: `self.legacy_mode = legacy_mode`

- Find and remove: `self.output_adapter = OutputAdapter()`

- Find: `from ..processors import OutputAdapter`
  Remove this import line

- Find the block starting with: `if self.settings.output_format_version == 'v1':`
  Remove the entire if block (4 lines) that converts to legacy format

- Find: `if result.success and self.settings.output_format_version == 'v2':`
  Replace with: `if result.success:`

- Find: `runner = RumiAIRunner(legacy_mode=legacy_mode)`
  Replace with: `runner = RumiAIRunner()`

- Find and remove entire method: `async def process_video_id(self, video_id: str) -> Dict[str, Any]:`
  (Remove from method definition through the end of the method)

- Find the legacy mode detection block that starts with:
  ```python
  legacy_mode = False
  ```
  And contains `legacy_mode = True` assignments
  Remove the entire block and replace with simplified argument parsing (see section 4.3)

#### 1.3 Update UnifiedAnalysis (analysis.py)
- Find: `def to_dict(self, legacy_mode: bool = False) -> Dict[str, Any]:`
  Replace with: `def to_dict(self) -> Dict[str, Any]:`

- Find: `self.timeline.to_dict(legacy_mode)`
  Replace with: `self.timeline.to_dict()`

- Find: `def save_to_file(self, file_path: str, legacy_mode: bool = False) -> None:`
  Replace with: `def save_to_file(self, file_path: str) -> None:`

- Find: `json.dump(self.to_dict(legacy_mode), tmp, indent=2)`
  Replace with: `json.dump(self.to_dict(), tmp, indent=2)`

#### 1.4 Update Timeline models (timeline.py)
- Find: `def to_dict(self, legacy_mode: bool = True) -> Dict[str, Any]:`
  Replace with: `def to_dict(self) -> Dict[str, Any]:`
  (Note: This appears twice in the file - update both occurrences)

- Find: `'start': self.start.to_json(legacy_mode),`
  Replace with: `'start': self.start.to_json(),`

- Find: `result['end'] = self.end.to_json(legacy_mode)`
  Replace with: `result['end'] = self.end.to_json()`

- Find: `'entries': [entry.to_dict(legacy_mode) for entry in self.entries]`
  Replace with: `'entries': [entry.to_dict() for entry in self.entries]`

#### 1.5 Update Timestamp model (timestamp.py)
- Find: `def to_json(self, legacy_mode: bool = True)`
  Replace with: `def to_json(self)`

- Find and remove the legacy format logic that checks `if legacy_mode:`
  Keep only: `return self.seconds`

#### 1.6 Update processors/__init__.py
- Find: `from .output_adapter import OutputAdapter`
  Remove this import line

- Find: `'OutputAdapter',` in the __all__ list
  Remove this entry

#### 1.7 Update test_e2e.py
- Find: `RumiAIRunner(legacy_mode=False)`
  Replace with: `RumiAIRunner()`

- Find and remove entire function: `async def test_legacy_mode():`
  (Remove from function definition through the end of the function)

- Find: `if not await test_legacy_mode():`
  Remove this line and any associated error handling

#### 1.8 Delete test_integration.py
```bash
rm /home/jorge/rumiaifinal/tests/test_integration.py
```

### Task 2: Hardcode Production Configuration

#### 2.1 Update settings.py
Replace lines 62-75 with:
```python
# ML Enhancement Feature Flags - ALWAYS ENABLED
self.use_ml_precompute = True  # HARDCODED
self.use_claude_sonnet = os.getenv('USE_CLAUDE_SONNET', 'false').lower() == 'true'
self.output_format_version = 'v2'  # HARDCODED

# All precompute prompts ALWAYS enabled
self.precompute_enabled_prompts = {
    'creative_density': True,
    'emotional_journey': True,
    'person_framing': True,
    'scene_pacing': True,
    'speech_analysis': True,
    'visual_overlay_analysis': True,
    'metadata_analysis': True
}
```

Replace line 57:
```python
self.temporal_markers_enabled = True  # HARDCODED (was from env var)
```

Replace line 79:
```python
self.enable_cost_monitoring = True  # HARDCODED (was from env var)
```

#### 2.2 Update to_dict() method (line 223)
Remove `'output_format_version': self.output_format_version,` since it's always v2

### Task 3: Simplify Model Selection

#### 3.1 Remove unused environment variable references
In settings.py, remove references to:
- `USE_ML_PRECOMPUTE` (line 62)
- `OUTPUT_FORMAT_VERSION` (line 64)
- `RUMIAI_TEMPORAL_MARKERS` (line 57)
- `ENABLE_COST_MONITORING` (line 79)
- All `PRECOMPUTE_*` individual flags (lines 67-75)

#### 3.2 Keep these environment variables:
```python
# Required
self.claude_api_key = os.getenv('CLAUDE_API_KEY', '')
self.apify_token = os.getenv('APIFY_API_TOKEN', '')

# Optional
self.use_claude_sonnet = os.getenv('USE_CLAUDE_SONNET', 'false').lower() == 'true'
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))

# Path configs (keep for flexibility)
self.output_dir = Path(os.getenv('RUMIAI_OUTPUT_DIR', 'outputs'))
self.temp_dir = Path(os.getenv('RUMIAI_TEMP_DIR', 'temp'))
self.unified_dir = Path(os.getenv('RUMIAI_UNIFIED_DIR', 'unified_analysis'))
self.insights_dir = Path(os.getenv('RUMIAI_INSIGHTS_DIR', 'insights'))
self.temporal_dir = Path(os.getenv('RUMIAI_TEMPORAL_DIR', 'temporal_markers'))

# Processing settings (keep for flexibility)
self.max_video_duration = int(os.getenv('RUMIAI_MAX_VIDEO_DURATION', '300'))
self.frame_sample_rate = float(os.getenv('RUMIAI_FRAME_SAMPLE_RATE', '1.0'))
self.prompt_delay = int(os.getenv('RUMIAI_PROMPT_DELAY', '10'))

# Operational settings (keep for flexibility)
self.strict_mode = os.getenv('RUMIAI_STRICT_MODE', 'false').lower() == 'true'
self.cleanup_video = os.getenv('RUMIAI_CLEANUP_VIDEO', 'false').lower() == 'true'
self.claude_cost_threshold = float(os.getenv('CLAUDE_COST_THRESHOLD', '0.10'))
```

### Task 4: Update Entry Point

**IMPORTANT**: The core v2 validation logic (lines 518-545) must be preserved! We're only removing the v1 conversion branch.

#### 4.1 Remove legacy mode from runner initialization
```python
# Line 50: Change from
def __init__(self, legacy_mode: bool = False):
# To:
def __init__(self):

# Remove line 57:
# self.legacy_mode = legacy_mode
```

#### 4.2 Simplify output format checks (lines 518-545)
Modify the existing block to remove v1 conversion:
```python
if result.success:  # Remove the version check
    # Validate response
    is_valid, parsed_data, validation_errors = ResponseValidator.validate_6block_response(
        result.response, 
        prompt_type.value
    )
    
    if is_valid and parsed_data:
        logger.info(f"Received valid 6-block response for {prompt_type.value}")
        
        # Store parsed data for later use
        result.parsed_response = parsed_data
        
        # REMOVE lines 531-534 (v1 conversion block)
    else:
        # Try to extract structure from text if JSON parsing failed
        extracted = ResponseValidator.extract_text_blocks(result.response)
        if extracted:
            logger.warning(f"Extracted 6-block structure from text for {prompt_type.value}")
            result.parsed_response = extracted
        else:
            logger.error(f"Invalid 6-block response for {prompt_type.value}: {', '.join(validation_errors)}")
            # Mark as failed if we can't parse the response
            result.success = False
            result.error = f"Invalid response format: {'; '.join(validation_errors)}"
```

#### 4.3 Simplify main() function
Find the block that starts with:
```python
legacy_mode = False
```
And contains multiple if/elif statements checking for video_id, video_url, etc.

Replace the entire block with:
```python
if args.video_url:
    video_url = args.video_url
elif args.video_input:
    video_url = args.video_input
else:
    print("Usage: rumiai_runner.py <video_url>", file=sys.stderr)
    return 1
```

#### 4.4 Simplify processing
Find the block that contains:
```python
if legacy_mode:
    logger.info(f"Running in legacy mode for video ID: {video_id}")
    result = asyncio.run(runner.process_video_id(video_id))
else:
    logger.info(f"Running in new mode for video URL: {video_url}")
    result = asyncio.run(runner.process_video_url(video_url))
```

Replace with:
```python
logger.info(f"Processing video URL: {video_url}")
result = asyncio.run(runner.process_video_url(video_url))
```

### Task 5: Clean Up Prompt System

#### 5.1 Check for v1 templates
```bash
# Find and remove any v1 templates
find prompt_templates -name "*v1.txt" -delete
find prompt_templates -name "*_legacy.txt" -delete
```

#### 5.2 Ensure prompt_manager.py only loads v2
- Check that `prompt_manager.py` only loads `*_v2.txt` files
- Remove any version selection logic

#### 5.3 Update prompt builder if needed
- Verify `prompt_builder.py` doesn't have v1/v2 conditionals

### Task 6: Update Documentation
- [ ] Update `Codemappingfinal.md` to reflect simplified architecture
- [ ] Update `RUMIAI_CORE_ARCHITECTURE_PRINCIPLES.md` to show single configuration
- [ ] Update `FlowStructure.md` to remove v1 references
- [ ] Create new `PRODUCTION_CONFIG.md` with the single supported configuration

### Task 7: Update Tests
- [ ] Remove tests for v1 format
- [ ] Remove tests for disabled features
- [ ] Update integration tests to assume production config

## [File References]

### Files to Modify
```
/rumiai_v2/config/settings.py                 # Hardcode production settings
/scripts/rumiai_runner.py                     # Remove legacy branches
/rumiai_v2/core/models/analysis.py           # Remove legacy_mode from to_dict()
/rumiai_v2/core/models/timeline.py            # Remove legacy_mode from to_dict()
/rumiai_v2/core/models/timestamp.py           # Remove legacy_mode from to_json()
/rumiai_v2/processors/__init__.py            # Remove OutputAdapter export
/test_e2e.py                                 # Remove legacy mode tests
```

### Files to Delete
```
/rumiai_v2/processors/output_adapter.py      # v1/v2 converter - DELETE
/prompt_templates/*v1.txt                    # Any v1 prompt templates - DELETE
/tests/test_integration.py                   # Legacy integration tests - DELETE
```

### Documentation to Update
```
/Codemappingfinal.md
/RUMIAI_CORE_ARCHITECTURE_PRINCIPLES.md
/FlowStructure.md
/.env.example                                # Simplify to essential vars only
```

## [Instructions]

### Step 1: Backup Current State
```bash
git add -A
git commit -m "Backup before v1 removal"
git branch backup-with-v1
```

### Step 2: Remove v1 Code

#### Search Commands to Find All References
```bash
# Find all legacy_mode references
grep -rn "legacy_mode" rumiai_v2/ scripts/

# Find all output format version references  
grep -rn "OUTPUT_FORMAT_VERSION\|output_format_version" rumiai_v2/ scripts/

# Find all v1/v2 conditionals
grep -rn "format.*==.*v1\|format.*==.*v2" rumiai_v2/ scripts/

# Find OutputAdapter imports
grep -rn "OutputAdapter\|output_adapter" rumiai_v2/ scripts/
```

#### Files to Modify (based on search results):
1. `scripts/rumiai_runner.py` - Remove legacy mode, output format checks
2. `rumiai_v2/config/settings.py` - Hardcode values
3. `rumiai_v2/core/models/analysis.py` - Remove legacy_mode parameter
4. `rumiai_v2/core/models/timeline.py` - Remove legacy_mode parameter
5. `rumiai_v2/core/models/timestamp.py` - Remove legacy_mode parameter
6. `rumiai_v2/processors/output_adapter.py` - DELETE entirely

### Step 3: Hardcode Settings
Edit `/rumiai_v2/config/settings.py`:
```python
class Settings:
    def __init__(self):
        # API Keys (still from env)
        self.claude_api_key = os.getenv('CLAUDE_API_KEY', '')
        self.apify_token = os.getenv('APIFY_API_TOKEN', '')
        
        # Model selection (simplified)
        self.use_claude_sonnet = os.getenv('USE_CLAUDE_SONNET', 'false').lower() == 'true'
        self.claude_model = 'claude-3-5-sonnet-20241022' if self.use_claude_sonnet else 'claude-3-haiku-20240307'
        
        # Hardcoded production settings
        self.use_ml_precompute = True  # ALWAYS TRUE
        self.output_format_version = 'v2'  # ALWAYS V2
        self.temporal_markers_enabled = True  # ALWAYS TRUE
        self.enable_cost_monitoring = True  # ALWAYS TRUE
        
        # All precompute flags enabled
        self.precompute_enabled_prompts = {
            'creative_density': True,
            'emotional_journey': True,
            'person_framing': True,
            'scene_pacing': True,
            'speech_analysis': True,
            'visual_overlay_analysis': True,
            'metadata_analysis': True
        }
```

### Step 4: Test Simplified System
```bash
# Test with minimal env vars
export CLAUDE_API_KEY=xxx
export APIFY_API_TOKEN=xxx
export USE_CLAUDE_SONNET=true  # or false for Haiku
python scripts/rumiai_runner.py [video_url]
```

### Step 5: Update .env.example
Create updated example with all supported variables:
```
# Required
CLAUDE_API_KEY=your_key_here
APIFY_API_TOKEN=your_token_here

# Optional - Model Selection
USE_CLAUDE_SONNET=true  # Use expensive model (default: false)

# Optional - Paths (all have sensible defaults)
RUMIAI_OUTPUT_DIR=outputs
RUMIAI_TEMP_DIR=temp
RUMIAI_UNIFIED_DIR=unified_analysis
RUMIAI_INSIGHTS_DIR=insights
RUMIAI_TEMPORAL_DIR=temporal_markers

# Optional - Processing Settings
RUMIAI_MAX_VIDEO_DURATION=300  # Max seconds (default: 300)
RUMIAI_FRAME_SAMPLE_RATE=1.0   # FPS for extraction (default: 1.0)
RUMIAI_PROMPT_DELAY=10         # Seconds between prompts (default: 10)

# Optional - Operational Settings
RUMIAI_STRICT_MODE=false       # Fail on config errors (default: false)
RUMIAI_CLEANUP_VIDEO=false     # Delete video after processing (default: false)
CLAUDE_COST_THRESHOLD=0.10     # Alert threshold (default: $0.10)
LOG_LEVEL=INFO                 # Logging level (default: INFO)
```

## [Safety Checks]

### Critical Code to Preserve
1. **6-block validation logic** (lines 520-545) - This is essential for v2!
2. **All precompute functions** in precompute_functions.py
3. **Temporal marker generation** logic
4. **ML services integration**
5. **ResponseValidator.validate_6block_response** method

### What We're Actually Removing
1. **OutputAdapter** - Only converts v2→v1 (not needed)
2. **legacy_mode parameters** - Only affect output format
3. **process_video_id method** - Old entry point
4. **v1 format conditionals** - Only in output generation

## [Expected Outcomes]

### Benefits
1. **Simpler codebase** - No conditional branches for features
2. **Fewer bugs** - Less configuration complexity
3. **Clearer documentation** - Single path through system
4. **Easier maintenance** - No legacy compatibility burden
5. **Better performance** - No runtime config checks

### Code Reduction Estimates
- ~300 lines removed from rumiai_runner.py
- ~200 lines removed from output_adapter.py (entire file)
- ~50 lines removed from settings.py
- ~100 lines removed from model classes
- **Total: ~650 lines of code removed**

### Risks to Monitor
1. **Breaking changes** for any external systems expecting v1 format
2. **Node.js integration** may expect certain outputs
3. **Existing data** in v1 format won't be readable
4. **Third-party integrations** may depend on removed features

## [Verification Checklist]

After changes:
- [ ] All 7 prompts generate v2 (6-block) format
- [ ] Temporal markers always generated
- [ ] Cost tracking always active
- [ ] No references to v1 or legacy in codebase
- [ ] ML precompute always enabled
- [ ] System works with simplified configuration

### Quick Test Command
```bash
# After changes, this should work:
export CLAUDE_API_KEY=your_key
export APIFY_API_TOKEN=your_token
export USE_CLAUDE_SONNET=true
python scripts/rumiai_runner.py https://www.tiktok.com/@user/video/123

# These are NO LONGER needed:
# export USE_ML_PRECOMPUTE=true  # Now hardcoded
# export OUTPUT_FORMAT_VERSION=v2  # Now hardcoded
# No need to set individual PRECOMPUTE_* flags
```

### Manual Testing Steps
1. Run a test video through the pipeline
2. Verify all 7 Claude prompts execute
3. Check that output files are in v2 (6-block) format
4. Confirm temporal markers are generated
5. Verify cost tracking appears in logs