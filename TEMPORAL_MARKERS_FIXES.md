# Temporal Markers Code Fixes Required

## Priority 1: Fix File Save Location with Unified Method

### File: `scripts/rumiai_runner.py`

**Step 1: Add unified save method after __init__ (around line 90):**
```python
def save_analysis_result(self, video_id: str, analysis_type: str, data: dict) -> Path:
    """Save any analysis result in the unified insights structure.
    
    This maintains consistency with how other insights (creative_density, 
    emotional_journey, etc.) are saved.
    """
    from datetime import datetime
    
    # Create directory structure: insights/{video_id}/{analysis_type}/
    analysis_dir = self.insights_handler.get_path(video_id, analysis_type)
    self.insights_handler.ensure_dir(analysis_dir)
    
    # Generate filename with timestamp (matching existing insights pattern)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{analysis_type}_complete_{timestamp}.json"
    filepath = analysis_dir / filename
    
    # Save using handler's method for atomic writes and error handling
    self.insights_handler.save_json(filepath, data)
    
    logger.info(f"Saved {analysis_type} to {filepath}")
    return filepath
```

**Step 2: Update temporal markers save (Lines 192-195):**
```python
# OLD CODE TO REMOVE:
temporal_path = self.temporal_handler.get_path(
    f"{video_id}_{int(time.time())}.json"
)
self.temporal_handler.save_json(temporal_path, temporal_markers)

# NEW CODE:
temporal_path = self.save_analysis_result(video_id, "temporal_markers", temporal_markers)
```

**Step 3: Update legacy mode (Lines 277-280):**
```python
# OLD CODE TO REMOVE:
temporal_path = self.temporal_handler.get_path(
    f"{video_id}_{int(time.time())}.json"
)
self.temporal_handler.save_json(temporal_path, temporal_markers)

# NEW CODE:
temporal_path = self.save_analysis_result(video_id, "temporal_markers", temporal_markers)
```

## Priority 2: Remove Misleading Node.js Comments

### File: `rumiai_v2/processors/temporal_markers.py`

**Lines 24-26 - Remove/Update Comment:**
```python
# CURRENT (WRONG):
    CRITICAL: This is called by Node.js via subprocess.
    MUST output valid JSON to stdout even on error.

# CHANGE TO:
    Generates temporal markers from unified ML analysis.
    Used in Python-only processing pipeline.
```

**Lines 35-36 - Remove/Update Comment:**
```python
# CURRENT (WRONG):
        CRITICAL: This is called by Node.js via subprocess.
        MUST output valid JSON to stdout even on error.

# CHANGE TO:
        Generate temporal markers from ML data for video analysis.
        Returns dictionary with first_5_seconds, cta_window, peak_moments, etc.
```

**Lines 77-78 - Update Error Comment:**
```python
# CURRENT:
            # CRITICAL: Log to stderr only, not stdout

# CHANGE TO:
            # Log error and return empty markers structure
```

**Lines 414-419 - Update main() Docstring:**
```python
# CURRENT:
    """
    Main entry point for command-line execution.
    
    This maintains compatibility with existing Node.js calls.
    """

# CHANGE TO:
    """
    Main entry point for command-line execution.
    
    For testing and standalone execution only.
    Production uses direct function calls from rumiai_runner.py.
    """
```

## Priority 3: Fix Incomplete Conditional in Main Function

### File: `scripts/rumiai_runner.py`

**Lines 443-446 - Fix Incomplete Conditional:**
```python
# CURRENT (INCOMPLETE):
    # Run processing
    
    logger.info(f"Running in legacy mode for video ID: {video_id}")
    result = asyncio.run(runner.process_video_id(video_id))

# CHANGE TO:
    # Run processing
    if video_url:
        logger.info(f"Processing video URL: {video_url}")
        result = asyncio.run(runner.process_video_url(video_url))
    elif video_id:
        logger.info(f"Running in legacy mode for video ID: {video_id}")
        result = asyncio.run(runner.process_video_id(video_id))
    else:
        logger.error("No video URL or ID provided")
        sys.exit(1)
```

## Priority 4: Remove Unused temporal_handler

### File: `scripts/rumiai_runner.py`

**Lines 49-50 - Remove temporal_handler initialization:**
```python
# REMOVE THESE LINES:
self.temporal_handler = DirectoryHandler(
    self.settings.temporal_dir, "temporal_markers")
```

**Line 38 - Update imports (remove DirectoryHandler if not used elsewhere):**
```python
# Check if DirectoryHandler is still needed for other handlers
# If not, remove from imports
```

## Optional: Add Temporal Markers to Summary Output

### File: `scripts/rumiai_runner.py`

**After Line 195 (and Line 280 for legacy), add:**
```python
# Add temporal markers to insights summary
if 'temporal_markers' not in insights:
    insights['temporal_markers'] = {}
insights['temporal_markers']['path'] = str(temporal_path)
insights['temporal_markers']['generated'] = True
```

## Testing Commands

After making these changes, test with:

```bash
# Test temporal marker generation
python scripts/rumiai_runner.py --video-id 7475724973378948374

# Verify output location
ls -la insights/7475724973378948374/temporal_markers/

# Check file content
cat insights/7475724973378948374/temporal_markers/temporal_markers_*.json | jq .metadata
```

## Expected Result

After these fixes:
1. Temporal markers will be saved in `insights/{video_id}/temporal_markers/` 
2. No more misleading Node.js/subprocess comments
3. Consistent file organization with other analysis types
4. Main function will properly handle both URL and ID inputs

## Notes

- The `temporal_markers/` root directory can be deleted after migration
- Old temporal marker files can be moved to new structure if needed
- No changes needed to `TemporalMarkerProcessor` logic - it works correctly
- No changes needed to settings.py - `temporal_markers_enabled = True` is correct