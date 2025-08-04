# Golden Dataset Validation Template Instructions

## Overview
This template helps you create comprehensive test data for validating RumiAI's 7 analysis dimensions. You don't need to fill every field - focus on what you can clearly observe.

## 🚨 CRITICAL: How to Fill Fields Correctly

### Understanding Field Formats

The template uses different field formats. **It's crucial to understand which format each field uses:**

#### 1. **Single Choice Fields** (contain `|` separator)
When you see: `"field": "option1|option2|option3"`
→ Choose **ONE** option and replace the entire string

**Correct ✅:**
```json
"pacing": "fast"
"shot_type": "close-up"
"purpose": "grab attention"
```

**Wrong ❌:**
```json
"pacing": "fast|medium"  // Don't keep multiple options
"pacing": "fast and medium"  // Don't combine with 'and'
"pacing": "very fast"  // Don't use your own words
"pacing": "fast|medium|slow"  // Don't leave the template unchanged
```

#### 2. **Boolean Fields** (true/false)
**Correct ✅:**
```json
"has_cta": true
"animated": false
```

**Wrong ❌:**
```json
"has_cta": "yes"  // Use true/false, not yes/no
"has_cta": 1  // Use true/false, not 1/0
```

#### 3. **Free Text Fields** (open-ended)
**Correct ✅:**
```json
"description": "Person holds product while speaking to camera"
"full_text": "Check out this amazing skincare routine! 🌟 Link in bio"
```

#### 4. **Array Fields** (lists)
**Correct ✅:**
```json
"texts": ["SALE", "50% OFF", "TODAY ONLY"]
"dominant_elements": ["person speaking", "text overlay"]
```

**Wrong ❌:**
```json
"texts": "SALE, 50% OFF"  // Must be array, not comma-separated string
"texts": [""]  // Don't include empty strings
```

#### 5. **Number Fields**
**Correct ✅:**
```json
"count": 3
"scene_count": 5
"duration": 30
```

**Wrong ❌:**
```json
"count": "three"  // Use numbers, not words
"count": "3"  // Use number type, not string
```

## How to Use This Template

### Step 1: Initial Setup (2 minutes)
1. Copy `golden_dataset_template.json` to a new file: `video_[ID]_validation.json`
2. Fill in basic video info (URL, duration, your name)
3. Copy the full caption and hashtags from TikTok

### Step 2: Watch Video Once - Take Notes (First Pass - 3 minutes)
Watch the video at normal speed and note:
- Overall structure (hook, main content, CTA)
- Obvious elements (people, products, text)
- Emotional tone changes
- Any standout moments

### Step 3: Detailed Observation (Second Pass - 10 minutes)
Watch with pause/rewind and fill in:

#### Timestamp Observations (Most Important!)
- Pick 5-7 key moments throughout the video
- Each timestamp is a SINGLE MOMENT (like a screenshot), not a range
- For each moment, note:
  - What appears on screen (people, objects, text, stickers)
  - What's being said
  - Camera angle and framing
  - Emotional tone

#### Required Observations per Timestamp:

**Correct Example ✅:**
```json
{
  "timestamp": "0:05",
  "description": "Person holds product up to camera",
  "elements_visible": {
    "people": {
      "count": 1,
      "positions": ["center"],
      "facing": "camera",
      "shot_type": "medium",
      "body_visible": "upper-body"
    },
    "objects": {
      "items": ["skincare bottle"],
      "positions": ["held in right hand"],
      "focus": "primary"
    },
    "text_overlays": {
      "texts": ["MY HOLY GRAIL", "⬇️ Link below"],
      "position": "top",
      "style": "animated",
      "size": "large",
      "colors": ["white", "yellow"]
    },
    "stickers": {
      "types": ["arrow"],
      "specific": ["pointing down arrow"],
      "animated": true
    }
  }
}
```

**Common Mistakes ❌:**
```json
{
  "timestamp": "0:05-0:10",  // Wrong: Don't use ranges here
  "facing": "camera|away",  // Wrong: Pick one option
  "position": "top/center",  // Wrong: Pick one, or use array if multiple
  "texts": "MY HOLY GRAIL, Link below",  // Wrong: Must be array
  "count": "one"  // Wrong: Use number 1
}
```

### Step 4: Section Analysis (5 minutes)
Divide video into 3-4 logical sections:
- **Hook** (first 3-5 seconds)
- **Main Content** (middle portion)
- **CTA/Outro** (last 5-10 seconds)

For each section, note:
- Number of scene changes (roughly)
- Pacing (fast/medium/slow)
- Dominant elements

### Step 5: Tracking Elements (5 minutes)

#### People Tracking
- Note when main person appears/disappears
- Track any additional people
- General position (center/left/right)

**Correct Example ✅:**
```json
"people_tracking": [
  {
    "person_id": "main_speaker",
    "appears_at": ["0:00-0:08", "0:12-0:20"],
    "total_screen_time": 16,
    "positions": ["center", "left"],
    "activities": ["speaking", "demonstrating"]
  }
]
```

**Wrong ❌:**
```json
"appears_at": "throughout video"  // Be specific with timestamps
"total_screen_time": "most of video"  // Use actual seconds
"positions": "center/left"  // Use array for multiple positions
```

#### Object Tracking
- Key products or props
- When they appear (rough timestamps)
- How prominent they are

**Correct Example ✅:**
```json
"object_tracking": [
  {
    "object": "iPhone",
    "appears_at": ["0:05-0:12", "0:20-0:22"],
    "total_screen_time": 9,
    "positions": ["hand", "table"],
    "prominence": "featured"
  }
]
```

#### Text/Sticker Persistence
- Which text stays on screen longest
- Recurring stickers or emojis

**Correct Example ✅:**
```json
"text_persistence": [
  {
    "text": "FOLLOW FOR MORE",
    "appears_at": ["0:25-0:30"],
    "duration": 5,
    "returns": false
  }
]
```

### Step 6: Speech Transcription (Optional but Valuable)
- Transcribe key phrases, especially:
  - Opening line
  - CTA phrases
  - Any repeated phrases
- Note speech patterns (fast/slow, enthusiastic/calm)

## What Each Field Validates

### For Creative Density Analysis:
- `timestamped_observations.elements_visible` - Validates element detection
- `visual_complexity.density_moments` - Validates density scoring
- `continuous_tracking.text_persistence` - Validates overlay duration

### For Emotional Journey:
- `emotional_progression.emotional_beats` - Validates emotion detection
- `timestamped_observations.emotional_indicators` - Validates expression analysis
- `speech_analysis.*.delivery` - Validates speech emotion

### For Person Framing:
- `timestamped_observations.people` - Validates person detection
- `continuous_tracking.people_tracking` - Validates tracking consistency
- Shot types and positions - Validates framing analysis

### For Scene Pacing:
- `scene_analysis.scene_changes` - Validates scene detection accuracy
- `scene_analysis.scene_rhythm` - Validates pacing calculations
- `section_analysis.*.scene_count` - Validates counting

### For Speech Analysis:
- `speech_analysis.speech_segments` - Validates transcription
- `speech_analysis.speech_patterns` - Validates pattern detection
- Speech coverage and speed - Validates timing analysis

### For Visual Overlay:
- `timestamped_observations.text_overlays` - Validates text detection
- `timestamped_observations.stickers` - Validates sticker recognition
- `visual_complexity.overlay_styles` - Validates style classification

### For Metadata Analysis:
- `metadata_observations` - Validates metadata extraction
- Caption and hashtag parsing - Validates text processing
- CTA detection - Validates call-to-action recognition

## Priority Fields (Must Fill)

### High Priority (Essential):
1. `timestamped_observations` - At least 5 moments
2. `metadata_observations.caption` - Full caption text
3. `metadata_observations.hashtags` - All hashtags
4. `section_analysis` - Basic structure
5. `continuous_tracking.people_tracking` - Main person at minimum

### Medium Priority (Valuable):
1. `speech_analysis.speech_segments` - Key phrases
2. `scene_analysis.scene_changes` - Rough count
3. `continuous_tracking.object_tracking` - Main products
4. `emotional_progression.emotional_beats` - Obvious emotions

### Low Priority (Nice to Have):
1. `quality_indicators` - Production quality
2. `special_techniques` - Advanced editing
3. `validation_notes` - Your confidence levels

## Common Field-Filling Examples

### Caption Length
```json
"caption_length": "medium"  // Correct: picked one option
"caption_length": "short|medium|long"  // Wrong: left template unchanged
```

### Hashtag Lists
```json
"list": ["#fyp", "#skincare", "#routine"]  // Correct: array of hashtags
"list": "#fyp #skincare #routine"  // Wrong: string instead of array
```

### Emotional Indicators
```json
"facial_expression": "smiling"  // Correct: picked one
"energy_level": "high"  // Correct: picked one
"facial_expression": "smiling/neutral"  // Wrong: used slash instead of picking one
```

### Scene Changes
```json
{"time": "0:03", "type": "cut", "from": "face", "to": "product"}  // Correct
{"time": "around 3 seconds", "type": "cut"}  // Wrong: use exact format "0:03"
```

### Speech Delivery
```json
"delivery": "enthusiastic"  // Correct: descriptive word
"speed": "fast"  // Correct: picked from fast|normal|slow
"speed": "very fast"  // Wrong: added modifier not in options
```

## Tips for Efficient Validation

1. **Use Timestamps**: Instead of describing "somewhere in the middle", use exact timestamps like "0:15"

2. **Be Specific with Text**: Write exact text when visible, even if briefly

3. **Count What You Can**: Even rough counts (5-7 scenes) are better than none

4. **Note Uncertainty**: Use the `validation_notes.uncertain_elements` field for things you're unsure about

5. **Focus on Obvious**: Don't stress about subtle details - obvious elements are most important

6. **Use Video Controls**: 
   - Pause at key moments
   - Use 0.5x speed for fast sections
   - Screenshot complex frames if needed

7. **When in Doubt**:
   - For choice fields: Pick the closest option
   - For arrays: Include only what you're sure about
   - For numbers: Estimate if needed, but note uncertainty

## Example Workflow

1. **First Watch** (2 min): Note structure and highlights
2. **Fill Metadata** (1 min): Caption, hashtags from description
3. **Timestamp Key Moments** (5 min): 5-7 important points
4. **Track People/Objects** (3 min): When they appear/disappear
5. **Note Speech** (3 min): Key phrases and delivery
6. **Scene Changes** (2 min): Rough count and types
7. **Review** (2 min): Check you have all priority fields

**Total Time: ~18 minutes per video**

## Validation Coverage

With this template filled, you'll validate:
- **Direct Observations**: 60% (what you explicitly noted)
- **Inferred Validations**: 25% (patterns between your observations)
- **Statistical Checks**: 15% (automatic consistency validation)
- **Total Coverage**: ~85-90% of system output

## Next Steps

1. Save your completed JSON file
2. Run validation: `python validate_golden.py video_001_validation.json rumiai_output.json`
3. Review the validation report
4. Iterate based on failures