# Golden Dataset V2 - Validation Testing Framework
## For Testing RumiAI's 10 Analysis Types with 10 Diverse Videos

### Target: 10 minutes per video for 10-video validation set

## Overview
This framework is designed to validate that RumiAI correctly analyzes videos across all 10 analysis types (including Temporal Markers, Feat Energy, and Librosa Energy). With only 10 videos, each should test different aspects of the system.

## Why This Approach Works for 10-Video Validation
- **Diagnostic Focus**: Each video tests specific system components
- **Comprehensive Coverage**: 10 diverse videos cover all analysis types
- **Reasonable Time**: 1.7 hours total (10 videos × 10 minutes)
- **Energy Validation**: Properly tests Feat and Librosa energy detection
- **Temporal Precision**: Validates opening hooks and closing retention
- **Quick Failure Detection**: Identifies which components need fixing

---

## Video Selection Strategy (10 Videos Total)

### Critical: Choose Diverse Videos That Test Different Components

#### Video Distribution:
1. **2 High-Energy Videos** (Rapid cuts, multiple elements, strong peaks)
   - Tests: Energy detection accuracy, peak identification, multimodal sync
   - Look for: Music-heavy content, dance, quick tutorials

2. **2 Tutorial/Educational** (Clear structure, steady pacing)
   - Tests: Speech analysis, section detection, text-overlay timing
   - Look for: How-to content, explanations, demonstrations

3. **2 Story/Narrative** (Emotional progression, varied pacing)
   - Tests: Emotional journey, temporal markers, engagement zones
   - Look for: Personal stories, transformations, reveals

4. **1 Product/Sales** (Strong CTA, urgency elements)
   - Tests: Closing retention, CTA detection, hook effectiveness
   - Look for: Product reviews, promotions, affiliate content

5. **1 Minimal/Artistic** (Few elements, deliberate pacing)
   - Tests: System handles sparse content, silence detection
   - Look for: Aesthetic content, single-take videos, minimal editing

6. **1 Text-Heavy** (Lots of overlays, complex visual information)
   - Tests: OCR accuracy, text persistence, reading flow
   - Look for: Tips videos, listicles, quote content

7. **1 Edge Case** (Unusual format)
   - Tests: System robustness, error handling
   - Look for: Slideshows, no speech, single frame, unusual aspect ratio

### Document Which Video Tests What:
Each video should note its primary test focus in the `test_focus` section of the template.

---

## Validation Workflow (10 minutes total)

### Phase 1: First Watch with Active Note-Taking (2 minutes)
**Watch at normal speed with finger on pause**

```json
{
  "initial_observations": {
    "immediate_impression": "High-energy product review with strong CTA",
    "video_type": "tutorial|review|story|entertainment|promotional|educational",
    "production_quality": "professional|good|amateur",
    "hook_strength": "strong|medium|weak",
    "first_3_seconds": {
      "grabbed_attention": true,
      "how": "Shocking question: 'Why is everyone wrong about skincare?'",
      "elements_shown": ["close-up face", "bold text", "dramatic music"]
    }
  }
}
```

### Phase 2: Detailed Timestamp Observations (4 minutes)
**5 key moments + energy tracking**

```json
{
  "timestamp_observations": [
    {
      "timestamp": "0:00",
      "section": "opening_hook",
      "energy_level": 8,  // 1-10 scale
      "elements": {
        "people": {"count": 1, "position": "center", "activity": "speaking directly"},
        "text": {"content": ["Why everyone's WRONG"], "style": "animated_pop"},
        "audio": {"music_intensity": "high", "speech": "questioning tone"}
      },
      "why_important": "Sets confrontational hook"
    },
    {
      "timestamp": "0:05",
      "section": "context_setup",
      "energy_level": 6,
      "elements": {
        "people": {"count": 1, "position": "left", "activity": "holding product"},
        "text": {"content": ["The Truth About"], "style": "static"},
        "objects": ["skincare bottle prominently shown"],
        "audio": {"music_intensity": "medium", "speech": "explanatory"}
      },
      "scene_change": true,
      "transition_type": "hard_cut"
    },
    {
      "timestamp": "0:12",
      "section": "peak_moment",
      "energy_level": 9,
      "elements": {
        "description": "Multiple elements converge - gesture, text, music peak",
        "people": {"gesture": "pointing at camera", "expression": "excited"},
        "text": {"content": ["THIS CHANGES EVERYTHING", "⚠️"], "style": "shake_effect"},
        "audio": {"music_intensity": "peak", "speech": "emphatic"}
      },
      "why_peak": "Maximum visual and audio stimulation"
    },
    {
      "timestamp": "0:20",
      "section": "value_delivery",
      "energy_level": 5,
      "elements": {
        "people": {"activity": "demonstrating product"},
        "text": {"content": ["Step 1", "Step 2"], "style": "bullet_points"},
        "audio": {"music_intensity": "low", "speech": "instructional"}
      }
    },
    {
      "timestamp": "0:27",
      "section": "closing_retention",
      "energy_level": 8,
      "elements": {
        "text": {"content": ["WAIT! Before you go", "50% OFF"], "style": "urgent_flash"},
        "audio": {"music_intensity": "rising", "speech": "urgent"},
        "cta_elements": ["follow button highlight", "link in bio arrow"]
      },
      "retention_hook": "Exclusive discount creates FOMO"
    }
  ]
}
```

### Phase 3: Energy Pattern Analysis (2 minutes)
**Track energy throughout video**

```json
{
  "energy_analysis": {
    "visual_energy": {
      "pattern": "high-low-high-medium-high",
      "checkpoints": [
        {"time": "0:00", "level": 8, "reason": "rapid cuts, motion"},
        {"time": "0:05", "level": 4, "reason": "static shot"},
        {"time": "0:10", "level": 9, "reason": "maximum elements"},
        {"time": "0:15", "level": 5, "reason": "steady demo"},
        {"time": "0:20", "level": 6, "reason": "moderate activity"},
        {"time": "0:25", "level": 8, "reason": "urgent CTA"}
      ],
      "highest_activity": "0:10-0:12",
      "lowest_activity": "0:05-0:07",
      "cut_rhythm": {
        "first_10_sec": 8,  // number of cuts
        "middle_10_sec": 4,
        "last_10_sec": 6
      }
    },
    "audio_energy": {
      "music_pattern": "crescendo-steady-peak-fade-crescendo",
      "checkpoints": [
        {"time": "0:00", "music": 7, "speech": 8},
        {"time": "0:10", "music": 9, "speech": 9},
        {"time": "0:20", "music": 4, "speech": 5},
        {"time": "0:28", "music": 8, "speech": 9}
      ],
      "speech_pacing": {
        "opening": "fast",
        "middle": "moderate",
        "closing": "fast"
      },
      "volume_dynamics": "variable",
      "has_silence_moments": true,
      "silence_timestamps": ["0:18-0:19"]
    },
    "multimodal_alignment": {
      "audio_visual_sync": [
        {"time": "0:00", "aligned": true, "description": "Music and cuts match"},
        {"time": "0:10", "aligned": true, "description": "Peak music with visual climax"},
        {"time": "0:20", "aligned": false, "description": "Calm music, but text appears"}
      ],
      "gesture_speech_sync": "high",
      "text_speech_alignment": "synchronized"
    }
  }
}
```

### Phase 4: Temporal Markers Validation (1 minute)
**Focus on engagement zones**

```json
{
  "temporal_markers": {
    "opening_analysis": {
      "first_5_seconds": {
        "hook_type": "controversy|question|shock|preview|story",
        "hook_effectiveness": 8,  // 1-10
        "immediate_value": "Challenges common belief",
        "visual_impact": ["face close-up", "bold text", "color contrast"],
        "audio_hook": ["dramatic music sting", "provocative question"],
        "cognitive_load": "high",  // low|medium|high
        "establishes_expectation": true
      }
    },
    "closing_analysis": {
      "last_15_percent_starts": "0:25",
      "retention_tactics": [
        "discount_offer",
        "next_video_tease",
        "urgency_creation"
      ],
      "energy_trend": "increasing",  // increasing|maintaining|decreasing
      "loop_probability": 6,  // 1-10
      "cta_clarity": 9,  // 1-10
      "final_frame_impact": "memorable"  // memorable|standard|weak
    },
    "engagement_zones": {
      "highest_engagement": ["0:00-0:03", "0:10-0:12", "0:27-0:30"],
      "potential_drop_offs": ["0:05-0:07", "0:18-0:20"],
      "surprise_moments": ["0:12"],
      "pattern_breaks": ["0:10", "0:25"]
    }
  }
}
```

### Phase 5: Detailed Element Tracking (1 minute)
**More precise than before, but still efficient**

```json
{
  "element_tracking": {
    "people": {
      "main_person_coverage": "75%",  // percentage of video
      "appearance_pattern": [[0, 8], [12, 20], [25, 30]],  // [start, end] ranges
      "framing_changes": 4,
      "dominant_framing": "medium_shot"
    },
    "text_overlays": {
      "total_unique": 12,
      "appearance_pattern": "front_loaded",  // front_loaded|even|back_loaded
      "longest_display": {"text": "50% OFF", "duration": 5},
      "text_density_score": 7  // 1-10
    },
    "objects": {
      "featured_products": ["skincare_bottle", "cotton_pad"],
      "screen_time": {"skincare_bottle": 15, "cotton_pad": 5},
      "prominence_pattern": "recurring"  // constant|recurring|single
    },
    "scene_changes": {
      "total_count": 12,
      "change_rate": 0.4,  // per second
      "longest_scene": {"start": 15, "end": 20, "duration": 5},
      "rhythm_consistency": "variable"  // consistent|variable|chaotic
    }
  }
}
```

### Phase 6: Speech & Audio Details (1 minute)
**Key phrases and patterns**

```json
{
  "speech_audio": {
    "speech_presence": "70%",  // percentage of video with speech
    "key_phrases": [
      {"time": "0:00", "text": "Why everyone's wrong about skincare"},
      {"time": "0:10", "text": "This changes everything"},
      {"time": "0:27", "text": "Get 50% off today only"}
    ],
    "speech_patterns": {
      "pace_variation": "high",  // low|medium|high
      "emphasis_technique": "volume_and_pause",
      "filler_usage": "minimal",  // none|minimal|moderate|heavy
      "clarity": 8  // 1-10
    },
    "music_characteristics": {
      "genre": "electronic_pop",
      "tempo_changes": 3,
      "builds_and_drops": true,
      "supports_narrative": true
    },
    "audio_effects": ["whoosh", "ding", "impact"],
    "effect_timestamps": ["0:03", "0:10", "0:25"]
  }
}
```

### Phase 7: Metadata & Summary (30 seconds)
**Quick final details**

```json
{
  "metadata_summary": {
    "caption_snippet": "First 100 chars of caption...",
    "hashtag_count": 8,
    "hashtag_strategy": "mix_broad_and_niche",
    "emoji_usage": "heavy",  // none|light|moderate|heavy
    "cta_in_caption": true,
    "link_mentioned": true
  },
  "validation_summary": {
    "video_purpose": "educate_and_sell",
    "target_audience": "skincare_enthusiasts",
    "viral_potential": 7,  // 1-10 based on patterns
    "production_effort": "medium",  // low|medium|high
    "confidence_in_observations": 8  // 1-10
  }
}
```

---

## Critical Validation Points for ML Training

### Must Capture Accurately (Non-Negotiable)
1. **Energy Levels** at 5+ timestamps (for energy model training)
2. **Opening Hook Type & Effectiveness** (for retention modeling)
3. **Peak Moment Identification** (for engagement modeling)
4. **Closing Retention Elements** (for completion rate modeling)
5. **Multimodal Alignment Points** (for synchronization detection)
6. **Scene Change Count** (±2 accuracy acceptable)
7. **Speech Coverage Percentage** (±10% accuracy acceptable)

### Should Capture (Important for ML)
1. **Text overlay frequency and timing**
2. **Person presence patterns**
3. **Audio-visual synchronization points**
4. **CTA clarity and placement**
5. **Energy pattern shape** (crescendo/steady/variable)

### Nice to Capture (Enhances ML)
1. **Exact transcription of key phrases**
2. **Specific gesture types**
3. **Color schemes and visual styles**
4. **Production techniques used**

---

## Validation Efficiency Tips

### Keyboard Shortcuts for Quick Marking
- Use video player with frame-by-frame controls
- Keep template open in split screen
- Use timestamps from player directly
- Mark energy levels while watching (don't rewatch for this)

### What NOT to Obsess Over
- Exact frame counts
- Perfect transcription of all speech
- Minor background elements
- Subjective quality assessments
- Frame-perfect transition timing

### What TO Focus On
- **Energy changes** - When does it peak/valley?
- **Attention moments** - What grabs you?
- **Synchronization** - When do audio/visual align?
- **Hooks and retention** - What makes you keep watching?
- **Pattern breaks** - What surprises you?

---

## Expected Outputs for ML Training

### Per-Video Validation Data
- **50-70 data points** per video (vs 150+ in old format, 25 in minimal)
- **10 energy measurements** (visual + audio at 5+ points)
- **5 timestamp observations** with rich context
- **3-5 engagement zones** identified
- **15-20 categorical classifications**
- **10-15 numerical measurements**

### Validation Set Statistics (10 videos)
- **500-700 total validation points**
- **100 energy measurements** across diverse video types
- **50 peak moments** for pattern validation
- **30+ hook analyses** for retention testing
- **30+ closing patterns** for completion testing
- **Coverage of all 10 analysis types**

---

## Quality Assurance Checklist

Before submitting validation:

- [ ] **5 timestamps** with energy levels recorded
- [ ] **Opening hook** type and effectiveness scored
- [ ] **Peak moment** identified with reason
- [ ] **Closing retention** elements documented
- [ ] **Audio-visual sync** noted at 3+ points
- [ ] **Scene change count** estimated (rough is OK)
- [ ] **Speech coverage** percentage estimated
- [ ] **Energy pattern** shape identified
- [ ] **Confidence score** self-assessed

---

## Time Management Guide

| Task | Time | Critical? |
|------|------|-----------|
| First watch + notes | 2 min | Yes |
| 5 timestamp observations | 4 min | Yes |
| Energy pattern tracking | 2 min | Yes |
| Temporal markers | 1 min | Yes |
| Element tracking | 1 min | Should |
| Speech/audio details | 1 min | Should |
| Metadata/summary | 30 sec | Nice |

**Total: 10-11 minutes**

If running behind, prioritize:
1. Energy measurements (MUST HAVE)
2. Temporal markers (MUST HAVE)
3. Peak moment identification (MUST HAVE)
4. Everything else (NICE TO HAVE)

---

## Quick Validation Mode (For 10-Video Set)

### After Creating Your Golden Data, Run These Quick Checks:

#### Binary Pass/Fail Checks:
- [ ] Person detected when visible?
- [ ] Text overlays detected?
- [ ] Speech transcribed when present?
- [ ] Scene count within ±3 of your count?
- [ ] Peak moment identified correctly (±2 seconds)?
- [ ] Opening hook type matches?
- [ ] Closing retention elements found?
- [ ] Energy pattern shape matches (crescendo/steady/variable)?

#### Major Failure Detection:
Document any complete failures:
- "No text detected despite heavy overlays"
- "Peak moment completely wrong section"
- "Energy pattern inverted"
- "No speech detected despite clear talking"

### Expected Accuracy Thresholds:
- **ML Detection (person/text/speech)**: >90% accuracy
- **Scene Count**: Within ±3 scenes
- **Peak Moment Timing**: Within ±2 seconds
- **Energy Pattern**: Correct general shape
- **Temporal Markers**: Correct hook/retention identification

---

## Validation File Naming

```
golden_v2_[video_type]_[test_focus]_[video_id].json
```

Examples: 
- `golden_v2_highenergy_peaks_7428757192624311594.json`
- `golden_v2_tutorial_speech_8521963074185296.json`
- `golden_v2_edgecase_silent_9632587410369852.json`

This helps identify what each video is testing.