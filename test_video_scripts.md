# Test Video Scripts for RumiAI Validation
## 10 Controlled Videos for Component Testing

---

## Video 1: HIGH ENERGY - Peak Detection Test
**Duration**: 30 seconds  
**Primary Test**: Energy peaks, multimodal synchronization  
**Filename**: `highenergy_peaks_test.mp4`

### Script & Actions:

| Time | Visual | Audio/Speech | Text Overlay | Energy Level |
|------|--------|--------------|--------------|--------------|
| 0:00-0:03 | Face close-up, excited expression | "WAIT WAIT WAIT!" (loud) + Music starts | "STOP SCROLLING" (shaking) | 9/10 |
| 0:03-0:05 | Pull back to medium shot | "You need to see this" (normal) | (none) | 5/10 |
| 0:05-0:08 | Quick cuts between 3 angles | "One... Two... Three..." | Numbers appear with each | 7/10 |
| 0:08-0:12 | **PEAK: Jump + effects** | "THIS IS IT!" + Music drop | "🔥🔥🔥 PEAK MOMENT" (flashing) | 10/10 |
| 0:12-0:15 | Slow, calm shot | "Now let me explain..." (quiet) | (none) | 3/10 |
| 0:15-0:20 | Building energy, lean forward | "It gets better..." (building) | "Watch this..." (fade in) | 6/10 |
| 0:20-0:23 | Second peak - clap hands | "BOOM!" + Sound effect | "SECOND PEAK" | 9/10 |
| 0:23-0:27 | Medium energy dancing | "Almost done but wait" | "One more thing" | 6/10 |
| 0:27-0:30 | Final burst + point at camera | "FOLLOW NOW!" | "⬇️ FOLLOW ⬇️" (pulsing) | 8/10 |

**Key Validation Points**:
- Peak at 0:08-0:12 should be detected
- Second peak at 0:20-0:23
- Energy dip at 0:12-0:15
- Multimodal sync at peaks (gesture + music + text)

---

## Video 2: HIGH ENERGY - Rapid Cuts Test  
**Duration**: 30 seconds  
**Primary Test**: Scene detection, cut rhythm  
**Filename**: `highenergy_cuts_test.mp4`

### Script & Actions:

| Time | Action | Cut Count |
|------|--------|-----------|
| 0:00-0:10 | Change location/angle every second (10 cuts) | 10 |
| 0:10-0:20 | Change every 2 seconds (5 cuts) | 5 |
| 0:20-0:30 | Change every 3-4 seconds (3 cuts) | 3 |

**Total**: 18 scene changes (system should detect 15-21)

---

## Video 3: TUTORIAL - Clear Speech Test
**Duration**: 30 seconds  
**Primary Test**: Speech transcription, text-speech alignment  
**Filename**: `tutorial_speech_test.mp4`

### Script & Actions:

| Time | Speech | Text Overlay | Notes |
|------|--------|--------------|-------|
| 0:00-0:03 | "Three tips to test speech recognition" | "3 TIPS" appears | Clear speech |
| 0:03-0:05 | (SILENCE) | "Tip #1" appears | Test silence detection |
| 0:05-0:10 | "First, speak very clearly" | "Speak clearly" appears at 0:07 | Test sync |
| 0:10-0:15 | "Second, vary your speaking speed fast then slow" | "Vary speed" | Speed change |
| 0:15-0:18 | "Third whisper section" (whispered) | "Whisper" | Volume test |
| 0:18-0:20 | (SILENCE) | "..." | Another silence |
| 0:20-0:25 | "Now normal voice with some uh filler words" | (none) | Filler test |
| 0:25-0:30 | "Follow for more tips and tricks okay?" | "FOLLOW" at 0:26 | CTA |

**Key Validation Points**:
- Speech coverage: ~73% (22/30 seconds)
- Silence periods: 0:03-0:05, 0:18-0:20
- Text-speech alignment at 0:07 and 0:26
- Speed variation detected
- Whisper section transcribed

---

## Video 4: TUTORIAL - Structure Test
**Duration**: 30 seconds  
**Primary Test**: Section detection, emotional journey  
**Filename**: `tutorial_structure_test.mp4`

### Script & Actions:

| Time | Section | Emotion | Visual | Speech |
|------|---------|---------|--------|--------|
| 0:00-0:05 | Hook | Curious/Excited | Face, question pose | "Want to learn something cool?" |
| 0:05-0:10 | Problem | Concerned | Frown, show issue | "Most people struggle with this" |
| 0:10-0:20 | Solution | Happy/Confident | Smile, demonstrate | "But here's the secret solution" |
| 0:20-0:25 | Result | Proud | Show outcome | "Look at this amazing result" |
| 0:25-0:30 | CTA | Urgent | Point at camera | "Follow before it's too late" |

**Emotional Journey**: Curious → Concerned → Happy → Proud → Urgent

---

## Video 5: STORY - Emotional Progression Test
**Duration**: 30 seconds  
**Primary Test**: Emotional journey, engagement zones  
**Filename**: `story_emotion_test.mp4`

### Script & Actions:

| Time | Emotion | Expression | Speech | Music |
|------|---------|------------|--------|-------|
| 0:00-0:05 | Sad | Look down, slow | "I used to feel so lost" | Sad piano |
| 0:05-0:10 | Hopeful | Look up slightly | "Then something changed" | Building |
| 0:10-0:15 | Excited | Smile growing | "I discovered this method" | Upbeat |
| 0:15-0:20 | Joy | Big smile, energetic | "It transformed everything!" | Happy |
| 0:20-0:25 | Grateful | Hand on heart | "I'm so thankful now" | Warm |
| 0:25-0:30 | Inspiring | Direct to camera | "You can do this too" | Motivational |

**Emotional Arc**: Sad → Hopeful → Excited → Joy → Grateful → Inspiring

---

## Video 6: STORY - Narrative Pacing Test
**Duration**: 30 seconds  
**Primary Test**: Temporal markers, retention  
**Filename**: `story_pacing_test.mp4`

### Script & Actions:

| Time | Story Beat | Hook/Retention Element |
|------|------------|------------------------|
| 0:00-0:03 | **Strong Hook** | "The day that changed my life" + dramatic pause |
| 0:03-0:08 | Setup | "It was a normal Tuesday..." |
| 0:08-0:15 | Build tension | "But then... something unexpected" |
| 0:15-0:20 | Climax | "I couldn't believe what happened" |
| 0:20-0:25 | Resolution | "That's when I realized..." |
| 0:25-0:30 | **Loop/Retention** | "Watch again to catch what you missed" + replay hint |

---

## Video 7: PRODUCT - CTA Heavy Test
**Duration**: 30 seconds  
**Primary Test**: CTA detection, closing retention  
**Filename**: `product_cta_test.mp4`

### Script & Actions:

| Time | Visual | Speech | Text Overlay | CTA Element |
|------|--------|--------|--------------|-------------|
| 0:00-0:05 | Product in hand | "This changed my life" | "LIFE CHANGING" | Curiosity |
| 0:05-0:10 | Demo product | "Let me show you how" | "WATCH THIS" | Demo |
| 0:10-0:15 | Show benefits | "Three amazing benefits" | "1. 2. 3." | Value |
| 0:15-0:20 | Urgency creation | "Limited time only" | "24 HOURS LEFT" | Urgency |
| 0:20-0:25 | Price reveal | "50% off today" | "50% OFF - CODE: TEST" | Offer |
| 0:25-0:30 | Final CTA | "Click the link now!" | "LINK IN BIO ⬇️" (flashing) | Action |

**CTA Escalation**: Curiosity → Demo → Value → Urgency → Offer → Action

---

## Video 8: MINIMAL - Sparse Content Test
**Duration**: 30 seconds  
**Primary Test**: System handles minimal elements  
**Filename**: `minimal_sparse_test.mp4`

### Script & Actions:

| Time | Visual | Audio | Text | Notes |
|------|--------|-------|------|-------|
| 0:00-0:10 | Single static shot, no person | Ambient music only | (none) | Test minimal |
| 0:10-0:15 | Slow pan, still no person | Music continues | "minimalism" (small) | One text |
| 0:15-0:20 | Person enters frame slowly | (No speech) | (none) | Person appears |
| 0:20-0:25 | Person makes one gesture | Single word: "Peace" | (none) | Minimal speech |
| 0:25-0:30 | Fade to black slowly | Music fades | "." | Almost nothing |

**Key Validation**: System should handle low energy without crashing

---

## Video 9: TEXT HEAVY - OCR Overload Test
**Duration**: 30 seconds  
**Primary Test**: OCR accuracy, text persistence  
**Filename**: `textheavy_ocr_test.mp4`

### Script & Actions:

| Time | Text Overlays | Position | Animation | Speech |
|------|---------------|----------|-----------|--------|
| 0:00-0:05 | "TEST 1" "TEST 2" "TEST 3" | Top, Middle, Bottom | Static | "Multiple texts" |
| 0:05-0:10 | "MOVING TEXT →" | Sliding across | Moving | "Moving text here" |
| 0:10-0:15 | "🔥 EMOJIS 😍 WORK 💯" | Center | Pop-in | "Emojis too" |
| 0:15-0:20 | "Small text" (tiny) + "BIG TEXT" | Various | Mixed | "Different sizes" |
| 0:20-0:25 | "PERSISTENT" (stays whole time) | Bottom | Static | "This stays" |
| 0:25-0:30 | 10 quick texts flashing | All over | Rapid | "Information overload" |

**Text Count Target**: 20+ unique text elements

---

## Video 10: EDGE CASE - Break the System Test
**Duration**: 30 seconds  
**Primary Test**: Robustness, error handling  
**Filename**: `edgecase_unusual_test.mp4`

### Unusual Elements:

| Time | Weird Thing | Why It's Testing |
|------|-------------|------------------|
| 0:00-0:05 | Completely black screen | No visual input |
| 0:05-0:08 | Strobe light effect | Rapid brightness changes |
| 0:08-0:10 | Upside down camera | Orientation detection |
| 0:10-0:15 | Multiple people talking over each other | Speech confusion |
| 0:15-0:18 | Extreme slow motion | Frame rate handling |
| 0:18-0:20 | Glitch effects/corrupted frames | Error resistance |
| 0:20-0:25 | No audio at all | Silent section handling |
| 0:25-0:30 | All previous effects combined | Maximum chaos |

---

## Production Tips for Easy Filming:

### Equipment Needed:
- Phone camera (vertical orientation)
- Tripod or phone stand (optional but helpful)
- Good lighting (window light works)
- Quiet room for clear audio

### Filming Order (Easiest to Hardest):
1. **Start with Minimal** - Just atmosphere shots
2. **Then Tutorials** - Simple talking head
3. **Then Story** - Emotional changes
4. **Then Product** - Props needed
5. **Then Text Heavy** - Post-production editing
6. **End with High Energy** - Most exhausting
7. **Edge Case** - Requires video effects

### Editing Apps for Effects:
- **CapCut** (Free) - Good for text overlays, cuts, effects
- **InShot** - Simple and effective
- **TikTok's built-in editor** - Has all needed effects

### Text Overlay Consistency:
- Use same font for each video (for easier OCR validation)
- High contrast (white text, black outline)
- Keep text on screen for at least 2 seconds
- Position consistently (top/middle/bottom)

### Audio Tips:
- Record speech clearly - speak toward camera
- Add music in post (TikTok's library is fine)
- For "silence" sections, actually have NO audio
- For peak moments, layer music + speech + effects

### Validation Helpers:
- Say timestamps out loud during filming ("This is the 10-second mark")
- Use finger counting for scene changes
- Exaggerate emotions for clarity
- Make CTAs super obvious

### File Organization:
After uploading to TikTok and downloading:
```
test_videos/
├── video_01_highenergy_peaks.mp4
├── video_02_highenergy_cuts.mp4
├── video_03_tutorial_speech.mp4
├── video_04_tutorial_structure.mp4
├── video_05_story_emotion.mp4
├── video_06_story_pacing.mp4
├── video_07_product_cta.mp4
├── video_08_minimal_sparse.mp4
├── video_09_textheavy_ocr.mp4
└── video_10_edgecase_unusual.mp4
```

---

## Quick Reference - What Each Video Tests:

| # | Type | Primary Test | Key Metric |
|---|------|--------------|------------|
| 1 | High Energy | Peak detection | Peaks at 0:08, 0:20 |
| 2 | High Energy | Scene detection | 18 cuts total |
| 3 | Tutorial | Speech transcription | 73% coverage |
| 4 | Tutorial | Structure | 5 clear sections |
| 5 | Story | Emotional journey | 6 emotions |
| 6 | Story | Temporal markers | Strong hook & loop |
| 7 | Product | CTA detection | 6 CTA elements |
| 8 | Minimal | Sparse content | Low energy handling |
| 9 | Text Heavy | OCR | 20+ text elements |
| 10 | Edge Case | Robustness | Doesn't crash |

This gives you exactly what to film, when to do what, and what each video should validate!