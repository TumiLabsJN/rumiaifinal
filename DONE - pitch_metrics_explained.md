# Understanding avg_pitch and pitch_range

## avg_pitch (Average Fundamental Frequency)

### What It Is

**Definition**: The average fundamental frequency (F0) of voiced speech in Hertz (Hz) over a time window.

```python
avg_pitch = mean(all_pitch_values_in_window)
# Example: 185.3 Hz
```

### What It Measures Physically

- **Fundamental Frequency (F0)**: How fast the vocal cords vibrate per second
- **Higher pitch** = Vocal cords vibrate faster (more cycles/second)
- **Lower pitch** = Vocal cords vibrate slower (fewer cycles/second)

### Real-World Examples

| Speaker Type | Typical avg_pitch | Example |
|-------------|------------------|----------|
| Adult Male | 85-155 Hz | Deep voice, news anchor |
| Adult Female | 165-255 Hz | Higher voice, typical female speaker |
| Child | 250-400 Hz | High-pitched child voice |
| Excited speech | +20-40 Hz above baseline | "OH MY GOD! This is AMAZING!" |
| Sad/depressed | -10-20 Hz below baseline | "I don't know... maybe..." |
| Question ending | +30-50 Hz rise | "You want to try this?" ⬆️ |
| Statement ending | -20-30 Hz fall | "This works perfectly." ⬇️ |

### Musical Context

- 110 Hz = A2 (low male singing)
- 220 Hz = A3 (male speaking/female low)
- 440 Hz = A4 (female high/singing)

### What avg_pitch Tells Us

1. **Speaker Demographics**
   - Gender (with ~80% accuracy)
   - Approximate age range
   - Physical characteristics

2. **Emotional State**
   - Higher than normal = Excitement, happiness, stress
   - Lower than normal = Sadness, boredom, authority
   - Very high = Surprise, fear

3. **Speaking Style**
   - Consistent avg_pitch = Professional, controlled
   - Variable avg_pitch across segments = Dynamic, engaging

### Example in Our Windows

```json
{
  "hook": {"avg_pitch": 195.2},      // Higher - grabbing attention
  "middle_segment_1": {"avg_pitch": 175.3}, // Normal - explaining
  "middle_segment_2": {"avg_pitch": 180.1}, // Slightly higher - emphasizing
  "closing": {"avg_pitch": 205.5}    // Highest - call-to-action excitement
}
```

**Pattern**: Rising pitch trajectory = Building excitement

---

## pitch_range (Normalized Pitch Range)

### What It Is

**Definition**: The difference between highest and lowest pitch in a window, normalized by the average.

```python
pitch_range = max_pitch - min_pitch
pitch_range_normalized = pitch_range / avg_pitch
# Example: (220 - 150) / 185 = 0.38
```

### Why Normalized?

- Male range 100-150 Hz = 50 Hz spread
- Female range 180-270 Hz = 90 Hz spread
- Same **absolute** range difference, but:
  - Male normalized: 50/125 = 0.40 (40% variation)
  - Female normalized: 90/225 = 0.40 (same 40% variation!)

### What It Measures

**Intonation span**: How much the voice moves up and down

| pitch_range_norm | Interpretation | Speaking Style |
|-----------------|----------------|----------------|
| < 0.10 | Very flat | Monotone, robotic, depressed |
| 0.10-0.20 | Low variation | Calm, controlled, factual |
| 0.20-0.35 | Normal | Conversational, natural |
| 0.35-0.50 | High variation | Animated, expressive, selling |
| > 0.50 | Very high | Theatrical, exaggerated, acting |

### Real-World Examples

#### Example 1: Monotone Statement
```
"Green tea boosts metabolism."
Pitch: 170, 172, 175, 173, 171 Hz
Range: 5 Hz
Normalized: 5/172 = 0.029 (very flat)
```

#### Example 2: Excited Question
```
"Want to burn MORE calories?!"
Pitch: 160, 175, 195, 210, 230 Hz
Range: 70 Hz  
Normalized: 70/195 = 0.359 (high variation)
```

#### Example 3: Authoritative Statement
```
"Follow these three steps."
Pitch: 140, 138, 135, 132 Hz
Range: 8 Hz
Normalized: 8/136 = 0.059 (controlled, serious)
```

### What pitch_range Tells Us

1. **In 3-Second Windows**
   - High range = Question or exclamation
   - Low range = Statement or monotone
   - Medium range = Normal conversational

2. **Engagement Indicators**
   - Very low (<0.10) = Risk of boring delivery
   - Optimal (0.25-0.40) = Dynamic but natural
   - Too high (>0.50) = May seem fake or over-acted

3. **Content Type Detection**
   - Tutorial: Lower range (0.15-0.25)
   - Sales pitch: Higher range (0.30-0.45)
   - Emotional story: High range (0.35-0.50)

---

## Comparison: Why Both Metrics?

### Scenario Analysis

| Scenario | avg_pitch | pitch_range_norm | What It Means |
|----------|-----------|------------------|---------------|
| Bored male speaker | 95 Hz (low) | 0.08 (flat) | Disengaged, monotone |
| Bored female speaker | 175 Hz (normal) | 0.08 (flat) | Different pitch, same boredom |
| Excited male speaker | 125 Hz (high for male) | 0.35 (dynamic) | Engaged, animated |
| Calm female teacher | 185 Hz (normal) | 0.15 (controlled) | Clear, professional |
| Question by anyone | Any Hz | 0.30+ (rising) | Interrogative pattern |

### They Capture Different Things

**avg_pitch alone**:
- ✅ Tells us WHO (demographics)
- ✅ Tells us baseline emotion
- ❌ Doesn't tell us HOW dynamic

**pitch_range alone**:
- ✅ Tells us HOW expressive
- ✅ Tells us engagement level
- ❌ Doesn't tell us WHO or baseline

**Together**:
- Complete picture of voice characteristics
- Robust across different speakers
- Better ML features (complementary information)

---

## Practical Examples in TikTok Context

### High-Performing Hook Pattern
```json
{
  "hook": {
    "avg_pitch": 205,          // Higher than normal
    "pitch_range_norm": 0.42   // Very dynamic
  }
  // Translation: Excited, attention-grabbing opening
  // "OMG, you NEED to see this trick!"
}
```

### Low-Performing Hook Pattern
```json
{
  "hook": {
    "avg_pitch": 165,          // Low-normal
    "pitch_range_norm": 0.09   // Flat
  }
  // Translation: Monotone, boring opening
  // "Here's another metabolism tip."
}
```

### Effective CTA Pattern
```json
{
  "closing": {
    "avg_pitch": 198,          // Rising excitement
    "pitch_range_norm": 0.38   // Dynamic call-to-action
  }
  // Translation: Energetic closing
  // "Try this TODAY and see the difference!"
}
```

---

## Implementation Formulas

### avg_pitch Calculation
```python
def calculate_avg_pitch(pitch_frames, start_frame, end_frame):
    # Extract window
    window_pitches = pitch_frames[:, start_frame:end_frame]
    
    # Get maximum pitch per frame (most prominent)
    max_pitches = np.max(window_pitches, axis=0)
    
    # Filter only voiced frames (pitch > 80 Hz)
    voiced_pitches = max_pitches[max_pitches > 80]
    
    if len(voiced_pitches) > 0:
        return float(np.mean(voiced_pitches))
    else:
        return 0.0  # No voiced speech
```

### pitch_range_norm Calculation
```python
def calculate_pitch_range_norm(pitch_frames, start_frame, end_frame):
    # Get voiced pitches (same as above)
    window_pitches = pitch_frames[:, start_frame:end_frame]
    max_pitches = np.max(window_pitches, axis=0)
    voiced_pitches = max_pitches[max_pitches > 80]
    
    if len(voiced_pitches) > 10:  # Need minimum frames
        pitch_max = float(np.max(voiced_pitches))
        pitch_min = float(np.min(voiced_pitches))
        pitch_avg = float(np.mean(voiced_pitches))
        
        if pitch_avg > 0:
            return (pitch_max - pitch_min) / pitch_avg
    
    return 0.0  # Not enough voiced speech
```

---

## Why These Work for Short Windows

### avg_pitch in 3 seconds
- **Reliable**: Even 1 second of speech gives stable average
- **Meaningful**: Indicates speaker and emotional baseline
- **ML-friendly**: Continuous value, always interpretable

### pitch_range_norm in 3 seconds
- **Captures the ONE pattern**: Question vs statement
- **Stable**: Max and min are robust statistics
- **Normalized**: Comparable across speakers
- **Better than variance**: Not affected by single outliers

### Why NOT pitch_variance in 3 seconds
- Needs multiple intonation patterns (5+ seconds)
- Single emphasis word corrupts the calculation
- Measures statistical noise, not speaking style
- Research shows <5s variance is unreliable

---

## Summary

**avg_pitch**: WHO is speaking and their emotional baseline
- 185 Hz = Female or excited male
- Gender/age indicator
- Emotion baseline (happy/sad/neutral)

**pitch_range_norm**: HOW dynamically they're speaking
- 0.35 = Animated, engaging delivery
- Intonation span (monotone vs expressive)
- Question vs statement detector

**Together**: Complete voice profile that works even in 3-second windows, giving ML models robust features for engagement prediction.