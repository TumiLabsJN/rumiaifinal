# Pitch Metrics → Actionable Creator Insights

## The Translation Challenge

ML discovers: `hook.avg_pitch = 198.5, hook.pitch_range_norm = 0.38`
Creator needs: "Start with an excited question to grab attention"

---

## Pattern Discovery → Creator Playbook

### 1. Hook Patterns (0-3 seconds)

#### Pattern A: "The Attention Grabber"
**ML Detects**: 
- `avg_pitch`: 15-25% above creator's baseline
- `pitch_range_norm`: > 0.35

**Creator Insight**:
"Start with a HIGH-ENERGY QUESTION that rises in pitch"

**Actionable Instructions**:
- ✅ "Want to know the SECRET to...?" (rising intonation)
- ✅ "You'll NEVER believe what happened!" (emphasis + rise)
- ❌ "Here's how to..." (flat, statement)
- ❌ "In this video..." (low energy start)

**Why it works**: The brain processes rising pitch as "incomplete" information, creating curiosity gap

---

#### Pattern B: "The Authority Hook" 
**ML Detects**:
- `avg_pitch`: 10-20% below baseline
- `pitch_range_norm`: < 0.15

**Creator Insight**: 
"Start with a DEEP, CONFIDENT STATEMENT"

**Actionable Instructions**:
- ✅ "Stop doing this immediately." (low, flat, commanding)
- ✅ "Three people died last week from..." (serious, low)
- ❌ High-pitched excitement for serious topics

**Why it works**: Lower pitch signals authority and importance

---

### 2. Middle Segment Patterns

#### Pattern C: "The Engagement Rollercoaster"
**ML Detects**:
- Alternating segments: High/Low/High pitch variance
- `pitch_range_norm`: Varies 0.15 → 0.35 → 0.20

**Creator Insight**:
"Alternate between EXPLANATION and EMPHASIS every 7-8 seconds"

**Actionable Script Structure**:
```
Segment 1 (3-10s): Calm explanation (lower pitch, steady)
"Green tea contains catechins that..."

Segment 2 (10-18s): Exciting result (higher pitch, dynamic)
"This means you burn 17% MORE calories!"

Segment 3 (18-26s): Return to teaching (normal pitch)
"The best time to drink it is..."
```

**Creator Checklist**:
- [ ] Every 7-8 seconds, change your energy level
- [ ] Use pitch peaks on KEY BENEFITS
- [ ] Return to baseline for explanations
- [ ] Never stay monotone for >10 seconds

---

#### Pattern D: "The Building Crescendo"
**ML Detects**:
- Progressive increase: 165 Hz → 175 Hz → 185 Hz → 195 Hz
- Each segment 5-10 Hz higher

**Creator Insight**:
"Build excitement gradually toward your CTA"

**How to execute**:
1. Start calm and informative
2. Add enthusiasm with each point
3. Peak energy at the revelation
4. Maximum excitement at CTA

---

### 3. Closing/CTA Patterns (Last 3 seconds)

#### Pattern E: "The Urgency Close"
**ML Detects**:
- `avg_pitch`: 20-30% above video average
- `pitch_range_norm`: > 0.40

**Creator Insight**:
"End with RISING EXCITEMENT and a question"

**Winning CTAs**:
- ✅ "Try this TODAY and tell me what happens!" (rising)
- ✅ "Which one will YOU try first?" (question up-pitch)
- ❌ "Thanks for watching." (falling, low energy)
- ❌ "See you next time." (flat, no urgency)

---

## Viral Video Formulas Discovered

### Formula 1: "The Shock Hook"
```
Hook: Very high pitch + high range (surprise/shock)
Middle: Normal pitch, moderate range (explanation)
Close: Rising pitch + question (engagement prompt)
```
**Use for**: Revealing surprising facts, debunking myths

### Formula 2: "The Teacher"
```
Hook: Moderate pitch + rising (gentle question)
Middle: Steady pitch, low range (clear teaching)
Close: Higher pitch + encouragement (motivation)
```
**Use for**: Tutorials, how-to content, education

### Formula 3: "The Storyteller"
```
Hook: Low pitch + narrow range (serious/mysterious)
Middle: Building pitch + increasing range (tension)
Close: High pitch + wide range (emotional payoff)
```
**Use for**: Personal stories, transformations

---

## Specific Actionable Insights

### For Energy/Enthusiasm Videos

**If ML finds top videos have:**
- Hook `avg_pitch` > 190 Hz (female) or > 140 Hz (male)
- Hook `pitch_range_norm` > 0.35

**Tell creators:**
1. "Start like you just won the lottery"
2. "Your first 3 words should POP with enthusiasm"
3. "End every sentence in the hook going UP"
4. "Imagine you're telling your best friend incredible news"

### For Authority/Trust Videos

**If ML finds top videos have:**
- Hook `avg_pitch` < 160 Hz (female) or < 110 Hz (male)  
- Middle segments `pitch_range_norm` < 0.20

**Tell creators:**
1. "Channel your inner news anchor"
2. "Speak like you're delivering serious news"
3. "Keep your voice steady and controlled"
4. "Lower your voice 10% from normal"

### For Question-Driven Engagement

**If ML finds pattern:**
- Segments ending with pitch rise > 15%
- `pitch_range_norm` spikes at segment ends

**Tell creators:**
1. "End each segment with a question"
2. "Your voice should go UP at the end"
3. "Make viewers mentally answer before continuing"
4. "Think: Statement → Statement → Question?"

---

## Platform-Specific Patterns

### TikTok Success Pattern
**ML Discovery**: 
- Successful videos show 25% higher pitch variance in first 3s
- avg_pitch correlates with younger demographic engagement

**Creator Translation**:
- "Be 2x more animated in your hook than feels natural"
- "If it feels over-the-top, it's probably perfect"
- "Match the energy of your target age group"

### YouTube Shorts Pattern
**ML Discovery**:
- Lower pitch variance but consistent energy
- Gradual pitch build toward CTA

**Creator Translation**:
- "Start conversational, build to excitement"
- "Save your highest energy for the final CTA"

---

## The Insights Dashboard

### What Creators Would See:

```
🎯 YOUR VIDEO ANALYSIS

❌ Problem Detected: Monotone Hook
Your hook pitch variation: 0.08 (flat)
Top performers: 0.35+ (dynamic)

💡 Fix: Add EMPHASIS and QUESTIONS
Instead of: "Here's how to boost metabolism"
Try: "Want to BURN more calories doing NOTHING?"

✅ Strength Found: Building Energy
Your pitch increased 15% from start to end
This matches viral video patterns!

📊 Benchmark Comparison:
Your avg energy: ████████░░ 75%
Your dynamics:  ███░░░░░░░ 30% (needs work)
Your CTA power: █████████░ 90%

🎬 QUICK FIXES:
1. Re-record your hook with a QUESTION
2. Add emphasis on benefit words (MORE, NEVER, SECRET)
3. Your middle segments are too similar - vary energy
```

---

## ROI for Creators

### Before (No Pitch Metrics):
- "Your video has low engagement"
- "Try being more energetic"
- Generic, non-actionable feedback

### After (With Pitch Metrics):
- "Your hook is 35% flatter than viral videos"
- "Successful creators rise 40 Hz on questions"
- "Your segment 3 energy dip loses viewers - keep it above 180 Hz"
- Specific, measurable, actionable

---

## Implementation Priority

### Must Have (MVP):
- `avg_pitch` per window → "Energy level tracking"
- Simple high/medium/low energy feedback

### Nice to Have (V2):
- `pitch_range_norm` → "Dynamism score"
- Question detection
- Pattern matching to successful formulas

### Future (V3):
- Real-time recording feedback
- "Your pitch is dropping - add energy!"
- Auto-script optimization suggestions

---

## Summary: Why These Metrics Matter

**Without pitch metrics**: "Be more engaging" (vague)

**With pitch metrics**: 
- "Start 20% higher than your normal voice"
- "End each segment with rising intonation"
- "Your monotone middle lost 45% of viewers at 12s"
- "Match this successful pattern: High → Low → High"

These features transform abstract "engagement" into concrete, replicable vocal techniques that creators can immediately implement.