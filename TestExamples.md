
# Tests D-Day

## Video Allocation Summary

### 11 Golden Dataset Videos:
- **Video 01: Speaker & Gaze** - Tests 10, 21-23, 39-40 (person_count, speech, gaze)
- **Video 02: Emotions Basic** - Tests 14-16 (expression_count, emotion_consistency, dominant_emotion)
- **Video 02B: Emotional Valence (50s)** - Tests 17-18 (emotional_valence with mixed emotions)
- **Video 03: Text & Captions** - Tests 1-6 (overlays, captions)
- **Video 04: Scenes & Density** - Tests 19-20, 35-38 (scene changes, density)
- **Video 05: Objects & Gestures** - Tests 7-9, 13 (object_count, gestures)
- **Video 06: Audio Energy** - Tests 24-26, 29-30, 47 (energy, pitch variations, TikTok music)
- **Video 07: Person Framing** - Tests 31-34 (face sizes)
- **Video 08: Gender Male** - Tests 27, 44-46 (male pitch, gender detection)
- **Video 09: Gender Female** - Tests 28, 41-43 (female pitch, gender detection)
- **Video 10: Two People** - Tests 11-12 (multiple person counting)

## Table

| To Test | Category       | Feature to Test                | Method                                                                 | What it Tests                              | Comments                               | Difficulty | Video ID |
|---------|----------------|--------------------------------|------------------------------------------------------------------------|--------------------------------------------|----------------------------------------|------------|----------|
| 1       | Text Overlay   | overlay_unique_count           | Use 1, 3, 5 in same Window                                             | Accelerated Overlays                       |                                        | Easy       | Video 03 |
| 2       | Text Overlay   | overlay_unique_count           | Input in different parts of screen                                     | Overlays in varied parts of screen         |                                        |            | Video 03 |
| 3       | Text Overlay   | overlay_unique_count           | Use emojis                                                             | Emojis as Overlays                         |                                        |            | Video 03 |
| 4       | Text Overlay   | overlay_unique_count           | Use Captions                                                           | Ensure captions are not counted as Overlay |                                        |            | Video 03 |
| 5       | Text Overlay   | overlay_coverage / persistence | Use overlays for 25%, 50%, 75% of window                               | If Coverage works                          |                                        |            | Video 03 |
| 6       | Text Overlay   | has_captions                   | Use Captions                                                           | Captions work                              |                                        | Easy       | Video 03 |
| 7       | Object         | object_count                   | Use 1, 3, 6                                                            | Object Count                               |                                        |            | Video 05 |
| 8       | Object         | object_count                   | Bring same object twice in same window                                 | Ensure it is not double counting objects   |                                        |            | Video 05 |
| 9       | Object         | object_count                   | Bring same object twice in multiple windows                            | ID relationship of Object between windows  |                                        |            | Video 05 |
| 10      | Object         | person_count                   | Film with 1 person                                                     | Count of single person                     |                                        | Easy       | Video 01 |
| 11      | Object         | person_count                   | Film with 2 people                                                     | Count of 2 people                          |                                        | Easy       | Video 10 |
| 12      | Object         | person_count                   | Make 1 person appear in one frame and disappear in other, within window| How a person is counted                    |                                        |            | Video 10 |
| 13      | Gesture        | gesture_count                  | Make 1, 3, 5 gestures                                                  | Count of Gestures                          |                                        | Easy       | Video 05 |
| 14      | Emotion        | expression_count               | Use 3, 5                                                               | Count of expressions                       | Sad → Happy → Angry → Sad (Count of 4) | Easy       | Video 02 |
| 15      | Emotion        | emotion_consistency            | Happy 70% then angry/sad                                               | Consistency of Emotion                     | Same test as feature before/after       |            | Video 02 |
| 16      | Emotion        | dominant_emotion_id            | Happy 70% then angry/sad                                               | If dominant emotion works                  |                                        |            | Video 02 |
| 17      | Emotion        | emotional_valence              | Happy 40% → negative (angry, sad, fear, disgust)                       |                                            |                                        |            | Video 02B |
| 18      | Emotion        | emotional_valence              | Happy 70% → negative                                                   |                                            |                                        |            | Video 02B |
| 19      | Scene          | scene_count                    | Make 1, 3, 5 scenes                                                    | Count of Scenes                            |                                        | Easy       | Video 04 |
| 20      | Scene          | scene_duration_variance        | Make 1, 3, 5 scenes                                                    | Variance of scenes                         | Same test as feature before             |            | Video 04 |
| 21      | Speech         | word_count                     | Video with loud music while speaking                                   | If transcript separates from music         |                                        | Easy       | Video 01 |
| 22      | Speech         | word_count                     | Say 5, 10, 15, 25 words per window                                     | Word Count working well                    |                                        | Easy       | Video 01 |
| 23      | Speech         | speech_coverage                | Say 5, 10, 15, 25 words per window                                     | Speech coverage working                    | Same test as feature before             |            | Video 01 |
| 24      | Audio          | energy_level                   | Speak with a lot of energy                                             | Energy working                             |                                        | Easy       | Video 06 |
| 25      | Audio          | energy_variance                | Speak with high energy and low energy                                  | Energy Variance working                    |                                        | Easy       | Video 06 |
| 26      | Audio          | energy_max                     | Yell in specific parts of video                                        | If energy max works                        |                                        |            | Video 06 |
| 27      | Audio          | avg_pitch_normalized           | You speak (Male)                                                       |                                            |                                        |            | Video 08 |
| 28      | Audio          | avg_pitch_normalized           | Nadia speaks (Female)                                                  |                                            |                                        |            | Video 09 |
| 29      | Audio          | pitch_range_norm               | Fluctuate from high to low energy                                      |                                            |                                        |            | Video 06 |
| 30      | Audio          | pitch_range_norm               | Stay monotone                                                          |                                            |                                        |            | Video 06 |
| 31      | Person Framing | average_face_size              | Face close to screen                                                   | Framing working                            |                                        | Easy       | Video 07 |
| 32      | Person Framing | average_face_size              | Face middle to screen                                                  | Framing working                            |                                        | Easy       | Video 07 |
| 33      | Person Framing | average_face_size              | Face far from screen                                                   | Framing working                            |                                        | Easy       | Video 07 |
| 34      | Person Framing | average_face_size              | Face very far from screen                                              | Framing working                            |                                        | Easy       | Video 07 |
| 35      | Calculation    | max_density                    | Calculation valid                                                      | Calculation Works                          |                                        | Easy       | Video 04 |
| 36      | Calculation    | min_density                    | Calculation valid                                                      | Calculation Works                          |                                        | Easy       | Video 04 |
| 37      | Scene          | shortest_scene                 | Time a short scene within a window                                     | Short scene being calculated well          |                                        | Easy       | Video 04 |
| 38      | Scene          | longest_scene                  | Time a long scene within a window                                      | Longest scene being calculated well        |                                        | Easy       | Video 04 |
| 39      | Gaze           | gaze_variance                  | Look at different things in video                                      | Gaze OK                                    |                                        | Easy       | Video 01 |
| 40      | Gaze           | eye_contact_rate               | Look at different things in video                                      | Eye contact rate OK                        | Same test as feature before             | Easy       | Video 01 |
| 41 | Gender | Gender | Have a window with a woman, in close range | Female testing | Gender in close range | Easy | Video 09 |
| 42 | Gender | Gender | Have a window with a woman, in medium range | Female testing | Gender in medium range | Easy | Video 09 |
| 43 | Gender | Gender | Have a window with a woman, in far range | Female testing | Gender in far range | Easy | Video 09 |
| 44 | Gender | Gender | Have a window with a man, in close | Male testing | Gender in close range | Easy | Video 08 |
| 45 | Gender | Gender | Have a window with a man, in medium | Male testing | Gender in medium range | Easy | Video 08 |
| 46 | Gender | Gender | Have a window with a man, in far | Male testing | Gender in far range | Easy | Video 08 |
| 47 | Speech | word_count | Speak with TikTok trending audio playing | If transcript separates from TikTok music | Test TikTok audio separation | Easy | Video 06 |


# Video Recording Scripts


## ML Features

### Video 01: Speaker & Gaze
**Duration:** 15 seconds (5 blocks of 3 seconds)
**Tests:** person_count (Test 10), word_count (Tests 21-22), speech_coverage (Test 23), gaze_variance (Test 39), eye_contact_rate (Test 40)
**Setup:**
- Sit at medium distance (like a video call)
- Plain background, good lighting on face
- Have music playing in background (test speech separation)
- Mark 3 spots to look at: Camera, Left wall, Right wall

**Recording Script:**

| Block | Time | Action | Say (with music playing) | Expected Results |
|-------|------|--------|--------------------------|------------------|
| 1 | 0-3s | LOOK AT CAMERA | "Hello everyone, this is test one" (5 words) | person_count=1, word_count=5, eye_contact_rate=high |
| 2 | 3-6s | LOOK LEFT | "Now I'm looking to the left side" (7 words) | word_count=7, eye_contact_rate=low, gaze_variance increases |
| 3 | 6-9s | LOOK RIGHT | "Looking right" (2 words) | word_count=2, gaze_variance increases more |
| 4 | 9-12s | LOOK AT CAMERA | (STAY SILENT) | word_count=0, speech_coverage drops, eye_contact_rate=high |
| 5 | 12-15s | KEEP LOOKING AT CAMERA | (STAY SILENT) | word_count=0, minimal gaze_variance |

**Validation Checklist:**
- [ ] Person detected throughout? (person_count = 1)
- [ ] Words detected in blocks 1-3 despite music? (5, 7, 2 words)
- [ ] No words detected in blocks 4-5? (word_count = 0)
- [ ] Speech only in first 9 seconds? (speech_coverage ≈ 0.6)
- [ ] Eye contact high in blocks 1, 4, 5? Low in blocks 2, 3?
- [ ] Gaze variance shows movement across blocks 1-3?

**Tips for Recording:**
- Use timer app with 3-second intervals
- Practice the word counts beforehand
- Keep music at moderate volume (not too loud)
- Make obvious head turns for gaze changes

---

### Video 02: Emotions
**Duration:** 15 seconds (5 blocks of 3 seconds)
**Tests:** expression_count (Test 14), emotion_consistency (Test 15), dominant_emotion_id (Test 16), emotional_valence (Tests 17-18)
**Setup:**
- Close-up on face (shoulders and up)
- Good lighting on face to show expressions clearly
- Plain background
- Practice expressions in mirror first

**Recording Script:**

| Block | Time | Facial Expression | Visual Cue | Expected Results |
|-------|------|------------------|------------|------------------|
| 1 | 0-3s | BIG SMILE | Show teeth, eyes crinkle | dominant_emotion="happy", emotional_valence=positive |
| 2 | 3-6s | KEEP SMILING | Maintain same smile | emotion_consistency=high (still happy) |
| 3 | 6-9s | KEEP SMILING | Maintain same smile | emotion_consistency=high (70% happy so far) |
| 4 | 9-12s | SAD FACE | Frown, eyes down | expression_count+=1, emotional_valence shifts negative |
| 5 | 12-15s | ANGRY FACE | Furrow brows, tight lips | expression_count+=1, emotional_valence=negative |

**Validation Checklist:**
- [ ] Happy detected in blocks 1-3? (9 seconds = 60% of video)
- [ ] Dominant emotion = happy? (60% of video time)
- [ ] Expression count = 3? (happy → sad → angry)
- [ ] Emotion consistency high in first 9s, then drops?
- [ ] Emotional valence positive (blocks 1-3) then negative (blocks 4-5)?
- [ ] Clear emotion changes detected at 9s and 12s?

**Tips for Recording:**
- Exaggerate expressions (make them obvious)
- Hold each expression steady for full 3 seconds
- Transition quickly between expressions at block boundaries
- Look directly at camera throughout
- Count in your head: "Happy one, happy two, happy three..."

---

### Video 02B: Emotional Valence (50 seconds)
**Duration:** 50 seconds
**Tests:** emotional_valence (Tests 17-18) - Testing mixed emotions formula
**Setup:**
- Close-up on face (shoulders and up)
- Good lighting on face
- Use timer with audio cues every 3 seconds
- Practice emotion transitions

**Recording Script:**

| Window | Time Range | Emotions Pattern | Expected emotional_valence |
|--------|------------|------------------|---------------------------|
| Hook | 0-3s | Happy (3s) | +1.0 (all positive) |
| Middle Seg 1 | 3-11.8s | Happy (6s) + Sad (2.8s) | +0.36 (positive dominant) |
| Middle Seg 2 | 11.8-20.6s | Happy (3s) + Angry (5.8s) | -0.34 (negative dominant) |
| Middle Seg 3 | 20.6-29.4s | Happy (6s) + Neutral (2.8s) | +0.68 (positive, neutral doesn't count) |
| Middle Seg 4 | 29.4-38.2s | Sad (4s) + Angry (4.8s) | -1.0 (all negative) |
| Middle Seg 5 | 38.2-47s | Happy (4s) + Sad (2s) + Angry (2.8s) | -0.09 (slightly negative) |
| Closing | 47-50s | Neutral (3s) | 0.0 (neutral) |

**Detailed Block Timing:**

| Block | Seconds | Action | Say/Think |
|-------|---------|--------|-----------|
| 1 | 0-3 | HAPPY | "Hook happy" |
| 2 | 3-9 | HAPPY | "Segment one happy" (count 6 seconds) |
| 3 | 9-11.8 | SAD | "Segment one sad" |
| 4 | 11.8-14.8 | HAPPY | "Segment two happy" |
| 5 | 14.8-20.6 | ANGRY | "Segment two angry" |
| 6 | 20.6-26.6 | HAPPY | "Segment three happy" |
| 7 | 26.6-29.4 | NEUTRAL | "Segment three neutral" |
| 8 | 29.4-33.4 | SAD | "Segment four sad" |
| 9 | 33.4-38.2 | ANGRY | "Segment four angry" |
| 10 | 38.2-42.2 | HAPPY | "Segment five happy" |
| 11 | 42.2-44.2 | SAD | "Segment five sad" |
| 12 | 44.2-47 | ANGRY | "Segment five angry" |
| 13 | 47-50 | NEUTRAL | "Closing neutral" |

**Validation Checklist:**
- [ ] Middle Seg 1: Positive valence (more happy than sad)?
- [ ] Middle Seg 2: Negative valence (more angry than happy)?
- [ ] Middle Seg 3: Strong positive (neutral doesn't affect calculation)?
- [ ] Middle Seg 4: Full negative (-1.0)?
- [ ] Middle Seg 5: Mixed slightly negative?
- [ ] Formula working: (positive - negative) / total

**Tips for Recording:**
- Use a metronome app or countdown timer
- Write emotion names on cards as reminders
- Say the segment names out loud to track position
- Record multiple takes if needed

---

### Video 03: Text & Captions
**Duration:** 15 seconds (5 blocks of 3 seconds)
**Tests:** overlay_unique_count (Tests 1-4), overlay_coverage/persistence (Test 5), has_captions (Test 6)
**Setup:**
- Film yourself or a background
- Have text overlay tools ready (editing app)
- Plan text positions (top, middle, bottom, corners)
- Prepare captions to add in post

**Recording Script:**

| Block | Time | Action | Text Overlays | Captions | Expected Results |
|-------|------|--------|---------------|----------|------------------|
| 1 | 0-3s | Show 1 text | "Subscribe" (top) | None | overlay_count=1 |
| 2 | 3-6s | Add 2 more | + "Follow Me" (middle) + "❤️" (corner) | None | overlay_count=3 total |
| 3 | 6-9s | Remove all, add new | "New Text" (bottom) | Start captions: "Hello everyone" | overlay_count=1, has_captions=true |
| 4 | 9-12s | Keep text | Same "New Text" | Continue: "This is a test" | overlay_persistence increases |
| 5 | 12-15s | Add emoji overlay | + "🔥🔥🔥" | Continue: "Thank you" | overlay_count=2, captions continue |

**Validation Checklist:**
- [ ] Overlays counted separately from captions?
- [ ] Emoji overlays detected?
- [ ] Overlay persistence tracked (text staying 6+ seconds)?
- [ ] Captions detected but not counted as overlays?
- [ ] Coverage calculation working?

---

### Video 04: Scenes & Density (50 seconds)
**Duration:** 50 seconds
**Tests:** scene_count (Tests 19-20), shortest/longest_scene (Tests 37-38), max/min_density (Tests 35-36)
**Setup:**
- Plan quick location/background changes (even just turning around)
- Gather objects, prepare gestures for density variation
- Use timer for scene change timing

**Window Structure (50s video):**
- Hook: 0-3s
- Middle Seg 1: 3-11.8s (8.8s)
- Middle Seg 2: 11.8-20.6s (8.8s)
- Middle Seg 3: 20.6-29.4s (8.8s)
- Middle Seg 4: 29.4-38.2s (8.8s)
- Middle Seg 5: 38.2-47s (8.8s)
- Closing: 47-50s

**Recording Script:**

| Window | Time | Scenes Within Window | Density Strategy | Expected Per-Window Results |
|--------|------|---------------------|------------------|----------------------------|
| Hook | 0-3s | 1 scene | Medium elements | scene_count=0 (no changes) |
| Middle Seg 1 | 3-11.8s | 3 scenes (3s each) + MAX DENSITY | Many objects + gestures + emotions | scene_count=2, max_density here |
| Middle Seg 2 | 11.8-20.6s | 1 scene only + MIN DENSITY | Just stand still | scene_count=0, min_density here |
| Middle Seg 3 | 20.6-29.4s | 5 quick scenes (1.7s each) | Medium density | scene_count=4, shortest_scene test |
| Middle Seg 4 | 29.4-38.2s | 2 scenes (4.4s each) | Medium density | scene_count=1 |
| Middle Seg 5 | 38.2-47s | 1 long scene | Medium density | scene_count=0, longest_scene test |
| Closing | 47-50s | 1 scene | Low density | scene_count=0 |

**Scene Change Schedule:**
- 0s: Scene 1 starts
- 3s: → Scene 2 (Middle Seg 1 starts)
- 6s: → Scene 3
- 9s: → Scene 4
- 11.8s: → Scene 5 (Middle Seg 2 starts, stays for whole segment)
- 20.6s: → Scene 6 (Middle Seg 3 starts)
- 22.3s: → Scene 7
- 24s: → Scene 8
- 25.7s: → Scene 9
- 27.4s: → Scene 10
- 29.4s: → Scene 11 (Middle Seg 4 starts)
- 33.8s: → Scene 12
- 38.2s: → Scene 13 (Middle Seg 5 starts, stays for whole segment)
- 47s: → Scene 14 (Closing)

**Total: 14 scenes, but distributed strategically across windows**

**Validation Checklist:**
- [ ] Middle Seg 1: scene_count=2? (changes at 6s, 9s)
- [ ] Middle Seg 2: scene_count=0? (no changes within segment)
- [ ] Middle Seg 3: scene_count=4? (changes at 22.3s, 24s, 25.7s, 27.4s)
- [ ] Middle Seg 4: scene_count=1? (change at 33.8s)
- [ ] Middle Seg 5: scene_count=0? (no changes)
- [ ] Max density in Seg 1? (scene changes + elements)
- [ ] Min density in Seg 2? (no changes, minimal elements)
- [ ] Shortest scene ~1.7s? (in Seg 3)
- [ ] Longest scene ~8.8s? (Seg 2 or 5)

---

### Video 05: Objects & Gestures
**Duration:** 15 seconds (5 blocks of 3 seconds)
**Tests:** object_count (Tests 7-9), gesture_count (Test 13)
**Setup:**
- Gather objects: apple, book, cup, phone, pen, bottle
- Practice hand gestures: pointing, thumbs up, peace sign, wave, fist

**Recording Script:**

| Block | Time | Objects | Gestures | Expected Results |
|-------|------|---------|----------|------------------|
| 1 | 0-3s | Hold 1 apple | Point at it | object_count=1, gesture_count=1 |
| 2 | 3-6s | Add book + cup (3 total) | Thumbs up | object_count=3, gesture_count=1 |
| 3 | 6-9s | Same 3 objects | Peace sign + Wave | object_count=3, gesture_count=2 |
| 4 | 9-12s | Remove all, add phone | Point + Thumbs up + Fist | object_count=1, gesture_count=3 |
| 5 | 12-15s | Add pen + bottle (3 total) | Wave | object_count=3, gesture_count=1 |

**Validation Checklist:**
- [ ] Objects counted correctly per window?
- [ ] Same object not double-counted?
- [ ] Gestures detected and counted?
- [ ] Objects tracked across windows?

---

### Video 06: Audio Energy & TikTok Music
**Duration:** 18 seconds (6 blocks of 3 seconds)
**Tests:** energy_level (Test 24), energy_variance (Test 25), energy_max (Test 26), pitch_range (Tests 29-30), TikTok music (Test 47)
**Setup:**
- Download a trending TikTok audio
- Practice volume control (whisper to loud)
- Plan energy variations

**Recording Script:**

| Block | Time | Voice Energy | TikTok Audio | Say | Expected Results |
|-------|------|-------------|--------------|-----|------------------|
| 1 | 0-3s | WHISPER | Music playing | "This is very quiet" | Low energy_level |
| 2 | 3-6s | NORMAL | Music playing | "Normal speaking voice" | Medium energy |
| 3 | 6-9s | LOUD | Music playing | "NOW I'M LOUD!" | High energy, energy_max here |
| 4 | 9-12s | WHISPER | Music playing | "Back to quiet" | Energy variance high |
| 5 | 12-15s | MONOTONE | Music fading | "Monotone voice here" | Low pitch_range |
| 6 | 15-18s | EXCITED | Music stops | "Super excited voice!" | High pitch_range |

**Validation Checklist:**
- [ ] Energy levels match volume changes?
- [ ] Energy_max detected at LOUD section?
- [ ] Energy_variance shows changes?
- [ ] Pitch range low when monotone?
- [ ] Pitch range high when excited?
- [ ] Words detected despite TikTok music?

---

### Video 07: Person Framing
**Duration:** 12 seconds (4 blocks of 3 seconds)
**Tests:** average_face_size (Tests 31-34)
**Setup:**
- Mark floor positions: Close (1ft), Medium (3ft), Far (6ft), Very Far (10ft)
- Use tripod or stable camera position
- Ensure face stays in frame at all distances

**Recording Script:**

| Block | Time | Position | Visual | Expected Results |
|-------|------|----------|--------|------------------|
| 1 | 0-3s | CLOSE (1ft) | Face fills frame | average_face_size ≈ 0.5-0.7 |
| 2 | 3-6s | MEDIUM (3ft) | Shoulders + head visible | average_face_size ≈ 0.2-0.3 |
| 3 | 6-9s | FAR (6ft) | Upper body visible | average_face_size ≈ 0.08-0.15 |
| 4 | 9-12s | VERY FAR (10ft) | Full body visible | average_face_size ≈ 0.02-0.05 |

**Validation Checklist:**
- [ ] Face size decreases with distance?
- [ ] Close = largest face_size?
- [ ] Very far = smallest face_size?
- [ ] Consistent detection at all distances?

---

### Video 08: Gender Male
**Duration:** 15 seconds (5 blocks of 3 seconds)
**Tests:** avg_pitch_normalized for male (Test 27), gender detection at 3 distances (Tests 44-46)
**Setup:**
- Male speaker required
- Mark 3 positions: Close, Medium, Far
- Normal speaking voice

**Recording Script:**

| Block | Time | Position | Say | Expected Results |
|-------|------|----------|-----|------------------|
| 1 | 0-3s | CLOSE | "Testing close position" | gender="male", high confidence |
| 2 | 3-6s | CLOSE | "Still in close position" | gender="male" consistent |
| 3 | 6-9s | MEDIUM | "Moving to medium" | gender="male", confidence may vary |
| 4 | 9-12s | FAR | "Now at far position" | gender="male", lower confidence? |
| 5 | 12-15s | MEDIUM | "Back to medium" | gender="male", avg_pitch normalized for male |

**Validation Checklist:**
- [ ] Gender detected as "male" throughout?
- [ ] Confidence varies with distance?
- [ ] Pitch normalized using male baseline?
- [ ] Detection works at all distances?

---

### Video 09: Gender Female
**Duration:** 15 seconds (5 blocks of 3 seconds)
**Tests:** avg_pitch_normalized for female (Test 28), gender detection at 3 distances (Tests 41-43)
**Setup:**
- Female speaker required (Nadia)
- Mark 3 positions: Close, Medium, Far
- Normal speaking voice

**Recording Script:**

| Block | Time | Position | Say | Expected Results |
|-------|------|----------|-----|------------------|
| 1 | 0-3s | CLOSE | "Testing close position" | gender="female", high confidence |
| 2 | 3-6s | CLOSE | "Still in close position" | gender="female" consistent |
| 3 | 6-9s | MEDIUM | "Moving to medium" | gender="female", confidence may vary |
| 4 | 9-12s | FAR | "Now at far position" | gender="female", lower confidence? |
| 5 | 12-15s | MEDIUM | "Back to medium" | gender="female", avg_pitch normalized for female |

**Validation Checklist:**
- [ ] Gender detected as "female" throughout?
- [ ] Confidence varies with distance?
- [ ] Pitch normalized using female baseline?
- [ ] Detection works at all distances?

---

### Video 10: Two People
**Duration:** 12 seconds (4 blocks of 3 seconds)
**Tests:** person_count with 2 people (Test 11), person appearing/disappearing (Test 12)
**Setup:**
- Need second person to help
- Plan entrance/exit timing
- Both people should be clearly visible

**Recording Script:**

| Block | Time | People | Action | Expected Results |
|-------|------|--------|--------|------------------|
| 1 | 0-3s | 1 person | You alone on camera | person_count=1 |
| 2 | 3-6s | 2 people | Second person enters frame | person_count=2 |
| 3 | 6-9s | 2 people | Both stay in frame | person_count=2 maintained |
| 4 | 9-12s | 1 person | Second person exits | person_count=1 |

**Validation Checklist:**
- [ ] Single person detected initially?
- [ ] Two people detected when both present?
- [ ] Count updates when person enters/exits?
- [ ] No double-counting of same person?

---

The temporal window testing strategy has been moved to a dedicated document for better organization and detail.

**See: [TestPOM.md](./TestPOM.md)** for the complete temporal window testing strategy, including:
- Evidence from production code
- Mathematical proof of correctness
- Testing efficiency gains
- Edge cases covered

The strategy proves that all windows use the same calculation function, so features only need testing once, not three times.


## Boundary Tests

### Video B1: Gesture Boundaries
**Duration:** 15 seconds (clean boundary test)
**Tests:** Temporal window boundary filtering for ALL discrete event features
**Setup:**
- Simple recording setup - just you and camera
- Use metronome app or count seconds out loud
- Clear hand gestures (thumbs up is easiest to detect)
- Good lighting on hands

**Recording Script:**

| Second | Time | Action | Expected Window | Expected Results |
|--------|------|--------|-----------------|------------------|
| 1-2 | 0-2s | NO GESTURES | Hook | gesture_count=0 |
| 3 | 2-3s | THUMBS UP 👍 | Hook (ends at 3s) | gesture_count=1 in hook |
| 4-6 | 3-6s | NO GESTURES | Middle Seg 1 | gesture_count=0 |
| 7 | 6-7s | THUMBS UP 👍 | Middle Seg 1/2 boundary | gesture_count=1 in seg 2 |
| 8-12 | 7-12s | NO GESTURES | Middle Seg 2-3 | gesture_count=0 |
| 13-15 | 12-15s | NO GESTURES | Closing | gesture_count=0 |

**What This Tests:**
- Gesture at second 3 (2-3s) should appear ONLY in hook
- Gesture at second 7 (6-7s) should appear ONLY in middle segment 2
- If gesture at 3s appears in middle seg 1 = boundary bug!
- If gesture at 7s appears in middle seg 1 = boundary bug!

**Validation Checklist:**
- [ ] Hook has gesture_count=1? (thumbs up at second 3)
- [ ] Middle segment 1 has gesture_count=0? (no gestures 3-6s)
- [ ] Middle segment 2 has gesture_count=1? (thumbs up at second 7)
- [ ] Middle segment 3 has gesture_count=0? (no gestures 9-12s)
- [ ] Closing has gesture_count=0? (no gestures 12-15s)

**Tips for Recording:**
- Count out loud: "One... Two... Three-THUMBS-UP... Four... Five... Six... Seven-THUMBS-UP..."
- Make gesture clear and hold for full second
- Keep hands down when not gesturing
- This single test validates boundary filtering for ALL discrete features!

---

### Video B2: Scene Spanning Test
**Duration:** 10 seconds (tests scene boundary spanning)
**Tests:** Scene features that can span across temporal window boundaries
**Setup:**
- Plan 3 distinct visual changes (easier than 4!)
- Each scene is longer and easier to film
- Make scenes visually distinct

**Recording Script:**

| Scene | Time | Location/Angle | Window Coverage | Expected Results |
|-------|------|----------------|-----------------|------------------|
| Scene 1 | 0-2s | Face camera (location A) | Hook only | scene_count=0 in hook |
| Scene 2 | 2-5s | Turn around or move (location B) | SPANS hook/middle | Appears in BOTH |
| Scene 3 | 5-10s | Different angle/room (location C) | Middle + Closing | Appears in both |

**Simpler Approach - What to Film:**
1. **Seconds 1-2**: Stand in one spot facing camera
2. **Seconds 2-5**: Turn around (showing your back) or move to different spot
3. **Seconds 5-10**: Turn back or move to third spot

**What This Tests:**
- Scene 2 (2s-5s) spans from hook into middle (crosses 3s boundary)
- Scene 3 (5s-10s) spans from middle into closing (crosses 7s boundary)
- This is CORRECT behavior - scenes can span boundaries

**Validation Checklist:**
- [ ] Hook has scene_count=1? (Scene change at 2s)
- [ ] Middle segment has scene_count=2? (Changes at 2s and 5s visible)
- [ ] Closing has scene_count=1? (Change at 5s visible)
- [ ] Scenes detected in multiple windows?

**Tips for Recording:**
- Count out loud: "One, Two-TURN, Three, Four, Five-TURN, Six..."
- Don't worry about exact precision - anywhere in those seconds works
- Scene detection will catch the visual change

---

### Video B3: Audio Boundary Test
**Duration:** 12 seconds (tests audio continuous sampling)
**Tests:** Audio feature boundary filtering
**Setup:**
- Just you and camera/phone
- Count out loud to track timing
- No editing needed!

**Simple Recording Script:**

| Seconds | What to Do | Expected Window | Expected Results |
|---------|------------|-----------------|------------------|
| 1-2 | Speak LOUD: "ONE! TWO!" | Hook | High energy_level |
| 3 | Say "three" quietly | Hook ends | Transition |
| 4-6 | COMPLETE SILENCE | Middle | Near-zero energy_level |
| 7 | Say "seven" normally | Middle ends | Transition |
| 8-9 | Speak NORMAL: "Eight, Nine" | Middle | Medium energy_level |
| 10-12 | Continue NORMAL: "Ten, Eleven, Twelve" | Closing | Medium energy_level |

**Even Simpler Alternative:**
1. **Seconds 1-2**: SHOUT the numbers
2. **Second 3**: Say "three" as transition
3. **Seconds 4-6**: Stay completely SILENT (just mouth the numbers)
4. **Second 7**: Say "seven" as transition
5. **Seconds 8-12**: Speak normally

**What This Tests:**
- Audio energy changes detected at window boundaries
- No need for precise 2.99s or 6.99s timing
- Natural transitions at boundary seconds (3 and 7)

**Validation Checklist:**
- [ ] Hook has high energy_level? (from shouting)
- [ ] Middle segment 1 has low energy_level? (mostly silence)
- [ ] Middle segment 2 has medium energy_level? (normal speech)
- [ ] Closing has medium energy_level? (normal speech)

**Tips for Recording:**
- The exact boundary timing doesn't matter much
- Audio features average over the window anyway
- Just make sure MOST of seconds 4-6 are silent
- This tests that audio respects window boundaries

---

## Edge Case Tests

### Video E1: Ultra Short Video (5 seconds)
**Duration:** 5 seconds
**Tests:** Temporal window calculation for videos with no middle segments
**Setup:**
- Quick recording, any location
- Just speak normally

**Expected Windows (5s video):**
- Hook: 0-3s
- No middle segments
- Closing: 3-5s

**Recording Script:**

| Time | Action | Say | Expected Window |
|------|--------|-----|-----------------|
| 0-3s | Look at camera | "Testing short video hook" | Hook only |
| 3-5s | Keep talking | "And closing" | Closing only |

**Validation Checklist:**
- [ ] Only hook and closing windows exist?
- [ ] No middle_segments in output?
- [ ] All features calculated for hook/closing?
- [ ] No crashes or errors?

---

### Video E2: Silent Video (No Audio)
**Duration:** 10 seconds
**Tests:** Zero audio energy handling, no speech detection
**Setup:**
- MUTE your recording or don't speak at all
- Just do visual actions (gestures, expressions)

**Recording Script:**

| Time | Visual Action | Expected Results |
|------|--------------|------------------|
| 0-3s | Wave hand | gesture_count=1, energy_level=0 |
| 3-7s | Smile | expression_count=1, word_count=0 |
| 7-10s | Thumbs up | gesture_count=1, speech_coverage=0 |

**Validation Checklist:**
- [ ] energy_level = 0 throughout?
- [ ] energy_variance = 0?
- [ ] word_count = 0 in all windows?
- [ ] speech_coverage = 0?
- [ ] avg_pitch_normalized = 0?
- [ ] Other features still work (gestures, expressions)?

---

### Video E3: No Face Detected
**Duration:** 10 seconds
**Tests:** Missing face/gender detection handling
**Setup:**
- Film with camera pointing away from you
- Or cover camera for portions
- Still speak to test audio

**Recording Script:**

| Time | Camera View | Say | Expected Results |
|------|------------|-----|------------------|
| 0-3s | Point at wall | "No face here" | person_count=0, no gender |
| 3-7s | Show objects only | "Still no face" | average_face_size=0 |
| 7-10s | Back to wall | "Testing done" | eye_contact_rate=0 |

**Validation Checklist:**
- [ ] Pipeline doesn't crash without face?
- [ ] gender_detection missing or null?
- [ ] average_face_size = 0?
- [ ] eye_contact_rate = 0?
- [ ] gaze_variance = 0?
- [ ] Audio features still work?

---

### Video E4: Extreme Density Explosion (50 seconds)
**Duration:** 50 seconds
**Tests:** Extreme density variations, stress testing element detection
**Setup:**
- Gather MANY objects (10+)
- Plan rapid scene changes
- Prepare text overlays in editing
- Practice quick gestures and expressions

**Window Structure (50s video):**
- Hook: 0-3s
- Middle Seg 1: 3-11.8s (8.8s)
- Middle Seg 2: 11.8-20.6s (8.8s)
- Middle Seg 3: 20.6-29.4s (8.8s)
- Middle Seg 4: 29.4-38.2s (8.8s)
- Middle Seg 5: 38.2-47s (8.8s)
- Closing: 47-50s

**Recording Script:**

| Window | Time | Strategy | Elements to Create | Expected Results |
|--------|------|----------|-------------------|------------------|
| Hook | 0-3s | Normal start | Few elements | Baseline density |
| Middle Seg 1 | 3-11.8s | **EXPLOSION** | 5 scene cuts + 10 objects + continuous gestures + emotions + 5 text overlays | **MAX DENSITY (20-30/sec)** |
| Middle Seg 2 | 11.8-20.6s | Empty | Stand still, neutral, no objects | **MIN DENSITY (0-1/sec)** |
| Middle Seg 3 | 20.6-29.4s | Medium | 2 scenes + 3 objects + occasional gesture | Normal density (5/sec) |
| Middle Seg 4 | 29.4-38.2s | High | 3 scenes + 5 objects + gestures | High density (10/sec) |
| Middle Seg 5 | 38.2-47s | Low | 1 scene, minimal movement | Low density (2/sec) |
| Closing | 47-50s | End | Fade out | Low density |

**Middle Seg 1 EXPLOSION Details (3-11.8s):**
- **Scene cuts**: Change location/angle every 1.5-2 seconds (5 total)
- **Objects**: Hold/show 10+ items continuously
- **Gestures**: Non-stop hand movements (pointing, waving, thumbs up)
- **Expressions**: Rapid emotion changes every 2 seconds
- **Text overlays**: Add 5+ different texts in post
- **Target**: 20-30 elements detected per second!

**Middle Seg 2 EMPTY Details (11.8-20.6s):**
- Stand perfectly still
- Neutral face
- No objects
- No gestures
- No text
- Just the person detection (1 element/sec)

**Validation Checklist:**
- [ ] Extreme max_density in Seg 1 (20-30)?
- [ ] Near-zero min_density in Seg 2 (0-1)?
- [ ] Huge variance between segments?
- [ ] System handles element explosion without crashing?
- [ ] All element types counted in density?
- [ ] Scene cuts add to density count?

**Tips for Recording:**
- Use timer with 3-second beeps
- Pre-arrange objects for quick grabbing
- Practice the "explosion" segment beforehand
- Don't worry about looking chaotic in Seg 1 - that's the point!

---

## Metadata Tests (Using Real TikTok Videos)

### Video M1: High Generic Hashtags
**URL:** [To be selected - find video with #fyp #viral #trending etc.]
**Tests:** hashtag_count, generic_hashtag_count, generic_ratio
**Expected Characteristics:**
- 10+ hashtags total
- 5+ generic hashtags from this list:
  - #fyp
  - #foryou
  - #foryoupage
  - #viral
  - #trending
  - #explore
  - #tiktok
  - #tiktokviral
  - #tiktokcreator
  - #contentcreator
  - #funny
  - #duet
  - #trendingvideo
  - #tiktokchallenge
- Few specific/niche hashtags

**Expected Results:**
- `hashtag_count` > 10
- `generic_hashtag_count` > 5
- `generic_ratio` > 0.5
- `specific_hashtag_count` < 5

---

### Video M2: Niche Specific Hashtags
**URL:** [To be selected - find cooking/fitness/tech video]
**Tests:** specific_hashtag_count, generic_ratio
**Expected Characteristics:**
- 5-10 hashtags total
- Mostly niche hashtags (#veganrecipes, #mealprep, #healthycooking)
- 0-2 generic hashtags max

**Expected Results:**
- `hashtag_count` = 5-10
- `generic_hashtag_count` < 2
- `generic_ratio` < 0.2
- `specific_hashtag_count` > 3

---

### Video M3: No/Minimal Hashtags
**URL:** [To be selected - find video with 0-2 hashtags]
**Tests:** Edge case for hashtag processing
**Expected Characteristics:**
- 0-2 hashtags only
- Tests graceful handling of minimal metadata

**Expected Results:**
- `hashtag_count` = 0-2
- `generic_ratio` = 0 or calculated correctly
- No crashes on empty hashtag list

---

### Video M4: High Engagement Video
**URL:** [To be selected - viral video with 1M+ views]
**Tests:** Engagement metric extraction
**Expected Characteristics:**
- High view count (1M+)
- High like count (100K+)
- High share count

**Expected Results:**
- `play_count` > 1000000
- `digg_count` > 100000
- `share_count` > 1000
- All metrics extracted correctly

---

### Video M5: Mixed Metadata Test
**URL:** [To be selected - regular creator video]
**Tests:** Complete metadata extraction
**Expected Characteristics:**
- Mix of generic and specific hashtags (5-8 total)
- Medium engagement (10K-100K views)
- Has description/caption
- Clear author info

**Expected Results:**
- `hashtag_count` = 5-8
- `generic_ratio` = 0.3-0.5
- `author` field populated
- `description` field populated
- `create_time` valid timestamp

---

### Metadata Test Execution Plan

1. **Select Videos**: Find 5 TikTok videos matching above criteria
2. **Document URLs**: Save URLs in test configuration
3. **Run Production Pipeline**:
   ```bash
   python rumiai_runner.py "https://tiktok.com/@user/video/id"
   ```
4. **Validate Output**: Check `/insights/{video_id}_temporal_windows_updated.json`
5. **Verify Metadata Section**: Confirm all fields populated correctly

### Validation Checklist:
- [ ] Hashtag metrics calculated correctly?
- [ ] Generic vs specific classification working?
- [ ] Engagement metrics extracted?
- [ ] Author/description fields populated?
- [ ] Gender detection in metadata (if applicable)?
- [ ] No crashes on edge cases (no hashtags)?

---

## Test Coverage Summary

### Total Test Videos: 22
- **Feature Tests (ML):** 11 videos (Videos 01-10, 02B) - LOCAL
- **Boundary Tests:** 3 videos (Videos B1-B3) - LOCAL
- **Edge Case Tests:** 4 videos (Videos E1-E4) - LOCAL
- **Metadata Tests:** 5 videos (Videos M1-M5) - REAL TIKTOK

**Note:** Total is 22 because we have Video 02 and Video 02B as separate tests

### Total Recording Time: ~6.5 minutes (local videos only)
- Feature videos: ~4 minutes
- Boundary videos: ~37 seconds
- Edge cases: ~1.5 minutes (includes 50s density test)
- Metadata videos: Use existing TikTok content (no recording)

### Features Covered: 47/47 ML tests + Metadata validation
### Test Categories:
1. **Local Golden Dataset** (15 videos):
   - All ML features
   - Boundary conditions
   - Edge cases
   - Run with: `python3 test_manual_videos.py`

2. **TikTok Metadata Tests** (5 videos):
   - Hashtag analysis
   - Engagement metrics
   - Author/description extraction
   - Run with: `python rumiai_runner.py [URL]`

### Edge Cases Covered:
- No middle window (ultra short)
- No audio (silent video)
- No face detection
- Extreme density explosion (stress test)
- No/minimal hashtags
- High generic hashtags
- Niche specific hashtags

### Critical Validations:
1. **Graceful Degradation:** System doesn't crash when features missing
2. **Boundary Accuracy:** Features assigned to correct temporal windows
3. **Formula Correctness:** Emotional valence, density calculations work
4. **Edge Handling:** Extreme/missing values handled properly