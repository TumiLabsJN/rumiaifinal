
## Video Allocation Summary

### 10 Golden Dataset Videos:
- **Video 01: Speaker & Gaze** - Tests 10, 21-23, 39-40 (person_count, speech, gaze)
- **Video 02: Emotions** - Tests 14-18 (expression_count, emotion features)
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
| 17      | Emotion        | emotional_valence              | Happy 40% → negative (angry, sad, fear, disgust)                       |                                            |                                        |            | Video 02 |
| 18      | Emotion        | emotional_valence              | Happy 70% → negative                                                   |                                            |                                        |            | Video 02 |
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





## Video Recording Scripts

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

## Temporal Window Testing Strategy

The temporal window testing strategy has been moved to a dedicated document for better organization and detail.

**See: [TestPOM.md](./TestPOM.md)** for the complete temporal window testing strategy, including:
- Evidence from production code
- Mathematical proof of correctness
- Testing efficiency gains
- Edge cases covered

The strategy proves that all windows use the same calculation function, so features only need testing once, not three times.


