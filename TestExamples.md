| To Test | Category       | Feature to Test                | Method                                                                 | What it Tests                              | Comments                               | Difficulty |
|---------|----------------|--------------------------------|------------------------------------------------------------------------|--------------------------------------------|----------------------------------------|------------|
| 1       | Text Overlay   | overlay_unique_count           | Use 1, 3, 5 in same Window                                             | Accelerated Overlays                       |                                        | Easy       |
| 2       | Text Overlay   | overlay_unique_count           | Input in different parts of screen                                     | Overlays in varied parts of screen         |                                        |            |
| 3       | Text Overlay   | overlay_unique_count           | Use emojis                                                             | Emojis as Overlays                         |                                        |            |
| 4       | Text Overlay   | overlay_unique_count           | Use Captions                                                           | Ensure captions are not counted as Overlay |                                        |            |
| 5       | Text Overlay   | overlay_coverage / persistence | Use overlays for 25%, 50%, 75% of window                               | If Coverage works                          |                                        |            |
| 6       | Text Overlay   | has_captions                   | Use Captions                                                           | Captions work                              |                                        | Easy       |
| 7       | Object         | object_count                   | Use 1, 3, 6                                                            | Object Count                               |                                        |            |
| 8       | Object         | object_count                   | Bring same object twice in same window                                 | Ensure it is not double counting objects   |                                        |            |
| 9       | Object         | object_count                   | Bring same object twice in multiple windows                            | ID relationship of Object between windows  |                                        |            |
| 10      | Object         | person_count                   | Film with 1 person                                                     | Count of single person                     |                                        | Easy       |
| 11      | Object         | person_count                   | Film with 2 people                                                     | Count of 2 people                          |                                        | Easy       |
| 12      | Object         | person_count                   | Make 1 person appear in one frame and disappear in other, within window| How a person is counted                    |                                        |            |
| 13      | Gesture        | gesture_count                  | Make 1, 3, 5 gestures                                                  | Count of Gestures                          |                                        | Easy       |
| 14      | Emotion        | expression_count               | Use 3, 5                                                               | Count of expressions                       | Sad → Happy → Angry → Sad (Count of 4) | Easy       |
| 15      | Emotion        | emotion_consistency            | Happy 70% then angry/sad                                               | Consistency of Emotion                     | Same test as feature before/after       |            |
| 16      | Emotion        | dominant_emotion_id            | Happy 70% then angry/sad                                               | If dominant emotion works                  |                                        |            |
| 17      | Emotion        | emotional_valence              | Happy 40% → negative (angry, sad, fear, disgust)                       |                                            |                                        |            |
| 18      | Emotion        | emotional_valence              | Happy 70% → negative                                                   |                                            |                                        |            |
| 19      | Scene          | scene_count                    | Make 1, 3, 5 scenes                                                    | Count of Scenes                            |                                        | Easy       |
| 20      | Scene          | scene_duration_variance        | Make 1, 3, 5 scenes                                                    | Variance of scenes                         | Same test as feature before             |            |
| 21      | Speech         | word_count                     | Video with loud music while speaking                                   | If transcript separates from music         |                                        | Easy       |
| 22      | Speech         | word_count                     | Say 5, 10, 15, 25 words per window                                     | Word Count working well                    |                                        | Easy       |
| 23      | Speech         | speech_coverage                | Say 5, 10, 15, 25 words per window                                     | Speech coverage working                    | Same test as feature before             |            |
| 24      | Audio          | energy_level                   | Speak with a lot of energy                                             | Energy working                             |                                        | Easy       |
| 25      | Audio          | energy_variance                | Speak with high energy and low energy                                  | Energy Variance working                    |                                        | Easy       |
| 26      | Audio          | energy_max                     | Yell in specific parts of video                                        | If energy max works                        |                                        |            |
| 27      | Audio          | avg_pitch_normalized           | You speak (Male)                                                       |                                            |                                        |            |
| 28      | Audio          | avg_pitch_normalized           | Nadia speaks (Female)                                                  |                                            |                                        |            |
| 29      | Audio          | pitch_range_norm               | Fluctuate from high to low energy                                      |                                            |                                        |            |
| 30      | Audio          | pitch_range_norm               | Stay monotone                                                          |                                            |                                        |            |
| 31      | Person Framing | average_face_size              | Face close to screen                                                   | Framing working                            |                                        | Easy       |
| 32      | Person Framing | average_face_size              | Face middle to screen                                                  | Framing working                            |                                        | Easy       |
| 33      | Person Framing | average_face_size              | Face far from screen                                                   | Framing working                            |                                        | Easy       |
| 34      | Person Framing | average_face_size              | Face very far from screen                                              | Framing working                            |                                        | Easy       |
| 35      | Calculation    | max_density                    | Calculation valid                                                      | Calculation Works                          |                                        | Easy       |
| 36      | Calculation    | min_density                    | Calculation valid                                                      | Calculation Works                          |                                        | Easy       |
| 37      | Scene          | shortest_scene                 | Time a short scene within a window                                     | Short scene being calculated well          |                                        | Easy       |
| 38      | Scene          | longest_scene                  | Time a long scene within a window                                      | Longest scene being calculated well        |                                        | Easy       |
| 39      | Gaze           | gaze_variance                  | Look at different things in video                                      | Gaze OK                                    |                                        | Easy       |
| 40      | Gaze           | eye_contact_rate               | Look at different things in video                                      | Eye contact rate OK                        | Same test as feature before             | Easy       |

## Temporal Window Testing Strategy

### Why Features Don't Need Window-Specific Testing

After analyzing the `temporal_compute.py` implementation, we discovered that **all temporal windows use the same `process_segment()` function**:

```python
# All three window types call the SAME function:
hook_data = process_segment(hook_bounds, ...)      # Line 1789
seg_data = process_segment(seg_bounds, ...)        # Line 1805
closing_data = process_segment(closing_bounds, ...) # Line 1819
```

#### What This Means:

1. **Feature Logic is Identical**: The calculation for `expression_count`, `word_count`, etc. is exactly the same regardless of window type
2. **Only Boundaries Differ**: The only difference is the time bounds passed to the function (0-3s for hook, last 3s for closing, etc.)
3. **No Window-Specific Code**: Features are not calculated differently in hook vs middle vs closing

#### What We Still Need to Test:

**Boundary Filtering** - The timestamp filtering logic:
```python
segment_expressions = [e for e in expression_timeline
                      if start <= e.get('timestamp', 0) < end]
```

We should verify that:
- Events at 2.9s appear in hook (< 3.0s boundary)
- Events at 3.1s appear in middle segments (>= 3.0s boundary)
- Events in last 3s appear in closing

### Optimized Testing Approach:

Instead of testing all 40 features across all windows (280 test cases!), we need:

1. **Feature Testing** (40 tests): Verify each feature calculates correctly using the test cases above
2. **Window Boundary Test** (1 test): Create ONE video that validates temporal boundaries work correctly:
   - 0-2.9s: Show 3 text overlays
   - 3.0-3.5s: Show 2 different text overlays
   - Last 3s: Show 1 text overlay
   - Verify correct counts in each window

This reduces testing from 280 cases to ~41 cases while maintaining confidence that the system works correctly.

### Conclusion:

Since `process_segment()` is reused for all windows, **if a feature works in one window, the calculation logic will work identically in all windows**. We only need to test that the boundary filtering correctly assigns events to the right windows.


