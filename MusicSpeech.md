# Music vs Speech Detection Investigation

**Created**: 2024-09-29
**Status**: Investigation Complete
**Finding**: No VAD or music filtering currently implemented

---

## 1. Documentation Claims vs Reality

### What AudioServices.md Claims (Line 108)
"Segmentation: Automatic by whisper.cpp VAD (Voice Activity Detection)"

### Investigation Results
- **NO VAD is actually implemented** in current whisper_cpp_service.py
- The main whisper.cpp binary doesn't have VAD options
- VAD is only available in example apps like stream/talk (not being used)
- No VAD parameters are being passed in current implementation

### Current Implementation (whisper_cpp_service.py:202)
```python
cmd = [
    str(self.binary_path),
    "-m", str(self.model_path),
    "-f", str(audio_path),
    "-t", "10",  # Use all 10 WSL2 cores
    "-bo", "1",  # Greedy decoding for maximum speed
    "-bs", "1",  # Greedy decoding (no beam search)
    "-oj",  # Output JSON to file
]
# NO VAD parameters present
```

---

## 2. Music Detection Investigation

### Test Case: Video03TextsCaptions.mp4
- **0-6s**: Background music only
- **6s**: "Hello, everyone" (actual speech)
- **10s**: "This is a test" (actual speech)
- **13s**: "Thank you" (actual speech)

### Whisper Output Analysis
When testing with music-only sections, Whisper sometimes outputs:
- `[Music]` markers
- `*outro music*` markers
- Sometimes hallucinates speech from music patterns

### Parameter Testing Results

| Configuration | Parameters | Result |
|--------------|------------|--------|
| Current (Greedy) | `-bo 1 -bs 1` | Sometimes adds [Music] markers |
| Beam Search | `-bo 3 -bs 3` | Similar behavior |
| With Entropy | `-bo 3 -bs 3 -et 2.0` | No improvement |
| Strict Thresholds | `-bo 3 -bs 3 -et 1.5 -lpt -0.5` | No improvement |
| No Fallback | `-bo 3 -bs 3 -et 1.5 -nf` | Changes [Music] to *outro music* |

---

## 3. Available Whisper.cpp Parameters for Music/Speech

### Parameters That Could Help (But Don't)
- `-et <float>`: Entropy threshold for decoder fail (default 2.4)
- `-lpt <float>`: Logprob threshold for decoder fail (default -1.0)
- `-wt <float>`: Word timestamp probability threshold (default 0.01)
- `-nf`: No timestamps for blank periods (no fallback)
- `-su`: Suppress blank output (NOT AVAILABLE in main binary)

### Why Parameters Don't Help
1. **Whisper is confident about music markers** - assigns high confidence
2. **Model expects speech in music** - trained on videos where background music often has speech
3. **No true VAD** - can't separate music from speech at audio level

---

## 4. Potential Solutions Explored

### Option 1: External VAD (Not Implemented)
- Could use Silero VAD or WebRTC VAD before Whisper
- Would filter out non-speech segments
- Requires additional processing step

### Option 2: Post-Processing Filters (Considered)
```python
def filter_music_markers(segments):
    """Remove segments that are just music markers"""
    filtered = []
    for segment in segments:
        text = segment.get('text', '').strip().lower()
        # Skip music markers
        if text in ['[music]', '[applause]', '*outro music*', '*intro music*']:
            continue
        filtered.append(segment)
    return filtered
```

### Option 3: Confidence Thresholds (Tested - Doesn't Work)
- Whisper is highly confident even about music markers
- Low entropy despite being non-speech
- Log probabilities pass thresholds

---

## 5. Current Status

### What We Have
- Whisper transcribes everything, including marking some music as `[Music]`
- No pre-filtering of audio
- No post-filtering of transcription results
- Music markers sometimes appear in transcription

### What We Don't Have
- Voice Activity Detection (VAD)
- Music vs speech classification
- Filtering of non-speech segments
- Confidence-based filtering (doesn't work anyway)

### Impact
- Music sections may be marked as `[Music]` in transcription
- These markers get counted in word_count
- Speech_coverage may be inflated if music is present
- But actual speech is correctly transcribed

---

## 6. Recommendations

### Short Term
1. **Document the limitation** - AudioServices.md should be corrected
2. **Accept music markers** - They're actually useful metadata
3. **Filter in post-processing** if needed for specific use cases

### Long Term
1. **Consider external VAD** if speech-only transcription becomes critical
2. **Upgrade Whisper model** - larger models handle music better
3. **Pre-process audio** - separate music track from speech if possible

---

## Key Takeaway

**There is NO music filtering or VAD currently implemented.** Whisper processes all audio and may output music markers like `[Music]` or transcribe actual speech over music. The documentation claiming VAD exists is incorrect and should be updated.