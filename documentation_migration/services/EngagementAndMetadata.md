# Engagement and Metadata Features

## 📊 Feature Overview Matrix

### ⚠️ CRITICAL: Feature Validation Required

This feature matrix is NOT just documentation - it requires:
1. **Statistical Analysis**: Calculate actual correlations between features
2. **Semantic Review**: Identify which features are interpretations vs measurements
3. **Dependency Tracking**: Verify which features are derivatives
4. **Quality Testing**: Run features through videos with known issues (no faces, no speech, etc.)
5. **Performance Profiling**: Measure actual processing time per feature

DO NOT trust feature descriptions at face value. Each feature must be:
- Traced to its source code
- Validated with test videos
- Checked for correlations
- Verified for reliability

| Feature Name | Category | Source Services | Dependencies | Temporal Type | Data Type & Range | ML Importance | Creator Benefit | Reliability | Doubtful | Comments | RF Transform | RF Complexity | KM Transform | KM Complexity | Feature Time |
|--------------|----------|-----------------|--------------|---------------|-------------------|---------------|----------------|-------------|----------|----------|--------------|---------------|--------------|---------------|--------------|
| digg_count | Virality Metrics | Apify Scraper | None | Global | Integer [0-∞] | Direct viral success indicator | Shows total likes received | High | None | Direct measurement from TikTok API | None | None | Log + scale | Low | Low |
| play_count | Virality Metrics | Apify Scraper | None | Global | Integer [0-∞] | View count drives algorithm ranking | Shows total video views | High | None | Direct measurement from TikTok API | None | None | Log + scale | Low | Low |
| collect_count | Virality Metrics | Apify Scraper | None | Global | Integer [0-∞] | Saves indicate high-value content | Shows bookmark/save actions | High | None | Direct measurement from TikTok API | None | None | Log + scale | Low | Low |
| share_count | Virality Metrics | Apify Scraper | None | Global | Integer [0-∞] | Shares drive organic distribution | Shows forward/share actions | High | None | Direct measurement from TikTok API | None | None | Log + scale | Low | Low |
| comment_count | Virality Metrics | Apify Scraper | None | Global | Integer [0-∞] | Comments indicate engagement depth | Shows comment thread activity | High | None | Direct measurement from TikTok API | None | None | Log + scale | Low | Low |
| create_time | Video Metadata | Apify Scraper | None | Global | String (ISO 8601) | Temporal patterns affect performance | Shows when content was published | High | None | Direct timestamp from TikTok metadata | Extract features | Medium | Cyclical encode | Medium | Low |
| author | Video Metadata | Apify Scraper | None | Global | String | Creator identity for account analysis | Shows username/unique identifier | High | None | Direct creator info from TikTok metadata | One-hot encode | High | Label encode | Medium | Low |
| description | Video Metadata | Apify Scraper | None | Global | String | Caption text for content analysis | Shows video description/caption | High | None | Direct caption text from TikTok metadata | Text features | High | Text encoding | High | Low |
| video_id | Video Metadata | Apify Scraper | None | Global | String | Unique identifier for tracking | Internal video identification | High | None | Direct TikTok video identifier | None | None | Label encode | Low | Low |
| duration | Video Metadata | Apify Scraper | None | Global | Float [1-600] | Video length affects engagement patterns | Shows total video duration in seconds | High | None | Direct duration from TikTok metadata | None | None | Scale [0-1] | Low | Low |
| gender_detection | Demographics | DeepFace | None | Global | Object | Required for pitch normalization | Shows detected gender and confidence | Medium | None | Gender classification from face analysis | Extract fields | Low | Label encode | Low | Medium |
| hashtag_analysis | Hashtags | Hashtag Analysis | Apify metadata | Global | Object | Strategy analysis for viral patterns | Shows hashtag strategy metrics | High | None | Analysis of hashtag types and patterns | Extract fields | Medium | Multiple features | Medium | Low |
| processing_timestamp | System Fields | System | None | Global | Float (timestamp) | Pipeline execution tracking | Internal processing metadata | High | None | System timestamp when processing completed | None | None | None | None | None |
| version | System Fields | System | None | Global | String | Pipeline version tracking | Internal version identification | High | None | Temporal compute version identifier | None | None | None | None | None |

---

# Virality Metrics

## 🎯 Feature Purpose & ML Value

### Business Question
How successful is this video in terms of platform engagement and viral reach?

### ML Significance
- **Predictive Power**: HIGH for success classification - engagement metrics are direct success indicators
- **Feature Type**: Count-based integers from TikTok platform data
- **Correlation with Success**: These ARE the success metrics - play_count drives algorithm ranking, engagement ratios predict viral potential

### Legacy ML Insights
```
⚠️ VERIFIED: From temporal_compute.py:1770-1775 and MetadataServices.md
- All metrics extracted from Apify TikTok scraper API
- Field name mapping: likes→digg_count, views→play_count, saves→collect_count
- No retry logic due to API costs - single extraction per video
```

## 📊 Feature Components

### Available Metrics in Metadata Section
```json
{
  "metadata": {
    "digg_count": 0-∞,        // Total likes received (mapped from 'likes')
    "play_count": 0-∞,        // Total video views (mapped from 'views')
    "collect_count": 0-∞,     // Total saves/bookmarks (mapped from 'saves')
    "share_count": 0-∞,       // Total shares/forwards
    "comment_count": 0-∞      // Total comments posted
  }
}
```

### Metric Definitions
⚠️ **VERIFIED: All features exist in temporal windows JSON output**
Reference: `/insights/7500252920844193067_temporal_windows_updated.json:352-356`

| Metric | Source Field (temporal_compute.py:1770-1775) | Range | Interpretation |
|--------|---------|-------|----------------|
| digg_count | metadata.get('likes', 0) | 0-∞ | Total likes received from viewers |
| play_count | metadata.get('views', 0) | 0-∞ | Total video views/impressions |
| collect_count | metadata.get('saves', 0) | 0-∞ | Total bookmarks/saves by users |
| share_count | metadata.get('shares', 0) | 0-∞ | Total forwards/shares to others |
| comment_count | metadata.get('comments', 0) | 0-∞ | Total comments in thread |

## 🔄 Data Pipeline

### Source to Feature Flow
```
Apify TikTok Scraper (MetadataServices.md)
    ↓ (TikTok API metadata extraction)
rumiai_runner.py (before ML services)
    ↓ (video metadata processing)
temporal_compute.py:1770-1775
    ↓ (field name mapping and validation)
Metadata Section Output
```

### Implementation Location
```python
# Engagement metrics extraction and mapping
/rumiai_v2/processors/temporal_compute.py:1770-1775
├── Field name mapping from Apify output
├── Default value handling (0 for missing metrics)
└── Direct assignment to calculated_metadata
```

## 🎨 Feature Engineering Opportunities

### Current Limitations
- Raw counts don't normalize for account size or posting time
- No engagement rate calculations (likes/views, comments/views)
- Missing temporal decay factors (recent vs old performance)
- No comparative metrics (performance vs creator average)

### Proposed Enhancements
- [ ] Add engagement_rate (likes + comments + shares) / views
- [ ] Implement save_rate (collect_count / play_count)
- [ ] Include viral_velocity (growth rate over time windows)
- [ ] Add creator_performance_ratio (vs creator's average metrics)

## 🔗 Cross-References

### Dependencies (from Phase 1)
- **Primary Service**: Apify Scraper (MetadataServices.md#Apify)
- **API Required**: TikTok data extraction via Apify
- **Performance Impact**: Blocks pipeline for up to 5 minutes on timeout
- **Data Flow**: Apify API → rumiai_runner.py → temporal_compute.py:1770

### Related Features
- **create_time**: Time of posting affects engagement accumulation
- **hashtag_analysis**: Strategy affects viral potential
- **author**: Creator identity influences baseline engagement

### Downstream Usage (for Phase 3)
- Used in ML models: Success classification, viral prediction
- API endpoints: Performance analytics dashboard
- Reports: Creator performance benchmarking

---

# Video Metadata

## 🎯 Feature Purpose & ML Value

### Business Question
What are the fundamental characteristics and context of this video content?

### ML Significance
- **Predictive Power**: MEDIUM to HIGH - metadata provides crucial context for all other features
- **Feature Type**: Mixed (timestamps, strings, numeric) requiring different preprocessing
- **Correlation with Success**: Duration and timing patterns strongly correlate with optimal performance windows

### Legacy ML Insights
```
⚠️ VERIFIED: From temporal_compute.py:1775-1777 and MetadataServices.md
- Author extraction handles nested author object with uniqueId and name fallbacks
- CreateTime supports both 'createTime' and 'createTimeISO' formats
- Duration drives frame sampling rates and temporal window calculations
```

## 📊 Feature Components

### Available Metrics in Metadata Section
```json
{
  "metadata": {
    "create_time": "2025-05-03T16:12:38+00:00",  // ISO 8601 timestamp
    "author": "janemukbangs",                    // Creator username
    "description": "We're back at the most...",  // Video caption
    "video_id": "7500252920844193067",           // TikTok video ID
    "duration": 73.0                            // Video length in seconds
  }
}
```

### Metric Definitions
⚠️ **VERIFIED: All features exist in temporal windows JSON output**
Reference: `/insights/7500252920844193067_temporal_windows_updated.json:350-351,357-359`

| Metric | Source Field (temporal_compute.py:1775-1777) | Range | Interpretation |
|--------|---------|-------|----------------|
| create_time | metadata.get('createTime', metadata.get('createTimeISO', '')) | ISO 8601 | When video was published |
| author | metadata.get('author', {}).get('uniqueId', ...get('name', '')) | String | Creator username/identifier |
| description | metadata.get('description', '') | String | Video caption/description text |
| video_id | analysis_dict.get('video_id', metadata.get('id', 'unknown')) | String | Unique TikTok video identifier |
| duration | analysis_dict.get('duration', 0) | 1-600s | Total video length |

## 🔄 Data Pipeline

### Source to Feature Flow
```
Apify TikTok Scraper (MetadataServices.md)
    ↓ (comprehensive video metadata)
rumiai_runner.py
    ↓ (metadata validation and processing)
temporal_compute.py:1775-1777
    ↓ (field extraction with fallbacks)
Metadata Section Output
```

### Implementation Location
```python
# Video metadata extraction with robust fallbacks
/rumiai_v2/processors/temporal_compute.py:1775-1777
├── Nested author object handling
├── Multiple timestamp format support
├── Fallback chains for missing fields
└── video_id from top-level vs metadata
```

## 🎨 Feature Engineering Opportunities

### Current Limitations
- create_time is string format (needs temporal feature extraction)
- author is raw username (no account characteristics)
- description is unprocessed text (needs NLP features)
- No derived timing features (hour of day, day of week)

### Proposed Enhancements
- [ ] Extract posting_hour, posting_day_of_week from create_time
- [ ] Add description_length, hashtag_count from description
- [ ] Implement creator_follower_tier classification
- [ ] Include video_age_days (time since posting)

---

# Demographics

## 🎯 Feature Purpose & ML Value

### Business Question
What are the demographic characteristics of the video creator for context-aware analysis?

### ML Significance
- **Predictive Power**: MEDIUM for content classification - enables gender-specific analysis
- **Feature Type**: Object with nested classification and confidence metrics
- **Correlation with Success**: Required for pitch normalization, affects audience targeting strategies

### Legacy ML Insights
```
⚠️ VERIFIED: From temporal_compute.py:1783-1787 and AnalysisServices.md
- DeepFace gender detection required for pitch metric normalization
- Subprocess isolation for stability (~500ms overhead per video)
- Confidence scores typically >95% for clear faces, lower for profile/occlusion
```

## 📊 Feature Components

### Available Metrics in Metadata Section
```json
{
  "metadata": {
    "gender_detection": {
      "gender": "female",           // Detected gender classification
      "confidence": 0.9975700,      // Classification confidence
      "method": "deepface"          // Detection method used
    }
  }
}
```

### Metric Definitions
⚠️ **VERIFIED: All features exist in temporal windows JSON output**
Reference: `/insights/7500252920844193067_temporal_windows_updated.json:360-364`

| Metric | Source Field (temporal_compute.py:1783-1787) | Range | Interpretation |
|--------|---------|-------|----------------|
| gender | gender_data.get('gender') | male/female/multiple_people | Primary gender classification |
| confidence | gender_data.get('confidence', 0.0) | 0.0-1.0 | Classification confidence score |
| method | gender_data.get('method', 'deepface') | String | Detection algorithm used |

## 🔄 Data Pipeline

### Source to Feature Flow
```
DeepFace Service (AnalysisServices.md)
    ↓ (face analysis with gender classification)
Subprocess Isolation (stability)
    ↓ (gender detection results)
temporal_compute.py:1783-1787
    ↓ (object construction with defaults)
Metadata Section Output
```

### Implementation Location
```python
# Gender detection object construction
/rumiai_v2/processors/temporal_compute.py:1783-1787
├── DeepFace data extraction from ml_data
├── Nested object creation with all fields
├── Default values for missing confidence/method
└── Required for pitch normalization pipeline
```

## 🎨 Feature Engineering Opportunities

### Current Limitations
- Binary gender classification (limited diversity representation)
- Single detection per video (no temporal changes considered)
- No age or other demographic features
- Confidence threshold not applied (low confidence still used)

### Proposed Enhancements
- [ ] Add confidence_threshold filtering for unreliable detections
- [ ] Implement multi_person_handling for videos with multiple people
- [ ] Include demographic_reliability_score based on face quality
- [ ] Add temporal_consistency check across video frames

---

# Hashtag Strategy

## 🎯 Feature Purpose & ML Value

### Business Question
What hashtag strategy is the creator using to drive discovery and engagement?

### ML Significance
- **Predictive Power**: HIGH for content strategy classification - hashtag patterns predict discovery potential
- **Feature Type**: Object with count and ratio metrics for strategy analysis
- **Correlation with Success**: Generic vs specific hashtag ratios correlate with different viral patterns

### Legacy ML Insights
```
⚠️ VERIFIED: From temporal_compute.py:150-210 and MetadataServices.md
- Expanded generic hashtag list: fyp, viral, trending, tiktok, tiktokviral, etc.
- Generic ratio calculation: generic_count / total_count
- Strategy classification based on ratio patterns (discovery vs niche targeting)
```

## 📊 Feature Components

### Available Metrics in Metadata Section
```json
{
  "metadata": {
    "hashtag_analysis": {
      "hashtag_count": 1,              // Total hashtags used
      "generic_hashtag_count": 0,      // Discovery-focused hashtags
      "specific_hashtag_count": 1,     // Niche/specific hashtags
      "generic_ratio": 0.0             // Generic / total ratio
    }
  }
}
```

### Metric Definitions
⚠️ **VERIFIED: All features exist in temporal windows JSON output**
Reference: `/insights/7500252920844193067_temporal_windows_updated.json:365-369`

| Metric | Formula (temporal_compute.py:206-210) | Range | Interpretation |
|--------|---------|-------|----------------|
| hashtag_count | len(hashtags) | 0-∞ | Total hashtags in video |
| generic_hashtag_count | count of generic tags | 0-∞ | Discovery-focused tags (fyp, viral, etc.) |
| specific_hashtag_count | total_count - generic_count | 0-∞ | Niche/topic-specific tags |
| generic_ratio | generic_count / total_count | 0-1 | Strategy classification metric |

## 🔄 Data Pipeline

### Source to Feature Flow
```
Apify TikTok Scraper (MetadataServices.md)
    ↓ (hashtag array from video metadata)
temporal_compute.py:150-210
    ↓ (hashtag analysis and classification)
Generic vs Specific Classification
    ↓ (strategy metrics calculation)
Metadata Section Output
```

### Implementation Location
```python
# Hashtag strategy analysis
/rumiai_v2/processors/temporal_compute.py:150-210
├── extract_hashtag_metrics() function
├── Generic hashtag list (lines 163-182)
├── String/dict format handling
└── Strategy ratio calculations
```

## 🎨 Feature Engineering Opportunities

### Current Limitations
- Binary generic/specific classification (limited strategy nuance)
- Static generic hashtag list (may become outdated)
- No trending vs evergreen hashtag analysis
- Missing hashtag effectiveness scoring

### Proposed Enhancements
- [ ] Add hashtag_trend_score based on current platform trends
- [ ] Implement hashtag_competition_level analysis
- [ ] Include branded_hashtag_count for business content detection
- [ ] Add hashtag_length_average for strategy complexity measurement

---

# System Fields

## 🎯 Feature Purpose & ML Value

### Business Question
What system metadata tracks the processing pipeline and data versioning?

### ML Significance
- **Predictive Power**: NONE for content analysis - purely operational metadata
- **Feature Type**: System timestamps and version identifiers
- **Correlation with Success**: No direct correlation - used for data quality and pipeline monitoring

### Legacy ML Insights
```
⚠️ VERIFIED: From temporal_compute.py:1806-1807
- processing_timestamp: time.time() when temporal computation completed
- version: '2.0.0' indicates temporal compute pipeline version
- Non-ML features marked for exclusion from model training
```

## 📊 Feature Components

### Available Metrics in Root Level
```json
{
  "processing_timestamp": 1758312056.8134718,  // Unix timestamp
  "version": "2.0.0"                           // Pipeline version
}
```

### Metric Definitions
⚠️ **VERIFIED: All features exist in temporal windows JSON output**
Reference: `/insights/7500252920844193067_temporal_windows_updated.json:372-373`

| Metric | Source (temporal_compute.py:1806-1807) | Range | Interpretation |
|--------|---------|-------|----------------|
| processing_timestamp | time.time() | Unix timestamp | When processing completed |
| version | '2.0.0' | Semantic version | Temporal compute version |

## 🔄 Data Pipeline

### Source to Feature Flow
```
System Clock / Code Version
    ↓ (runtime metadata generation)
temporal_compute.py:1806-1807
    ↓ (timestamp and version injection)
Root Level Output
```

### Implementation Location
```python
# System metadata injection
/rumiai_v2/processors/temporal_compute.py:1806-1807
├── processing_timestamp: time.time()
└── version: '2.0.0' (hardcoded pipeline version)
```

## 🎨 Feature Engineering Opportunities

### Current Limitations
- No processing duration tracking
- Missing pipeline component versions
- No data quality metrics
- Static version string (not dynamic)

### Proposed Enhancements
- [ ] Add processing_duration_seconds for performance monitoring
- [ ] Include service_versions for each ML component
- [ ] Implement data_quality_score based on feature completeness
- [ ] Add pipeline_mode indicator (parallel vs sequential)

## 📊 Validation & Testing

### Feature Presence Verification
```python
# Verify all engagement and metadata features exist
import json
with open('insights/[video_id]_temporal_windows_updated.json') as f:
    data = json.load(f)

metadata = data.get('metadata', {})

# Check Virality Metrics
virality_metrics = ['digg_count', 'play_count', 'collect_count', 'share_count', 'comment_count']
for metric in virality_metrics:
    assert metric in metadata
    assert metadata[metric] >= 0

# Check Video Metadata
assert 'create_time' in metadata
assert 'author' in metadata
assert 'description' in metadata
assert 'video_id' in metadata
assert 'duration' in metadata

# Check Demographics
if 'gender_detection' in metadata:
    gender_data = metadata['gender_detection']
    assert 'gender' in gender_data
    assert 'confidence' in gender_data
    assert 'method' in gender_data
    assert gender_data['gender'] in ['male', 'female', 'multiple_people']

# Check Hashtag Analysis
if 'hashtag_analysis' in metadata:
    hashtag_data = metadata['hashtag_analysis']
    required_fields = ['hashtag_count', 'generic_hashtag_count', 'specific_hashtag_count', 'generic_ratio']
    for field in required_fields:
        assert field in hashtag_data

# Check System Fields
assert 'processing_timestamp' in data
assert 'version' in data
```

### Value Range Validation
```python
# Ensure engagement and metadata features are properly bounded
# Virality metrics should be non-negative
for metric in virality_metrics:
    assert metadata[metric] >= 0

# Duration should be reasonable for TikTok
assert 1 <= metadata['duration'] <= 600

# Gender confidence should be 0-1
if 'gender_detection' in metadata:
    assert 0 <= metadata['gender_detection']['confidence'] <= 1

# Hashtag ratios should be 0-1
if 'hashtag_analysis' in metadata:
    hashtag_data = metadata['hashtag_analysis']
    assert 0 <= hashtag_data['generic_ratio'] <= 1
    assert hashtag_data['hashtag_count'] >= 0
    assert hashtag_data['generic_hashtag_count'] <= hashtag_data['hashtag_count']
    assert hashtag_data['specific_hashtag_count'] <= hashtag_data['hashtag_count']
    assert hashtag_data['generic_hashtag_count'] + hashtag_data['specific_hashtag_count'] == hashtag_data['hashtag_count']
```

### Dependency Validation
```python
# Check critical dependencies for metadata features
# Apify scraper required for most metadata
assert metadata['video_id'] != 'unknown', "Apify scraper failed to provide video_id"

# DeepFace required for pitch normalization
if any('pitch' in str(data).lower() for temporal_window in data.get('temporal_windows', {}).values()):
    assert 'gender_detection' in metadata, "Pitch metrics require DeepFace gender detection"

# System fields should always be present
assert data['processing_timestamp'] > 0, "Processing timestamp missing"
assert data['version'], "Version identifier missing"
```

## 🚀 Feature Importance Ranking

### For Success Prediction
1. **play_count**: Primary success metric - view count drives everything
2. **digg_count + comment_count**: Engagement depth indicators
3. **share_count + collect_count**: Viral spread and value indicators
4. **duration**: Optimal length varies by content type (15s, 30s, 60s sweet spots)

### For Content Strategy Classification
1. **hashtag_analysis.generic_ratio**: Discovery vs niche targeting strategy
2. **description length + hashtag_count**: Professional vs casual content
3. **create_time patterns**: Optimal posting timing analysis
4. **author consistency**: Creator vs brand account identification

### For Demographic Targeting
1. **gender_detection**: Required for normalized voice analysis
2. **author + virality correlation**: Creator performance patterns
3. **description sentiment**: Content tone classification
4. **hashtag strategy + engagement**: Strategy effectiveness measurement

### Cross-Feature Correlations to Monitor
1. **Virality metrics intercorrelation**: All engagement metrics should correlate positively
2. **Duration vs engagement**: Different optimal lengths for different content types
3. **Hashtag strategy vs success**: Generic vs specific strategy effectiveness
4. **Gender detection confidence vs pitch metrics**: Low confidence affects pitch reliability

### API and System Dependencies
1. **Apify service reliability**: 5-minute timeout risk blocks entire pipeline
2. **DeepFace for pitch features**: Gender detection failure breaks pitch normalization
3. **TikTok API changes**: Field name mapping may break with platform updates
4. **Processing timestamp accuracy**: System clock critical for temporal analysis