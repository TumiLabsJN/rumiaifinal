## 🎥 C. Video Selection Criteria & Apify Integration

### Apify TikTok Scraping Investigation Results

#### Volume Limits Analysis

**CRITICAL LIMITATION DISCOVERED:**
- **Hard Limit**: 400-800 videos per hashtag maximum (TikTok platform limitation, not Apify)
- **Our Requirement**: 240 videos (60 per duration bucket × 4 buckets: 40 top + 20 bottom each)
- **Status**: ✅ Within limits, but limited headroom for filtering

#### Date Range Filtering Limitation

**MAJOR CONSTRAINT:**
- ❌ **No date filtering available for hashtag searches** 
- ✅ **Date filtering only available for profile scraping**
- **Impact**: Cannot filter "videos posted after 01/05/2025" during scraping

**Required Workaround - Post-Processing Date Filter:**
```python
def filter_by_date_after_scraping(videos, min_date):
    """
    Client-side date filtering since Apify cannot filter hashtag results by date
    """
    filtered = []
    for video in videos:
        # Convert video.creation_date to datetime if needed
        if video.creation_date >= min_date:
            filtered.append(video)
    return filtered

def select_videos_with_date_constraint(hashtag, min_date, target_per_bucket=50):
    # 1. Scrape maximum available from hashtag (400-800 videos)
    all_videos = apify_scraper.scrape_hashtag(hashtag, max_count=800)
    
    # 2. Filter by date client-side (REQUIRED step)
    recent_videos = filter_by_date_after_scraping(all_videos, min_date)
    
    # 3. Calculate engagement rates for selection
    for video in recent_videos:
        video.engagement_rate = (video.likes + video.comments + video.shares) / video.views
    
    # 4. Sort and select by duration buckets
    return select_top_by_duration_buckets(recent_videos, target_per_bucket)
```

#### Scraper Cost Comparison & Alternative

**Regular TikTok Hashtag Scraper:**
- **Cost**: $0.005 per video
- **300 videos**: $1.50 per hashtag analysis
- **Reliability**: Official Apify scraper, well-tested

**Super TikTok Scraper Alternative:**
- **Cost**: $0.0005 per video (10x cheaper)
- **300 videos**: $0.15 per hashtag analysis  
- **Savings**: 90% cost reduction for production volume
- **Trade-offs**: Third-party developer, potentially slower, less support

**Cost Analysis for Scale:**
```python
# For 10 hashtags (typical client analysis):
regular_scraper_cost = 10 * $1.25 = $12.50
super_scraper_cost = 10 * $0.125 = $1.25
annual_savings = ($12.50 - $1.25) * 52 weeks = $585 per client
```

#### Available Engagement Metrics (Verified)

**✅ All Required Data Fields Available:**
- **Views**: Available as `plays` field
- **Likes**: Available as `diggCount` field  
- **Comments**: Available as `commentCount` field
- **Shares**: Available as `shareCount` field
- **Duration**: Available for bucket sorting (0-15s, 16-30s, etc.)
- **Creation Date**: Available for post-processing date filter
- **Video URL**: Available for download and RumiAI analysis

### 🎥 C.1 Video Selection Strategy

#### Recency Handling: User-Controlled Date Cutoff
**No complex weighting needed** - The user specifies the date cutoff during setup configuration. Videos older than this date are simply excluded from analysis. This gives the user full control over the recency/freshness of patterns being analyzed.

#### Primary Selection Criterion: Engagement Rate

**Recommended Approach:**
```python
def calculate_engagement_rate(video):
    """
    Primary metric for top-performing video selection
    """
    total_engagement = video.likes + video.comments + video.shares
    return total_engagement / video.views if video.views > 0 else 0

def select_top_videos_by_engagement(videos, date_cutoff, min_thresholds=True):
    """
    Select videos using engagement rate with quality filters
    User specifies date_cutoff during setup - no complex recency weighting needed
    """
    qualified_videos = []
    
    for video in videos:
        # User-defined recency cutoff (specified during setup)
        if video.created_date < date_cutoff:
            continue  # Skip videos older than user-specified date
            
        # Quality filters
        if min_thresholds:
            if video.views < 1000:  # Minimum sample size
                continue
                
        engagement_rate = calculate_engagement_rate(video)
        
        # Minimum engagement threshold (filter dead content)
        if engagement_rate < 0.02:  # 2% minimum
            continue
            
        qualified_videos.append({
            "video": video,
            "engagement_rate": engagement_rate,
            "composite_score": engagement_rate * (1 + shares_boost_factor(video))
        })
    
    return sorted(qualified_videos, key=lambda x: x["composite_score"], reverse=True)
```

#### Duration Bucket Distribution

**Process:**
1. **Scrape hashtag**: Get 400-800 videos maximum
2. **Filter by date**: Apply client-side date constraints  
3. **Calculate engagement**: Rate all remaining videos
4. **Segment by duration**: Sort into 4 buckets (0-15s, 16-30s, 31-60s, 61-120s)
5. **Select top 50**: From each bucket by engagement rate

**Risk Mitigation:**
- **Insufficient recent videos**: Some duration buckets may have <50 videos after date filtering
- **Solution**: Lower date constraints or accept fewer videos per bucket
- **Monitoring**: Track actual video counts per bucket for each hashtag

#### Implementation Recommendation

**Phase 1: Validation (Start Here)**
- Use **Regular TikTok Hashtag Scraper** for first 2-3 clients
- Validate data quality and engagement rate accuracy
- Confirm date filtering workflow effectiveness

**Phase 2: Scale Optimization**
- Migrate to **Super TikTok Scraper** for 90% cost savings
- Implement batch processing for multiple hashtags
- Monitor performance and reliability differences

**Phase 3: Advanced Filtering**
- Consider multiple scraping sessions over time for better date coverage
- Implement dynamic thresholds based on hashtag performance
- Add viral velocity metrics (engagement rate over time)

---

### 🎥 C.2 Checkpoint & Resume System for Sequential Processing

#### The Challenge

When processing 160 videos (40 per bucket × 4 buckets):
- Video #80 fails due to bug (YOLO crash, MediaPipe error, etc.)
- System fails fast to identify bug
- After fixing, need to resume from video #81, not restart

#### Simple Checkpoint Manager for One-by-One Processing

```python
class SimpleCheckpointManager:
    """
    Lightweight checkpoint system for sequential video processing
    Saves progress after each successful video
    """
    def __init__(self, hashtag_id, run_id):
        self.checkpoint_file = Path(f"checkpoints/{hashtag_id}_{run_id}.json")
        self.completed_file = Path(f"checkpoints/{hashtag_id}_{run_id}_completed.jsonl")
        
    def save_progress(self, video_id, bucket, position, features):
        # Append completed video to JSONL (one line per video)
        with open(self.completed_file, 'a') as f:
            f.write(json.dumps({
                "position": position,
                "video_id": video_id,
                "bucket": bucket,
                "features": features,
                "timestamp": datetime.now().isoformat()
            }) + '\n')
        
        # Update checkpoint with latest position
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                "last_position": position,
                "last_video_id": video_id,
                "last_bucket": bucket,
                "total_processed": position + 1
            }, f)
    
    def get_resume_point(self):
        if not self.checkpoint_file.exists():
            return 0, None
        
        with open(self.checkpoint_file) as f:
            checkpoint = json.load(f)
        
        return checkpoint["last_position"] + 1, checkpoint["last_bucket"]
    
    def load_completed_features(self):
        """Load all previously processed features for ML training"""
        if not self.completed_file.exists():
            return []
        
        features = []
        with open(self.completed_file) as f:
            for line in f:
                video_data = json.loads(line)
                features.append(video_data["features"])
        return features
```

#### Integration with Sequential Processing

```python
async def process_hashtag_videos_with_checkpoint(hashtag_id, videos_by_bucket):
    """
    Process 200 videos sequentially with checkpoint/resume
    """
    run_id = str(uuid.uuid4())
    checkpoint = SimpleCheckpointManager(hashtag_id, run_id)
    
    # Check for existing progress
    start_position, last_bucket = checkpoint.get_resume_point()
    
    if start_position > 0:
        logger.info(f"✓ Resuming from position {start_position}/200")
        logger.info(f"✓ Found {start_position} completed videos")
    
    position = start_position
    
    for bucket, videos in videos_by_bucket.items():
        # Skip completed buckets
        if last_bucket and bucket < last_bucket:
            continue
        
        # Calculate starting index within bucket
        start_index = position % 40 if bucket == last_bucket else 0
        
        for video in videos[start_index:]:
            try:
                # Process single video
                logger.info(f"Processing video {position+1}/200: {video.id}")
                features = await extract_features(video)
                
                # Save immediately after success
                checkpoint.save_progress(video.id, bucket, position, features)
                
                logger.info(f"✓ Completed {position+1}/200: {video.id}")
                position += 1
                
            except Exception as e:
                # Fail fast with clear resume instructions
                logger.error(f"✗ Failed at position {position}, video {video.id}")
                logger.error(f"Error: {e}")
                logger.info(f"To resume after fix: run with same hashtag_id")
                logger.info(f"Progress saved: {position} videos completed")
                raise  # Fail fast for debugging
    
    logger.info(f"✅ Successfully processed all 200 videos!")
    
    # Load all features for ML training
    all_features = checkpoint.load_completed_features()
    return all_features
```

#### Benefits of Sequential Processing with Checkpoints

1. **Simple Implementation**: Single-threaded, easy to debug
2. **Immediate Recovery**: Each video saved independently
3. **Clear Progress**: Know exactly where failure occurred
4. **Zero Re-processing**: Never repeat completed videos
5. **Fail-Fast Compatible**: Bugs identified immediately
6. **Cost Efficient**: No wasted API calls or processing

#### Checkpoint File Structure

```
checkpoints/
├── nutrition_hashtag_uuid123.json          # Current position
├── nutrition_hashtag_uuid123_completed.jsonl  # All completed videos
└── completed/                              # Successful runs moved here
    └── nutrition_hashtag_uuid123/
```

**Usage Example:**
```bash
# First run - fails at video 80
> python process_hashtag.py --hashtag nutrition
Processing video 80/200: 7374651255392210219
✗ Failed: YOLO detection error
To resume: run with same hashtag_id

# After fixing bug
> python process_hashtag.py --hashtag nutrition --resume
✓ Resuming from position 80/200
✓ Found 79 completed videos
Processing video 80/200: 7374651255392210219
✓ Completed 80/200
...
✅ Successfully processed all 200 videos!
```

### 🎥 C.3 Temporal Window Data Validation

Before ML training, validate that temporal windows are correctly extracted:

```python
def validate_temporal_windows(video_features, video_duration):
    """
    Ensure temporal windows are correctly extracted based on MLMVP2 architecture
    """
    validations = {
        'hook_present': all(f in video_features for f in [
            'hook_0to3s_density', 'hook_effectiveness_score'
        ]),
        'middle_consistent': (
            video_duration <= 6 or 'middle_is_present' in video_features
        ),
        'closing_present': all(f in video_features for f in [
            'closing_3s_density', 'closing_3s_has_cta'
        ])
    }
    
    # Duration-specific validations
    if video_duration >= 16 and video_duration <= 30:
        # Should have bins but not piecewise
        validations['has_bins'] = 'middle_early_density' in video_features
        validations['no_piecewise'] = 'middle_slope_early' not in video_features
        
    elif video_duration >= 31 and video_duration <= 60:
        # Should have bins AND piecewise
        validations['has_bins'] = 'middle_early_density' in video_features
        validations['has_piecewise'] = 'middle_slope_early' in video_features
        validations['no_rhythm'] = 'middle_burstiness' not in video_features
        
    elif video_duration >= 61:
        # Should have everything
        validations['has_bins'] = 'middle_early_density' in video_features
        validations['has_piecewise'] = 'middle_slope_early' in video_features
        validations['has_rhythm'] = 'middle_burstiness' in video_features
    
    # Log validation results
    if not all(validations.values()):
        logger.warning(f"Temporal validation failed for {video_duration}s video:")
        for check, passed in validations.items():
            if not passed:
                logger.warning(f"  ❌ {check}")
    
    return all(validations.values())

# Usage in pipeline
for video in videos:
    features = extract_temporal_features(video.timeline, video.type, video.duration)
    if not validate_temporal_windows(features, video.duration):
        raise ValueError(f"Invalid temporal extraction for video {video.id}")
```

### 🎥 C.4 Feature Scaling Strategy for Ensemble Models

#### Why Scaling is Required

Our MVP ensemble includes models with different scaling requirements:
```python
models = {
    "random_forest": RandomForestRegressor(),  # ✅ Doesn't need scaling
    "decision_tree": DecisionTreeRegressor(),  # ✅ Doesn't need scaling  
    "linear_model": LinearRegression(),        # ⚠️ Benefits from scaling
    "clustering": KMeans(n_clusters=5)         # 🔴 BREAKS without scaling!
}
```

**The Problem**: Our features have wildly different scales:
- `views`: 10,000,000 (millions)
- `overlayDensity`: 0.448 (fraction)
- `totalOverlays`: 26 (count)

**Critical Issue**: KMeans clustering uses Euclidean distance - without scaling, `views` will completely dominate all distance calculations, making clustering meaningless.

#### RobustScaler: Optimal for Social Media Data

```python
from sklearn.preprocessing import RobustScaler
import joblib
import numpy as np

def prepare_ml_features(features_list):
    """
    Scale all features using RobustScaler
    Handles viral outliers common in social media metrics
    
    Args:
        features_list: List of feature dictionaries from processed videos (>100 features each)
    
    Returns:
        X_scaled: Scaled feature matrix ready for ML
        scaler: Fitted scaler for inference
    """
    # Convert to numpy matrix
    X = np.array([list(f.values()) for f in features_list])
    
    # RobustScaler uses median and IQR, robust to outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler for inference on new videos
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    
    return X_scaled, scaler
```

**Why RobustScaler?**

1. **Viral Outliers Are Common**:
   - 99% of videos: 1K - 100K views
   - 1% viral videos: 1M - 10M views
   - RobustScaler uses median/IQR instead of mean/std
   - Outliers remain outliers (important signal)

2. **Power-Law Distributions**:
   - Social media metrics follow power law
   - StandardScaler would be skewed by top 1%
   - RobustScaler centers on the typical 99%

3. **Works for All Models**:
   - KMeans: Gets properly scaled distances
   - LinearRegression: Gets normalized coefficients
   - Tree models: Unaffected (split points just shift)

#### Implementation in Training Pipeline

```python
async def train_ensemble_with_scaling(hashtag_id):
    """
    Complete training pipeline with scaling
    """
    # 1. Load processed features from checkpoint
    checkpoint = SimpleCheckpointManager(hashtag_id, run_id)
    features_list = checkpoint.load_completed_features()
    
    # 2. Extract feature matrix and target
    X = extract_all_features(features_list)
    y = extract_engagement_targets(features_list)
    
    # 3. Scale features for ensemble
    X_scaled, scaler = prepare_ml_features(X)
    
    # 4. Train all models with scaled features
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        ).fit(X_scaled, y),
        
        "decision_tree": DecisionTreeRegressor(
            max_depth=8, 
            random_state=42
        ).fit(X_scaled, y),
        
        "linear_model": LinearRegression().fit(X_scaled, y),
        
        "clustering": KMeans(
            n_clusters=5, 
            random_state=42
        ).fit(X_scaled)
    }
    
    # 5. Save models and scaler
    for name, model in models.items():
        joblib.dump(model, f'models/{hashtag_id}_{name}.pkl')
    
    return models, scaler
```

#### Inference with Saved Scaler

```python
def predict_new_video(video_features, hashtag_id):
    """
    Predict performance for new video using saved models
    """
    # Load saved scaler
    scaler = joblib.load('models/feature_scaler.pkl')
    
    # Scale new video features using same scaler
    X_new = extract_all_features([video_features])
    X_scaled = scaler.transform(X_new)
    
    # Load and predict with each model
    predictions = {}
    for model_name in ['random_forest', 'decision_tree', 'linear_model']:
        model = joblib.load(f'models/{hashtag_id}_{model_name}.pkl')
        predictions[model_name] = model.predict(X_scaled)[0]
    
    # Clustering assignment
    kmeans = joblib.load(f'models/{hashtag_id}_clustering.pkl')
    cluster = kmeans.predict(X_scaled)[0]
    
    return predictions, cluster
```

#### Key Implementation Notes

1. **Scale Once**: Apply same scaling to all models (simpler pipeline)
2. **Save Scaler**: Critical for consistent inference on new videos
3. **Robust to Outliers**: Viral videos won't distort scaling
4. **No Feature Selection**: Use all features (>100, exact count TBC) - let models decide importance

### 🎥 C.5 Missing Data Handling: Simplified by Service Contracts

#### Upstream Service Contracts Guarantee Valid Data

Our ML pipeline benefits from robust service contracts in upstream services (YOLO, MediaPipe, Whisper, OCR). These contracts ensure:
- **No null/undefined values** - Services always return valid data structures
- **No error states** - Exceptions are caught and handled upstream
- **Consistent schemas** - All required fields are always present

#### Valid Empty Results vs Errors

**What ML Pipeline Receives (ALL VALID):**
```python
# These are valid video characteristics, not errors:
{
    "objectTimeline": {},        # ✅ No objects in video (abstract content)
    "gestureTimeline": {},       # ✅ No gestures detected (product-only)
    "speechTimeline": {},        # ✅ No speech (music-only video)
    "textOverlays": 0,          # ✅ No text overlays (visual-only)
    "emotions": [],             # ✅ No faces detected (landscape)
    "densityCurve": []          # ✅ No density variations
}
```

**What We'll NEVER Receive (Caught Upstream):**
```python
# Service contracts prevent these from reaching ML:
{
    "objectTimeline": null,      # ❌ Service contract prevents
    "gestureTimeline": "error",  # ❌ Caught upstream
    "speechTimeline": undefined, # ❌ Never happens
    "features": NaN             # ❌ Validated upstream
}
```

#### Simplified Feature Extraction

```python
def extract_features_from_validated_data(raw_output):
    """
    Simple feature extraction - trust upstream contracts
    Empty collections are valid video characteristics
    
    No error handling needed - data is pre-validated
    """
    features = {}
    duration = raw_output.get("duration", 1)  # Prevent div by zero
    
    # Object features - empty dict = no objects (valid)
    object_timeline = raw_output.get("objectTimeline", {})
    features["object_count"] = len(object_timeline)
    features["object_density"] = len(object_timeline) / duration
    features["has_objects"] = 1 if object_timeline else 0
    
    # Gesture features - empty dict = no gestures (valid)
    gesture_timeline = raw_output.get("gestureTimeline", {})
    features["gesture_count"] = len(gesture_timeline)
    features["gesture_variety"] = len(set(gesture_timeline.values())) if gesture_timeline else 0
    features["has_gestures"] = 1 if gesture_timeline else 0
    
    # Speech features - empty dict = silence (valid)
    speech_timeline = raw_output.get("speechTimeline", {})
    features["speech_density"] = len(speech_timeline) / duration
    features["words_count"] = sum(len(text.split()) for text in speech_timeline.values()) if speech_timeline else 0
    features["has_speech"] = 1 if speech_timeline else 0
    
    # Overlay features - zero = no overlays (valid)
    features["overlay_count"] = raw_output.get("totalOverlays", 0)
    features["overlay_density"] = raw_output.get("overlayDensity", 0.0)
    
    # Array features - empty = no variations (valid)
    density_curve = raw_output.get("densityCurve", [])
    if density_curve:
        features["density_mean"] = np.mean([d["density"] for d in density_curve])
        features["density_std"] = np.std([d["density"] for d in density_curve])
        features["density_max"] = max(d["density"] for d in density_curve)
    else:
        features["density_mean"] = 0.0
        features["density_std"] = 0.0
        features["density_max"] = 0.0
    
    return features  # All values guaranteed valid numbers
```

#### Benefits of Service Contract Approach

1. **No Try/Catch Blocks**: Errors caught upstream
2. **No Null Checks**: Service contracts guarantee non-null
3. **No Validation**: Data pre-validated by services
4. **Clean Code**: Focus on transformation, not error handling
5. **Clear Semantics**: Empty = valid characteristic, not error

#### Integration with ML Pipeline

```python
async def process_video_for_ml(video_data):
    """
    Process video data for ML training
    Trusts upstream service contracts
    """
    # Extract features (no error handling needed)
    features = extract_features_from_validated_data(video_data)
    
    # All features guaranteed to be valid numbers
    # Empty detections already encoded as zeros
    
    # Continue with scaling and model training
    return features
```

#### Key Principle

**Empty ≠ Error**: 
- Empty timelines represent actual video content (no objects/speech/gestures)
- These are valid data points that help models learn what makes videos without these elements successful
- A video with zero gestures but high engagement teaches the model that gestures aren't always necessary

This simplified approach reduces code complexity and focuses on the actual ML logic rather than defensive programming.

```
Validating 60s video features:
✓ hook_present: All 8 hook features found
✓ middle_consistent: middle_is_present = true
✓ has_bins: middle_early_density present
✓ has_piecewise: middle_slope_early present  
✓ has_rhythm: middle_burstiness present
✓ closing_present: All 8 closing features found
✅ Temporal validation PASSED
```

This demonstrates how the temporal window architecture provides rich, duration-appropriate insights that generic feature extraction would miss.

### 🎥C.6 Pattern Aggregation via Claude API

#### The Role of Claude in Pattern Generation

After ML training, Claude serves as our pattern aggregation engine, transforming statistical insights into actionable creative strategies.

```python
def prepare_patterns_for_claude(model, features, engagement_rates):
    """
    Prepare ML results for Claude to interpret into 10 creative reports
    """
    # Statistical summaries from ML models
    pattern_data = {
        "feature_importance": dict(zip(feature_names, model.feature_importances_)),
        "top_20_features": get_top_features(model, 20),
        "engagement_tiers": {
            "top_10_percent": analyze_tier(features, engagement_rates, 90, 100),
            "top_20_percent": analyze_tier(features, engagement_rates, 80, 90),
            "average_performers": analyze_tier(features, engagement_rates, 40, 60)
        },
        "cluster_analysis": {
            "num_clusters": 5,
            "cluster_summaries": get_cluster_characteristics(features, model.clustering)
        },
        "duration_bucket_patterns": analyze_by_duration_bucket(features, engagement_rates)
    }
    
    # Request to Claude
    pattern_data["request"] = """
    Based on these ML insights, generate 10 distinct creative strategy reports:
    1. Hook Optimization Strategy
    2. CTA Effectiveness Guide  
    3. Pacing & Rhythm Patterns
    4. Visual Element Coordination
    5. Emotional Journey Mapping
    6. Text Overlay Best Practices
    7. Trend-Jacking Opportunities
    8. Duration-Specific Tactics
    9. Engagement Acceleration Techniques
    10. Viral Replication Framework
    
    Each report should include:
    - Specific, actionable recommendations
    - Statistical backing from the data
    - Examples from top performers
    - Clear do's and don'ts
    """
    
    return pattern_data

async def generate_creative_reports(hashtag_id):
    """
    Complete flow from ML to creative reports via Claude
    """
    # 1. Load ML results
    model = load_model(hashtag_id)
    features = load_features(hashtag_id)
    engagement_rates = load_engagement_data(hashtag_id)
    
    # 2. Prepare pattern data
    pattern_data = prepare_patterns_for_claude(model, features, engagement_rates)
    
    # 3. Send to Claude for interpretation
    reports = await claude_api.generate_strategies(
        pattern_data,
        num_reports=10,
        report_style="actionable_creative_guide"
    )
    
    # 4. Save reports
    save_creative_reports(hashtag_id, reports)
    
    return reports
```

#### Why Claude for Pattern Aggregation?

**We provide the statistics:**
- Feature importance scores
- Cluster assignments
- Performance tier comparisons
- Statistical correlations

**Claude provides the interpretation:**
- Translates statistics into creative language
- Identifies non-obvious pattern combinations
- Generates actionable recommendations
- Creates narrative structure for reports

**Benefits:**
- No complex aggregation logic needed in our code
- Claude's language skills create better reports
- Flexible report generation based on findings
- Natural language output ready for clients

### 🎥C.7 Engagement Data Source

#### Engagement Score Calculation

For **top mode** video selection, engagement is calculated using a composite score that weights shares heavily (viral indicator).

**Formula**: `engagement_score = views × (1 + share_rate × 10)`

**Complete implementation details**: See [MLAnalysisMode.md - Engagement Score Calculation](./MLAnalysisMode.md#engagement-score-calculation)

#### Engagement Metrics from Apify

All engagement data comes directly from Apify's TikTok scraper output:

```python
# Apify provides these metrics for each video:
{
    "playCount": 3200000,      # → views
    "diggCount": 346500,       # → likes
    "commentCount": 872,        # → comments
    "shareCount": 15500         # → shares
}
```

#### Data Flow for Engagement Metrics

```python
# 1. Apify scrapes TikTok video
apify_data = await apify_client.scrape_video(video_url)

# 2. Parse into VideoMetadata
video = VideoMetadata.from_apify_data(apify_data)
# Automatically maps: playCount→views, diggCount→likes, etc.

# 3. Calculate engagement rate during metadata analysis
metadata_analysis = {
    "CoreMetrics": {
        "engagementRate": 11.34,  # Calculated
        "viewCount": 3200000,      # From Apify
    },
    "Interactions": {
        "likeCount": 346500,       # From Apify
        "commentCount": 872,        # From Apify
        "shareCount": 15500         # From Apify
    }
}

# 4. Use as ML target variable
X = extract_all_features(video_analyses)
y = [video["engagementRate"] for video in metadata_analyses]
model.fit(X, y)  # Predict engagement rate
```

#### Engagement Data Characteristics

**Reliability:**
- ✅ Apify always provides these metrics (core TikTok data)
- ✅ If missing, video is skipped (not processed)
- ✅ Service contracts ensure valid numbers (0 if truly zero)

**Freshness:**
- Point-in-time snapshot when scraped
- Sufficient for MVP (analyzing established patterns)
- No need to track changes over time initially

**Usage in ML Pipeline:**
```python
def select_top_videos_by_engagement(videos):
    """
    Primary selection criterion for "top performing" videos
    """
    for video in videos:
        # Calculate engagement rate from Apify data
        engagement_rate = (
            video.likes + 
            video.comments + 
            video.shares
        ) / video.views
        
        video.engagement_rate = engagement_rate
    
    # Select top 50 per bucket by engagement rate
    return sorted(videos, key=lambda x: x.engagement_rate, reverse=True)[:50]
```

This engagement rate becomes the target variable that our ML models learn to predict based on the creative features (>100 features, exact count TBC).

### 🎥C.8 Data Storage Architecture

#### MVP: File-Based Storage (Recommended)

For the MVP phase, use structured file storage to avoid database complexity:

```python
class MVPDataStore:
    """
    Simple file-based storage for MVP
    No database required, human-readable JSON files
    """
    def __init__(self, base_path="data"):
        self.base_path = Path(base_path)
    
    def save_video_features(self, client, hashtag, video_id, features):
        """Save extracted features for a video"""
        path = self.base_path / client / hashtag / "features" / f"{video_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        feature_record = {
            "video_id": video_id,
            "features": features,
            "extraction_date": datetime.now().isoformat(),
            "feature_version": "v1.0"
        }
        
        with open(path, 'w') as f:
            json.dump(feature_record, f, indent=2)
    
    def load_hashtag_features(self, client, hashtag):
        """Load all features for ML training"""
        path = self.base_path / client / hashtag / "features"
        features = []
        
        for file in sorted(path.glob("*.json")):
            with open(file) as f:
                features.append(json.load(f))
        
        return features
    
    def save_ml_model(self, client, hashtag, models, scaler):
        """Save trained models and scaler"""
        model_path = self.base_path / client / hashtag / "models"
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for name, model in models.items():
            joblib.dump(model, model_path / f"{name}_model.pkl")
        
        # Save scaler
        joblib.dump(scaler, model_path / "feature_scaler.pkl")
        
        # Save metadata
        metadata = {
            "training_date": datetime.now().isoformat(),
            "model_version": "v1.0",
            "feature_count": "TBC (>100)",
            "video_count": len(list((self.base_path / client / hashtag / "features").glob("*.json")))
        }
        
        with open(model_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_patterns(self, client, hashtag, patterns):
        """Save discovered patterns"""
        pattern_path = self.base_path / client / hashtag / "patterns"
        pattern_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(pattern_path / f"patterns_{timestamp}.json", 'w') as f:
            json.dump(patterns, f, indent=2)
```

**Directory Structure:**
```
data/
├── nutritional_supplements/           # Client
│   └── nutrition/                    # Hashtag
│       ├── features/                 # Extracted features
│       │   ├── 7274651255392210219.json
│       │   ├── 7274651255392210220.json
│       │   └── ... (200 videos)
│       ├── models/                   # Trained ML models
│       │   ├── random_forest_model.pkl
│       │   ├── decision_tree_model.pkl
│       │   ├── linear_model.pkl
│       │   ├── clustering_model.pkl
│       │   ├── feature_scaler.pkl
│       │   └── metadata.json
│       └── patterns/                 # Discovered patterns
│           └── patterns_20250115_143022.json
```

**Benefits for MVP:**
- ✅ **Zero setup** - Start immediately, no database required
- ✅ **Human readable** - JSON files can be inspected/edited
- ✅ **Git friendly** - Can version control data and models
- ✅ **Easy debugging** - See exactly what's stored
- ✅ **Simple backup** - Just copy files

#### Production: PostgreSQL with JSONB (Future)

For production scale, migrate to PostgreSQL:

```sql
-- Future production schema
CREATE TABLE clients (
    client_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE hashtags (
    hashtag_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID REFERENCES clients(client_id),
    name VARCHAR(255) NOT NULL,
    tiktok_url VARCHAR(500),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE videos (
    video_id VARCHAR(50) PRIMARY KEY,
    hashtag_id UUID REFERENCES hashtags(hashtag_id),
    duration_segment VARCHAR(20),  -- '0-15s', '16-30s', etc.
    engagement_metrics JSONB,      -- views, likes, shares, etc.
    extracted_features JSONB,      -- All ML features (>100, exact count TBC)
    processing_date TIMESTAMP,
    INDEX idx_segment (duration_segment),
    INDEX idx_hashtag (hashtag_id)
);

CREATE TABLE ml_models (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hashtag_id UUID REFERENCES hashtags(hashtag_id),
    model_type VARCHAR(50),        -- 'random_forest', 'kmeans', etc.
    model_binary BYTEA,            -- Serialized model
    performance_metrics JSONB,
    feature_importance JSONB,
    training_date TIMESTAMP DEFAULT NOW(),
    model_version VARCHAR(20)
);

CREATE TABLE discovered_patterns (
    pattern_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hashtag_id UUID REFERENCES hashtags(hashtag_id),
    pattern_type VARCHAR(100),
    pattern_data JSONB,
    confidence_score FLOAT,
    videos_supporting INTEGER,
    discovered_date TIMESTAMP DEFAULT NOW()
);
```

**Migration Path:**
1. **MVP Phase**: Use file-based storage
2. **Validation Phase**: Prove ML value with real clients
3. **Scale Phase**: Migrate to PostgreSQL when handling multiple clients
4. **Migration Script**: Simple script to load JSON files into database

**Why This Approach:**
- Start simple, scale when needed
- Avoid premature optimization
- Focus on ML value first, infrastructure later
- Easy migration path when ready

### 🎥C.9 Statistical Significance & Pattern Validation

#### Sample-Size-Adjusted Significance Thresholds

**Challenge**: Small datasets are naturally harder to achieve statistical significance, but we shouldn't penalize genuine patterns just because we have limited data.

**Solution**: Adjust p-value thresholds based on available sample size while maintaining meaningful effect size requirements.

```python
def classify_pattern_strength(p_value, effect_size, sample_size):
    """
    Sample-size-adjusted pattern classification
    Prevents small datasets from being unfairly penalized
    """
    # Always require meaningful business impact
    if abs(effect_size) < 0.15:  # Less than 15% improvement
        return "NEGLIGIBLE - Too small to matter"
    
    # Adjust significance thresholds based on sample reality
    if sample_size >= 80:
        # Large sample: strict academic standards
        thresholds = {"high": 0.01, "moderate": 0.05, "preliminary": 0.10}
    elif sample_size >= 40:
        # Medium sample: relaxed thresholds
        thresholds = {"high": 0.05, "moderate": 0.10, "preliminary": 0.15}
    else:
        # Small sample: very relaxed but still meaningful
        thresholds = {"high": 0.10, "moderate": 0.15, "preliminary": 0.20}
    
    # Classify based on adjusted thresholds
    if p_value < thresholds["high"]:
        return f"HIGH CONFIDENCE ({sample_size} videos)"
    elif p_value < thresholds["moderate"]:
        return f"MODERATE CONFIDENCE ({sample_size} videos)"
    elif p_value < thresholds["preliminary"]:
        return f"PRELIMINARY ({sample_size} videos)"
    else:
        return f"INCONCLUSIVE ({sample_size} videos)"
```

#### Cross-Validation Strategy

**Adaptive approach** based on available data per bucket:

```python
def select_validation_method(n_samples):
    """
    Choose appropriate validation based on sample size
    """
    if n_samples >= 50:
        return "StratifiedKFold", {"n_splits": 5}
    elif n_samples >= 30:
        return "StratifiedKFold", {"n_splits": 3}
    elif n_samples >= 20:
        return "Bootstrap", {"n_iterations": 100}
    else:
        return "LeaveOneOut", {}
```

#### Pattern Confidence Reporting

**Clear communication** to end users about pattern reliability:

```python
# Example output format
pattern_report = {
    "pattern": "Videos with 4+ text overlays",
    "effect": "+28% engagement increase",
    "confidence": "HIGH CONFIDENCE (85 videos)",
    "p_value": 0.003,
    "effect_size": 0.28,
    "recommendation": "IMPLEMENT - Strong evidence supports this strategy"
}

preliminary_report = {
    "pattern": "Hook timing at 2-3 seconds",
    "effect": "+19% share increase", 
    "confidence": "PRELIMINARY (34 videos)",
    "p_value": 0.08,
    "effect_size": 0.19,
    "recommendation": "TEST CAREFULLY - Promising but needs more data"
}
```

#### Statistical Test Selection

**Appropriate tests** for different pattern types:

```python
def test_pattern_significance(pattern_type, data_high, data_low):
    """
    Select appropriate statistical test based on data type
    """
    if pattern_type == "continuous":
        # T-test for numeric features (overlay count, timing, etc.)
        from scipy.stats import ttest_ind
        statistic, p_value = ttest_ind(data_high, data_low)
        
    elif pattern_type == "categorical":
        # Chi-square for categorical features (strategy types, etc.)
        from scipy.stats import chi2_contingency
        statistic, p_value, _, _ = chi2_contingency(data_high, data_low)
        
    elif pattern_type == "proportion":
        # Proportion test for binary outcomes
        from statsmodels.stats.proportion import proportions_ztest
        statistic, p_value = proportions_ztest(data_high, data_low)
    
    return statistic, p_value
```

#### Implementation Priority

**MVP Requirements:**
- ✅ Effect size threshold (15% minimum)
- ✅ Sample-size-adjusted p-values
- ✅ Clear confidence reporting
- ✅ Adaptive cross-validation

**Benefits:**
- **Fair evaluation** regardless of sample size
- **Business-focused** pattern detection
- **Transparent confidence** communication
- **Scientific rigor** without over-conservatism

---

