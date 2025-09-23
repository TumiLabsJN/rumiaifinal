## G. Competitor Handle Analysis System

#### Overview
Analyze competitor TikTok accounts to extract successful creative patterns and identify high-performing hashtag strategies, providing competitive intelligence for content strategy development.

#### Business Problem
- **Current Gap**: Only analyzing hashtag-based content, missing competitor-specific strategies
- **Opportunity**: Direct competitors have proven what works for the target audience
- **Value**: Reverse-engineer successful competitor strategies and hashtag selection

#### Proposed Solution: Handle-Based ML Analysis

**Architecture:**
```python
class CompetitorHandleAnalyzer:
    """
    Analyze competitor TikTok handles to extract patterns and hashtag strategies
    Similar to hashtag analysis but with additional competitive intelligence features
    """
    
    def __init__(self, client, competitor_handles):
        self.client = client
        self.competitor_handles = competitor_handles  # e.g., [@competitorA, @competitorB]
        self.handle_models = {}  # Structure: {handle: {bucket: model}}
        self.hashtag_intelligence = {}
        
    def analyze_competitor_handle(self, handle):
        """
        Full analysis pipeline for a competitor handle
        """
        analysis_results = {
            "handle": handle,
            "analysis_date": datetime.now(),
            "video_analysis": {},
            "hashtag_strategy": {},
            "content_patterns": {},
            "performance_insights": {}
        }
        
        # Step 1: Scrape competitor's videos (using Apify profile scraper)
        videos = self.scrape_handle_videos(handle, max_videos=500)
        
        # Step 2: Segment by duration buckets (same as hashtag analysis)
        videos_by_bucket = self.segment_by_duration(videos)
        
        # Step 3: Train bucket-specific models for this handle
        for bucket, bucket_videos in videos_by_bucket.items():
            if len(bucket_videos) >= 20:
                # Run through RumiAI pipeline
                analyzed_videos = self.run_rumiai_analysis(bucket_videos)
                
                # Train ML model for this bucket
                model = self.train_bucket_model(analyzed_videos, bucket)
                self.handle_models[handle][bucket] = model
                
                # Extract creative patterns
                analysis_results["content_patterns"][bucket] = {
                    "sample_size": len(bucket_videos),
                    "avg_engagement": self.calculate_avg_engagement(bucket_videos),
                    "top_patterns": model.get_top_patterns(),
                    "unique_strategies": self.identify_unique_strategies(model)
                }
        
        # Step 4: Extract hashtag intelligence
        analysis_results["hashtag_strategy"] = self.extract_hashtag_intelligence(videos)
        
        return analysis_results
    
    def extract_hashtag_intelligence(self, videos):
        """
        Identify which hashtags correlate with high performance
        """
        hashtag_performance = {}
        
        # Group videos by performance tier
        videos_sorted = sorted(videos, key=lambda v: v.engagement_rate, reverse=True)
        top_20_percent = videos_sorted[:int(len(videos) * 0.2)]
        
        # Extract hashtags from top performers
        for video in top_20_percent:
            for hashtag in video.hashtags:
                if hashtag not in hashtag_performance:
                    hashtag_performance[hashtag] = {
                        "frequency": 0,
                        "avg_engagement": [],
                        "video_count": 0,
                        "duration_distribution": {}
                    }
                
                hashtag_performance[hashtag]["frequency"] += 1
                hashtag_performance[hashtag]["avg_engagement"].append(video.engagement_rate)
                hashtag_performance[hashtag]["video_count"] += 1
                
                # Track which durations use this hashtag
                bucket = self.get_duration_bucket(video.duration)
                if bucket not in hashtag_performance[hashtag]["duration_distribution"]:
                    hashtag_performance[hashtag]["duration_distribution"][bucket] = 0
                hashtag_performance[hashtag]["duration_distribution"][bucket] += 1
        
        # Calculate metrics and rank hashtags
        ranked_hashtags = []
        for hashtag, data in hashtag_performance.items():
            avg_engagement = np.mean(data["avg_engagement"])
            ranked_hashtags.append({
                "hashtag": hashtag,
                "frequency_in_top_content": data["frequency"],
                "avg_engagement_rate": avg_engagement,
                "usage_rate": data["frequency"] / len(top_20_percent),
                "best_duration": max(data["duration_distribution"], 
                                   key=data["duration_distribution"].get),
                "recommendation": self.generate_hashtag_recommendation(data)
            })
        
        # Sort by engagement rate
        ranked_hashtags.sort(key=lambda x: x["avg_engagement_rate"], reverse=True)
        
        return {
            "top_performing_hashtags": ranked_hashtags[:10],
            "hashtag_combinations": self.analyze_hashtag_combinations(top_20_percent),
            "optimal_hashtag_count": self.calculate_optimal_hashtag_count(videos),
            "strategic_insights": self.generate_hashtag_insights(ranked_hashtags)
        }
    
    def analyze_hashtag_combinations(self, top_videos):
        """
        Identify which hashtag combinations appear together in successful content
        """
        from itertools import combinations
        
        combo_performance = {}
        
        for video in top_videos:
            # Look at 2-hashtag and 3-hashtag combinations
            for r in [2, 3]:
                for combo in combinations(video.hashtags, r):
                    combo_key = tuple(sorted(combo))
                    if combo_key not in combo_performance:
                        combo_performance[combo_key] = {
                            "count": 0,
                            "avg_engagement": []
                        }
                    combo_performance[combo_key]["count"] += 1
                    combo_performance[combo_key]["avg_engagement"].append(video.engagement_rate)
        
        # Find high-performing combinations
        successful_combos = []
        for combo, data in combo_performance.items():
            if data["count"] >= 3:  # Appears in at least 3 videos
                successful_combos.append({
                    "hashtags": list(combo),
                    "frequency": data["count"],
                    "avg_engagement": np.mean(data["avg_engagement"])
                })
        
        successful_combos.sort(key=lambda x: x["avg_engagement"], reverse=True)
        return successful_combos[:5]  # Top 5 combinations
    
    def compare_competitor_strategies(self, handles):
        """
        Compare strategies across multiple competitors
        """
        comparison = {
            "common_hashtags": {},
            "unique_strategies": {},
            "performance_benchmarks": {},
            "content_mix": {}
        }
        
        # Find hashtags used by multiple competitors
        all_hashtags = {}
        for handle in handles:
            handle_hashtags = self.hashtag_intelligence[handle]["top_performing_hashtags"]
            for hashtag_data in handle_hashtags:
                hashtag = hashtag_data["hashtag"]
                if hashtag not in all_hashtags:
                    all_hashtags[hashtag] = []
                all_hashtags[hashtag].append(handle)
        
        # Identify common vs unique hashtags
        comparison["common_hashtags"] = {
            hashtag: handles for hashtag, handles in all_hashtags.items()
            if len(handles) > 1
        }
        
        # Content mix comparison
        for handle in handles:
            comparison["content_mix"][handle] = self.calculate_duration_mix(handle)
        
        return comparison
```

#### Implementation Strategy

**Phase 1: Competitor Identification & Scraping**
```python
# Identify key competitors for client
competitors = {
    "direct_competitors": ["@competitor1", "@competitor2"],  # Same product category
    "aspirational_competitors": ["@marketleader1"],          # Where client wants to be
    "adjacent_competitors": ["@related1", "@related2"]       # Similar audience
}

# Scrape using Apify profile scraper (supports date filtering)
for handle in competitors["direct_competitors"]:
    videos = apify.scrape_profile(
        handle=handle,
        max_videos=500,
        date_from="2024-10-01"  # Last 3 months
    )
```

**Phase 2: Pattern Extraction & Hashtag Analysis**
- Run RumiAI analysis on competitor videos
- Train bucket-specific models (same as hashtag approach)
- Extract hashtag usage patterns from high performers

**Phase 3: Competitive Intelligence Report**
```json
{
  "competitor_analysis": {
    "handle": "@competitorA",
    "videos_analyzed": 247,
    "avg_engagement_rate": "5.8%",
    "content_strategy": {
      "0-15s": "45% of content, 7.2% avg engagement",
      "16-30s": "30% of content, 5.4% avg engagement",
      "31-60s": "20% of content, 4.1% avg engagement",
      "61-120s": "5% of content, 2.8% avg engagement"
    },
    "hashtag_strategy": {
      "top_hashtags": [
        {
          "hashtag": "#protein",
          "usage_rate": "82%",
          "avg_engagement": "7.1%",
          "insight": "Core hashtag, used in most content"
        },
        {
          "hashtag": "#fitness",
          "usage_rate": "45%",
          "avg_engagement": "6.8%",
          "insight": "Secondary hashtag, high performance"
        },
        {
          "hashtag": "#nutrition",
          "usage_rate": "38%",
          "avg_engagement": "5.2%",
          "insight": "Supporting hashtag"
        }
      ],
      "winning_combinations": [
        ["#protein", "#fitness", "#gym"],
        ["#nutrition", "#healthylifestyle", "#wellness"]
      ],
      "optimal_hashtag_count": 5.2
    },
    "creative_patterns": {
      "0-15s": [
        "Product reveal in first 2 seconds",
        "User testimonial format",
        "Before/after transformation"
      ],
      "16-30s": [
        "Tutorial with product integration",
        "Science explanation format",
        "Comparison with competitors"
      ]
    },
    "strategic_insights": [
      "Competitor focuses heavily on short-form content (75% under 30s)",
      "#protein is their anchor hashtag - appears in 82% of top content",
      "They avoid saturated hashtags like #fitness on longer videos",
      "Product placement always in first 5 seconds"
    ]
  },
  "recommended_actions": {
    "adopt_hashtags": ["#protein", "#supplements", "#preworkout"],
    "avoid_hashtags": ["#gym", "#bodybuilding"],  // Oversaturated for this competitor
    "content_mix_adjustment": "Increase 0-15s content to 40% (currently 25%)",
    "pattern_adoption": [
      "Implement product-first approach in opening",
      "Test testimonial format for 0-15s content"
    ]
  }
}
```

#### Business Value

**Competitive Intelligence:**
- Understand what's working for successful competitors
- Identify hashtag gaps and opportunities
- Benchmark performance expectations

**Hashtag Strategy Optimization:**
- Data-driven hashtag selection based on competitor success
- Understand hashtag combinations that drive engagement
- Avoid oversaturated or underperforming hashtags

**Content Strategy Refinement:**
- Learn from competitor's duration mix
- Adapt successful patterns while maintaining uniqueness
- Identify whitespace opportunities competitors missed

#### Technical Considerations

**Apify Integration:**
- Use profile scraper instead of hashtag scraper
- Can filter by date range for recent content
- Returns all necessary engagement metrics

**Storage Requirements:**
- Separate storage for competitor data (privacy/organization)
- Track analysis history for trend detection
- Maintain competitor performance benchmarks

**Ethical Considerations:**
- Only analyze publicly available content
- Focus on pattern learning, not copying
- Respect intellectual property

This system would provide crucial competitive intelligence, helping clients understand not just what works in their hashtags, but what's working for their successful competitors, enabling more strategic content planning and hashtag selection.
