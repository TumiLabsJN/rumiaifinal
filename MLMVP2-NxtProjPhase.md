
+++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---+++---------

MLMVP2 Next Phase

Very likely to have skeletons in the process of selecting features. Features who per the 1st report, and 2nd review:
- Are actually hardcoded
- Aren't compatible with ML

Important Points:
1. Revise for duplicity / similar features 
2. DOUBLECHECK: Will these selected features adapt to our temporal windows (first seconds + middle content + last seconds analysis?)
3. Data Transformation Coding to do to adapt to Kmeans and RF
4. Do we have architectural inconsistencies where we have features OUTSIDe the temporal window architecture?
    4.1   Make temporal windows the single source of truth
    4.2   Remove redundancy between global and window metrics
             FeaturesMLMVP.md --> ### Architectural Principle: Temporal Windows as Single Source of Truth
5. Window level counts vs Syncronization metrics of features

MVP: Window level counts
##We're moving overlay-specific metrics to windows
##Should have overlay peaks in temporal windows

Examples:
1) We won't be able to identify acceleration of any feature within windows, but between windows we can

+++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---+++---------
IMPORTANT KNOWLEDGE

POINT 1 - TEMPORAL WINDOW FEATURES
---
Total[Overlay/Words/Scenechanges]
Needs architecture change to be included in temporal windows
- And total to be sum of what is found in temporal windows 
    - To be architecturally clean and not have Global Features vs Local Features

POINT 2 - GLOBAL FEATURES
---
uniqueOverlayCount . Need Global video access to match how many unique stickers / text occurred in video. If limit by temporal, it will not work

  Global vs Temporal Features - The Reality:

  Inherently Global Features (legitimate exceptions):
  1. duration_sec - Video length, needed for rate calculations
  2. uniqueOverlayCount - Requires cross-window content comparison
  3. uniqueOverlayRatio - Derived from unique count
  4. aspectRatio - Video format property
  5. resolution - Technical property
  6. hashtagCount - Metadata not temporal


Won't damage ML if done correctly:

  1. ML algorithms handle mixed features fine:
    - RF: Treats each feature independently
    - K-means: After scaling, all features equal
    - Both can process global + temporal together
  2. Actually reflects reality:
    - Some patterns ARE global (video quality)
    - Some patterns ARE temporal (pacing)
    - Forcing everything into windows loses signal
  3. The key principle:
    - Temporal events → Windows (overlays appearing)
    - Cross-window properties → Global (uniqueness)
    - Technical properties → Global (resolution)
    - No redundant storage (don't store both)


POINT 3 - Phase 1 Limitation
---
Phase 2 will capture precise timing relationships between multimodal events (text, speech, gestures) with synchronization metrics like sync_rate and avg_distance.

  Phase 1 (MVP) only captures co-occurrence within temporal windows, such as:
  - Hook: 4 text overlays and 3 gestures appeared (somewhere in those 3 seconds)
  - Middle: Speech was present 75% of the time
  - Closing: CTA text and gestures both occurred

  Phase 1 tells us WHAT appears together in a window.
  Phase 2 will tell us HOW PRECISELY they align (within 0.5s, 1s, etc.)"

+++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---+++---------

STRUCTURE
Hook Window (0-3)
Middle: Piecewise Segments/Subwindows
    - Slope Early
    - Slope Mid
    - Slope late
        Example 1: 31 - 60s videos = Middle divided into 3 segments
        Example 2: 61 - 120s videos: Middle divided into 3 - 5 segments
CTA (last seconds)
  
Technical term: These are called "piecewise segments" or "sub-windows" within the middle window.



+++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---+++---------

---

GOAL
---
Revise our shortlist of feature found in FeaturesMLMVP.md 

CONTEXT
---
We will follow three steps.
1. Revise the features as a whole (considering all flows) and we will predetermine which features seem repetitive
2. Revise the features to ensure they can adapt to our temporal windows architecture
3. Create a AdaptFeatureData.md to map out how we will adapt the Features' data to each ML Algorithm, K-means and RF per MLMVP2.md

INSTRUCTION
---


+++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---+++---------

PEAKS = Tend to be interpretative Schema
  Other Variants to Cancel: climaxTiming, emotionalPeaks, accelerationPoints, Hooks, 
  1. We define what constitutes a "peak" (threshold = interpretation)
  2. We decide peak window size (0.5s vs 1s = interpretation)
  3. We assume peaks matter (maybe valleys matter more!)
     The Exception: Audio energy peaks
    Other Variants to Cancel: climaxTiming, emotionalPeaks, accelerationPoints, Hooks, 




+++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---+++---------
Step of adapting features to Kmeans + RF is NOT Called ETF,

Its called Feature Transformations

+++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---+++---------

 The Key Distinction:

  - Uses temporal data ≠ Is temporal feature
  - It's like calculating an average: you use multiple numbers but get ONE result
  - The feature itself is Global (stored once), even though it's derived from comparing temporal windows

  Think of it like:
  - Temperature readings every hour = Temporal (24 values)
  - "Today was warmer than yesterday" = Global (1 conclusion from comparing data)

+++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---+++---------

Avoids multicollinearity issues

● Multicollinearity Explained

  Definition: Multicollinearity occurs when two or more features in your dataset are highly correlated
  with each other - they essentially contain the same information expressed differently.

  Simple Example

  Imagine predicting house prices with these features:
  - house_size_sqft: 2000
  - house_size_sqm: 185.8 (same house, metric units)
  - number_of_rooms: 6

  The first two features are perfectly multicollinear - they're the same measurement in different units.
  Including both doesn't add new information.

  The RumiAI Example

  In your emotional features:
  # These features contain overlapping information:
  emotionalDistribution = {joy: 0.6, neutral: 0.3, surprise: 0.1}
  emotionalDiversity = 0.5  # Calculated FROM the distribution

  # emotionalDiversity is just a mathematical summary of emotionalDistribution
  # Like calculating average from a list of numbers - redundant if you have the list

  Why It's Problematic

  1. For Linear Models (Regression)

  - Can't determine which feature is actually important
  - Coefficients become unstable - small data changes cause huge swings
  - Standard errors inflate, making features seem less significant

  2. For K-means Clustering

  - Double-counts the same signal
  - If you have joy_ratio and emotional_diversity, you're essentially voting twice for "variety matters"
  - Distance calculations get skewed toward the duplicated information

  3. For Random Forest

  - Less problematic - RF randomly selects features at each split
  - Might waste splits on redundant features
  - Feature importance scores get diluted across correlated features

  Visual Analogy

  Imagine judging a cooking competition:
  - Judge A rates "flavor"
  - Judge B rates "taste" (same as flavor)
  - Judge C rates "presentation"

  With multicollinearity, it's like flavor gets 2 votes while presentation gets 1. The final score is
  biased toward the duplicated criterion.

  Detection Methods

  1. Correlation Matrix: Values > 0.8 indicate high multicollinearity
  2. VIF (Variance Inflation Factor): Values > 5-10 indicate problems
  3. Domain Knowledge: "These two features measure the same thing"

  +++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---+++---------

Three Categories of Fixes
Architectural improvements (temporal windows as source of truth)
Raw data additions (multimodal counts, quiet periods)
ML-Compatible Transformations (variable to fixed arrays)

+++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---+++---------
+++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---++++++---+++---------

