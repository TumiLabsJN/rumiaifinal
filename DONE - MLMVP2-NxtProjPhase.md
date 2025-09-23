

# Interesting Point 1

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


# Interesting Point 2

PEAKS = Tend to be interpretative Schema. Not the best type of features as they are interprative. 
  Other Variants to Cancel: climaxTiming, emotionalPeaks, accelerationPoints, Hooks, 
  1. We define what constitutes a "peak" (threshold = interpretation)
  2. We decide peak window size (0.5s vs 1s = interpretation)
  3. We assume peaks matter (maybe valleys matter more!)
     The Exception: Audio energy peaks
    Other Variants to Cancel: climaxTiming, emotionalPeaks, accelerationPoints, Hooks, 


# Interesting Point 3

 The Key Distinction:

  - Uses temporal data ≠ Is temporal feature
  - It's like calculating an average: you use multiple numbers but get ONE result
  - The feature itself is Global (stored once), even though it's derived from comparing temporal windows

  Think of it like:
  - Temperature readings every hour = Temporal (24 values)
  - "Today was warmer than yesterday" = Global (1 conclusion from comparing data)


# Interesting Point 4

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


