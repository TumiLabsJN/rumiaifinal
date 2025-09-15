Viable designs
1) Middle quartiles inside the middle only

• Split the middle window into four equal parts of the middle window itself
• Emit density or other metrics per quartile

"middle_quartiles": {
  "q1_density": ...,
  "q2_density": ...,
  "q3_density": ...,
  "q4_density": ...
}


• Guardrail: only compute if middle_len ≥ 8 s. Otherwise fall back to thirds or a single middle metric and set a mask

Pros
• Very interpretable shape over time
Cons
• Short videos will often fail the guardrail

2) Key shape statistics in the middle

• Summarize dynamics without extra bins

"middle_shape": {
  "peak_value": ...,
  "peak_position": 0.58,   // normalized 0 to 1 within middle
  "valley_value": ...,
  "oscillations": 2,       // peak valley cycles over a prominence threshold
  "trend_slope": ...,
  "variance": ...,
  "cv": ...                // coefficient of variation
}


Pros
• Few columns, strong signal for RF and K means
Cons
• Less granular than bins for diagnostics

3) Hybrid windows plus proportional bins that exclude hook and closing

• Keep hook 0 to 3 s and closing last 3 s
• Partition only the middle into equal parts
• Use a 30 40 30 split or simple thirds inside the middle

"middle_bins": {
  "early_part_density": ...,
  "mid_part_density": ...,
  "late_part_density": ...
}


Pros
• No overlap, clean story arc
Cons
• For short middles one or more bins can be very short

4) Piecewise linear fit of the middle

• Fit a two break line to the middle timeline
• Emit slopes and breakpoint positions

"middle_piecewise": {
  "slope_early": ...,
  "slope_mid": ...,
  "slope_late": ...,
  "break_pos_1": 0.33,   // within middle, normalized
  "break_pos_2": 0.72
}


Pros
• Captures rise then fall or slow burn in very few numbers
Cons
• Slightly more compute and needs a simple fitting routine

5) Adaptive bin count by duration

• Decide middle granularity based on video length
• Always keep the same output columns by padding with nulls and masks

if L ≤ 15 s:      no middle bins, only middle_shape
if 16 to 30 s:    middle thirds
if 31 to 60 s:    middle quartiles
if 61 to 120 s:   middle quintiles


Pros
• Uses detail where it matters, stays stable for short videos
Cons
• More rules to maintain