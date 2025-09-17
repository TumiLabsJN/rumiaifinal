# Improved Metaprompt for Feature Evaluation

## Your Original Prompt

```
Since the end goal is to transform the patterns ML will obtain into actionable human insights to replicate the patterns of the most viral videos,

What would these features tell us?
```

### Strengths ✓
- Clear end goal (actionable insights)
- Ties to business value (viral videos)
- Forces practical thinking

### Weaknesses ✗
- Too open-ended
- Doesn't guide specific analysis
- Missing evaluation criteria
- No structure for response

---

## Improved Metaprompt V2

```
Our goal is to transform ML patterns into actionable creator insights for replicating viral video success.

For these proposed features, analyze:

1. PATTERN DETECTION: What specific viral patterns could these features identify?
   - Example: "Hook energy 25% above baseline" → "Start with excitement"

2. ACTIONABILITY: How would creators change their behavior?
   - Bad: "Be more engaging" (vague)
   - Good: "Raise pitch 30Hz on key benefits" (specific)

3. MEASURABILITY: Can creators self-assess improvement?
   - Can they hear/feel the difference?
   - Is it practicable without tools?

4. PLATFORM ROI: What engagement metrics would improve?
   - View duration? CTR? Shares? Comments?
   - Evidence from similar features?

5. IMPLEMENTATION COST vs VALUE:
   - Processing complexity vs insight quality
   - Feature redundancy with existing metrics

Provide concrete examples of insights these features would generate.
```

---

## Even Better: Structured Evaluation Framework

```
# Feature Evaluation for Creator Insights

Context: We're building ML features to help creators replicate viral video patterns.

For [FEATURE NAME], evaluate:

## 1. Viral Pattern Discovery
□ What top 10% vs bottom 10% patterns would this reveal?
□ Example: "Top videos show X, bottom videos show Y"

## 2. Creator Translation
□ ML finding: [technical metric]
□ Creator sees: [human instruction]
□ Specific example with numbers

## 3. Actionability Score (1-10)
□ How easily can creators implement this?
□ What specific actions would they take?
□ Before/After example

## 4. Uniqueness Check
□ What insights does this provide that current features don't?
□ Risk of redundancy with: [existing features]
□ Correlation analysis needed

## 5. Success Metrics
□ Which KPIs would improve? (completion rate, shares, etc.)
□ Expected lift: X%
□ Based on: [evidence/research]

## 6. Cost-Benefit
□ Implementation complexity: Low/Medium/High
□ Processing time impact: +Xms
□ Value to creators: Low/Medium/High
□ Decision: Include/Exclude/Defer

Provide dashboard mockup showing how creators would see this insight.
```

---

## Best Version: The "So What?" Test

```
# The Creator Value Test

We're evaluating: [FEATURE NAME]

Answer these questions AS A CREATOR would:

1. "My video underperformed. What specifically does this feature tell me to fix?"
   → Answer must be an action, not an observation

2. "I'm recording tomorrow. How do I use this insight?"
   → Answer must be practicable without special tools

3. "Why should I care about this metric?"
   → Show correlation with views/engagement

4. "How is this different from 'be more energetic'?"
   → Prove it's specific and measurable

5. "Show me a before/after example"
   → Demonstrate clear improvement

If you can't answer ALL five with specific, actionable responses, the feature fails the test.
```

---

## Why Your Original Prompt Worked (But V2 is Better)

### Your prompt triggered good analysis because:
1. **Outcome focus**: "actionable human insights"
2. **Context setting**: "viral videos" 
3. **Open exploration**: Allowed comprehensive thinking

### V2 improvements add:
1. **Structured thinking**: Specific evaluation criteria
2. **Concrete examples**: Forces real scenarios
3. **ROI focus**: Ties to business metrics
4. **Redundancy check**: Prevents feature bloat
5. **Creator perspective**: Not just technical accuracy

---

## Template for Future Feature Discussions

```
# Quick Feature Evaluation

Feature: [NAME]
Hypothesis: This will help creators by [SPECIFIC BENEFIT]

Rapid Assessment:
1. Viral pattern it reveals: ___
2. Creator instruction: "Do ___ when ___"
3. Measurable without tools? Y/N
4. Redundant with [FEATURE]? Y/N  
5. Worth the complexity? Y/N

Decision: Build / Skip / Research more
```

---

## Examples of Using Improved Prompt

### Example 1: Evaluating Pitch Features

**Original prompt response**: "These features tell us about voice dynamics"

**V2 prompt response**: 
- Pattern: "Viral hooks show 0.35+ pitch range vs 0.10 failures"
- Action: "Start with a rising question, emphasize 3 key words"
- Measurable: "Creators can hear pitch rise, practice with phone recorder"
- ROI: "42% better retention in first 3 seconds"
- Cost/Value: Medium complexity, High value → BUILD

### Example 2: Evaluating ZCR Feature

**Original prompt response**: "ZCR shows frequency content"

**V2 prompt response**:
- Pattern: "Correlates -0.4 with energy (redundant signal)"
- Action: "Unclear - 'adjust consonant ratio'?" (too abstract)
- Measurable: "Creators can't hear ZCR directly"
- ROI: "No clear engagement correlation"
- Cost/Value: Low complexity, Low value → SKIP

---

## The Ultimate Metaprompt

```
You're a product manager building features for creators.
Every feature must pass this test:

"A creator's video flopped. They open our app.
In 10 seconds, what SPECIFIC instruction does this feature generate?
Will they know exactly what to do differently next time?"

If yes → Build it
If maybe → Research more  
If no → Skip it

Now evaluate: [FEATURE]
```

This forces:
- Creator-centric thinking
- Actionability over accuracy
- Speed to insight
- Clear decision making