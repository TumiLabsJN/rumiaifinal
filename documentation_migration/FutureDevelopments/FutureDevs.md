# RumiAI Future Developments

This document tracks planned features and enhancements for RumiAI's ML training pipeline. Features are organized by category and phase.

---

## Phase 2 Features

### Top Video Selection Formula Validation
**Category:** Video Selection
**Objective:** Validate optimal engagement formula for selecting viral training videos through multi-formula testing
**Why:** Ensure ML models train on truly viral content, not just high view counts or arbitrary metrics
**What:** Implement validation framework comparing multiple engagement formulas (views-only, views×shares, weighted-all) using feature separation analysis and RF performance metrics. Select formula that produces clearest creative pattern differences between top/bottom performers, ensuring actionable ML insights for creators.
**Importance:** 7/10

#### Explanation

1. **Multi-Formula Testing Framework**: Implement system to test 3-4 engagement formulas simultaneously (views-only, views×share_rate, weighted engagement including likes/comments/shares, engagement rate). Each formula ranks videos differently, affecting which content trains the ML models.

2. **Feature Separation Analysis**: Measure Cohen's d effect size between top 40 and bottom 20 videos across all creative features. Formula producing highest average separation (d > 0.5) indicates clearest pattern differences, enabling stronger ML training signals.

3. **RF Performance Comparison**: Train Random Forest models using videos selected by each formula. Compare accuracy, F1 scores, and feature importance clarity. Winner is formula where RF achieves highest classification performance, indicating meaningful pattern separation.

4. **Expert Review Validation**: Domain experts review top 40 videos selected by winning formula, rating viral-worthiness (1-5 scale). Calculate correlation between formula scores and expert ratings. High correlation (r > 0.7) confirms formula aligns with human judgment.

5. **Automated Validation Pipeline**: Create `test_formula_validation.py` script that scrapes 500 videos, applies all formulas, measures separation metrics, generates comparison report. Enable quick pre-launch validation before committing to ML training run.

6. **Documentation of Results**: Add "Formula Validation Results" section to MLAnalysisMode.md documenting which formula won, separation metrics achieved, and rationale for selection. Provides transparency for future iterations and client reporting.

7. **Iterative Refinement Strategy**: Post-launch tracking of creator performance after following ML recommendations. If real-world outcomes don't match expectations, re-run validation with adjusted formulas. Close feedback loop between formula selection and business outcomes.

---
