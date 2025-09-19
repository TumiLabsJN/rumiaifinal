# MASTER PROMPT: Service Documentation Migration (11-Document Strategy)

## ⚠️ CRITICAL CONTEXT
This is a MASTER PROMPT providing big-picture guidance for a 3-phase interactive documentation migration.
**DO NOT EXECUTE WITHOUT PHASE-SPECIFIC STRUCTURE FILES** that define document templates.

### This is an INTERACTIVE PROCESS
- We will create each document together
- Deep discovery will be run for each section
- User will be consulted at decision points
- No automatic assumptions

### Legacy Warning
**ALL existing documentation was written for precompute functions that are being DELETED.**
- DO NOT copy sections without verification
- DO NOT assume any data flow is current
- DO NOT trust function names or file paths without checking
- ALWAYS run deep discovery (30 min investigation) to verify current implementation

## 🚨 LEGACY DANGER CHECKLIST
If you encounter ANY of these patterns in legacy docs, STOP and investigate:
- `precompute_*` functions → Being deleted
- `compute_metrics()` → Old architecture
- `blocks['opening']` or any 6-block reference → Deprecated structure
- `/rumiai/precompute/` paths → Old location
- `combined_metrics` → Deprecated
- `aggregate_*` → Likely outdated
- Block names: `opening`, `development`, `peak`, `climax`, `resolution`, `closing`
- `compute_creative_density()`, `compute_speech_analysis()` → Old functions
- References to "6-block temporal structure"
- Any function starting with `analyze_` in precompute context

## 🎯 OVERALL GOAL
Migrate from legacy feature-based documentation (8 docs) to consolidated service-based architecture (11 docs), ensuring:
1. No precompute dependencies remain
2. All valuable ML insights are preserved
3. Current temporal_compute flow is accurately documented
4. Each service is self-contained and optimized

## 📁 PROMPT STRUCTURE FILES

This migration uses a hierarchical prompt system:

### Required Files (Must exist before starting):
1. **MASTER_PROMPT.md** (this file) - Overall strategy and warnings
2. **PHASE_1_SERVICE_STRUCTURE.md** - Exact template for service documentation
3. **PHASE_2_FEATURE_STRUCTURE.md** - Exact template for feature documentation
4. **PHASE_3_ARCHITECTURE_STRUCTURE.md** - Exact template for architecture documentation
5. **MIGRATION_LOG.md** - Track progress & discoveries

### Execution Order:
1. Read MASTER_PROMPT.md first (context and warnings)
2. When starting Phase 1: Load PHASE_1_SERVICE_STRUCTURE.md
3. When starting Phase 2: Load PHASE_2_FEATURE_STRUCTURE.md
4. When starting Phase 3: Load PHASE_3_ARCHITECTURE_STRUCTURE.md

## 📋 DOCUMENT INVENTORY

### What We're Starting With (Legacy - DO NOT TRUST)
```
⚠️ LEGACY (contains outdated precompute references):
- GestureAnalysis.md     → Mine carefully for ML insights only
- SpeechAnalysis.md      → Mine carefully for ML insights only
- CreativeDensity.md     → Mine carefully for ML insights only
- PersonFraming.md       → Mine carefully for ML insights only
- ScenePacing.md         → Mine carefully for ML insights only
- EmotionService.md      → Mine carefully for ML insights only
- MetadataAnalysis.md    → Mine carefully for ML insights only
- VisualOverlay.md       → Mine carefully for ML insights only
```

### What We're Creating (New Structure - 10 Documents Total)
```
/services (4 docs - Phase 1)
  1. VisionServices.md (YOLO, MediaPipe, OCR, Scene Detection)
  2. AudioServices.md (Whisper, Audio Energy)
  3. AnalysisServices.md (FEAT, DeepFace)
  4. MetadataServices.md (Hashtags, Engagement)

/features (4 docs - Phase 2)
  5. VisualFeatures.md (Framing, density, overlays, scenes, objects)
  6. AudioFeatures.md (Speech, energy, pitch)
  7. BehavioralFeatures.md (Gestures, gaze, emotion)
  8. EngagementMetadata.md (Virality metrics, hashtags, demographics)

/architecture (2 docs - Phase 3)
  9. SystemArchitecture.md (Complete flow, dependencies)
  10. MigrationGuide.md (Deprecation list, breaking changes)
```

## 🤝 INTERACTIVE MIGRATION PROCESS

### Working Method:
1. **Start together**: Begin each document with user present
2. **Deep discovery**: 30-minute investigation for each service/feature
3. **Discuss findings**: Review discoveries before documenting
4. **Validate sections**: Confirm each section before proceeding
5. **Complete together**: Final review before next document

### Decision Points Requiring User Input:
- Optimization priorities (what to fix now vs. later)
- Performance thresholds (acceptable execution times)
- Feature importance (which ML features are critical)
- Breaking changes (what can be deprecated)
- Unclear implementations (when code is ambiguous)

### Document Creation Order:
```
Phase 1 (Services):
1. VisionServices.md (most complex, establish patterns)
2. AudioServices.md
3. AnalysisServices.md
4. MetadataServices.md
[PHASE 1 GATE - Review all service docs together]

Phase 2 (Features):
5. VisualFeatures.md
6. AudioFeatures.md
7. BehavioralFeatures.md
8. EngagementMetadata.md
[PHASE 2 GATE - Review all feature docs together]

Phase 3 (Architecture):
9. SystemArchitecture.md
10. MigrationGuide.md
[PHASE 3 GATE - Final validation]
```

## 🔍 DEEP DISCOVERY METHODOLOGY

### For EVERY service/feature (30-minute investigation):

1. **Code Analysis (10 min)**
   ```python
   # Find current implementation
   grep -r "service_name" /rumiai_v2/ --include="*.py"

   # Trace imports and dependencies
   grep -r "from.*service_name" /rumiai_v2/

   # Check timeline integration
   grep -r "service_name" /rumiai_v2/processors/timeline_builder.py

   # Verify temporal_compute usage
   grep -r "service_name" /rumiai_v2/processors/temporal_compute.py
   ```

2. **Execution Testing (15 min)**
   ```python
   # Test service in isolation
   # Measure execution time for 1-minute video
   # Document input/output structure
   # Check memory usage
   # Test without precompute imports
   ```

3. **Data Flow Verification (5 min)**
   ```python
   # Trace: Service → Timeline → Temporal → Output
   # Document actual structure at each step
   # Verify features appear in temporal windows
   ```

## 📝 MINING STRATEGY FOR LEGACY DOCS

### Three-Pass Approach:

#### Pass 1: Identify Valuable Content
- ML insights and mathematical formulas
- Business value descriptions
- Feature importance explanations
- Performance benchmarks (verify if still relevant)

#### Pass 2: Verify Everything
```python
# For each claimed feature:
- Does this service still exist?
- Is this feature still computed?
- Has the implementation moved?
- Is the formula still accurate?
- Are the file paths correct?
```

#### Pass 3: Rewrite for Current Architecture
- Update all file paths to current locations
- Replace old function names with current ones
- Rewrite data flow for temporal_compute architecture
- Remove all precompute references

## 📝 AMENDMENT PROTOCOL

### If we discover issues with earlier documentation:
1. **STOP and notify**: "Found issue with [document]: [description]"
2. **User decision**: "Should we update [document] now or note for later?"
3. **If updating now**:
   - Make changes together
   - Mark with "AMENDED: [date] - [reason]"
   - Update MIGRATION_LOG.md
4. **If noting for later**:
   - Add to MIGRATION_LOG.md under "Pending Amendments"
   - Continue current document
   - Address in final review

## ✅ PHASE COMPLETION GATES

### Phase 1 Completion (Services):
Before starting Phase 2, verify:
- [ ] All 4 service docs created
- [ ] Each service tested without precompute
- [ ] Performance metrics documented (execution time for 1-min video)
- [ ] Optimization issues listed with priority
- [ ] File paths verified and current
- [ ] Self-containment validated
- [ ] MIGRATION_LOG.md updated with discoveries

### Phase 2 Completion (Features):
Before starting Phase 3, verify:
- [ ] All 4 feature docs created
- [ ] Features mapped to source services
- [ ] ML value preserved from legacy (where valid)
- [ ] Temporal window mapping complete
- [ ] Cross-references between services and features accurate
- [ ] No precompute dependencies remain
- [ ] MIGRATION_LOG.md updated

### Phase 3 Completion (Architecture):
Final validation:
- [ ] System architecture diagram complete and accurate
- [ ] Migration guide lists all deprecations
- [ ] Breaking changes documented
- [ ] All 10 documents internally consistent
- [ ] No contradictions between documents
- [ ] Ready for production use

## 🚫 COMMON PITFALLS TO AVOID

1. **Copying without verifying** - Every legacy section needs deep discovery
2. **Assuming file paths** - Always check current locations with grep
3. **Trusting function names** - Many have been renamed/removed
4. **Missing dependencies** - Test each service in complete isolation
5. **Outdated metrics** - Some features no longer exist in temporal_compute
6. **Shallow investigation** - Always do 30-min deep discovery
7. **Ignoring performance** - Document actual execution times
8. **Skipping user consultation** - Ask when uncertain

## 📊 VALIDATION CHECKLIST

### After EACH document:
- [ ] Deep discovery completed (30 min)?
- [ ] Performance metrics measured and documented?
- [ ] Legacy content verified before inclusion?
- [ ] Cross-references to other docs accurate?
- [ ] No precompute dependencies?
- [ ] Optimization issues documented?
- [ ] User consulted on decisions?
- [ ] MIGRATION_LOG.md updated?

## 🔄 FEEDBACK AND ITERATION

This is a living process. As we discover issues:
1. Update affected documentation immediately
2. Learn from each document to improve the next
3. Keep MIGRATION_LOG.md as source of truth for decisions
4. Refine templates if patterns emerge

---

## 📌 REMEMBER

1. **This is interactive** - We do this together, not automated
2. **Deep discovery always** - 30 minutes investigation minimum
3. **Legacy is dangerous** - Verify everything from old docs
4. **User decides** - Consult on optimization priorities and unclear points
5. **Document everything** - Keep MIGRATION_LOG.md updated

**NEXT STEP:** Load PHASE_1_SERVICE_STRUCTURE.md when ready to begin Phase 1