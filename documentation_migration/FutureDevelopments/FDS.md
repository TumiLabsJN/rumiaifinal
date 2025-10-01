# Final Document Structure (FDS)

**Purpose**: Master reference showing relationships between ML planning and future development documents
**Last Updated**: 2025-10-01
**Status**: Living document - update when adding/reorganizing docs

---

## Document Hierarchy

```
MLPlanning.md (Master Planning)
│
├── Analysis Mode System
│   ├── MLAnalysisMode.md (HLD)
│   └── MLAnalysisModeTI.md (Technical Implementation)
│
├── Checkpoint Resume System
│   ├── MLCheckpointResume.md (HLD)
│   └── MLCheckpointResumeTI.md (Technical Implementation)
│
├── Creator Match Analysis
│   └── MLCreatorMatch.md (HLD)
│
└── Creative Reports Generation
    └── MLCreativeReports.md (Brainstorm/Planning)
```

---

## Core Documents

### 1. MLPlanning.md
**Location**: `/home/jorge/rumiaifinal/MLPlanning.md`

**Type**: Master Planning Document

**Purpose**:
- Central planning hub for ML batch processing system
- Tracks implementation status of all ML features
- Documents decisions, priorities, and next steps

**Contents**:
- System 1: Sequential Processing (IMPLEMENTED)
- System 2: Fail-Fast Architecture (IMPLEMENTED)
- System 3: Analysis Mode System (PLANNED)
  - Section 3.1: Apify Scraper Investigation
- System 4+: Future systems

**Audience**: Product managers, developers, stakeholders

**Maintenance**: Update when features move from planned → in progress → completed

---

## Feature Documentation (HLD + TI Pattern)

### 2. Analysis Mode System

#### 2A. MLAnalysisMode.md (HLD)
**Location**: `/home/jorge/rumiaifinal/documentation_migration/FutureDevelopments/MLAnalysisMode.md`

**Type**: High-Level Design

**Purpose**:
- Business context for top vs recent analysis modes
- Design decisions and rationale
- Use cases per analysis type (hashtag, competitor, creator)

**Contents**:
- Business problem statement
- Two modes: top (engagement) vs recent (date)
- Default modes per analysis type
- Engagement score calculation (formula and rationale)
- Implementation design (conceptual)
- Edge cases and handling strategies

**Audience**: Product managers, business stakeholders, architects

**No Code**: All implementation code removed, references TI document

**Cross-References**:
- Technical Implementation → MLAnalysisModeTI.md
- Checkpoint Integration → MLCheckpointResume.md

---

#### 2B. MLAnalysisModeTI.md (Technical Implementation)
**Location**: `/home/jorge/rumiaifinal/documentation_migration/FutureDevelopments/MLAnalysisModeTI.md`

**Type**: Technical Implementation

**Purpose**:
- Production-ready code for analysis mode system
- Integration examples and usage patterns
- Testing scripts and validation

**Contents**:
- Section 1: Apify Client Implementation (dual-scraper support)
- Section 2: Client-Side Date Filtering
- Section 3: CLI Argument Parsing
- Section 4: Checkpoint Integration (brief usage example)
- Section 5: End-to-End Workflow Example
- Section 6: Testing Scripts

**Audience**: Developers, DevOps engineers

**Code-Heavy**: All implementation code lives here

**Cross-References**:
- High-Level Design → MLAnalysisMode.md
- Checkpoint Implementation → MLCheckpointResumeTI.md

---

### 3. Checkpoint Resume System

#### 3A. MLCheckpointResume.md (HLD)
**Location**: `/home/jorge/rumiaifinal/documentation_migration/FutureDevelopments/MLCheckpointResume.md`

**Type**: High-Level Design

**Purpose**:
- Business context for checkpoint/resume functionality
- System architecture and strategy
- User experience (CLI usage, auto-resume behavior)

**Contents**:
- Business problem (6-8 hour batch interruptions)
- Checkpoint strategy (when/how checkpoints are created)
- Checkpoint file structure (schema)
- CLI usage examples (start, auto-resume, force restart)
- Edge cases (config mismatch, corruption, disk full)
- Performance considerations
- Future enhancements

**Audience**: Product managers, business stakeholders, operations

**No Code**: All implementation code removed, references TI document

**Cross-References**:
- Technical Implementation → MLCheckpointResumeTI.md
- Related System → MLAnalysisMode.md (checkpoint validates analysis_mode)

---

#### 3B. MLCheckpointResumeTI.md (Technical Implementation)
**Location**: `/home/jorge/rumiaifinal/documentation_migration/FutureDevelopments/MLCheckpointResumeTI.md`

**Type**: Technical Implementation

**Purpose**:
- Production-ready CheckpointManager class
- Integration with batch processing workflow
- Testing suite for checkpoint functionality

**Contents**:
- Complete `CheckpointManager` class implementation
- Key features:
  - Analysis mode validation
  - JSONL format for completed videos
  - Fail-fast architecture support
- Integration example (usage in batch processing)
- Comprehensive unit tests

**Audience**: Developers

**Code-Heavy**: Single source of truth for checkpoint implementation

**Cross-References**:
- High-Level Design → MLCheckpointResume.md
- Used By → MLAnalysisModeTI.md (checkpoint integration example)

---

## Feature Documentation (HLD Only - No TI Yet)

### 4. MLCreatorMatch.md
**Location**: `/home/jorge/rumiaifinal/documentation_migration/FutureDevelopments/MLCreatorMatch.md`

**Type**: High-Level Design

**Purpose**:
- Creator vetting system for affiliate hiring
- Analyze creator's natural style vs hashtag/competitor patterns
- Generate compatibility score and hiring recommendation

**Contents**:
- Business problem (hiring wrong creators)
- Data collection strategy (recent 40 videos)
- Analysis logic (style profiling, pattern matching)
- Compatibility scoring algorithm
- Report format (compatibility score, recommendations, coaching notes)

**Status**: HLD complete, no TI document yet

**Audience**: Product managers, business stakeholders

**Future Work**: Create MLCreatorMatchTI.md when implementing

**Cross-References**:
- Depends On → MLAnalysisMode.md (uses "recent" mode for creator analysis)
- Related To → MLCreativeReports.md (creator reports format)

---

## Brainstorm / Planning Documents

### 5. MLCreativeReports.md
**Location**: `/home/jorge/rumiaifinal/documentation_migration/FutureDevelopments/MLCreativeReports.md`

**Type**: Brainstorm / Planning

**Purpose**:
- Explore creative report formats and content strategies
- Align reports with ML analysis flows (hashtag, competitor, creator)

**Contents**:
- Hashtag Analysis Reports (5 reports per bucket = 40 total)
- Report format options:
  - Option 1: Pattern-based reports
  - Option 2: Element-focused reports
  - Option 3: Narrative arc reports
- Content considerations per analysis type

**Status**: Early planning/brainstorm phase

**Audience**: Product, content strategists, UX

**Future Work**:
- Finalize report format strategy
- Create HLD: MLCreativeReportsHLD.md
- Create TI: MLCreativeReportsTI.md (when implementing generation logic)

**Cross-References**:
- Context From → MLPlanning.md (hashtag, competitor, creator flows)
- Related To → MLCreatorMatch.md (creator compatibility reports)

---

## Document Type Definitions

### High-Level Design (HLD)
**Purpose**: Business context, architecture decisions, design rationale

**Characteristics**:
- ✅ Business problem statement
- ✅ Stakeholder value proposition
- ✅ System design (conceptual)
- ✅ Use cases and workflows
- ✅ Edge cases and handling strategies
- ✅ Architecture diagrams (conceptual)
- ❌ NO implementation code
- ❌ NO detailed technical specs

**Audience**: Product managers, architects, business stakeholders

**Examples**: MLAnalysisMode.md, MLCheckpointResume.md, MLCreatorMatch.md

---

### Technical Implementation (TI)
**Purpose**: Production-ready code, integration guides, testing

**Characteristics**:
- ✅ Complete class implementations
- ✅ Function/method code with docstrings
- ✅ Integration examples (usage patterns)
- ✅ Configuration details (file paths, parameters)
- ✅ Testing scripts (unit tests, integration tests)
- ✅ Command-line examples
- ❌ Minimal business context (reference HLD for that)

**Audience**: Developers, DevOps engineers

**Examples**: MLAnalysisModeTI.md, MLCheckpointResumeTI.md

---

### Master Planning
**Purpose**: Track status, priorities, and coordination across features

**Characteristics**:
- ✅ Feature status tracking (planned/in progress/completed)
- ✅ Implementation priorities
- ✅ Decision log
- ✅ Cross-feature dependencies
- ✅ Next steps and action items
- ✅ Brief summaries (not deep dives)

**Audience**: Product managers, team leads, stakeholders

**Examples**: MLPlanning.md

---

### Brainstorm / Planning
**Purpose**: Explore options, gather requirements, align on direction

**Characteristics**:
- ✅ Multiple options presented
- ✅ Open questions
- ✅ Requirements gathering
- ✅ Stakeholder considerations
- ⚠️ Not finalized (work in progress)

**Audience**: Product, design, stakeholders

**Examples**: MLCreativeReports.md

---

## Document Relationships

### Parent-Child Relationships

**MLPlanning.md** (Parent)
- ├─ **MLAnalysisMode.md** (Child - HLD)
- ├─ **MLCheckpointResume.md** (Child - HLD)
- ├─ **MLCreatorMatch.md** (Child - HLD)
- └─ **MLCreativeReports.md** (Child - Brainstorm)

**MLAnalysisMode.md** (Parent - HLD)
- └─ **MLAnalysisModeTI.md** (Child - TI)

**MLCheckpointResume.md** (Parent - HLD)
- └─ **MLCheckpointResumeTI.md** (Child - TI)

---

### Cross-Feature Dependencies

**MLCheckpointResume → MLAnalysisMode**
- Checkpoint validates `analysis_mode` on resume
- Must ensure top/recent mode consistency

**MLCreatorMatch → MLAnalysisMode**
- Creator analysis uses "recent" mode by default
- Fetches 40 most recent videos (not top performers)

**MLCreativeReports → MLAnalysisMode**
- Report generation happens after analysis completes
- Different report formats per analysis type (hashtag/competitor/creator)

**MLCreativeReports → MLCreatorMatch**
- Creator compatibility reports need special format
- Include coaching notes and compatibility scores

---

## Document Naming Conventions

### Pattern
```
[Feature Name][Document Type].md

Examples:
- MLAnalysisMode.md (HLD implied when no suffix)
- MLAnalysisModeTI.md (TI = Technical Implementation)
- MLCheckpointResume.md (HLD implied)
- MLCheckpointResumeTI.md (TI = Technical Implementation)
```

### When to Split HLD/TI

**Split when**:
- ✅ Feature has >300 lines of implementation code
- ✅ Business stakeholders need to review design without code
- ✅ Multiple developers will implement (TI = reference doc)

**Don't split when**:
- ❌ Feature is still in brainstorm phase
- ❌ Implementation is <100 lines (keep in HLD)
- ❌ Single developer, rapid prototyping

---

## Maintenance Guidelines

### Adding New Features

1. **Create HLD first** (e.g., `MLNewFeature.md`)
   - Business context
   - Design decisions
   - Use cases

2. **Link from MLPlanning.md**
   - Add to appropriate system section
   - Update status (PLANNED)

3. **Create TI when implementing** (e.g., `MLNewFeatureTI.md`)
   - Extract code from HLD if it exists
   - Clean up HLD (remove code, add TI references)

4. **Update status in MLPlanning.md**
   - Move from PLANNED → IN PROGRESS → COMPLETED

---

### Updating Existing Features

**If changing design**:
- Update HLD with new rationale
- Update TI with new implementation
- Update MLPlanning.md if status/priority changes

**If adding code**:
- Add to TI document (not HLD)
- Update cross-references if needed

**If fixing bugs**:
- Update TI document only
- No HLD changes unless design changes

---

### Document Review Checklist

**Before committing HLD**:
- [ ] No implementation code (only conceptual descriptions)
- [ ] Business problem clearly stated
- [ ] Design decisions explained (why, not just what)
- [ ] References to TI document for implementation
- [ ] Cross-references to related features

**Before committing TI**:
- [ ] Complete implementation code included
- [ ] All functions have docstrings
- [ ] Integration examples provided
- [ ] Testing scripts included
- [ ] References back to HLD for context

**Before committing to MLPlanning.md**:
- [ ] Feature status accurate (PLANNED/IN PROGRESS/COMPLETED)
- [ ] Links to HLD documents working
- [ ] Priority clearly stated
- [ ] Dependencies noted

---

## Quick Reference

### "I need to understand what Analysis Mode does"
→ Read: **MLAnalysisMode.md** (HLD)

### "I need to implement Analysis Mode"
→ Read: **MLAnalysisModeTI.md** (TI)

### "I need to understand checkpoint strategy"
→ Read: **MLCheckpointResume.md** (HLD)

### "I need to implement checkpoints"
→ Read: **MLCheckpointResumeTI.md** (TI)

### "I need to see overall ML roadmap"
→ Read: **MLPlanning.md** (Master Planning)

### "I need to understand creator vetting"
→ Read: **MLCreatorMatch.md** (HLD)

### "I need to plan creative reports"
→ Read: **MLCreativeReports.md** (Brainstorm)

---

## Document Locations

```
/home/jorge/rumiaifinal/
├── MLPlanning.md
└── documentation_migration/
    └── FutureDevelopments/
        ├── FDS.md (this file)
        ├── MLAnalysisMode.md
        ├── MLAnalysisModeTI.md
        ├── MLCheckpointResume.md
        ├── MLCheckpointResumeTI.md
        ├── MLCreatorMatch.md
        └── MLCreativeReports.md
```

---

## Future Document Roadmap

### Immediate (Next 2-4 Weeks)
- [ ] **MLCreatorMatchTI.md** - Implementation for creator compatibility analysis
- [ ] **MLCreativeReportsHLD.md** - Finalize creative report design
- [ ] **MLCreativeReportsTI.md** - Report generation implementation

### Short-Term (1-2 Months)
- [ ] **MLModelTraining.md** - Random Forest + K-means training pipeline
- [ ] **MLFeatureExtraction.md** - 60+ feature calculation logic
- [ ] **MLDurationBucketing.md** - 8-bucket system design + implementation

### Medium-Term (2-4 Months)
- [ ] **MLPatternAggregation.md** - Claude API integration for pattern synthesis
- [ ] **MLStatisticalValidation.md** - P-value thresholds, confidence scoring
- [ ] **MLOutputFormats.md** - PDF reports, JSON exports, API responses

---

## Document Statistics

| Document | Type | Lines | Code Blocks | Status |
|----------|------|-------|-------------|--------|
| MLPlanning.md | Master Planning | ~670 | ~10 | Active |
| MLAnalysisMode.md | HLD | ~550 | 0 | Complete |
| MLAnalysisModeTI.md | TI | ~850 | ~12 | Complete |
| MLCheckpointResume.md | HLD | ~410 | 0 | Complete |
| MLCheckpointResumeTI.md | TI | ~450 | ~8 | Complete |
| MLCreatorMatch.md | HLD | ~350 | ~3 | Complete |
| MLCreativeReports.md | Brainstorm | ~150 | 0 | In Progress |
| **Total** | - | **~3,430** | **~33** | - |

---

## Contact & Maintenance

**Document Owner**: Development Team
**Last Major Refactor**: 2025-10-01 (HLD/TI separation pattern established)
**Review Frequency**: Update when adding/reorganizing docs
**Questions**: Reference this FDS.md first, then check individual HLD/TI documents
