# Phase 3: Child HLD Generation Instructions

> **Purpose**: Generate complete, production-ready Child HLD document
> **Output**: {ComponentName}CHILD.md (10 sections + 2 appendices)
> **Next Phase**: Phase 4 (Optional review & refinement)

---

## Your Role

Senior Technical Architect writing production-ready High-Level Design. Generate a PERFECT draft with NO TODOs.

---

## Inputs Required

### Required
1. **Mother Document**: {MotherHLD.md}
   - Section X.Y: Component to document
   - Part 1: Foundation (system context)
2. **Template**: ChildTemplate.md (10-section structure to follow)
3. **Phase 1 Output**: Critique_{ComponentName}.md (business decisions)
4. **Phase 2 Output**: QA_{ComponentName}.md (all technical clarifications)

---

## Your Task

### Step 1: Verify Inputs Complete

**Check Phase outputs:**
- Critique_{ComponentName}.md → Status = "COMPLETE", Final Decision = "Proceed to Phase 2: YES"
- QA_{ComponentName}.md → Status = "COMPLETE", Ready for HLD Generation = "YES"

**If NOT complete**: STOP and inform user
- "Cannot generate HLD. Phase {X} not complete. Please finish Phase {X} first."

**If complete**: Proceed to Step 2

---

### Step 2: Generate Complete Child HLD

Create file: `{ComponentName}CHILD.md`

Follow ChildTemplate.md structure (10 sections + 2 appendices).

Use this generation guide:

---

## Section-by-Section Generation Guide

### **Section 1: Context & Business Goal**

#### Section 1.1: What Problem Does This Solve?

**Sources**:
- Critique_{ComponentName}.md → Component Summary (refined problem statement)
- Critique_{ComponentName}.md → Q&A answers (business justification)

**Generate**:
```markdown
### 1.1 What Problem Does This Solve?

{2-3 sentences from Critique explaining the problem}

Example:
"FEAT emotion detection causes 15% of video processing failures due to timeouts.
These failures are silent until Stage 3 aggregation, wasting compute and time.
This component detects anomalies in real-time during Stage 2, enabling early
intervention and preventing bad data from entering ML training."
```

#### Section 1.2: Where This Fits in Pipeline

**Sources**:
- MotherHLD.md Section X.Y (stage flow context)
- MotherHLD.md Part 1 (Foundation - which sections this component uses)
- QA_{ComponentName}.md (dependencies identified)

**Generate**:
```markdown
### 1.2 Where This Fits in Pipeline

**Foundation Dependencies**: This component depends on MotherHLD.md Part 1 for:
- System architecture (Part 1, Section: {X} - {specific topic})
- Configuration patterns (Part 1, Section: {Y} - {specific topic})
- {List all Part 1 sections referenced in Q&A answers}

\`\`\`
Stage {N-1}: [Previous Stage Name]
   ↓ Output: [format from Mother Doc]
Stage {N}: [THIS COMPONENT]
   ↓ Output: [format from Q&A answers]
Stage {N+1}: [Next Stage Name]
\`\`\`
```

#### Section 1.3: Success Criteria

**Sources**:
- Critique_{ComponentName}.md → Final Decision (accepted constraints/goals)
- QA_{ComponentName}.md → Performance targets, validation criteria

**Generate**:
```markdown
### 1.3 Success Criteria

- [ ] {Measurable criterion from Q&A - e.g., "Process 300 videos in < 5 minutes"}
- [ ] {Criterion from Critique - e.g., "No data loss on checkpoint resume"}
- [ ] {Criterion from Q&A - e.g., "Flag < 10% of videos as anomalies (target false positive rate)"}
```

---

### **Section 2: Architecture & Design**

#### Section 2.1: High-Level Approach

**Sources**:
- QA_{ComponentName}.md → Answers about overall approach/strategy

**Generate** (3-5 sentences):
```markdown
### 2.1 High-Level Approach

{Extract from Q&A answers about technical strategy}

Example:
"We maintain incremental statistics (mean, std, quartiles) for each feature across
all processed videos in the bucket. After each video completes, we validate its
features against rolling statistics using IQR-based outlier detection. Flagged videos
trigger investigation package creation (centralized troubleshooting folder). Pipeline
continues without halting—flagged videos marked for manual review."
```

#### Section 2.2: Data Flow

**Sources**:
- QA_{ComponentName}.md → Input/output answers
- MotherHLD.md Part 1 → Directory paths

**Generate**:
```markdown
### 2.2 Data Flow

\`\`\`
Input: {format from Q&A}
       Schema: {shape/size from Q&A}
       Location: {path from Mother Part 1}
   ↓
Process Step 1: {action from Q&A}
   ↓
Process Step 2: {action from Q&A}
   ↓
Process Step 3: {action from Q&A}
   ↓
Output: {format from Q&A}
        Schema: {shape/size from Q&A}
        Location: {path from Mother Part 1}
\`\`\`
```

#### Section 2.3: Detailed Process

**Sources**:
- QA_{ComponentName}.md → Detailed logic answers

**Generate**: One subsection (2.3.X) per major sub-process identified in Q&A

For each subsection:
```markdown
#### Step 2.3.{X}: [Sub-process Name from Q&A]

**Purpose**: {One line from Q&A}

**Logic**:
\`\`\`python
# 10-20 lines pseudocode from Q&A answers
# Use realistic variable names
# Show exact structure, edge cases from Q&A

# Example:
def validate_input(df, required_cols):
    """
    {Purpose from Q&A}
    """
    # Check required columns exist (from Q&A answer about validation)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Validate duration range (from Q&A answer about edge cases)
    if not df['duration'].between(3, 120).all():
        invalid = df[~df['duration'].between(3, 120)]
        raise ValueError(f"Invalid duration in {len(invalid)} rows")

    # Check for nulls (from Q&A answer about null handling)
    if df.isnull().any().any():
        raise ValueError("Found null values in input")
\`\`\`

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| {From Q&A edge case answers} | {From Q&A} | {From Q&A or Critique} |
```

---

### **Section 3: Dependencies & Integration**

#### Section 3.1: Input Dependencies

**Sources**:
- QA_{ComponentName}.md → Input format answers
- MotherHLD.md Part 1 → Architecture, configuration

**Generate** (Table format):
```markdown
### 3.1 Input Dependencies

| Dependency | Source | Format | Required Fields | Failure Mode |
|------------|--------|--------|-----------------|--------------|
| **System setup** | MotherHLD.md Part 1 (Section {X}) | Directory structure + config | {From Q&A: which Part 1 items used} | Fail-fast if directories don't exist |
| {Input name from Q&A} | {Source from Q&A} | {Format from Q&A} | {Fields from Q&A} | {Failure mode from Q&A} |
```

**Rules**:
- FIRST ROW: Always "System setup" referencing Mother Part 1
- Remaining rows: From Q&A answers about inputs

#### Section 3.2: Output Contracts

**Sources**:
- QA_{ComponentName}.md → Output format answers
- MotherHLD.md Part 1 → Directory paths

**Generate** (Table format):
```markdown
### 3.2 Output Contracts

| Output | Format | Schema | Consumers | Validation |
|--------|--------|--------|-----------|------------|
| {Output name from Q&A} | {Format from Q&A} | {Schema from Q&A} | {Consumers from Q&A} | {Validation from Q&A} |
```

**Use paths from Mother Part 1** for file locations

#### Section 3.3: Cross-Stage Dependencies

**Sources**:
- QA_{ComponentName}.md → Dependencies on other components
- MotherHLD.md Section X.Y → Mentions of other stages

**Generate**:
```markdown
### 3.3 Cross-Stage Dependencies

**This feature depends on**:
- **Stage {N} ({Name})**: {Requirement from Q&A}
- {List all upstream dependencies from Q&A}

**This feature is required by**:
- **Stage {M} ({Name})**: {How they use this from Q&A}
- {List all downstream consumers from Q&A}

**Failure Impact**:
- If this stage fails: {Impact from Q&A or Critique}
- Checkpoint: {Resume strategy from Q&A}
```

#### Section 3.4: External Dependencies

**Sources**:
- QA_{ComponentName}.md → External services/APIs mentioned

**Generate**:
```markdown
### 3.4 External Dependencies

**Python Libraries**:
\`\`\`python
{From Q&A answers - exact import statements with versions}
import pandas as pd  # 2.0.0+
import numpy as np  # 1.24.0+
\`\`\`

**File System**:
- Read access: {From Q&A + Mother Part 1 paths}
- Write access: {From Q&A + Mother Part 1 paths}

**Environment Variables**:
- {From Q&A answers about config}

**External Services**: {From Q&A - or "None" if no external services}
```

---

### **Section 4: Configuration & Parameters**

**Sources**:
- MotherHLD.md Part 1 → Base CLI parameters (if applicable)
- QA_{ComponentName}.md → Component-specific parameters

**Generate**:
```markdown
## 4. Configuration & Parameters

### 4.1 CLI Parameters (if applicable)

{If component has CLI params from Q&A, create table}
| Parameter | Type | Default | Valid Values | Impact |
|-----------|------|---------|--------------|--------|
| {From Q&A} | {Type} | {Default} | {Range} | {Impact} |

{If no CLI params: "This component uses configuration from Mother Part 1 Section {X}."}

### 4.2 Internal Configuration

\`\`\`python
# Constants from Q&A answers
{CONSTANT_NAME} = {value}  # {Purpose from Q&A}
\`\`\`
```

---

### **Section 5: Data Schemas**

#### Section 5.1: Input Schema

**Sources**:
- QA_{ComponentName}.md → Exact input format answers (this is CRITICAL)

**Generate** (Complete table):
```markdown
### 5.1 Input Schema

**File**: {filename from Q&A}

| Column | Type | Range | Nulls? | Description | Example |
|--------|------|-------|--------|-------------|---------|
| {From Q&A - exact column name} | {Type from Q&A} | {Range from Q&A} | {Yes/No from Q&A} | {Description from Q&A} | {Example from Q&A} |
```

**Rules**:
- Include EVERY column from Q&A answer
- Use EXACT column names (case-sensitive)
- Copy types, ranges, descriptions verbatim from Q&A

#### Section 5.2: Output Schema

**Sources**:
- QA_{ComponentName}.md → Exact output format answers

**Generate** (Complete table - same format as 5.1):
```markdown
### 5.2 Output Schema

**File**: {filename from Q&A}

{Same table structure as 5.1, using Q&A output answers}
```

---

### **Section 6: Error Handling & Validation**

#### Section 6.1: Input Validation

**Sources**:
- QA_{ComponentName}.md → Validation rules answers

**Generate** (Pseudocode):
```markdown
### 6.1 Input Validation

\`\`\`python
def validate_input(data):
    """
    Validate input before processing.
    Source: QA Q{N}, Q{M}
    """
    # {Each validation check from Q&A answers}
    # Extract from Q&A answers about validation, edge cases

    assert condition, "error_message"  # From Q&A
\`\`\`
```

#### Section 6.2: Error Cases

**Sources**:
- QA_{ComponentName}.md → Error handling answers

**Generate** (Table):
```markdown
### 6.2 Error Cases

| Error | Detection | Handling | User Message | Exit Code |
|-------|-----------|----------|--------------|-----------|
| {From Q&A} | {Detection method from Q&A} | {Handling strategy from Q&A} | {EXACT message from Q&A} | {Code from Q&A} |
```

**Rules**:
- Use EXACT error messages from Q&A (copy verbatim)
- Include ALL error scenarios from Q&A

#### Section 6.3: Output Validation

**Sources**:
- QA_{ComponentName}.md → Output validation answers

**Generate** (Pseudocode - similar to 6.1)

---

### **Section 7: Performance & Scalability**

#### Section 7.1: Performance Targets

**Sources**:
- QA_{ComponentName}.md → Performance target answers

**Generate**:
```markdown
### 7.1 Performance Targets

- **Throughput**: {From Q&A - e.g., "300 videos in < 5 minutes"}
- **Memory**: {From Q&A - e.g., "Peak < 2GB"}
- **Disk**: {From Q&A - e.g., "< 1GB output"}
- **CPU**: {From Q&A - if specified}
```

#### Section 7.2: Measured Performance (if available)

{Skip if no data in Q&A}

#### Section 7.3: Bottlenecks & Mitigations

**Sources**:
- QA_{ComponentName}.md → Bottleneck answers
- Critique_{ComponentName}.md → Performance concerns raised

**Generate** (Table):
```markdown
### 7.3 Bottlenecks & Mitigations

| Bottleneck | Impact | Cause | Mitigation | Priority |
|------------|--------|-------|------------|----------|
| {From Q&A or Critique} | {Impact from Q&A} | {Cause from Q&A} | {Solution from Q&A} | {Priority} |
```

---

### **Section 8: Testing Strategy**

**Sources**:
- QA_{ComponentName}.md → Testing answers

**Generate**:
```markdown
## 8. Testing Strategy

### 8.1 Unit Tests

{From Q&A test scenario answers - list test cases}

- [ ] Test {scenario from Q&A}
- [ ] Test {edge case from Q&A}

### 8.2 Integration Tests

{From Q&A integration test answers}

- [ ] End-to-end: {Flow from Q&A}

### 8.3 Test Data

{From Q&A realistic test data answers}

**File**: {test file name}
\`\`\`csv
{Sample data from Q&A - minimum 3 rows}
\`\`\`

### 8.4 Test Execution

\`\`\`bash
{Test commands from Q&A or infer from tech stack in Mother Part 1}
\`\`\`
```

---

### **Section 9: Future Enhancements**

**Sources**:
- Critique_{ComponentName}.md → Deferred improvements
- QA_{ComponentName}.md → [NICE] priority items

**Generate**:
```markdown
## 9. Future Enhancements

### 9.1 Planned Improvements

{From Critique or Q&A [NICE] items}

- **Phase {N}**: {Enhancement}
  - Current: {Current state}
  - Future: {Desired state}
  - Impact: {Expected improvement}

### 9.2 Known Limitations

{From Critique concerns or Q&A}

- **{Limitation}**: {Description}
```

---

### **Section 10: References & Related Docs**

**Generate**:
```markdown
## 10. References & Related Docs

### 10.1 Parent Document

- **MotherHLD.md Section X.Y "{Section Title}"**
  - High-level component overview
  - Stage position in pipeline

### 10.2 Mother Document Foundation

- **MotherHLD.md Part 1: Foundation**
  - Section: {X} ({Topic}) - {How this component uses it}
  - Section: {Y} ({Topic}) - {How this component uses it}
  - {List ALL Part 1 sections referenced in Q&A}

**Note**: After first components complete, extract Part 1 → FoundationCHILD.md for reusability

### 10.3 Related Child Docs

{From Mother Doc - upstream and downstream components}

- **{UpstreamComponent}CHILD.md** (Stage {N}) - Produces input for this component
- **{DownstreamComponent}CHILD.md** (Stage {M}) - Consumes output from this component
```

---

### **Appendix A: Example Data**

**Sources**:
- QA_{ComponentName}.md → Realistic test data answers

**Generate**:
```markdown
## Appendix A: Example Data

### A.1 Sample Input ({N} rows)

**File**: {filename}

\`\`\`csv
{From Q&A - real column names, realistic values, minimum 3 rows}
\`\`\`

### A.2 Sample Output ({N} rows)

**File**: {filename}

\`\`\`csv
{From Q&A - corresponding output for above input}
\`\`\`
```

**Rules**:
- Use REAL data from Q&A (not "video_123" placeholders)
- Minimum 3 rows
- Match schemas from Section 5

---

### **Appendix B: Pseudocode (Complete)**

**Sources**:
- Section 2.3 pseudocode (expanded)
- QA_{ComponentName}.md → Detailed logic answers

**Generate** (30-50 lines):
```markdown
## Appendix B: Pseudocode (Complete)

### B.1 Full Pipeline

\`\`\`python
def process_component(input_path, output_path, config):
    """
    Complete implementation logic.

    Sources: Q&A Q{N}, Q{M}, Q{P}
    """
    # Expand Section 2.3 pseudocode to 30-50 lines
    # Include all edge cases from Q&A
    # Use realistic variable names
    # Show complete flow start to finish
\`\`\`
```

---

## Step 3: Anti-Hallucination Validation

Before outputting file, verify:

**Schema Check**:
- [ ] Every field name in Section 5 traced to Q&A answer
- [ ] Every type, range copied exactly from Q&A
- [ ] No invented columns

**Path Check**:
- [ ] Every file path uses Mother Part 1 directory structure
- [ ] No invented paths

**Error Message Check**:
- [ ] Every error message copied exactly from Q&A
- [ ] No invented error codes or messages

**Performance Check**:
- [ ] Every performance number from Q&A (not invented)

**Reference Check**:
- [ ] Section 10.2 lists ALL Part 1 sections mentioned in Q&A

**TODO Check**:
- [ ] ZERO TODOs in document (all info from Q&A and Critique)
- [ ] If info genuinely missing: Stop and create `Phase2_FollowUp_{ComponentName}.md` with missing questions

---

## Step 4: Output Complete File

If validation passes:

Output complete `{ComponentName}CHILD.md` with:
- All 10 sections filled
- 2 appendices with realistic examples
- NO TODOs
- NO placeholders

If validation fails (missing info):

**DO NOT generate incomplete HLD**

Instead:
1. Create `Phase2_FollowUp_{ComponentName}.md` with missing questions
2. Tell user: "Cannot generate complete HLD. Need answers to {N} additional questions. Please review Phase2_FollowUp_{ComponentName}.md and provide answers."

---

## Output File

**File**: `{ComponentName}CHILD.md`

**Status**: COMPLETE (perfect draft, ready for Phase 4 review if user wants changes)

---

## Key Rules

### Perfect Draft Rules
1. **NO TODOs** - All sections must be complete
2. **NO placeholders** - Use real data from Q&A
3. **Realistic examples** - From Q&A, not "example_123"
4. **Complete schemas** - ALL columns with types, ranges
5. **Exact error messages** - Copy from Q&A verbatim

### Source Tracing Rules
1. **Add source comments** - `# Source: Q&A Q5` in pseudocode
2. **Reference HLD sections** - "For HLD Section 5.1" in tables
3. **Cite Mother Part 1** - List specific sections in 10.2
4. **Credit Phase 1** - Use Critique for business context

### Anti-Hallucination Rules
1. **Don't invent field names** - Use Q&A exactly
2. **Don't invent paths** - Use Mother Part 1 exactly
3. **Don't invent messages** - Use Q&A exactly
4. **Don't invent numbers** - Use Q&A exactly
5. **Don't skip validation** - Check before outputting

---

## Completion Criteria

Before outputting:
- [ ] All 10 sections complete
- [ ] 2 appendices with realistic examples (min 3 rows each)
- [ ] ZERO TODOs
- [ ] All schemas complete (every column with type, range)
- [ ] All error messages exact (from Q&A)
- [ ] All paths use Mother Part 1 structure
- [ ] Section 10.2 lists ALL Part 1 sections used
- [ ] Anti-hallucination validation passed

---

## Next Phase

**Phase 4 (Optional)**: User reviews and requests specific section changes

If user requests changes:
- User invokes Phase 4 with:
  - Phase4_ReviewRefinement.md
  - {ComponentName}CHILD.md (this output)
  - Specific feedback

If no changes needed:
- {ComponentName}CHILD.md is ready for TI generation

---

**Version**: 1.0
**Last Updated**: 2025-01-28
**Applies To**: All projects using this documentation system
