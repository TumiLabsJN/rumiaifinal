# Phase 2: Clarification Q&A Instructions

> **Purpose**: Fill knowledge gaps before writing detailed HLD
> **Output**: QA_{ComponentName}.md with complete answers
> **Next Phase**: Phase 3 (Child HLD Generation)

---

## Your Role

Requirements Engineer preparing for detailed HLD creation. Ask questions to eliminate ALL ambiguities.

---

## Inputs Required

### Required
1. **Mother Document**: {MotherHLD.md}
   - Section X.Y: Component to clarify
   - Part 1: Foundation (system context)
2. **Phase 1 Output**: Critique_{ComponentName}.md (business decisions from Phase 1)

---

## Your Task

### Step 1: Read Inputs

**From Critique_{ComponentName}.md:**
- Final Decision (must be "Proceed to Phase 2: YES")
- Business concerns raised
- Q&A answers from Phase 1 (context about decisions made)

**From MotherHLD.md Section X.Y:**
- Component description
- Input/Output mentioned
- Process logic described

**From MotherHLD.md Part 1:**
- System architecture (directory structure, tech stack)
- Configuration patterns (how config is managed)
- Shared schemas (data structures used across components)

---

### Step 2: Identify Knowledge Gaps

Scan for missing details needed to write HLD Sections 2-8:

#### Category 1: Input/Output Contracts
**For HLD Sections**: 3.1 (Input Dependencies), 5.1 (Input Schema), 3.2 (Output Contracts), 5.2 (Output Schema)

Missing details:
- Exact input format? (CSV columns? JSON fields? API response structure?)
- Exact output format? (What columns? What fields? What structure?)
- File paths? (Where do files come from? Where do they go?)
- What if input malformed? (Missing fields? Wrong type? Out of range?)
- What if input missing entirely? (Fail? Use default? Skip?)

#### Category 2: Dependencies & Integration
**For HLD Sections**: 3.3 (Cross-Stage Dependencies), 3.4 (External Dependencies), 10.2 (References)

Missing details:
- Which Mother Part 1 sections does this use? (Architecture? Configuration? Schemas?)
- Which other components does this consume data from?
- Which other components consume data from this?
- External APIs/services involved?
- What if upstream component fails mid-process?

#### Category 3: Edge Cases & Validation
**For HLD Sections**: 6.1 (Input Validation), 6.2 (Error Cases), 6.3 (Output Validation)

Missing details:
- Min/max input size? (1 video? 1000 videos?)
- Empty input? (0 rows? Empty file?)
- Null/missing fields? (Fail? Use default? Skip row?)
- Invalid values? (Negative numbers? Out of range?)
- Duplicate records? (Skip? Error? Take first?)

#### Category 4: Performance & Scale
**For HLD Sections**: 7.1 (Performance Targets), 7.3 (Bottlenecks)

Missing details:
- Expected throughput? (e.g., "300 videos in < 5 minutes")
- Memory constraints? (Max peak? When to fail?)
- Disk constraints? (Max file size? Cleanup strategy?)
- What's "too slow"? (User-facing definition)
- What's "acceptable performance"?

#### Category 5: Error Handling
**For HLD Section**: 6.2 (Error Cases)

Missing details:
- Which errors are recoverable? (Retry? Fallback?)
- Which errors are fatal? (Exit immediately?)
- User-facing error messages? (Exact wording?)
- Retry strategies? (Max attempts? Backoff?)
- Logging requirements? (What to log? What level?)

#### Category 6: Testing
**For HLD Section**: 8 (Testing Strategy)

Missing details:
- Realistic test scenario? (Real data to use?)
- Edge cases to test? (From Category 3)
- Definition of "working correctly"? (What proves it works?)
- Integration test setup? (What other components needed?)

---

### Step 3: Categorize & Prioritize Questions

Mark each question:
- **[CRITICAL]** - Must answer before HLD (blocks HLD creation)
- **[HIGH]** - Should answer for complete HLD (incomplete without it)
- **[NICE]** - Can defer to implementation phase

**Priority Logic**:
- [CRITICAL]: Without this answer, cannot write corresponding HLD section
- [HIGH]: Can write section, but will have TODOs or gaps
- [NICE]: Doesn't affect HLD content, only implementation details

---

### Step 4: Create QA Document

Create file: `QA_{ComponentName}.md`

Write structure:

```markdown
# Clarification Q&A: {ComponentName}

> **Mother Doc**: {MotherHLD.md} Section X.Y "{Section Title}"
> **Phase 1**: Critique_{ComponentName}.md
> **Date**: {current_date}
> **Status**: IN PROGRESS

## Questions by Category

### Input/Output Contracts

[Questions will be filled iteratively]

### Dependencies & Integration

[Questions will be filled iteratively]

### Edge Cases & Validation

[Questions will be filled iteratively]

### Performance & Scale

[Questions will be filled iteratively]

### Error Handling

[Questions will be filled iteratively]

### Testing

[Questions will be filled iteratively]

## Completeness Check

[Will be filled at end - see Step 6]

## Proceed to Phase 3

[Will be filled at end - see Step 6]
```

---

### Step 5: Iterative Q&A Protocol (CRITICAL)

**ONE QUESTION AT A TIME** to prevent context loss.

#### Process:

1. **Ask first [CRITICAL] question** under appropriate category
   - Be specific
   - Reference Mother Doc content
   - Explain which HLD section needs this answer
   - Example: "[CRITICAL] Q1: Section X.Y mentions 'temporal_windows_updated.json' as input (line 715). What exact JSON fields are required? (For HLD Section 5.1: Input Schema)"

2. **WAIT for user answer**

3. **IMMEDIATELY update QA_{ComponentName}.md** under appropriate category:
   ```markdown
   ### Input/Output Contracts

   #### Q1: [CRITICAL] {question}
   **Answer**: {user's answer}
   **For HLD Section**: 5.1 (Input Schema)
   **Notes**: {any interpretation or follow-up needed}
   ```

4. **Ask next [CRITICAL] question** (if any)

5. **When all [CRITICAL] answered**, ask [HIGH] questions

6. **Repeat** until all [CRITICAL] and [HIGH] questions answered

#### Question Guidelines:

**Good questions:**
- Specific: "What exact columns in aggregated_features.csv?" (not "What's the input?")
- Referenced: "Section X.Y line 750 mentions 'bucket_base path'—is this from Mother Part 1 Section 2?"
- Purposeful: "For HLD Section 5.1 Input Schema—need column: name, type, range"
- Evidence-seeking: "What validates this 5-minute target? Current benchmarks?"

**Bad questions:**
- Generic: "What does this component do?"
- Already answered: Don't re-ask what's in Mother Doc or Critique
- Vague: "How should this work?"

---

### Step 6: Completeness Check

After all [CRITICAL] and [HIGH] questions answered, validate you can write HLD:

Update `QA_{ComponentName}.md`:

```markdown
## Completeness Check

Can write these HLD sections without TODOs or gaps?

- [ ] Section 2 (Architecture & Design)?
  - 2.1: High-level approach - {YES/NO - if NO, what's missing?}
  - 2.2: Data flow - {YES/NO - if NO, what's missing?}
  - 2.3: Detailed process - {YES/NO - if NO, what's missing?}

- [ ] Section 3 (Dependencies & Integration)?
  - 3.1: Input dependencies - {YES/NO - if NO, what's missing?}
  - 3.2: Output contracts - {YES/NO - if NO, what's missing?}
  - 3.3: Cross-stage dependencies - {YES/NO - if NO, what's missing?}
  - 3.4: External dependencies - {YES/NO - if NO, what's missing?}

- [ ] Section 5 (Data Schemas)?
  - 5.1: Input schema - {YES/NO - if NO, what's missing?}
  - 5.2: Output schema - {YES/NO - if NO, what's missing?}

- [ ] Section 6 (Error Handling)?
  - 6.1: Input validation - {YES/NO - if NO, what's missing?}
  - 6.2: Error cases - {YES/NO - if NO, what's missing?}
  - 6.3: Output validation - {YES/NO - if NO, what's missing?}

- [ ] Section 8 (Testing Strategy)?
  - 8.1-8.3: Test cases - {YES/NO - if NO, what's missing?}

## Proceed to Phase 3

**Ready for HLD Generation**: [YES | NO]

**If NO**: Missing information:
- {List what's still unclear}
- {Ask follow-up [CRITICAL] questions}

**If YES**: All critical info gathered. Ready for Phase 3.

**Status**: COMPLETE
```

**If NO**: Ask follow-up questions, repeat Step 5

**If YES**: Mark Status: COMPLETE

---

## Output File Format

**File**: `QA_{ComponentName}.md`

**Complete Structure**:
```markdown
# Clarification Q&A: {ComponentName}

> **Mother Doc**: {MotherHLD.md} Section X.Y "{Section Title}"
> **Phase 1**: Critique_{ComponentName}.md
> **Date**: {timestamp}
> **Status**: [IN PROGRESS | COMPLETE]

## Questions by Category

### Input/Output Contracts

#### Q1: [CRITICAL] What are the exact columns in the input CSV?
**Answer**: {user answer with column names, types, ranges}
**For HLD Section**: 5.1 (Input Schema)

#### Q2: [HIGH] What if the 'duration' column is missing?
**Answer**: {user answer: fail-fast vs use default}
**For HLD Section**: 6.1 (Input Validation), 6.2 (Error Cases)

### Dependencies & Integration

#### Q3: [CRITICAL] Which Mother Part 1 sections does this component use?
**Answer**: {user lists: Architecture Section 2 for paths, Configuration Section 4 for CLI params}
**For HLD Section**: 10.2 (References), 3.1 (Input Dependencies)

#### Q4: [CRITICAL] Where do input files come from? (Which stage/component?)
**Answer**: {user answer: Stage 3 produces aggregated_features.csv}
**For HLD Section**: 3.3 (Cross-Stage Dependencies)

### Edge Cases & Validation

#### Q5: [CRITICAL] Min/max number of videos to process?
**Answer**: {user answer: Min 10, Max 500, fail if outside range}
**For HLD Section**: 6.1 (Input Validation), 6.2 (Error Cases)

#### Q6: [HIGH] What if input CSV has 0 rows?
**Answer**: {user answer: fail-fast with error "No videos to process"}
**For HLD Section**: 6.2 (Error Cases)

### Performance & Scale

#### Q7: [CRITICAL] What's the performance target?
**Answer**: {user answer: 300 videos in < 5 minutes, fail if > 10 minutes}
**For HLD Section**: 7.1 (Performance Targets)

#### Q8: [HIGH] What's the memory constraint?
**Answer**: {user answer: Peak < 2GB, warn at 1.5GB}
**For HLD Section**: 7.1 (Performance Targets), 7.3 (Bottlenecks)

### Error Handling

#### Q9: [CRITICAL] Which errors are recoverable vs fatal?
**Answer**: {user answer: Network timeout = retry 3x, Invalid schema = fatal}
**For HLD Section**: 6.2 (Error Cases)

#### Q10: [HIGH] What are the exact user-facing error messages?
**Answer**: {user provides exact wording for 3-5 error scenarios}
**For HLD Section**: 6.2 (Error Cases)

### Testing

#### Q11: [HIGH] What's a realistic test scenario with sample data?
**Answer**: {user provides: 10 videos, bucket 18-33s, CSV with 185 columns}
**For HLD Section**: 8.3 (Test Data)

#### Q12: [HIGH] What edge cases must be tested?
**Answer**: {user lists: empty input, missing columns, out-of-range values}
**For HLD Section**: 8.1 (Unit Tests)

## Completeness Check

- [✓] Section 2 (Architecture & Design): YES
- [✓] Section 3 (Dependencies & Integration): YES
- [✓] Section 5 (Data Schemas): YES
- [✓] Section 6 (Error Handling): YES
- [✓] Section 8 (Testing Strategy): YES

## Proceed to Phase 3

**Ready for HLD Generation**: YES
**Status**: COMPLETE
```

---

## Key Rules

### Question Protocol Rules
1. **Ask [CRITICAL] first** - Block on must-haves before asking nice-to-haves
2. **ONE question at a time** - Write answer before asking next question
3. **Reference HLD sections** - Show where each answer will be used
4. **Be specific** - Ask for exact formats, exact wording, exact numbers
5. **Validate completeness** - Check if you can write each HLD section

### Integration with Mother Doc Part 1
1. **Reference Foundation** - Ask which Part 1 sections this component uses
2. **Use Foundation paths** - Ask about file locations using Part 1 directory structure
3. **Use Foundation config** - Ask about configuration using Part 1 patterns
4. **Use Foundation schemas** - Ask if shared schemas from Part 1 are used

### Anti-Hallucination Rules
1. **Don't assume** - If not in Mother Doc or Critique, ASK
2. **Don't invent schemas** - Ask for exact field names, types, ranges
3. **Don't invent error messages** - Ask for exact wording
4. **Don't invent performance targets** - Ask for specific numbers with units
5. **Don't invent paths** - Use Mother Part 1 architecture, ask if unclear

---

## Examples

### Good Question (Specific, References HLD Section)
```markdown
#### Q3: [CRITICAL] Section X.Y "{Section Title}" mentions "bucket_base path" for file storage.
Is this the path from Mother Part 1 Section 2.2 "Client Architecture" (`/data/clients/{id}/buckets/{bucket}/`)?
If yes, which subdirectory under bucket_base? (e.g., ml_analysis/, models/, reports/)
**For HLD Section**: 3.2 (Output Contracts), 8 (File Structure)
```

### Bad Question (Generic, No Context)
```markdown
#### Q3: Where do files go?
```

### Good Answer Documentation
```markdown
#### Q5: [CRITICAL] What are the exact columns in aggregated_features.csv input?
**Answer**:
Bucket 18-33s has 185 columns:
- hook_scene_count (int, 0-20)
- hook_eye_contact_rate (float, 0.0-1.0)
- middle_1_scene_count (int, 0-20)
- middle_2_scene_count (int, 0-20)
- middle_3_scene_count (int, 0-20)
- middle_4_scene_count (int, 0-20)
- closing_scene_count (int, 0-20)
- duration (float, 18.0-33.0 for this bucket)
- create_time (datetime)
- [... user lists all 185 columns with types and ranges]

**For HLD Section**: 5.1 (Input Schema) - will create complete table
```

---

## Completion Criteria

Before marking Status: COMPLETE:
- [ ] All [CRITICAL] questions asked AND answered
- [ ] All [HIGH] questions asked AND answered
- [ ] Answers include exact details (not vague descriptions)
- [ ] Each answer notes which HLD section(s) it feeds
- [ ] Completeness Check performed (can write all HLD sections?)
- [ ] Proceed to Phase 3 decision made (YES/NO)

---

## Next Phase

If "Ready for HLD Generation: YES":
- User invokes Phase 3 with:
  - Phase3_ChildHLDGeneration.md
  - ChildTemplate.md
  - MotherHLD.md
  - Critique_{ComponentName}.md (Phase 1 output)
  - QA_{ComponentName}.md (this output)

If "Ready for HLD Generation: NO":
- Ask remaining critical questions
- User answers
- Re-run Completeness Check

---

**Version**: 1.0
**Last Updated**: 2025-01-28
**Applies To**: All projects using this documentation system
