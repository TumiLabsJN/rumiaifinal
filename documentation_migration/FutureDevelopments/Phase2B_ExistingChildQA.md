# Phase 2B: Existing Child Document - Gap-Filling Q&A

> **Purpose**: Fill knowledge gaps identified in existing Child HLD during Phase 1B audit
> **Output**: QA_{ComponentName}.md with complete answers for updating existing doc
> **Next Phase**: Phase 4 (Review & Refinement to apply fixes)

---

## Your Role

Requirements Engineer gathering missing information to complete an existing Child HLD document.

---

## Inputs Required

### Required
1. **Existing Child HLD**: ChildHLD_{ComponentName}.md (or {ComponentName}CHILD.md)
   - The document with gaps/issues identified in Phase 1B
2. **Phase 1B Output**: Critique_ExistingChild_{ComponentName}.md
   - Critical/High priority issues that need answers
3. **Mother Document**: {MotherHLD.md}
   - Section X.Y: Component description (for validation)
   - Part 1: Foundation (system context)

---

## Your Task

### Step 1: Read Inputs

**From Critique_ExistingChild_{ComponentName}.md:**
- Status (must be "COMPLETE")
- Ready for TI Generation (must be "NO" - otherwise skip Phase 2B)
- [CRITICAL] issues list - gaps that block TI generation
- [HIGH] issues list - gaps that reduce quality
- Q&A answers from Phase 1B (context about issues found)

**From Existing ChildHLD_{ComponentName}.md:**
- What sections exist vs missing
- What content is incomplete (TODOs, placeholders, partial schemas)
- What content contradicts Mother doc or Foundation

**From MotherHLD.md Part 1:**
- System architecture (directory structure, tech stack)
- Configuration patterns (how config is managed)
- Shared schemas (data structures used across components)

---

### Step 2: Map Issues to Knowledge Gaps

**For each [CRITICAL] and [HIGH] issue from Phase 1B, determine what information is needed.**

#### Example Mapping (from Phase 1B issues):

**Phase 1B Issue**: "[CRITICAL] Completeness: Section 5.1 lists 185 columns but only 12 defined in table"
→ **Knowledge Gap**: Need exact names, types, and ranges for missing 173 columns
→ **Category**: Input/Output Contracts (Category 1)

**Phase 1B Issue**: "[HIGH] Missing Appendix B (Decision Log)"
→ **Knowledge Gap**: Need 3-5 key design decisions with rationale
→ **Category**: Design Decisions (Category 7)

**Phase 1B Issue**: "[CRITICAL] Traceability: Section 10.2 references non-existent Mother Part 1 Section 2.3"
→ **Knowledge Gap**: Which Mother Part 1 sections does this component actually use?
→ **Category**: Dependencies & Integration (Category 2)

---

### Step 3: Organize Gaps by Category

Use same 7 categories as Phase 2, but focus on **filling existing doc gaps** instead of writing from scratch:

#### Category 1: Input/Output Contracts
**For existing HLD Sections**: 3.1, 3.2, 5.1, 5.2

**Missing details from Phase 1B audit**:
- Incomplete schema tables (missing columns, types, ranges)
- Vague input/output descriptions (no exact format specified)
- Missing file paths or data sources
- Undefined error handling for malformed input

**Questions to ask**:
- What are ALL columns in input CSV/JSON? (Not just subset)
- What exact types and ranges for each field?
- What exact file paths for input/output?

#### Category 2: Dependencies & Integration
**For existing HLD Sections**: 3.1, 3.2, 3.3, 3.4, 10.2

**Missing details from Phase 1B audit**:
- Broken Mother Part 1 references
- Unverified upstream/downstream dependencies
- Missing external dependency specifications

**Questions to ask**:
- Which Mother Part 1 sections does this component use? (Verify Section 10.2)
- Which other components provide input to this? (Verify Section 3.1)
- Which components consume output from this? (Verify Section 3.2)

#### Category 3: Edge Cases & Validation
**For existing HLD Sections**: 6.1, 6.2, 6.3

**Missing details from Phase 1B audit**:
- Incomplete validation rules
- Vague error messages ("Error occurred" instead of specific text)
- Missing edge case handling

**Questions to ask**:
- What exact validation rules for input fields?
- What exact error messages for each error case?
- What edge cases are missing from Section 6?

#### Category 4: Performance & Scale
**For existing HLD Sections**: 7.1, 7.2, 7.3

**Missing details from Phase 1B audit**:
- Vague performance targets ("should be fast" instead of "< 5 min")
- No scalability limits specified
- Missing bottleneck analysis

**Questions to ask**:
- What exact performance target? (Number + unit)
- What max scale? (Max input size, max throughput)
- What are known bottlenecks?

#### Category 5: Error Handling
**For existing HLD Sections**: 6.2

**Missing details from Phase 1B audit**:
- Generic error messages without specifics
- No retry strategies specified
- Unclear which errors are recoverable

**Questions to ask**:
- What exact wording for each user-facing error message?
- Which errors retry vs fail-fast?
- What retry strategy? (Max attempts, backoff)

#### Category 6: Testing
**For existing HLD Sections**: 8.1, 8.2, 8.3

**Missing details from Phase 1B audit**:
- No test data provided
- Test cases don't cover error scenarios from Section 6
- Unrealistic test examples (placeholders instead of real data)

**Questions to ask**:
- What realistic test data? (Actual sample rows with real values)
- What test cases for each error scenario?
- What edge cases to test?

#### Category 7: Design Decisions & Documentation
**For existing HLD Sections**: Appendix A (Glossary), Appendix B (Decision Log)

**Missing details from Phase 1B audit**:
- Missing Appendix A (Glossary)
- Missing Appendix B (Decision Log)
- Incomplete or inconsistent terminology

**Questions to ask**:
- What component-specific terms need definition? (3-5 terms for Appendix A)
- What 3-5 key design decisions for Appendix B?
- For each decision: What alternatives considered? What rationale? What trade-offs?

---

### Step 4: Prioritize Questions

Mark each question based on Phase 1B issue priority:

- **[CRITICAL]**: Question answers a [CRITICAL] issue from Phase 1B (blocks TI generation)
- **[HIGH]**: Question answers a [HIGH] issue from Phase 1B (quality improvement)
- **[NICE]**: Additional clarification not tied to Phase 1B issue

**Priority Logic**:
- [CRITICAL]: Must answer to unblock TI generation
- [HIGH]: Should answer to improve doc quality
- [NICE]: Can skip if time-constrained

---

### Step 5: Create QA Document

Create file: `QA_{ComponentName}.md`

Write structure:

```markdown
# Gap-Filling Q&A: {ComponentName} (Existing Doc Update)

> **Existing Doc**: ChildHLD_{ComponentName}.md (v{X.Y})
> **Phase 1B**: Critique_ExistingChild_{ComponentName}.md
> **Date**: {current_date}
> **Status**: IN PROGRESS

## Phase 1B Issues Summary

**[CRITICAL] Issues**: {count} - Block TI generation
**[HIGH] Issues**: {count} - Reduce quality
**[LOW] Issues**: {count} - Minor improvements

## Questions by Category

### Category 1: Input/Output Contracts

[Questions will be filled iteratively]

### Category 2: Dependencies & Integration

[Questions will be filled iteratively]

### Category 3: Edge Cases & Validation

[Questions will be filled iteratively]

### Category 4: Performance & Scale

[Questions will be filled iteratively]

### Category 5: Error Handling

[Questions will be filled iteratively]

### Category 6: Testing

[Questions will be filled iteratively]

### Category 7: Design Decisions & Documentation

[Questions will be filled iteratively]

## Completeness Check

[Will be filled at end - see Step 7]

## Proceed to Phase 4

[Will be filled at end - see Step 7]
```

---

### Step 6: Iterative Q&A Protocol (CRITICAL)

**ONE QUESTION AT A TIME** to prevent context loss.

**Scope**: Ask about [CRITICAL] and [HIGH] priority gaps only. Skip [LOW] (user can address later).

#### Process:

1. **Ask first [CRITICAL] question** under appropriate category
   - Reference the Phase 1B issue this addresses
   - Be specific about what existing doc is missing
   - Example: "[CRITICAL] Q1: Phase 1B found Section 5.1 lists 185 columns but only 12 defined. What are the exact names, types, and ranges for all 185 columns? (Addresses Phase 1B Issue: Completeness-1)"

2. **WAIT for user answer**

3. **IMMEDIATELY update QA_{ComponentName}.md** under appropriate category:
   ```markdown
   ### Category 1: Input/Output Contracts

   #### Q1: [CRITICAL] {question}
   **Addresses Phase 1B Issue**: {Issue ID from Critique_ExistingChild}
   **Answer**: {user's answer}
   **For HLD Section**: 5.1 (Input Schema)
   **Action**: Add 173 missing columns to Section 5.1 table
   ```

4. **Ask next [CRITICAL] question** (if any)

5. **When all [CRITICAL] answered**, ask [HIGH] questions

6. **Repeat** until all [CRITICAL] and [HIGH] questions answered

#### Question Guidelines:

**Good questions:**
- Specific: "What exact columns in aggregated_features.csv?" (not "What's the input?")
- Referenced: "Phase 1B Issue Completeness-3 flagged missing schema - need all 185 columns"
- Purposeful: "For updating HLD Section 5.1 - need column: name, type, range"
- Tied to Phase 1B: "Addresses [CRITICAL] issue about incomplete schema"

**Bad questions:**
- Generic: "Is the doc complete?"
- Not tied to audit: Asking about things Phase 1B didn't flag
- Vague: "How should this work?"

---

### Step 7: Completeness Check

After all [CRITICAL] and [HIGH] questions answered, validate you have info to fix all issues:

Update `QA_{ComponentName}.md`:

```markdown
## Completeness Check

Can fix all [CRITICAL] and [HIGH] issues from Phase 1B?

### [CRITICAL] Issues Resolution
- [ ] Issue 1: {Issue description} - {YES/NO - if NO, what's missing?}
- [ ] Issue 2: {Issue description} - {YES/NO - if NO, what's missing?}
- [Continue for all CRITICAL issues]

**All [CRITICAL] issues have answers**: {YES | NO}

### [HIGH] Issues Resolution
- [ ] Issue 1: {Issue description} - {YES/NO - if NO, what's missing?}
- [ ] Issue 2: {Issue description} - {YES/NO - if NO, what's missing?}
- [Continue for all HIGH issues]

**All [HIGH] issues have answers**: {YES | NO}

## Proceed to Phase 4

**Ready to Update Child HLD**: [YES | NO]

**If NO**: Missing information:
- {List what's still unclear}
- {Ask follow-up [CRITICAL] questions}

**If YES**: All critical gaps filled. Ready for Phase 4 (Review & Refinement).

**Status**: COMPLETE
```

**If NO**: Ask follow-up questions, repeat Step 6

**If YES**: Mark Status: COMPLETE

---

## Output File Format

**File**: `QA_{ComponentName}.md`

**Complete Structure**:
```markdown
# Gap-Filling Q&A: {ComponentName} (Existing Doc Update)

> **Existing Doc**: ChildHLD_{ComponentName}.md (v{version})
> **Phase 1B**: Critique_ExistingChild_{ComponentName}.md
> **Date**: {timestamp}
> **Status**: [IN PROGRESS | COMPLETE]

## Phase 1B Issues Summary
**[CRITICAL] Issues**: 3 - Block TI generation
**[HIGH] Issues**: 5 - Reduce quality
**[LOW] Issues**: 2 - Minor improvements

## Questions by Category

### Category 1: Input/Output Contracts

#### Q1: [CRITICAL] Phase 1B found Section 5.1 lists 185 columns but only 12 defined. What are all 185 column names, types, and ranges?
**Addresses Phase 1B Issue**: Completeness-1 (Section 5.1 incomplete schema)
**Answer**: {user lists all 185 columns with types and ranges}
**For HLD Section**: 5.1 (Input Schema)
**Action**: Add 173 missing columns to table

#### Q2: [HIGH] What exact output file path for rf_transformed.csv?
**Addresses Phase 1B Issue**: Accuracy-2 (vague output path)
**Answer**: {user provides exact path}
**For HLD Section**: 3.2 (Output Contracts)
**Action**: Update output path in Section 3.2

### Category 2: Dependencies & Integration

#### Q3: [CRITICAL] Section 10.2 references "Part 1 Section 2.3" which doesn't exist. Which Part 1 sections does this component use?
**Addresses Phase 1B Issue**: Traceability-1 (broken Mother Part 1 reference)
**Answer**: {user lists: Section 2 "Client Architecture", Section 4 "CLI Command Structure"}
**For HLD Section**: 10.2 (Mother Document Foundation)
**Action**: Replace Section 2.3 reference with Section 2 and Section 4

[... more Q&A ...]

### Category 7: Design Decisions & Documentation

#### Q12: [HIGH] Appendix B (Decision Log) is missing. What 3-5 key design decisions shaped this component?
**Addresses Phase 1B Issue**: Missing-Appendix-B
**Answer**: {user lists 3 decisions: Sequential vs parallel, Skip-on-fail policy, Checkpoint frequency}
**For HLD Section**: Appendix B (Decision Log)
**Action**: Create Appendix B with 3 decision entries

## Completeness Check

### [CRITICAL] Issues Resolution
- [✓] Issue 1: Incomplete schema (185 columns) - YES (Q1 answered)
- [✓] Issue 2: Broken Mother Part 1 reference - YES (Q3 answered)
- [✓] Issue 3: Missing error messages - YES (Q7 answered)

**All [CRITICAL] issues have answers**: YES

### [HIGH] Issues Resolution
- [✓] Issue 1: Vague output path - YES (Q2 answered)
- [✓] Issue 2: Missing Appendix B - YES (Q12 answered)
- [✓] Issue 3: No test data - YES (Q10 answered)
- [✓] Issue 4: Incomplete validation rules - YES (Q8 answered)
- [✓] Issue 5: Generic error messages - YES (Q7 answered)

**All [HIGH] issues have answers**: YES

## Proceed to Phase 4

**Ready to Update Child HLD**: YES
**Status**: COMPLETE
```

---

## Key Rules

### Gap Identification Rules
1. **Focus on Phase 1B issues** - Don't ask about things audit didn't flag
2. **Map issues to categories** - Each question addresses specific Phase 1B issue
3. **Prioritize correctly** - [CRITICAL] issues from Phase 1B → [CRITICAL] questions
4. **Be specific** - Reference exact sections/issues from Phase 1B audit

### Q&A Protocol Rules
1. **ONE question at a time** - Prevents context loss during compaction
2. **Write immediately** - Update file after EACH answer before asking next question
3. **Track resolution** - Note which Phase 1B issue each Q&A resolves
4. **Completeness check** - Verify all [CRITICAL]+[HIGH] issues have answers

### Anti-Hallucination Rules
1. **Don't invent fixes** - If Phase 1B flagged it, ask user for answer
2. **Don't assume content** - If existing doc is incomplete, ask for complete data
3. **Trace to Phase 1B** - Every question must map to specific Phase 1B issue
4. **Validate against Mother** - Cross-check answers with Mother doc when possible

---

## Examples

### Good Question (Tied to Phase 1B Issue)

```markdown
#### Q3: [CRITICAL] Phase 1B Issue Completeness-1 flagged Section 5.1 has incomplete schema (only 12 of 185 columns defined).

What are the exact names, types, ranges, and nullability for ALL 185 columns in aggregated_features.csv?

**For HLD Section**: 5.1 (Input Schema) - need complete table with:
- Column name (exact field name)
- Type (int, float, str, datetime)
- Range (min-max or valid values)
- Nulls? (Yes/No)
- Description
- Example value

**Addresses Phase 1B Issue**: Completeness-1
```

### Bad Question (Not Tied to Audit)

```markdown
#### Q3: What should the schema be?
```

---

## Completion Criteria

Before marking Status: COMPLETE:
- [ ] All [CRITICAL] questions asked AND answered (tied to Phase 1B issues)
- [ ] All [HIGH] questions asked AND answered (tied to Phase 1B issues)
- [ ] Answers include exact details (not vague descriptions)
- [ ] Each answer notes which HLD section to update
- [ ] Each answer notes which Phase 1B issue it resolves
- [ ] Completeness Check performed (all issues resolvable?)
- [ ] Proceed to Phase 4 decision made (YES/NO)

---

## Next Phase

If "Ready to Update Child HLD: YES":
- User invokes Phase 4 with:
  - Phase4_ReviewRefinement.md
  - ChildHLD_{ComponentName}.md (existing doc to update)
  - QA_{ComponentName}.md (this output - gap-filling answers)
  - Critique_ExistingChild_{ComponentName}.md (Phase 1B output - issue list)

If "Ready to Update Child HLD: NO":
- Ask remaining questions to fill gaps
- Re-run Completeness Check

---

## Differences from Phase 2 (New Doc)

| Aspect | Phase 2 (New Doc) | Phase 2B (Existing Doc) |
|--------|-------------------|-------------------------|
| **Input** | Mother Doc Section X.Y + Critique | Existing Child HLD + Critique_ExistingChild |
| **Goal** | Gather info to WRITE new HLD | Gather info to FIX existing HLD |
| **Questions** | "What's needed for all HLD sections?" | "What's missing from flagged sections?" |
| **Scope** | All 10 sections + 2 appendices | Only sections with issues from Phase 1B |
| **Next Phase** | Phase 3 (Generate new HLD) | Phase 4 (Update existing HLD) |
| **Category 7** | Not in Phase 2 | Added for Glossary/Decision Log gaps |

---

**Version**: 1.0
**Last Updated**: 2025-01-28
**Applies To**: Existing Child HLD documents requiring gap-filling after Phase 1B audit
**Related**: Phase2_ClarificationQA.md (for new documents), Phase1B_ExistingChildCritique.md, Phase4_ReviewRefinement.md
