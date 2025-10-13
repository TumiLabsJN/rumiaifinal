# Phase 1B: Existing Child Document Critique Instructions

> **Purpose**: Quality audit and validation of existing Child HLD documents
> **Output**: Critique_ExistingChild_{ComponentName}.md with findings and recommendations
> **Use Cases**: Legacy docs, quality audits, pre-TI validation

---

## Your Role

Quality Auditor reviewing existing Child HLD documents for completeness, accuracy, and alignment with documentation standards.

---

## Inputs Required

### Required
1. **Existing Child HLD**: ChildHLD_{ComponentName}.md or {ComponentName}CHILD.md
2. **Mother Document**: {MotherHLD.md} (for validation against source)

### Optional (Recommended)
1. **ChildTemplate.md**: To check if existing doc follows standard structure
2. **Phase 1 Output**: Critique_{ComponentName}.md (if exists - original business critique)
3. **Phase 2 Output**: QA_{ComponentName}.md (if exists - original Q&A)

---

## Your Task

### Step 1: Document Assessment

**Read Existing Child HLD:**
- Does it follow 10-section + 2 appendices structure?
- What sections are present?
- What sections are missing?
- Document version and last updated date

**Read Mother Document:**
- Find corresponding section (referenced in Child Section 10.1)
- Read Mother Part 1 (Foundation sections referenced in 10.2)
- Verify alignment between Mother and Child

---

### Step 2: Quality Audit (7 Dimensions)

Audit the existing Child HLD on 7 dimensions:

#### 1. Completeness
- Are all 10 required sections present?
- Are both appendices present?
- Are there TODOs or placeholders?
- Are schema tables complete (all columns defined)?
- Are all cross-references valid?

#### 2. Accuracy
- Does Section 1 (Context) match Mother Doc description?
- Do dependencies in Section 3 match actual Mother Part 1 sections?
- Do schemas in Section 5 match what's described in Section 2?
- Do error messages in Section 6 align with validation rules?
- Do test cases in Section 8 cover scenarios from Section 6?

#### 3. Traceability
- Section 10.1: Does Mother Doc section title match what's referenced?
- Section 10.2: Do all Mother Part 1 references point to real sections?
- Are configuration references traceable to Mother Part 1?
- Can you trace each schema field back to requirements?

#### 4. Consistency
- Do input schemas (5.1) match input dependencies (3.1)?
- Do output schemas (5.2) match output contracts (3.2)?
- Do validation rules (6.1/6.3) match schema constraints (5.1/5.2)?
- Do examples use consistent field names across sections?
- Are terms in Appendix A (Glossary) used consistently?

#### 5. Testability
- Section 8.1: Are unit tests realistic and specific?
- Section 8.2: Are integration tests defined?
- Section 8.3: Is test data provided with realistic values?
- Do tests cover error cases from Section 6?
- Do tests validate performance targets from Section 7?

#### 6. Implementation Readiness
- Can a developer implement from this HLD without guessing?
- Are error messages specific enough (not "Error occurred")?
- Are performance targets quantified (not "should be fast")?
- Are all dependencies clearly identified?
- Is configuration fully specified?

#### 7. Business Alignment
- Section 1.1: Does business goal align with Mother Doc system goals?
- Does component still solve the stated problem?
- Are risks acknowledged in Appendix B (Decision Log)?
- Are future enhancements (Section 9) realistic?

---

### Step 3: Create Audit Report

Create file: `Critique_ExistingChild_{ComponentName}.md`

Write structure:

```markdown
# Existing Child Document Critique: {ComponentName}

> **Target Document**: {ChildHLD_ComponentName.md}
> **Mother Doc**: {MotherHLD.md} Section X.Y
> **Audit Date**: {current_date}
> **Status**: IN PROGRESS

## Document Information

**File**: {ChildHLD_ComponentName.md}
**Version**: {version from doc metadata}
**Last Updated**: {date from doc metadata}
**Status**: {status from doc metadata}

## Structure Assessment

**Sections Present**: [list sections found]
**Sections Missing**: [list missing sections]
**Template Compliance**: [FULL | PARTIAL | NON-COMPLIANT]

## Quality Audit Findings

### 1. Completeness: [PASS | ISSUES FOUND]

**Issues**:
- [Issue 1: specific finding with section reference]
- [Issue 2: specific finding with section reference]

**Missing Elements**:
- [List TODOs, placeholders, incomplete tables]

### 2. Accuracy: [PASS | ISSUES FOUND]

**Alignment Issues**:
- [Discrepancy between Child and Mother Doc]
- [Inconsistency within Child doc sections]

**Verification Results**:
- Mother Doc Section X.Y reference: [VALID | INVALID | OUTDATED]
- Mother Part 1 references: [VALID | INVALID | MISSING]

### 3. Traceability: [PASS | ISSUES FOUND]

**Broken References**:
- [Section 10.1: Mother section reference broken]
- [Section 3.1: Dependency not found in Mother Part 1]

**Untraceable Elements**:
- [Schema fields with no source documentation]
- [Configuration parameters not in Mother Part 1]

### 4. Consistency: [PASS | ISSUES FOUND]

**Internal Inconsistencies**:
- [Schema field "X" in 5.1 but not validated in 6.1]
- [Different field names for same data: "user_id" vs "userId"]

### 5. Testability: [PASS | ISSUES FOUND]

**Test Gaps**:
- [Section 6.2 has 8 error cases, Section 8 only tests 3]
- [No integration test for cross-stage dependency]

**Unrealistic Tests**:
- [Test data uses placeholders, not realistic values]

### 6. Implementation Readiness: [READY | GAPS FOUND | NOT READY]

**Ambiguities**:
- [Error message too vague: "Invalid input"]
- [Performance target not quantified: "should be fast"]

**Missing Details**:
- [No retry strategy specified for external API calls]
- [No max file size specified for input validation]

### 7. Business Alignment: [ALIGNED | CONCERNS FOUND]

**Alignment Issues**:
- [Business goal doesn't match Mother Doc system goals]
- [Component addresses outdated requirement]

## Critical Issues (Must Fix Before TI)

[CRITICAL] = Blocks TI generation, must fix

1. **[CRITICAL]** {Issue category}: {Specific issue}
   - **Impact**: {Why this blocks implementation}
   - **Location**: {Section X.Y of Child HLD}
   - **Fix Required**: {What needs to change}

2. **[CRITICAL]** {Issue category}: {Specific issue}
   - **Impact**: {Why this blocks implementation}
   - **Location**: {Section X.Y of Child HLD}
   - **Fix Required**: {What needs to change}

[Continue for all CRITICAL issues]

## High-Priority Issues (Should Fix)

[HIGH] = Doesn't block TI, but creates technical debt

1. **[HIGH]** {Issue category}: {Specific issue}
   - **Impact**: {What could go wrong}
   - **Location**: {Section X.Y of Child HLD}
   - **Recommendation**: {Suggested fix}

[Continue for all HIGH issues]

## Low-Priority Issues (Nice to Fix)

[LOW] = Minor quality improvements

1. **[LOW]** {Issue category}: {Specific issue}
   - **Recommendation**: {Suggested fix}

[Continue for all LOW issues]

## Validation Questions & Answers

[Will be filled iteratively - see Step 4]

## Final Assessment

[Will be filled after Q&A complete - see Step 5]
```

---

### Step 4: Iterative Q&A Protocol (CRITICAL)

**ONE QUESTION AT A TIME** to clarify issues found.

**Scope**: Ask about [CRITICAL] and [HIGH] priority issues only. Skip [LOW] priority issues (user can address those later if desired).

#### Process:

1. **Ask Question 1** based on [CRITICAL] and [HIGH] priority issues found
   - Start with [CRITICAL] issues first, then [HIGH] issues
   - Make it specific and actionable
   - Reference exact section and line numbers
   - Example: "Section 5.1 lists 'aggregated_features.csv' with 185 columns, but only 12 are defined in the table. Should I list all 185, or were only 12 intended?"

2. **WAIT for user answer**

3. **IMMEDIATELY update Critique_ExistingChild_{ComponentName}.md** with:
   ```markdown
   ### Q1: {your question}
   **Answer**: {user's answer}
   **LLM Analysis**: {your interpretation}
   **Action Required**: {What needs to change in Child HLD}
   ```

4. **Ask Question 2**

5. **Repeat** until all [CRITICAL] and [HIGH] issues clarified

#### Question Guidelines:

**Ask about:**
- Missing information needed for implementation
- Inconsistencies between sections
- Broken or invalid references
- Ambiguous specifications
- Untraceable data elements

**Make questions:**
- Specific (reference exact section/line numbers)
- Actionable (clear what's needed to fix)
- Evidence-based (point to exact discrepancy)
- Prioritized (ask about [CRITICAL] issues first, then [HIGH] issues, skip [LOW] issues)

---

### Step 5: Final Assessment

After all Q&A complete, update `Critique_ExistingChild_{ComponentName}.md`:

```markdown
## Final Assessment

**Overall Quality**: [EXCELLENT | GOOD | NEEDS WORK | POOR]

**Summary**:
Based on audit findings:
- Completeness: {summary}
- Accuracy: {summary}
- Traceability: {summary}
- Consistency: {summary}
- Testability: {summary}
- Implementation Readiness: {summary}
- Business Alignment: {summary}

**Recommended Actions**:
1. **CRITICAL**: {Action 1 - must do before TI}
2. **CRITICAL**: {Action 2 - must do before TI}
3. **HIGH**: {Action 3 - should do}
4. **HIGH**: {Action 4 - should do}

**Ready for TI Generation**: [YES | NO - FIX CRITICAL ISSUES FIRST]

**If NO**: Critical issues that block TI:
- {Issue 1}
- {Issue 2}

**If YES**: Document meets minimum standards for TI generation.
Recommended: Address [HIGH] priority issues to improve quality.

**Next Steps**:
- [If YES]: Proceed to TI generation using TI_Generation_Prompt.md
- [If NO]: Use Phase 4 (Review & Refinement) to fix critical issues
- [If major gaps]: Consider re-running Phase 2 & 3 to regenerate HLD

**Status**: COMPLETE
```

---

## Output File Format

**File**: `Critique_ExistingChild_{ComponentName}.md`

**Complete Structure**:
```markdown
# Existing Child Document Critique: {ComponentName}

> **Target Document**: {ChildHLD_ComponentName.md}
> **Mother Doc**: {MotherHLD.md} Section X.Y
> **Audit Date**: {timestamp}
> **Status**: [IN PROGRESS | COMPLETE]

## Document Information
- File: {path}
- Version: {version}
- Last Updated: {date}
- Status: {status}

## Structure Assessment
- Sections Present: [list]
- Sections Missing: [list]
- Template Compliance: [FULL | PARTIAL | NON-COMPLIANT]

## Quality Audit Findings

### 1. Completeness: [PASS | ISSUES FOUND]
[Details]

### 2. Accuracy: [PASS | ISSUES FOUND]
[Details]

### 3. Traceability: [PASS | ISSUES FOUND]
[Details]

### 4. Consistency: [PASS | ISSUES FOUND]
[Details]

### 5. Testability: [PASS | ISSUES FOUND]
[Details]

### 6. Implementation Readiness: [READY | GAPS FOUND | NOT READY]
[Details]

### 7. Business Alignment: [ALIGNED | CONCERNS FOUND]
[Details]

## Critical Issues (Must Fix Before TI)
1. **[CRITICAL]** {category}: {issue} - {fix required}
2. [Continue...]

## High-Priority Issues (Should Fix)
1. **[HIGH]** {category}: {issue} - {recommendation}
2. [Continue...]

## Low-Priority Issues (Nice to Fix)
1. **[LOW]** {category}: {issue} - {recommendation}
2. [Continue...]

## Validation Questions & Answers

### Q1: {question}
**Answer**: {user answer}
**LLM Analysis**: {interpretation}
**Action Required**: {what to change}

### Q2: {question}
**Answer**: {user answer}
**LLM Analysis**: {interpretation}
**Action Required**: {what to change}

[Continue for all questions]

## Final Assessment

**Overall Quality**: [EXCELLENT | GOOD | NEEDS WORK | POOR]
**Summary**: {paragraph summarizing findings}
**Recommended Actions**: [numbered list]
**Ready for TI Generation**: [YES | NO]
**Next Steps**: [specific guidance]
**Status**: COMPLETE
```

---

## Key Rules

### Audit Rules
1. **Be objective** - Base findings on observable issues, not preferences
2. **Be specific** - Reference exact sections, line numbers, field names
3. **Be actionable** - Every issue should have clear fix/recommendation
4. **Use evidence** - Quote exact text showing the issue
5. **Prioritize correctly** - [CRITICAL] only for TI blockers

### Q&A Protocol Rules
1. **ONE question at a time** - Prevents context loss during compaction
2. **Write immediately** - Update file after EACH answer before asking next question
3. **Focus on clarification** - Ask to understand issues, not to challenge design
4. **Prioritize [CRITICAL]** - Ask about TI blockers first
5. **5-10 questions max** - Enough to clarify, not exhausting

### Anti-Hallucination Rules
1. **Verify references exist** - Check that Mother Doc sections are real
2. **Don't invent standards** - Only flag issues based on ChildTemplate.md or common sense
3. **Don't assume intent** - If ambiguous, ask user rather than assume wrong
4. **Trace everything** - Verify claims by checking source documents
5. **Stay in scope** - Audit THIS document, don't redesign the system

---

## Examples

### Good Finding (Specific, Evidence-Based)

```markdown
**[CRITICAL]** Traceability: Broken Mother Part 1 reference
- **Issue**: Section 10.2 references "Mother Part 1 Section 2.3: Client Architecture"
  but MLPlanningv2.md Part 1 only has Sections 1-2 (verified by reading Mother doc).
  Section 2.3 doesn't exist.
- **Impact**: Blocks implementation - can't find client architecture patterns
- **Location**: ChildHLD_VideoProcessing.md Section 10.2, line 485
- **Fix Required**: Update reference to correct Mother section, or add missing
  architecture details to Section 2 of this Child HLD
```

### Bad Finding (Generic, No Evidence)

```markdown
**[CRITICAL]** Documentation: References might be wrong
- **Issue**: Some references don't look right
- **Impact**: Could cause problems
- **Fix Required**: Fix references
```

### Good Question (Specific, Actionable)

```markdown
Q3: Section 5.1 Input Schema lists "aggregated_features.csv" with 185 columns
(line 287), but the table only defines 12 columns (lines 289-312).

Should I:
1. List all 185 columns in the table (complete schema)?
2. Keep only these 12 columns (partial schema for key fields)?
3. Reference another document that has the full 185-column schema?

For HLD completeness, I need the full schema or a valid reference to it.
```

### Bad Question (Generic, Vague)

```markdown
Q3: Is the schema complete?
```

---

## Special Cases

### Case 1: Child HLD created without Phase 1-3 (Legacy Doc)

**Response**: "This Child HLD appears to be a legacy document created outside the Phase 1-3 workflow. I'll audit it against ChildTemplate.md standards, but note:
- No Critique_{ComponentName}.md exists (can't verify business decisions)
- No QA_{ComponentName}.md exists (can't verify schema sources)
- Will flag untraceable elements as [HIGH] priority issues"

**Process**: Continue audit, but flag all untraceable data as issues requiring user clarification.

---

### Case 2: Child HLD has custom sections (not in template)

**Response**: "This Child HLD has custom sections not in ChildTemplate.md:
- Section 11: {Custom Section Name}
- Section 12: {Custom Section Name}

Should I:
1. Audit these as-is (accept custom structure)?
2. Recommend moving content to standard sections?
3. Flag as template non-compliance?"

**Wait for user guidance** before proceeding.

---

### Case 3: Multiple critical issues found (10+ issues)

**Response**: "Audit found 12 [CRITICAL] issues blocking TI generation. This suggests fundamental HLD quality problems.

Recommended approach:
1. **Option A**: Fix issues one-by-one using Phase 4 (time-consuming)
2. **Option B**: Re-run Phase 2 & 3 to regenerate HLD (faster, cleaner)

Which approach do you prefer?"

**Wait for user decision** before continuing with Q&A.

---

### Case 4: Mother Doc section referenced in 10.1 no longer exists

**Response**: "Section 10.1 references 'MotherHLD.md Section 2.4 \"Validation Logic\"' but when I read the Mother doc, Section 2.4 doesn't exist (doc only has sections 1-3).

Possible causes:
1. Mother doc was restructured after Child HLD created
2. Reference typo (should be Section 2.3?)
3. Child HLD is for different Mother doc version

Can you clarify which Mother doc section this Child HLD should reference?"

**Wait for user answer**, then update audit findings.

---

## Completion Criteria

Before marking Status: COMPLETE:
- [ ] All 7 quality dimensions audited
- [ ] All issues categorized ([CRITICAL] | [HIGH] | [LOW])
- [ ] All [CRITICAL] and [HIGH] issues have specific fix recommendations
- [ ] 5-10 validation questions asked AND answered
- [ ] Each Q&A has action required documented
- [ ] Final assessment made (Overall Quality rating)
- [ ] Ready for TI decision made (YES/NO)
- [ ] Next steps specified

---

## Next Steps After Audit

**If "Ready for TI Generation: YES":**
- All [CRITICAL] issues resolved (or none found)
- User can proceed directly to TI generation with TI_Generation_Prompt.md
- Optionally address [HIGH]/[LOW] priority issues first using Phase 4

**If "Ready for TI Generation: NO":**
- [CRITICAL] or [HIGH] issues block TI generation
- **Proceed to Phase 2B** (Existing Child Gap-Filling Q&A) to gather missing information
  - User invokes: "Follow Phase2B_ExistingChildQA.md for {ComponentName}"
  - Phase 2B asks questions to fill gaps identified in this audit
  - Output: QA_{ComponentName}.md with answers for all [CRITICAL]+[HIGH] issues
- **Then proceed to Phase 4** (Review & Refinement) to apply fixes using Q&A answers
  - User invokes: "Follow Phase4_ReviewRefinement.md for ChildHLD_{ComponentName}.md"
  - Phase 4 updates existing doc using QA answers
  - Output: Updated ChildHLD_{ComponentName}.md (v1.1 or v2.0)
- After fixes complete, optionally re-run Phase 1B to verify all issues resolved

**If "Needs Work" or "Poor" quality rating:**
- Recommend re-running Phase 2 & 3 to regenerate HLD
- Faster and cleaner than fixing extensive issues
- Preserves original as reference

---

## Usage Example

**User Prompt**:
```
Follow Phase1B_ExistingChildCritique.md instructions for:

/home/jorge/rumiaifinal/documentation_migration/FutureDevelopments/ChildDocs/VideoProcessingCHILD.md

Attach:
- Phase1B_ExistingChildCritique.md (instructions)
- VideoProcessingCHILD.md (target document)
- MLPlanningv2.md (Mother document for validation)
- ChildTemplate.md (optional - template reference)
```

**LLM Response**:
1. Reads VideoProcessingCHILD.md
2. Reads MLPlanningv2.md to validate references
3. Audits on 7 dimensions
4. Creates Critique_ExistingChild_VideoProcessing.md
5. Asks questions to clarify issues found
6. Updates critique file after each answer
7. Provides final assessment and next steps

---

**Version**: 1.0
**Last Updated**: 2025-01-28
**Applies To**: Existing Child HLD documents requiring quality audit
**Related**: Phase1_BusinessCritique.md (for new components), Phase4_ReviewRefinement.md (for fixing issues)
