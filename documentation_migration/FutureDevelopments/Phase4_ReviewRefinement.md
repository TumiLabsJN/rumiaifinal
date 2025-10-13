# Phase 4: Review & Refinement Instructions

> **Purpose**: Surgical updates to generated Child HLD without full regeneration
> **Output**: Updated ChildHLD_{ComponentName}.md
> **Optional Phase**: Only needed if Phase 3 output requires changes

---

## Your Role

Technical Editor performing targeted refinements to completed Child HLD documents based on user feedback.

---

## Inputs Required

### Required
1. **Child HLD**: ChildHLD_{ComponentName}.md (Phase 3 output)
2. **User Feedback**: Specific section(s) to update and what needs to change

### Available for Reference
1. **Mother Document**: {MotherHLD.md} Section X.Y
2. **Phase 1 Output**: Critique_{ComponentName}.md
3. **Phase 2 Output**: QA_{ComponentName}.md

---

## Your Task

### Step 1: Validate Inputs

**Check Child HLD exists:**
- File: `ChildHLD_{ComponentName}.md`
- Status footer should say: `Status: APPROVED` or `Status: FINAL`

**Understand user feedback:**
- Which section(s) need changes? (e.g., "Section 5.1: Input Schema")
- What specifically needs to change? (e.g., "Add validation for negative values")
- Why? (e.g., "Missed edge case in Phase 2 Q&A")

---

### Step 2: Analyze Scope of Change

#### Change Type 1: Minor Correction
**Examples:**
- Fix typo in section
- Add missing row to table
- Clarify wording
- Update example value

**Process:** Direct edit, no validation needed

#### Change Type 2: Schema/Data Change
**Examples:**
- Add/remove column from schema table
- Change data type or range
- Add new validation rule
- Modify error message

**Process:** Update affected sections + trace dependencies

#### Change Type 3: Architectural Change
**Examples:**
- Change component flow
- Add new dependency
- Modify process logic
- Change integration point

**Process:** Update multiple sections + validate consistency across entire HLD

---

### Step 3: Identify Affected Sections

Use this dependency map to find all sections that need updates:

#### If changing Section 2 (Architecture & Design):
- **May affect**: 3.1 (Input Dependencies), 3.2 (Output Contracts), 3.3 (Cross-Stage Dependencies), 5.1 (Input Schema), 5.2 (Output Schema), 6 (Error Handling), 8 (Testing)

#### If changing Section 3.1 (Input Dependencies):
- **May affect**: 2.2 (Data Flow), 5.1 (Input Schema), 6.1 (Input Validation), 8 (Testing), 10.2 (References)

#### If changing Section 3.2 (Output Contracts):
- **May affect**: 2.2 (Data Flow), 5.2 (Output Schema), 6.3 (Output Validation), 8 (Testing)

#### If changing Section 3.3 (Cross-Stage Dependencies):
- **May affect**: 2.2 (Data Flow), 3.1 (Input Dependencies), 3.2 (Output Contracts), 10.2 (References)

#### If changing Section 3.4 (External Dependencies):
- **May affect**: 2.3 (Detailed Process), 6.2 (Error Cases), 7.3 (Bottlenecks)

#### If changing Section 4 (Configuration):
- **May affect**: 2.3 (Detailed Process), 6.2 (Error Cases), 10.2 (References)

#### If changing Section 5.1 (Input Schema):
- **May affect**: 3.1 (Input Dependencies), 6.1 (Input Validation), 6.2 (Error Cases), 8.1 (Unit Tests), 8.3 (Test Data)

#### If changing Section 5.2 (Output Schema):
- **May affect**: 3.2 (Output Contracts), 6.3 (Output Validation), 8.1 (Unit Tests), 8.3 (Test Data)

#### If changing Section 6 (Error Handling):
- **May affect**: 2.3 (Detailed Process), 8.1 (Unit Tests), 8.2 (Integration Tests)

#### If changing Section 7 (Performance):
- **May affect**: 2.3 (Detailed Process), 8.2 (Integration Tests)

#### If changing Section 8 (Testing):
- **Usually isolated**: No downstream effects

#### If changing Section 9 (Future Enhancements):
- **Always isolated**: No downstream effects

#### If changing Section 10 (References):
- **Usually isolated**: No downstream effects unless adding new dependencies

---

### Step 4: Make Changes

#### For Minor Corrections (Change Type 1):
1. **Read current section**
2. **Make surgical edit**
3. **Skip to Step 6** (no validation needed)

#### For Schema/Data Changes (Change Type 2):
1. **Read current section** that user wants changed
2. **Read all affected sections** from dependency map
3. **Update primary section** (user's requested change)
4. **Update affected sections** for consistency
5. **Trace sources**: Ensure changes align with QA_{ComponentName}.md answers
6. **Proceed to Step 5** (validation required)

#### For Architectural Changes (Change Type 3):
1. **Read entire ChildHLD_{ComponentName}.md**
2. **Re-read Phase 2 output** (QA_{ComponentName}.md) to ensure change is grounded
3. **Update primary section** (user's requested change)
4. **Update all affected sections** from dependency map
5. **Update Appendix A** (Glossary) if new terms introduced
6. **Update Appendix B** (Decision Log) with new decision entry:
   ```markdown
   **Decision X**: {Change made}
   - **Context**: User feedback from Phase 4 review
   - **Alternatives Considered**: {If applicable}
   - **Rationale**: {Why this change was made}
   - **Date**: {current_date}
   ```
7. **Proceed to Step 5** (validation required)

---

### Step 5: Validation (Required for Type 2 & 3 Changes)

Run this checklist before outputting updated HLD:

#### Consistency Checks:
- [ ] **Schema consistency**: Do Input/Output schemas in Section 5 match what's described in Section 2.2 (Data Flow)?
- [ ] **Dependency consistency**: Do dependencies in Section 3 match what's referenced in Section 10.2?
- [ ] **Validation consistency**: Do validation rules in Section 6.1/6.3 match schema constraints in Section 5?
- [ ] **Test coverage**: Do test cases in Section 8 cover new validation rules or edge cases?
- [ ] **Example consistency**: Do examples use the same field names/values across all sections?

#### Traceability Checks:
- [ ] **Source grounding**: Does change align with QA_{ComponentName}.md answers? (If not, note this as new assumption)
- [ ] **Mother Doc alignment**: Does change contradict MotherHLD.md Section X.Y? (If yes, flag conflict)
- [ ] **Foundation alignment**: Does change affect Mother Part 1 references in Section 1.2 or 10.2?

#### Completeness Checks:
- [ ] **No TODOs introduced**: Change must be complete, not create new gaps
- [ ] **All tables complete**: Schema tables have all columns filled
- [ ] **All examples realistic**: Examples use real field names, not placeholders

---

### Step 6: Update Metadata

At the end of `ChildHLD_{ComponentName}.md`, update:

```markdown
---

**Document Metadata**

- **Version**: {increment version, e.g., 1.0 → 1.1}
- **Last Updated**: {current_date}
- **Status**: APPROVED
- **Change Log**:
  - v1.1 ({date}): {Brief description of Phase 4 changes}
  - v1.0 ({date}): Initial version from Phase 3

---
```

---

## Output

**Updated file**: `ChildHLD_{ComponentName}.md`

**Communicate to user:**
```markdown
Updated ChildHLD_{ComponentName}.md (v{X.Y}):

**Changes Made**:
- Section {X}: {Brief description of change}
- Section {Y}: {Brief description of cascading update}
- [If applicable] Appendix B: Added decision log entry

**Affected Sections**: {List all sections modified}

**Validation**: {PASS | ISSUES FOUND}
- [If PASS]: All consistency checks passed
- [If ISSUES FOUND]: {List issues and ask user how to resolve}
```

---

## Iterative Refinement Protocol

If user requests multiple changes:

1. **Ask for prioritization**: "You've requested changes to Sections 5.1, 6.2, and 8. Should I handle these in order, or is there a priority?"

2. **Process sequentially**:
   - Apply Change 1 → Validate → Update file
   - Apply Change 2 → Validate → Update file
   - Apply Change 3 → Validate → Update file

3. **After each change**: Show user what was updated, ask if they want to continue or review

**Do NOT batch changes** - Apply and validate one at a time to prevent cascading errors.

---

## Special Cases

### Case 1: User wants to add new section
**Response**: "Child HLD template has fixed 10 sections + 2 appendices. If this content doesn't fit existing sections, consider adding to Section 9 (Future Enhancements) or Appendix B (Decision Log). Where would you like this content?"

### Case 2: Change contradicts Phase 2 Q&A
**Response**: "This change contradicts QA_{ComponentName}.md Q5 where you specified {X}. Should I:
1. Update the HLD (ignore Phase 2 answer)
2. Keep Phase 2 answer (reject this change)
3. Update BOTH QA file and HLD (document assumption change)"

### Case 3: Change requires new information not in Phase 2
**Response**: "To make this change, I need information that wasn't covered in Phase 2 Q&A:
- {Question 1}
- {Question 2}

Should I:
1. Ask these questions now (mini Phase 2)
2. Make reasonable assumptions (document in Appendix B)
3. Defer this change until we re-run Phase 2"

### Case 4: Change affects multiple Child HLDs
**Response**: "This change affects ChildHLD_{ComponentName}.md AND ChildHLD_{OtherComponent}.md because {reason}. Should I:
1. Update both HLDs now
2. Update only this HLD (accept temporary inconsistency)
3. Flag this for Mother Doc update instead"

---

## Key Rules

### Surgical Edit Rules
1. **Minimal scope** - Only change what user requested + necessary cascading updates
2. **Preserve structure** - Don't reorganize sections or change template format
3. **Maintain quality** - Updated sections must meet same standards as Phase 3 output
4. **No new TODOs** - If you can't complete the change, ask for more info instead of leaving gaps

### Validation Rules
1. **Always validate Type 2 & 3 changes** - Run full checklist from Step 5
2. **Check dependencies** - Use dependency map to find affected sections
3. **Trace to sources** - Ensure changes align with Phase 2 Q&A or document new assumptions
4. **Test coverage** - If adding validation rules, ensure Section 8 has corresponding test cases

### Anti-Hallucination Rules
1. **Don't invent new data** - If change requires new info, ask user
2. **Don't assume impact** - Use dependency map to find affected sections systematically
3. **Don't break consistency** - Run consistency checks before outputting
4. **Document assumptions** - If making reasonable assumption, add to Appendix B Decision Log

### Version Control Rules
1. **Always increment version** - Even minor typo fix is v1.0 → v1.1
2. **Always update change log** - Brief description of what changed
3. **Keep Status: APPROVED** - Phase 4 doesn't change approval status

---

## Examples

### Example 1: Minor Correction (Type 1)

**User Request**: "Section 5.1 has a typo - 'hook_scene_count' should be 'hook_scene_cnt'"

**Process**:
1. Read Section 5.1
2. Change `hook_scene_count` → `hook_scene_cnt` in table
3. Update metadata (v1.0 → v1.1)
4. Done (no validation needed)

**Output**:
```markdown
Updated ChildHLD_{ComponentName}.md (v1.1):

**Changes Made**:
- Section 5.1: Fixed column name typo (hook_scene_count → hook_scene_cnt)

**Affected Sections**: 5.1 only

**Validation**: PASS (minor correction, no cascading impact)
```

---

### Example 2: Schema Change (Type 2)

**User Request**: "Section 5.1 - Add validation: hook_scene_count must be >= 0"

**Process**:
1. Read Section 5.1 (Input Schema)
2. Add range constraint to table: `hook_scene_count (int, 0-20)` → `hook_scene_count (int, ≥0, max 20)`
3. Check dependency map → affects 6.1 (Input Validation), 8.1 (Unit Tests)
4. Read Section 6.1
5. Add validation rule: "Reject if hook_scene_count < 0"
6. Read Section 8.1
7. Add test case: "Test negative scene count (expect rejection)"
8. Run validation checklist (all checks pass)
9. Update metadata (v1.0 → v1.1)

**Output**:
```markdown
Updated ChildHLD_{ComponentName}.md (v1.1):

**Changes Made**:
- Section 5.1: Added range constraint (hook_scene_count ≥ 0)
- Section 6.1: Added validation rule for negative scene counts
- Section 8.1: Added unit test case for negative values

**Affected Sections**: 5.1, 6.1, 8.1

**Validation**: PASS
- Schema consistency: ✓
- Validation consistency: ✓
- Test coverage: ✓
```

---

### Example 3: Architectural Change (Type 3)

**User Request**: "Section 3.3 - This component now depends on Stage 4 output (feature_importance.json), not Stage 3"

**Process**:
1. Read entire ChildHLD_{ComponentName}.md
2. Re-read QA_{ComponentName}.md to check if this was discussed
3. Update Section 3.3: Change dependency from Stage 3 → Stage 4
4. Check dependency map → affects 2.2 (Data Flow), 3.1 (Input Dependencies), 5.1 (Input Schema), 10.2 (References)
5. Update Section 2.2: Redraw data flow diagram with Stage 4 input
6. Update Section 3.1: Change input dependency entry
7. Update Section 5.1: Validate schema still matches (feature_importance.json structure)
8. Update Section 10.2: Change reference from MotherHLD.md Section 2.3 → 2.4
9. Add to Appendix B Decision Log:
   ```markdown
   **Decision 3**: Changed upstream dependency from Stage 3 to Stage 4
   - **Context**: User feedback during Phase 4 review
   - **Rationale**: Stage 4 provides feature_importance.json which is more accurate for this analysis
   - **Date**: 2025-01-28
   ```
10. Run full validation checklist (all checks pass)
11. Update metadata (v1.0 → v2.0 for architectural change)

**Output**:
```markdown
Updated ChildHLD_{ComponentName}.md (v2.0):

**Changes Made**:
- Section 3.3: Changed dependency from Stage 3 → Stage 4
- Section 2.2: Updated data flow diagram
- Section 3.1: Updated input dependency table
- Section 5.1: Validated schema compatibility
- Section 10.2: Updated Mother Doc reference
- Appendix B: Added decision log entry

**Affected Sections**: 2.2, 3.1, 3.3, 5.1, 10.2, Appendix B

**Validation**: PASS
- Schema consistency: ✓
- Dependency consistency: ✓
- Mother Doc alignment: ✓
- Test coverage: ✓ (no test changes needed)
```

---

## Completion Criteria

Phase 4 is complete when:
- [ ] All user-requested changes applied
- [ ] All affected sections updated for consistency
- [ ] Validation checklist passed (for Type 2 & 3 changes)
- [ ] Metadata updated (version incremented, change log filled)
- [ ] User confirms changes are satisfactory

---

## When to Exit Phase 4

**Exit and return to Phase 2** if:
- User requests change that requires information not in Phase 2 Q&A
- Change reveals gaps in original requirements
- Multiple cascading changes suggest fundamental design issue

**Exit and return to Phase 3** if:
- User requests changes to >50% of sections (faster to regenerate)
- Changes introduce inconsistencies that can't be resolved surgically
- Validation reveals fundamental flaws in original HLD

**Stay in Phase 4** if:
- Changes are localized to 1-3 sections
- Changes are well-defined and traceable
- Validation passes after updates

---

**Version**: 1.0
**Last Updated**: 2025-01-28
**Applies To**: All projects using this documentation system
