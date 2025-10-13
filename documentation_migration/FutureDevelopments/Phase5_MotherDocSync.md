# Phase 5: Document Synchronization (Three-Tier)

> **Purpose**: Maintain consistency across three-tier architecture: Mother HLD ← Foundation Child ← Component Children
> **When to Use**: After Phase 3 or Phase 4 when Child work reveals Foundation or Mother doc gaps, contradictions, or outdated content
> **Output**: FoundationSync_{ComponentName}.md OR MotherSync_{ComponentName}.md (proposed changes) + Updated docs (if approved)

---

## Your Role

Documentation System Maintainer ensuring all three tiers stay accurate, complete, and consistent.

---

## Three-Tier Architecture

```
Mother HLD Part 1 (Foundation)
    ↓ defines system-wide patterns
FoundationCHILD.md (Shared Foundation)
    ↓ detailed foundation used by all components
Component Child HLDs (Individual Components)
    ReviewCSVGenerationCHILD.md
    FeatureAggregationCHILD.md
    MLTrainingCHILD.md
    ... etc
```

**Key Principle**: Foundation content flows downward. Issues discovered at Component level may require Foundation or Mother updates.

---

## Inputs Required

### Required
1. **Component Child HLD**: ChildHLD_{ComponentName}.md (newly created or recently updated)
2. **Foundation Child HLD**: FoundationCHILD.md (shared foundation document)
3. **Mother HLD**: {MotherHLD.md} (source of truth, includes Part 1: Foundation)

---

## Your Task

### Step 1: Identify Documentation Issues

**Compare Component Child → Foundation Child → Mother HLD** to find discrepancies, gaps, or contradictions.

**Method**: Read all three documents and look for patterns where content diverges or contradicts.

---

#### Category 1: Broken References (Component → Foundation)

**Symptom**: Component Child references FoundationCHILD.md sections that don't exist

**Examples**:
- "Component Section 10.2 references 'FoundationCHILD.md Section 2.3' but Foundation only has Sections 2.1, 2.2"
- "Component Section 1.2 references 'Foundation Section 5: Glossary' which doesn't exist"
- "Component Section 3.1 lists dependency on 'FoundationCHILD.md Section 4.2' but Foundation ends at Section 4.1"

**Detection**:
- Read Component Section 10.2 (Foundation references)
- Check if referenced Foundation sections actually exist
- Read Component Section 1.2 (Where This Fits in Pipeline) for Foundation section references
- Verify all Foundation section numbers against actual FoundationCHILD.md structure

**Fix Location**: FoundationCHILD.md (add missing sections)

---

#### Category 2: Broken References (Component → Mother)

**Symptom**: Component Child references Mother sections that don't exist

**Examples**:
- "Component Section 10.2 references 'Mother Part 1 Section 2.3' but Mother Part 1 only has Sections 1-2"
- "Component Section 1.2 references 'Mother Section 2.5: Bucket Definitions' which doesn't exist"

**Detection**:
- Read Component Section 10.2 (Mother Document Foundation references)
- Check if referenced Mother sections actually exist
- Verify all Mother section numbers against actual Mother doc structure

**Fix Location**: Mother HLD (add missing sections)

---

#### Category 3: Outdated Information (Mother)

**Symptom**: Mother doc describes old architecture, old processes, or deprecated features

**Examples**:
- "Mother Section 2.3 says '8 buckets' but Component Section 1.2 says 'only 3 buckets processed'"
- "Mother Part 1 Section 2 shows old directory structure, but Component Section 2.2 uses different paths"
- "Mother describes manual process, but Component Section 2.3 implements automated version"

**Detection**:
- Compare Component Section 2 (Architecture & Design) with Mother Section X.Y
- Check if Component uses different architecture than Mother describes
- Look for Component Decision Log (Appendix A) entries noting "Mother says X but we do Y"
- Compare technical details (bucket counts, directory paths, processing modes)

**Fix Location**: Mother HLD (update outdated content)

---

#### Category 4: Outdated Information (Foundation)

**Symptom**: FoundationCHILD.md doesn't match Mother Part 1 or Component implementations

**Examples**:
- "FoundationCHILD.md Section 2 shows old directory structure, but Component uses different paths"
- "Foundation Glossary defines 'temporal window' differently than Mother Part 1"
- "Foundation Section 3 lists 8 buckets, but Mother Part 1 now says 3 buckets processed"

**Detection**:
- Compare FoundationCHILD.md with Mother Part 1 - do they align?
- Check if Component Child uses different architecture than Foundation describes
- Look for Component Decision Log noting "Foundation says X but we do Y"

**Fix Location**: FoundationCHILD.md (update to match Mother Part 1)

---

#### Category 5: Duplicate Foundation Content (Component)

**Symptom**: Component Child duplicates content that already exists in FoundationCHILD.md

**Examples**:
- "Component Appendix A defines 'temporal window' - but FoundationCHILD.md Glossary already defines it"
- "Component Section 2.2 lists directory structure - should just reference FoundationCHILD.md Section 2"
- "Component Section 6 lists 8 bucket definitions - FoundationCHILD.md Section 3 already has this"

**Detection**:
- Compare Component content with FoundationCHILD.md - is there duplication?
- Check if Component has detailed content that Foundation already covers
- Look for Component sections that could be replaced with "See FoundationCHILD.md Section X"

**Fix Location**: Component Child (remove duplication, add references to Foundation)

---

#### Category 6: Missing Foundation Content

**Symptom**: Information duplicated across 3+ Component Children that should be in FoundationCHILD.md

**Examples**:
- "3 Component Children all define 'temporal window' in their Appendix A - should be in Foundation Glossary"
- "5 Component Children all list 8 bucket definitions - should be in Foundation Section 3"
- "4 Component Children duplicate directory structure - should be in Foundation Section 2"

**Detection**:
- Compare multiple Component Child docs - do they define same terms?
- Check if Components have identical Foundation content that could be shared
- If you notice identical content in 3+ Component Children, it should be in FoundationCHILD.md

**Fix Location**: FoundationCHILD.md (add shared content), then update Components to reference Foundation

---

#### Category 7: Missing Mother Foundation Content

**Symptom**: Information that should be in Mother Part 1 but isn't, forcing duplication in Foundation and Components

**Examples**:
- "FoundationCHILD.md Appendix A defines 'temporal window' but Mother Part 1 has no Glossary"
- "Foundation Section 3 lists 8 bucket definitions - Mother Part 1 should define this"
- "Foundation duplicates directory structure from Component specs - Mother Part 1 should have canonical version"

**Detection**:
- Check if FoundationCHILD.md has extensive content that seems system-defining (not implementation)
- If Foundation content is describing "what the system is" (not "how to implement"), it might belong in Mother Part 1
- Look for Foundation content that feels like it's defining the domain, not implementing it

**Fix Location**: Mother Part 1 (add canonical definition), then update FoundationCHILD.md to reference Mother

---

#### Category 8: Contradictions (Foundation ↔ Mother)

**Symptom**: Foundation and Mother say conflicting things, Component had to choose

**Examples**:
- "Mother Part 1 says 'fail-fast' but Foundation Section 4 says 'skip-on-fail' - Component implements skip-on-fail"
- "Mother Part 1 lists 8 buckets but Foundation Section 3 lists 6 buckets - Component uses 6"
- "Foundation Glossary defines 'bucket' differently than Mother Part 1 Glossary"

**Detection**:
- Compare FoundationCHILD.md with Mother Part 1 - do they contradict?
- Read Component Decision Log (Appendix A) for notes like "Foundation and Mother contradict on X"
- If Component explicitly states "Following Foundation (not Mother)" - indicates contradiction

**Fix Location**: Resolve in Mother Part 1 (source of truth), then update Foundation to align

---

#### Category 9: Contradictions (Mother Internal)

**Symptom**: Different Mother sections say conflicting things, Component had to choose

**Examples**:
- "Mother Section 2.3 says 'sequential processing' but Section 2.5 says 'parallel processing'"
- "Mother Part 1 says 'fail-fast' but Section 3.2 says 'skip-on-fail'"
- "Mother Part 1 lists 8 buckets but Section 2.1 lists 6 buckets"

**Detection**:
- Read Component Decision Log for notes like "Mother contradicts itself on X"
- Compare Mother Part 1 with Mother Part 2 sections - do they align?

**Fix Location**: Mother HLD (resolve internal contradictions)

---

#### Category 10: Incomplete Specifications (Mother)

**Symptom**: Mother provides high-level description but missing critical details

**Examples**:
- "Mother mentions 'bucket definitions' but Component Section 1.2 has to define all 8 buckets"
- "Mother says 'uses config.json' but Component Section 5 shows complete schema"
- "Mother describes Stage 3 but Component Section 3 has detailed I/O contracts Mother lacks"

**Detection**:
- Compare level of detail: Is Component HLD much more detailed than Mother section?
- Check if Component has complete schemas/specs that Mother only mentions vaguely
- If Component Section 5 (schemas) has extensive content Mother doesn't show, Mother is incomplete

**Fix Location**: Mother HLD (add missing details)

---

#### Category 11: Incomplete Specifications (Foundation)

**Symptom**: Foundation provides high-level description but missing critical details

**Examples**:
- "Foundation mentions 'directory structure' but Component has detailed paths"
- "Foundation says 'uses temporal windows' but Component has complete window definitions"
- "Foundation describes config.json but Component Section 5 has complete schema"

**Detection**:
- Compare level of detail: Is Component HLD more detailed than Foundation?
- Check if Component has complete specs that Foundation only mentions vaguely

**Fix Location**: FoundationCHILD.md (add missing details)

---

### Step 2: Analyze Impact Scope

For each identified issue, determine impact level:

---

**Impact Level 1: Single Component**
- Issue affects only this Component Child doc
- Example: Component Section 2.3 has outdated info specific to this component only
- **Action**: Update Component Child only (via Phase 4)
- **Cascade**: None

---

**Impact Level 2: Multiple Components**
- Issue affects several Component Children (but not all)
- Example: 3 Component Children reference Foundation Section 2.3 which is outdated
- **Action**: Update Foundation or Mother section, re-audit affected Components
- **Cascade**: Re-audit specific Component Children

---

**Impact Level 3: Foundation Child**
- Issue affects FoundationCHILD.md, which impacts ALL Component Children
- Example: Foundation Glossary is missing terms, all Components duplicate definitions
- **Action**: Update FoundationCHILD.md, re-audit ALL Component Children
- **Cascade**: Re-audit ALL Component Children

---

**Impact Level 4: Mother Part 1**
- Issue affects Mother HLD Part 1 (Foundation), which impacts Foundation + ALL Components
- Example: Mother Part 1 directory structure is outdated
- **Action**: Update Mother Part 1, re-sync FoundationCHILD.md, re-audit ALL Component Children
- **Cascade**: Update Mother → Update Foundation → Re-audit ALL Components

---

### Step 3: Determine Fix Location

**Decision Tree**:

```
Issue identified in Component Child
    ↓
Question: Is this content duplicated in 3+ Component Children?
    YES → Check Foundation
        ↓
        Question: Does Foundation have this content?
            NO → Add to FoundationCHILD.md (Level 3)
                  Create: FoundationSync_{ComponentName}.md
            YES → Question: Is Foundation content outdated/wrong?
                YES → Update FoundationCHILD.md (Level 3)
                      Create: FoundationSync_{ComponentName}.md
                NO → Update Components to reference Foundation (Level 2)
                     No sync file needed - use Phase 4
    ↓
    NO → Check if issue is with Foundation or Mother references
        ↓
        Question: Does Foundation or Mother have this content?
            Foundation has it → Question: Is Foundation content outdated/wrong?
                YES → Update FoundationCHILD.md (Level 3)
                      Create: FoundationSync_{ComponentName}.md
                NO → Component has wrong reference (Level 1)
                     No sync file needed - use Phase 4
            ↓
            Mother has it → Question: Is Mother content outdated/wrong?
                YES → Update Mother HLD (Level 4)
                      Create: MotherSync_{ComponentName}.md
                NO → Component has wrong reference (Level 1)
                     No sync file needed - use Phase 4
            ↓
            Neither has it → Question: Is this system-defining content?
                YES → Add to Mother Part 1 (Level 4)
                      Create: MotherSync_{ComponentName}.md
                NO → Add to FoundationCHILD.md (Level 3)
                     Create: FoundationSync_{ComponentName}.md
```

---

### Step 4: Create Sync Proposal

**Based on Decision Tree outcome:**

---

#### Option A: Foundation Child Update (Level 3)

Create file: `FoundationSync_{ComponentName}.md`

```markdown
# Foundation Document Sync: Proposed Changes from {ComponentName} Work

> **Trigger**: Component Child HLD work revealed Foundation doc issues
> **Component**: {ComponentName}
> **Phase Outputs Reviewed**:
>   - Critique_{ComponentName}.md (Phase 1)
>   - QA_{ComponentName}.md (Phase 2)
>   - ChildHLD_{ComponentName}.md (Phase 3)
> **Date**: {current_date}
> **Status**: PENDING APPROVAL

## Summary

**Total Changes Proposed**: {count}
**Impact Scope**: Level 3 (Foundation Child - affects ALL Component Children)

**Affected Docs**:
- FoundationCHILD.md (direct changes)
- ALL Component Children (require re-audit after Foundation update)

## Proposed Changes

### Change 1: [{Category}] {Foundation Section to Update}

**Issue Type**: {Broken Reference | Outdated Info | Missing Content | Contradiction | Incomplete Spec | Duplicate Content}

**Current State**:
- **Foundation Section**: {Section number and title}
- **Current Text**:
  ```
  {quote exact current text from Foundation, or "MISSING" if section doesn't exist}
  ```

**Problem Discovered**:
- **By Comparing**: Component {ComponentName} Section X vs FoundationCHILD.md Section Y
- **Evidence**: {Quote from Component doc showing the discrepancy}
  - Example: "Component Section 10.2 references 'Foundation Section 2.3' but Foundation ends at Section 2.2"
  - Example: "3 Component Children all define 'temporal window' - should be in Foundation Glossary"

**Proposed Update**:
```markdown
{new or revised Foundation content - show exact text to add/replace}
```

**Rationale**: {Why this change is needed - what problem it solves}

**Impact**: ALL Component Children must be re-audited to:
- Remove duplicate content now in Foundation
- Update broken references to new Foundation sections
- Verify they correctly reference updated Foundation content

**Priority**: [CRITICAL | HIGH | LOW]
- CRITICAL: Broken references that block Component work
- HIGH: Missing content duplicated across 3+ Components
- LOW: Minor clarifications, nice-to-have additions

---

[Continue for all proposed Foundation changes]

## Change Summary by Priority

### [CRITICAL] Changes (must apply)
1. Change X: {Brief description}

### [HIGH] Changes (should apply)
1. Change A: {Brief description}

### [LOW] Changes (optional)
1. Change M: {Brief description}

## Recommended Action

**Option A: Apply All Changes**
- Update FoundationCHILD.md with all {count} proposed changes
- Re-audit ALL {count} Component Children
- Estimated effort: {time estimate}

**Option B: Apply [CRITICAL] + [HIGH] Only**
- Update FoundationCHILD.md with {count} critical/high priority changes
- Skip [LOW] priority changes for now
- Re-audit ALL {count} Component Children
- Estimated effort: {time estimate}

**Option C: Apply [CRITICAL] Only**
- Fix broken references only
- Defer [HIGH] and [LOW] changes
- Re-audit ALL {count} Component Children
- Estimated effort: {time estimate}

**Option D: Reject Changes**
- Keep FoundationCHILD.md as-is
- Component Children work around Foundation limitations
- No re-audit needed

## User Decision

**Selected Option**: [A | B | C | D]

**Changes to Apply**: [List change numbers approved]

**Status**: [PENDING | APPROVED | REJECTED]

---

## Cascade: Component Children Requiring Re-Audit

**Due to FoundationCHILD.md updates:**

- [ ] {ComponentChild1}.md - Impact: {what changed in Foundation}
- [ ] {ComponentChild2}.md - Impact: {what changed in Foundation}
- [ ] {ComponentChild3}.md - Impact: {what changed in Foundation}
... (list ALL Component Children)

**Re-audit approach**:
1. Run Phase 1B on each Component Child
2. Check if broken Foundation references are now fixed
3. Check if duplicate content can be removed (now in Foundation)
4. Check if outdated Foundation info is now current
```

---

#### Option B: Mother HLD Update (Level 4)

Create file: `MotherSync_{ComponentName}.md`

(Use original Phase 5 format from current file, lines 135-266)

**Additional Section for Three-Tier**:

```markdown
## Three-Tier Cascade Plan

**Since Mother Part 1 is being updated:**

1. **Update Mother HLD Part 1** (this sync)
2. **Re-sync FoundationCHILD.md** (Foundation must reflect Mother Part 1)
   - Create follow-up: `FoundationSync_FromMother_{date}.md`
   - Update Foundation sections that reference changed Mother Part 1 content
3. **Re-audit ALL Component Children** (after Foundation is re-synced)

**Estimated Total Effort**: {time for Mother + Foundation + ALL Components}
```

---

### Step 5: User Review and Approval

**Present FoundationSync_{ComponentName}.md OR MotherSync_{ComponentName}.md to user with:**

1. **Summary**: X changes proposed, Y docs affected
2. **Priority breakdown**: CRITICAL, HIGH, LOW counts
3. **Recommended action**: Usually Option B (CRITICAL + HIGH)
4. **Estimated effort**: How long updates and re-audits will take
5. **Cascade complexity**: If Mother Part 1 changes, explain three-tier cascade

**User reviews and decides**:
- Which changes to approve (all, some, or none)
- When to apply changes (now or later)
- Whether to batch with other pending sync files

---

### Step 6: Apply Approved Changes

#### If updating FoundationCHILD.md:

1. **Update FoundationCHILD.md**:
   - Apply each approved change to corresponding Foundation section
   - Respect Foundation doc architecture (don't reorganize structure)
   - If adding new Foundation section, follow existing numbering
   - Ensure Foundation aligns with Mother Part 1 (no contradictions)

2. **Update Foundation metadata**:
   ```markdown
   **Version**: {increment version}
   **Last Modified**: {current_date}

   ## Change Log
   | Version | Date | Author | Changes |
   |---------|------|--------|---------|
   | 1.1 | {date} | {name} | Updated from {ComponentName} feedback: {brief summary} |
   ```

3. **Update FoundationSync_{ComponentName}.md**:
   - Change Status: PENDING → APPLIED
   - Add "Applied Date" field
   - Note which changes were applied vs deferred

4. **Document cascade requirements**:
   - Create list of ALL Component Children to re-audit

---

#### If updating Mother HLD:

1. **Update Mother HLD** (Part 1 or other sections):
   - Apply each approved change to corresponding Mother section
   - Respect Mother doc architecture
   - If adding new Mother Part 1 section, follow existing numbering
   - Update Mother metadata (version, date, change log)

2. **Update MotherSync_{ComponentName}.md**:
   - Change Status: PENDING → APPLIED
   - Add "Applied Date" field

3. **Create follow-up Foundation sync** (if Mother Part 1 changed):
   - Create `FoundationSync_FromMother_{date}.md`
   - List Foundation sections that need updating to align with new Mother Part 1 content

4. **Document cascade requirements**:
   - Update FoundationCHILD.md (if Mother Part 1 changed)
   - Re-audit ALL Component Children (after Foundation updated)

---

### Step 7: Cascade Analysis (Three-Tier)

**For Level 1 changes (Single Component)**:
- Only update that Component Child (via Phase 4)
- No Foundation or Mother changes needed
- Example: Component Section 2.3 typo fix

**For Level 2 changes (Multiple Components)**:
- Update specific Component Children (via Phase 4)
- Might trigger Foundation update if pattern emerges
- Example: 3 Components reference wrong Foundation section number

**For Level 3 changes (Foundation Child)**:
```markdown
## Cascade Path: Foundation Update

1. **Update FoundationCHILD.md** ✓
   - Applied changes from FoundationSync_{ComponentName}.md

2. **Re-audit ALL Component Children**:
   - [ ] VideoDiscoveryCHILD.md - Impact: {change details}
   - [ ] VideoProcessingCHILD.md - Impact: {change details}
   - [ ] FeatureAggregationCHILD.md - Impact: {change details}
   - [ ] ReviewCSVGenerationCHILD.md - Impact: {change details}
   - [ ] FeatureTransformationCHILD.md - Impact: {change details}
   - [ ] MLTrainingCHILD.md - Impact: {change details}
   ... (list ALL Component Children)

**Re-audit approach**:
1. Run Phase 1B on each Component Child
2. Check if broken Foundation references are now fixed
3. Remove duplicate content (now in Foundation)
4. Update references to new Foundation sections
```

**For Level 4 changes (Mother Part 1)**:
```markdown
## Cascade Path: Mother Part 1 Update

1. **Update Mother HLD Part 1** ✓
   - Applied changes from MotherSync_{ComponentName}.md

2. **Re-sync FoundationCHILD.md** (NEXT STEP)
   - Create: FoundationSync_FromMother_{date}.md
   - Align Foundation with updated Mother Part 1
   - Example changes:
     - Update Foundation Section 2 (directory structure) to match new Mother Part 1 Section 2
     - Update Foundation Glossary to align with new Mother Part 1 Glossary

3. **Re-audit ALL Component Children** (AFTER Foundation synced)
   - Same checklist as Level 3
   - Verify Components reference correct Foundation sections
   - Verify Components reference correct Mother Part 1 sections

**Total Cascade**:
- Mother Part 1 → FoundationCHILD.md → ALL Component Children
- Estimated effort: {time for Foundation sync + time for ALL Component re-audits}
```

---

## Output File Formats

### File 1: FoundationSync_{ComponentName}.md

(See Option A template in Step 4)

### File 2: MotherSync_{ComponentName}.md

(See Option B template in Step 4 - uses original Phase 5 format)

---

## Key Rules

### Issue Identification Rules
1. **Evidence-based** - Every proposed change must cite specific Phase output or Component Child content
2. **Impact-aware** - Classify as Level 1/2/3/4 to understand cascade scope
3. **Prioritized** - Mark CRITICAL (broken refs, contradictions) vs HIGH vs LOW
4. **Specific** - Quote exact current text and exact proposed replacement
5. **Three-tier aware** - Check Component → Foundation → Mother before proposing changes

### Proposal Rules
1. **Show current state** - Quote what Foundation or Mother currently says (or "MISSING")
2. **Show proposed state** - Provide exact new content
3. **Explain rationale** - Why this change solves the problem
4. **List affected docs** - Which docs need re-audit after change (be specific about Foundation vs Components)

### Fix Location Rules
1. **Component-only issue?** - Use Phase 4, no sync file needed
2. **Duplicated across 3+ Components?** - Add to Foundation (Level 3)
3. **Foundation outdated?** - Update Foundation (Level 3)
4. **Mother Part 1 outdated?** - Update Mother (Level 4), then cascade to Foundation
5. **Foundation contradicts Mother?** - Fix Mother (source of truth), then update Foundation

### Three-Tier Update Rules
1. **Respect hierarchy** - Mother Part 1 is source of truth, Foundation reflects Mother, Components reflect Foundation
2. **Cascade downward** - Mother changes cascade to Foundation, Foundation changes cascade to Components
3. **Never contradict upward** - Foundation cannot contradict Mother Part 1, Components cannot contradict Foundation
4. **Document cascades** - Always list full cascade path for Level 3/4 changes

### Anti-Hallucination Rules
1. **Don't invent problems** - Only propose changes backed by actual Component Child content
2. **Don't assume impact** - Check which docs actually reference affected sections
3. **Don't over-correct** - If doc is vague but not wrong, consider if change is necessary
4. **Verify before proposing** - Read actual Foundation/Mother section before claiming it's wrong
5. **Don't mix levels** - Keep Foundation fixes separate from Mother fixes

---

## Completion Criteria

Before marking Status: APPLIED:

**For Foundation Updates:**
- [ ] All approved changes applied to FoundationCHILD.md
- [ ] Foundation metadata updated (version, date, change log)
- [ ] FoundationSync file updated (Status: APPLIED, Applied Date added)
- [ ] Cascade analysis complete (list of ALL Component Children to re-audit created)

**For Mother Updates:**
- [ ] All approved changes applied to Mother HLD
- [ ] Mother metadata updated (version, date, change log)
- [ ] MotherSync file updated (Status: APPLIED, Applied Date added)
- [ ] If Mother Part 1 changed: FoundationSync_FromMother_{date}.md created
- [ ] Cascade analysis complete (Foundation + ALL Component Children)

---

## When to Use Phase 5

**Invoke Phase 5 when:**
1. Phase 2 Q&A revealed Foundation or Mother doc gaps (many [CRITICAL] questions on basic info)
2. Phase 1B audit found same broken reference in multiple Component Children
3. Phase 3 generated duplicate content across 3+ Component Children (should be in Foundation)
4. Component HLD contradicts Foundation or Mother and user confirms Component is correct
5. Multiple sync issues accumulating - time to batch update Foundation or Mother

**Skip Phase 5 when:**
1. Component HLD work didn't reveal any Foundation or Mother issues
2. Issues are minor and affect only 1 Component (Level 1, LOW priority) - use Phase 4 instead
3. Foundation/Mother is intentionally high-level (not a bug, just different abstraction level)

---

## Next Phase

**After Phase 5 completes:**

**For Foundation Updates (Level 3)**:
1. **Update FoundationCHILD.md**: Apply approved changes
2. **Re-audit ALL Component Children**: Run Phase 1B on all Components
3. **Update Component Children**: Use Phase 4 to fix Components based on new Foundation
4. **Proceed to TI generation**: Once Foundation and Components are synchronized

**For Mother Updates (Level 4)**:
1. **Update Mother HLD**: Apply approved changes to Mother Part 1
2. **Create Foundation sync**: Generate FoundationSync_FromMother_{date}.md
3. **Update FoundationCHILD.md**: Apply Foundation sync to align with Mother Part 1
4. **Re-audit ALL Component Children**: Run Phase 1B on all Components (after Foundation synced)
5. **Update Component Children**: Use Phase 4 to fix Components
6. **Proceed to TI generation**: Once all three tiers are synchronized

---

## Examples

### Example 1: Missing Foundation Content (Level 3)

**Trigger**: ReviewCSVGenerationCHILD.md has Glossary (Appendix A) with term "temporal window"

**Analysis**:
- Check other Component Children → VideoProcessingCHILD.md and FeatureAggregationCHILD.md also define "temporal window"
- Check FoundationCHILD.md → No Glossary section exists
- Decision: Add Glossary to FoundationCHILD.md

**Output**: `FoundationSync_ReviewCSVGeneration.md`

**Cascade**: Update Foundation → Re-audit ALL 3 Component Children to remove duplicate definitions

---

### Example 2: Outdated Mother Part 1 (Level 4)

**Trigger**: FeatureAggregationCHILD.md Section 2.2 shows different directory structure than Mother Part 1 Section 2

**Analysis**:
- Check Mother Part 1 Section 2 → Shows old directory structure
- Check FoundationCHILD.md Section 2 → References Mother Part 1 (also outdated)
- Check other Components → All use new directory structure from FeatureAggregation
- Decision: Update Mother Part 1 Section 2

**Output**: `MotherSync_FeatureAggregation.md`

**Cascade**:
1. Update Mother Part 1 Section 2
2. Create `FoundationSync_FromMother_{date}.md` to update Foundation Section 2
3. Update FoundationCHILD.md Section 2
4. Re-audit ALL Component Children to verify references

---

### Example 3: Component-Only Issue (Level 1)

**Trigger**: ReviewCSVGenerationCHILD.md Section 10.2 references "FoundationCHILD.md Section 2.3" but Foundation has Section 2.3

**Analysis**:
- Check FoundationCHILD.md → Section 2.3 exists and is correct
- Check ReviewCSVGeneration content → Reference is correct
- Issue: False alarm, no sync needed

**Output**: None

**Action**: Inform user that docs are already synchronized

---

**Version**: 2.0 (Three-Tier Architecture)
**Last Updated**: 2025-01-29
**Applies To**: All projects using three-tier documentation system (Mother ← Foundation ← Component Children)
**Related**: Phase3_ChildHLDGeneration.md, Phase4_ReviewRefinement.md, Phase1B_ExistingChildCritique.md
