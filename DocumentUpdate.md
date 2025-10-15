# Documentation Update Process Template

> **Purpose**: Standardized workflow for updating official repository documentation after implementing new features
>
> **Version**: 1.0
>
> **Created**: 2025-10-13
>
> **Use Case**: Any major feature implementation that requires documentation updates

# Brainstorm
## Sequential Update A (Simple)
**This is for traditional PlanetHLD(Child) creation > Star(Mother) Document & StarFoundation alignment**
We have existing documentation for this:
\\wsl$\Ubuntu\home\jorge\rumiaifinal\documentation_migration\FutureDevelopments\Phase5_MotherDocSync
  .md

## Sequential Update B (Complex)
**This is for Loose Product Improvements/Implementations**
Loose means that the Planet TI, PlanetHLD, Star and StarFoundation documentation already exist. 
After their creation, a feature updated was identified that after implementation meant reupdating our documentation.

The prompt should be structured to carry out the Documentation Update one document at a time, 
- Step 1: Check ChildTI Doc with NewTI Doc
- Step 2: Ensure NewTI Doc and ChildHLD Doc match
    - If the process is being implemented for Loose_HLD implementation
        - NewHLD (HLD of NewTI) must match NewTI
            - NewHLD must match ChildHLD

---

## When to Use This Process

Use this process when:
1. ✅ A new feature has been implemented in code
2. ✅ Official documentation needs updating to reflect the changes
3. ✅ Multiple documents need coordinated updates
4. ✅ Conflicting/outdated content needs removal

**Do NOT use this process for**:
- Minor bug fixes with no architectural changes
- Internal code comments or docstrings
- Temporary/experimental features

---

## Overview of Process

### Phase 1: Complete Reading (MANDATORY)

**Goal**: Fully understand current documentation state without assumptions

**Steps**:
1. Identify all target documents
2. Count total lines per document (`wc -l file`)
3. Calculate reading strategy for files >2000 lines
4. Read EVERY section systematically (use offset/limit for large files)
5. DO NOT skip sections due to file size
6. DO NOT assume content based on section titles
7. Verify completion by tracking last line read

**Critical Rule**: ❌ **NEVER hallucinate content**. If a document exceeds ~25,000 tokens (approximately >2,000 lines), it MUST be read in chunks using offset/limit parameters.

**Why This Matters**: Large files (>25k tokens) cannot be read in a single Read call due to token limits. Attempting to read them at once will result in truncation, causing you to miss critical sections and hallucinate about their contents.

**Section-by-Section Reading Strategy**:

```bash
# Step 1: Count lines
wc -l VideoDiscoveryCHILDTI.md
# Output: 3874 lines

# Step 2: Calculate chunk size (use 500-line chunks for safety)
# 3874 lines ÷ 500 = ~8 chunks needed

# Step 3: Read systematically with offset/limit
Read(file, offset=1, limit=500)      # Lines 1-500
Read(file, offset=501, limit=500)    # Lines 501-1000
Read(file, offset=1001, limit=500)   # Lines 1001-1500
Read(file, offset=1501, limit=500)   # Lines 1501-2000
Read(file, offset=2001, limit=500)   # Lines 2001-2500
Read(file, offset=2501, limit=500)   # Lines 2501-3000
Read(file, offset=3001, limit=500)   # Lines 3001-3500
Read(file, offset=3501, limit=500)   # Lines 3501-3874

# Step 4: Verify completion
# Last read covered line 3874 ✓
```

**Efficient Chunk Sizing**:
- Files <2000 lines: Read in one call
- Files 2000-4000 lines: Use 500-line chunks (4-8 reads)
- Files >4000 lines: Use 500-1000 line chunks (adjust based on content density)

**Tracking Progress**:
```
Document: VideoDiscoveryCHILDTI.md (3,874 lines)
[✓] Lines 1-500 (APIFY constants section)
[✓] Lines 501-1000 (Logging specs)
[✓] Lines 1001-1500 (Dependencies section)
...
[✓] Lines 3501-3874 (Traceability matrix)
Status: COMPLETE - All 3,874 lines read
```

**Why This Matters**: Partial reading leads to:
- Missed conflicting statements
- Duplicate content
- Incorrect assumptions about current state

---

### Phase 2: Identify Changes (Additions + Removals)

**Goal**: Create complete change proposal including both additions AND removals

**Steps**:
1. Analyze what needs to be ADDED (new features, capabilities)
2. Analyze what needs to be REMOVED (conflicting statements, outdated info)
3. Analyze what needs to be MODIFIED (updated values, corrected descriptions)

**Critical Question**: "Did we remove any functionality or specific implementation that should be removed for clarity?"

**Example from Hashtag Cluster Implementation**:

| Type | Change | Reason |
|------|--------|--------|
| **ADD** | Cluster Mode section | New feature documentation |
| **REMOVE** | Two-actor table | We now use single unified actor |
| **REMOVE** | "TBD" actor warnings | No longer needed |
| **MODIFY** | Actor configuration constants | Update to unified APIFY_ACTOR_ID |

**Common Pitfalls**:
- ❌ Only documenting additions (forgetting to remove contradictions)
- ❌ Leaving outdated "TODO" or "TBD" references
- ❌ Keeping deprecated workflow descriptions

---

### Phase 3: Create Comprehensive Proposal

**Goal**: Present user with complete picture (additions + modifications + removals)

**Structure**:

```markdown
## DOCUMENT X: [Name]

### Changes Summary
- Additions: X lines
- Modifications: Y sections
- Removals: Z lines
- Net change: +N lines

### Change 1.1: [Title]
**Type**: Addition
**Location**: After line 65
**Content**: [exact content to add]
**Reason**: [why this is needed]

### Change 1.2: [Title]
**Type**: Removal
**Location**: Lines 1003-1013
**Current Content**: [show what will be removed]
**Reason**: [why removal is necessary]

### Change 1.3: [Title]
**Type**: Modification
**Location**: Line 298
**Current**: [old content]
**New**: [new content]
**Reason**: [why modification is needed]
```

**Validation Checklist Before Presenting**:
- [ ] All target documents read completely
- [ ] Both additions and removals identified
- [ ] Conflicting statements addressed
- [ ] Deprecated references removed
- [ ] Updated values reflect current implementation
- [ ] No "TBD" or placeholder content remains

---

### Phase 4: Execute Changes

**Goal**: Apply all approved changes systematically

**Steps**:
1. Start with removals/modifications (clean up first)
2. Then apply additions (add new content)
3. Update one document at a time
4. Use Edit tool for modifications
5. Verify each change after execution

**Best Practices**:
- ✅ Use exact strings from Read output (avoid typos)
- ✅ Update TodoWrite tool to track progress
- ✅ Test critical changes (grep for references, verify consistency)
- ✅ Mark each document complete before moving to next

**Error Recovery**:
- If Edit fails (string not found), re-read the section to get exact formatting
- If uncertain about impact, read surrounding context
- If change affects multiple files, update all consistently

---

### Phase 5: Create Process Documentation

**Goal**: Document the methodology for future use

**Steps**:
1. Create DocumentUpdate.md template (this file)
2. Include real examples from current implementation
3. Document lessons learned
4. Provide reusable structure

---

## Real Example: Hashtag Cluster Implementation (2025-10-13)

### Context
- **Feature**: Multi-hashtag cluster scraping with provenance tracking
- **Target Documents**: VideoDiscoveryCHILD.md, VideoDiscoveryCHILDTI.md, MLPlanningv2.md
- **Key Insight**: We changed from two-actor architecture to single unified actor

### Phase 1: Complete Reading
- VideoDiscoveryCHILD.md: Read all 1,967 lines
- VideoDiscoveryCHILDTI.md: Read all 3,874 lines (in chunks of 500 lines)
- MLPlanningv2.md: Read all 2,815 lines (in chunks of 500 lines)

**Time Invested**: ~15 minutes of systematic reading
**Result**: Discovered two-actor references that needed removal

### Phase 2: Change Analysis

**Additions**:
- Cluster Mode overview section (VideoDiscoveryCHILD.md)
- Cluster scraping functions (VideoDiscoveryCHILDTI.md)
- Implementation status tracker (MLPlanningv2.md)

**Removals** (Critical - would have been missed without complete reading):
- Two-actor table (VideoDiscoveryCHILD.md line 154)
- "TBD" actor configuration (VideoDiscoveryCHILD.md lines 1006-1013)
- Separate hashtag scraper constants (VideoDiscoveryCHILDTI.md lines 2897-2912)
- Dual actor service documentation (VideoDiscoveryCHILDTI.md lines 3526-3538)

**Modifications**:
- Actor table: 3 separate rows → unified row
- Actor constants: APIFY_PROFILE_SCRAPER_ID + APIFY_HASHTAG_SCRAPER_ID → APIFY_ACTOR_ID
- Validation date: 2025-01-28 → 2025-10-13

### Phase 3: Proposal
Total changes across 3 documents:
- **Additions**: ~480 lines
- **Removals**: ~115 lines
- **Net**: +365 lines

### Phase 4: Execution
Order of updates:
1. VideoDiscoveryCHILD.md: 3 changes (cluster section, scraper table, actor config)
2. VideoDiscoveryCHILDTI.md: 2 changes (actor constants, service documentation)
3. MLPlanningv2.md: 3 changes (status tracker, scraper table, date filtering note)

**Execution Time**: ~20 minutes
**Result**: All documents updated successfully

### Lessons Learned
1. ✅ **Complete reading is non-negotiable** - We discovered two-actor references only through systematic reading
2. ✅ **Ask "what should be removed?"** - User's question caught missing removals
3. ✅ **Track both actor changes** - Unified actor affected more sections than initially thought
4. ✅ **Update validation dates** - Small details matter for accuracy

---

## Process Metrics

### Effort Breakdown
| Phase | Time | Critical? |
|-------|------|-----------|
| Complete Reading | 30% | ✅ YES |
| Change Analysis | 20% | ✅ YES |
| Proposal Creation | 20% | ✅ YES |
| Execution | 25% | ✅ YES |
| Documentation | 5% | Optional |

### Quality Indicators
- ✅ Zero hallucinated content (complete reading)
- ✅ Both additions and removals addressed
- ✅ User question prompted critical review
- ✅ Process documented for reuse

---

## Template Checklist

When using this process for future implementations:

### Pre-Work
- [ ] Feature is fully implemented and tested
- [ ] All target documents identified
- [ ] Decision made to proceed with updates

### Phase 1: Reading
- [ ] Document sizes counted (line numbers)
- [ ] Every document read completely (no skipping)
- [ ] Reading strategy documented (offset/limit for large files)
- [ ] No assumptions made about content

### Phase 2: Analysis
- [ ] Additions identified
- [ ] Modifications identified
- [ ] **Removals identified** (ASK: "what should be removed?")
- [ ] Conflicting statements found
- [ ] Deprecated content marked

### Phase 3: Proposal
- [ ] Summary statistics calculated (additions, removals, net)
- [ ] Change list created (type, location, content, reason)
- [ ] Validation checklist completed
- [ ] User approval obtained

### Phase 4: Execution
- [ ] TodoWrite tool tracking progress
- [ ] Changes applied systematically
- [ ] Each document verified after updates
- [ ] No errors or inconsistencies

### Phase 5: Documentation
- [ ] Process template updated (if needed)
- [ ] Lessons learned captured
- [ ] Example added to template
- [ ] Metrics documented

---

## Anti-Patterns to Avoid

### ❌ Partial Reading
**Problem**: "This document is too long, I'll just read the sections I think need updating"
**Result**: Missed conflicting statements, incomplete updates
**Solution**: Use offset/limit to read entire document systematically

### ❌ Addition-Only Mindset
**Problem**: "I'll just add the new sections, no need to remove anything"
**Result**: Contradictory documentation, confusion about current state
**Solution**: Always ask "what should be removed?" before executing

### ❌ Assuming Current State
**Problem**: "Based on the section title, I know what's in there"
**Result**: Incorrect assumptions, mismatched updates
**Solution**: Read actual content, don't trust section titles alone

### ❌ Batch Updates Without Review
**Problem**: "I'll update all three documents at once and show the user"
**Result**: User can't review changes, no chance for feedback
**Solution**: Present comprehensive proposal first, execute after approval

---

## Success Criteria

A documentation update is successful when:
1. ✅ All new features documented accurately
2. ✅ All conflicting/outdated content removed
3. ✅ No "TBD" or placeholder content remains
4. ✅ Updated values reflect current implementation
5. ✅ User can understand changes made
6. ✅ Process documented for future use

---

## Appendix: Common Documentation Update Scenarios

### Scenario 1: New Feature Implementation
- **Read**: All affected documents completely
- **Add**: Feature description, examples, configuration
- **Remove**: "Coming soon" or "planned" markers
- **Modify**: Status trackers, feature lists

### Scenario 2: Architecture Change
- **Read**: All architectural documents
- **Add**: New architecture descriptions
- **Remove**: Old architecture references, outdated diagrams
- **Modify**: Design decisions, trade-off analyses

### Scenario 3: Dependency Update
- **Read**: All dependency references
- **Add**: New dependency documentation
- **Remove**: Deprecated dependencies
- **Modify**: Version numbers, configuration examples

### Scenario 4: API/Interface Change
- **Read**: All API documentation
- **Add**: New parameters, return values
- **Remove**: Deprecated parameters
- **Modify**: Function signatures, examples

---

## Conclusion

This process ensures:
1. **Completeness**: Nothing is missed through systematic reading
2. **Accuracy**: No conflicting information remains
3. **Clarity**: Deprecated content is removed
4. **Reusability**: Template can be applied to future updates

**Key Takeaway**: Invest time in complete reading and removal identification upfront to avoid confusion and errors downstream.

---

**Process Owner**: Development Team
**Last Updated**: 2025-10-13
**Next Review**: After next major feature implementation
