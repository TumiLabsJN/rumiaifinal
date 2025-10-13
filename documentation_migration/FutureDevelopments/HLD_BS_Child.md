# Child Document Creation Workflow - High-Level Design

> **Purpose**: Blueprint for creating Child HLD documents from Mother HLD using 4-phase iterative process
> **Audience**: LLMs and human developers using this documentation system
> **Related**: DevSystem.md, Phase1-4 instruction files, ChildTemplate.md

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CHILD DOCUMENT CREATION                       │
│                         (4-Phase Workflow)                           │
└─────────────────────────────────────────────────────────────────────┘

  INPUTS (User Provides)
  ┌──────────────────────────────────────────────────────────────────┐
  │ - MotherHLD.md (contains Part 1: Foundation + Parts 2-N)         │
  │ - Section X.Y (component to develop)                             │
  └──────────────────────────────────────────────────────────────────┘
                                    ↓
  ╔══════════════════════════════════════════════════════════════════╗
  ║                    PHASE 1: BUSINESS CRITIQUE                    ║
  ║                  (Aggressive Challenge & Validation)             ║
  ╚══════════════════════════════════════════════════════════════════╝
                                    ↓
  ┌──────────────────────────────────────────────────────────────────┐
  │ OUTPUT: Critique_ComponentName.md (5-10 Q&A + Final Decision)    │
  │ Status: COMPLETE ✓                                               │
  │ Proceed to Phase 2: [YES | NO]                                   │
  └──────────────────────────────────────────────────────────────────┘
                                    ↓
  ╔══════════════════════════════════════════════════════════════════╗
  ║                   PHASE 2: CLARIFICATION Q&A                     ║
  ║                   (Fill Knowledge Gaps for HLD)                  ║
  ╚══════════════════════════════════════════════════════════════════╝
                                    ↓
  ┌──────────────────────────────────────────────────────────────────┐
  │ OUTPUT: QA_ComponentName.md (Categorized Q&A + Completeness)     │
  │ Status: COMPLETE ✓                                               │
  │ Proceed to Phase 3: [YES | NO]                                   │
  └──────────────────────────────────────────────────────────────────┘
                                    ↓
  ╔══════════════════════════════════════════════════════════════════╗
  ║                  PHASE 3: CHILD HLD GENERATION                   ║
  ║              (Perfect Draft - 10 Sections + 2 Appendices)        ║
  ╚══════════════════════════════════════════════════════════════════╝
                                    ↓
  ┌──────────────────────────────────────────────────────────────────┐
  │ OUTPUT: ChildHLD_ComponentName.md (Complete, No TODOs)           │
  │ Status: APPROVED ✓                                               │
  └──────────────────────────────────────────────────────────────────┘
                                    ↓
  ╔══════════════════════════════════════════════════════════════════╗
  ║              PHASE 4: REVIEW & REFINEMENT (Optional)             ║
  ║                    (Surgical Updates Only)                       ║
  ╚══════════════════════════════════════════════════════════════════╝
                                    ↓
  ┌──────────────────────────────────────────────────────────────────┐
  │ FINAL OUTPUT: ChildHLD_ComponentName.md (FINALIZED)              │
  │ Ready for TI Generation                                          │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Business Critique - Detailed Flow

```
╔══════════════════════════════════════════════════════════════════════╗
║                         PHASE 1: BUSINESS CRITIQUE                   ║
║  Goal: Challenge design decisions before development starts          ║
║  Output: Critique_ComponentName.md with Q&A                          ║
╚══════════════════════════════════════════════════════════════════════╝

USER INVOCATION
┌────────────────────────────────────────────────────────────────────┐
│ Prompt: "Follow Phase1_BusinessCritique.md instructions            │
│          for MotherHLD.md Section X.Y"                             │
│                                                                     │
│ Documents Attached:                                                 │
│  - Phase1_BusinessCritique.md (instructions)                       │
│  - MotherHLD.md (reads Part 1 + Section X.Y)                       │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ LLM READS INSTRUCTIONS                                              │
│  1. Extract component info from Section X.Y                        │
│  2. Extract Foundation context from Part 1                         │
│  3. Analyze on 6 dimensions: Necessity, Business Value, Risk,      │
│     Alternatives, Architectural Fit, Dependencies                  │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ LLM CREATES OUTPUT FILE                                             │
│                                                                     │
│ File: Critique_ComponentName.md                                    │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ # Business Critique: ComponentName                             │ │
│ │                                                                 │ │
│ │ ## Component Summary                                           │ │
│ │ - Name: ComponentName                                          │ │
│ │ - Purpose: [1-line from Section X.Y]                           │ │
│ │                                                                 │ │
│ │ ## Critical Analysis                                           │ │
│ │ ### Overall Assessment: NEEDS REFINEMENT                       │ │
│ │                                                                 │ │
│ │ ### Critical Concerns                                          │ │
│ │ 1. [Concern 1]                                                 │ │
│ │ 2. [Concern 2]                                                 │ │
│ │ 3. [Concern 3]                                                 │ │
│ │                                                                 │ │
│ │ ## Validation Questions & Answers                              │ │
│ │ [Empty - will fill iteratively]                                │ │
│ │                                                                 │ │
│ │ Status: IN PROGRESS                                            │ │
│ └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ ITERATIVE Q&A PROTOCOL (ONE QUESTION AT A TIME)                    │
└────────────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │ LLM: "Q1: You assume FEAT timeouts are the main failure │
    │      cause. What data validates this assumption?"       │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │ USER: "We see 78% of retries are FEAT-related based on  │
    │       Stage 0 logs from last 30 days."                  │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │ LLM IMMEDIATELY UPDATES FILE                            │
    │ ┌─────────────────────────────────────────────────────┐ │
    │ │ ### Q1: [question]                                  │ │
    │ │ **Answer**: We see 78% of retries are FEAT-related │ │
    │ │ **LLM Analysis**: Data-backed assumption, valid     │ │
    │ └─────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │ LLM: "Q2: What's the cost of a false positive timeout  │
    │      detection vs false negative?"                      │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │ USER: [answers Q2]                                      │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │ LLM IMMEDIATELY UPDATES FILE (Q2 added)                 │
    └─────────────────────────────────────────────────────────┘
                              ↓
           [Repeat for Q3, Q4, Q5... up to 5-10 questions]
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ ALL Q&A COMPLETE → LLM MAKES FINAL DECISION                        │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ FINAL OUTPUT: Critique_ComponentName.md (COMPLETE)                 │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ ## Validation Questions & Answers                              │ │
│ │ ### Q1: [question]                                             │ │
│ │ **Answer**: [answer]                                           │ │
│ │ **LLM Analysis**: [interpretation]                             │ │
│ │ [... Q2-Q10 ...]                                               │ │
│ │                                                                 │ │
│ │ ## Final Decision                                              │ │
│ │ **Overall Assessment**: APPROVE                                │ │
│ │ **Reasoning**: Based on Q&A, component is justified            │ │
│ │ **Proceed to Phase 2**: YES ✓                                  │ │
│ │                                                                 │ │
│ │ Status: COMPLETE ✓                                             │ │
│ └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ USER APPROVES → PROCEED TO PHASE 2                                 │
└────────────────────────────────────────────────────────────────────┘
```

---

## Phase 2: Clarification Q&A - Detailed Flow

```
╔══════════════════════════════════════════════════════════════════════╗
║                      PHASE 2: CLARIFICATION Q&A                      ║
║  Goal: Fill knowledge gaps needed for complete HLD                   ║
║  Output: QA_ComponentName.md with categorized Q&A                    ║
╚══════════════════════════════════════════════════════════════════════╝

USER INVOCATION
┌────────────────────────────────────────────────────────────────────┐
│ Prompt: "Follow Phase2_ClarificationQA.md instructions             │
│          for MotherHLD.md Section X.Y"                             │
│                                                                     │
│ Documents Attached:                                                 │
│  - Phase2_ClarificationQA.md (instructions)                        │
│  - MotherHLD.md (reads Part 1 + Section X.Y)                       │
│  - Critique_ComponentName.md (Phase 1 output)                      │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ LLM READS INPUTS                                                    │
│  1. From Critique: Business decisions made                         │
│  2. From Section X.Y: Component description, I/O, process          │
│  3. From Part 1: System architecture, config patterns, schemas    │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ LLM IDENTIFIES KNOWLEDGE GAPS                                       │
│  - Category 1: Input/Output Contracts (for HLD Sections 3, 5)     │
│  - Category 2: Dependencies & Integration (for HLD Section 3, 10)  │
│  - Category 3: Edge Cases & Validation (for HLD Section 6)         │
│  - Category 4: Performance & Scale (for HLD Section 7)             │
│  - Category 5: Error Handling (for HLD Section 6)                  │
│  - Category 6: Testing (for HLD Section 8)                         │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ LLM PRIORITIZES QUESTIONS                                           │
│  - [CRITICAL]: Must answer before HLD (blocks HLD creation)        │
│  - [HIGH]: Should answer for complete HLD                          │
│  - [NICE]: Can defer to implementation phase                       │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ LLM CREATES OUTPUT FILE                                             │
│                                                                     │
│ File: QA_ComponentName.md                                          │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ # Clarification Q&A: ComponentName                             │ │
│ │                                                                 │ │
│ │ ## Questions by Category                                       │ │
│ │                                                                 │ │
│ │ ### Input/Output Contracts                                     │ │
│ │ [Will be filled iteratively]                                   │ │
│ │                                                                 │ │
│ │ ### Dependencies & Integration                                 │ │
│ │ [Will be filled iteratively]                                   │ │
│ │                                                                 │ │
│ │ ### Edge Cases & Validation                                    │ │
│ │ [Will be filled iteratively]                                   │ │
│ │                                                                 │ │
│ │ ### Performance & Scale                                        │ │
│ │ [Will be filled iteratively]                                   │ │
│ │                                                                 │ │
│ │ ### Error Handling                                             │ │
│ │ [Will be filled iteratively]                                   │ │
│ │                                                                 │ │
│ │ ### Testing                                                    │ │
│ │ [Will be filled iteratively]                                   │ │
│ │                                                                 │ │
│ │ Status: IN PROGRESS                                            │ │
│ └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ ITERATIVE Q&A PROTOCOL (ONE [CRITICAL] QUESTION AT A TIME)         │
└────────────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │ LLM: "[CRITICAL] Q1: Section X.Y mentions               │
    │       'aggregated_features.csv' as input. What exact    │
    │       columns are required? (For HLD Section 5.1)"      │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │ USER: "185 columns: hook_scene_count (int, 0-20),       │
    │        hook_eye_contact_rate (float, 0.0-1.0), ..."     │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │ LLM IMMEDIATELY UPDATES FILE                            │
    │ ┌─────────────────────────────────────────────────────┐ │
    │ │ ### Input/Output Contracts                          │ │
    │ │ #### Q1: [CRITICAL] [question]                      │ │
    │ │ **Answer**: 185 columns: hook_scene_count...        │ │
    │ │ **For HLD Section**: 5.1 (Input Schema)             │ │
    │ └─────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │ LLM: "[CRITICAL] Q2: Which Mother Part 1 sections       │
    │       does this component use? (For HLD 10.2)"          │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │ USER: [answers Q2]                                      │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │ LLM IMMEDIATELY UPDATES FILE (Q2 added under category)  │
    └─────────────────────────────────────────────────────────┘
                              ↓
       [Repeat for all [CRITICAL] questions, then [HIGH] questions]
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ ALL CRITICAL + HIGH ANSWERED → LLM RUNS COMPLETENESS CHECK         │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ FINAL OUTPUT: QA_ComponentName.md (COMPLETE)                       │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ ## Questions by Category                                       │ │
│ │ ### Input/Output Contracts                                     │ │
│ │ #### Q1: [CRITICAL] [question]                                 │ │
│ │ **Answer**: [detailed answer]                                  │ │
│ │ **For HLD Section**: 5.1                                       │ │
│ │ [... more questions ...]                                       │ │
│ │                                                                 │ │
│ │ ### Dependencies & Integration                                 │ │
│ │ [... Q&A ...]                                                  │ │
│ │ [... other categories ...]                                     │ │
│ │                                                                 │ │
│ │ ## Completeness Check                                          │ │
│ │ - [✓] Section 2 (Architecture & Design): YES                   │ │
│ │ - [✓] Section 3 (Dependencies & Integration): YES              │ │
│ │ - [✓] Section 5 (Data Schemas): YES                            │ │
│ │ - [✓] Section 6 (Error Handling): YES                          │ │
│ │ - [✓] Section 8 (Testing Strategy): YES                        │ │
│ │                                                                 │ │
│ │ ## Proceed to Phase 3                                          │ │
│ │ **Ready for HLD Generation**: YES ✓                            │ │
│ │ **Status**: COMPLETE ✓                                         │ │
│ └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ USER APPROVES → PROCEED TO PHASE 3                                 │
└────────────────────────────────────────────────────────────────────┘
```

---

## Phase 3: Child HLD Generation - Detailed Flow

```
╔══════════════════════════════════════════════════════════════════════╗
║                    PHASE 3: CHILD HLD GENERATION                     ║
║  Goal: Generate perfect draft HLD (no TODOs, complete schemas)       ║
║  Output: ChildHLD_ComponentName.md (10 sections + 2 appendices)      ║
╚══════════════════════════════════════════════════════════════════════╝

USER INVOCATION
┌────────────────────────────────────────────────────────────────────┐
│ Prompt: "Follow Phase3_ChildHLDGeneration.md instructions          │
│          for MotherHLD.md Section X.Y"                             │
│                                                                     │
│ Documents Attached:                                                 │
│  - Phase3_ChildHLDGeneration.md (instructions)                     │
│  - ChildTemplate.md (10-section structure)                         │
│  - MotherHLD.md (reads Part 1 + Section X.Y)                       │
│  - Critique_ComponentName.md (Phase 1 output)                      │
│  - QA_ComponentName.md (Phase 2 output)                            │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ LLM VALIDATES INPUTS                                                │
│  ✓ Critique status = COMPLETE, Proceed to Phase 2 = YES            │
│  ✓ QA status = COMPLETE, Proceed to Phase 3 = YES                  │
│  ✓ All [CRITICAL] questions answered                               │
│  ✓ Completeness check passed                                       │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ LLM EXTRACTS SOURCE DATA                                            │
│  From Critique: Business goals, concerns, decisions                │
│  From QA: Schemas, validation rules, performance targets, tests    │
│  From Mother Section X.Y: Component description, process logic     │
│  From Mother Part 1: Architecture, config, shared schemas          │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ LLM GENERATES COMPLETE HLD USING ChildTemplate.md                  │
│                                                                     │
│ Section-by-section generation:                                     │
│  1. Context & Business Goal ← Critique                             │
│  2. Architecture & Design ← QA + Section X.Y + Part 1              │
│  3. Dependencies & Integration ← QA + Part 1                       │
│  4. Configuration & Parameters ← QA + Part 1                       │
│  5. Data Schemas ← QA (EXACT field names, types, ranges)           │
│  6. Error Handling & Validation ← QA (EXACT error messages)        │
│  7. Performance & Scalability ← QA + Critique                      │
│  8. Testing Strategy ← QA (realistic test scenarios)               │
│  9. Future Enhancements ← Critique + QA                            │
│ 10. References & Related Docs ← Mother Part 1 + Section X.Y        │
│                                                                     │
│ Appendix A: Glossary ← All technical terms used                    │
│ Appendix B: Decision Log ← Critique + QA key decisions             │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ LLM RUNS ANTI-HALLUCINATION VALIDATION                              │
│  □ No TODOs or placeholders?                                       │
│  □ All schema tables complete (all columns from Q&A)?              │
│  □ All examples use real field names (not "field_1", "field_2")?   │
│  □ All error messages match Q&A answers (not invented)?            │
│  □ All Mother Part 1 references accurate (traced to sections)?     │
│  □ All cross-references valid (sections exist)?                    │
│  □ Source comments present (# Source: Q&A Q5)?                     │
│                                                                     │
│  If ANY check fails → Create Phase2_FollowUp instead of HLD        │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ FINAL OUTPUT: ChildHLD_ComponentName.md (COMPLETE)                 │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ # ComponentName - High-Level Design                            │ │
│ │                                                                 │ │
│ │ ## 1. Context & Business Goal                                  │ │
│ │ [From Critique: Why this component, business value]            │ │
│ │                                                                 │ │
│ │ ## 2. Architecture & Design                                    │ │
│ │ ### 2.1 High-level Approach                                    │ │
│ │ [From QA: Overall strategy]                                    │ │
│ │ ### 2.2 Data Flow                                              │ │
│ │ [Diagram from QA answers]                                      │ │
│ │ ### 2.3 Detailed Process                                       │ │
│ │ [Step-by-step from Section X.Y + QA]                           │ │
│ │                                                                 │ │
│ │ ## 3. Dependencies & Integration                               │ │
│ │ ### 3.1 Input Dependencies                                     │ │
│ │ | Dependency | Source | Provides | Notes |                     │ │
│ │ | Mother Part 1 | Section 2 | Directory structure | ... |     │ │
│ │ [From QA Category 2]                                           │ │
│ │ ### 3.2 Output Contracts                                       │ │
│ │ [From QA Category 1]                                           │ │
│ │ ### 3.3 Cross-Stage Dependencies                               │ │
│ │ [From QA Category 2]                                           │ │
│ │                                                                 │ │
│ │ ## 4. Configuration & Parameters                               │ │
│ │ [From QA + Mother Part 1 Section 4]                            │ │
│ │                                                                 │ │
│ │ ## 5. Data Schemas                                             │ │
│ │ ### 5.1 Input Schema                                           │ │
│ │ | Column | Type | Range | Nulls? | Description | Example |     │ │
│ │ | hook_scene_count | int | 0-20 | No | ... | 12 |              │ │
│ │ [EXACT columns from QA Q1 - all 185 columns listed]           │ │
│ │ ### 5.2 Output Schema                                          │ │
│ │ [From QA Category 1]                                           │ │
│ │                                                                 │ │
│ │ ## 6. Error Handling & Validation                              │ │
│ │ ### 6.1 Input Validation                                       │ │
│ │ [From QA Category 3]                                           │ │
│ │ ### 6.2 Error Cases                                            │ │
│ │ [From QA Category 5 - EXACT error messages]                   │ │
│ │ ### 6.3 Output Validation                                      │ │
│ │ [From QA Category 3]                                           │ │
│ │                                                                 │ │
│ │ ## 7. Performance & Scalability                                │ │
│ │ ### 7.1 Performance Targets                                    │ │
│ │ [From QA Q7: "300 videos in < 5 minutes"]                      │ │
│ │ ### 7.2 Scalability Considerations                             │ │
│ │ [From Critique concerns]                                       │ │
│ │ ### 7.3 Bottlenecks & Mitigations                              │ │
│ │ [From QA Category 4]                                           │ │
│ │                                                                 │ │
│ │ ## 8. Testing Strategy                                         │ │
│ │ ### 8.1 Unit Tests                                             │ │
│ │ [From QA Category 6]                                           │ │
│ │ ### 8.2 Integration Tests                                      │ │
│ │ [From QA Category 6]                                           │ │
│ │ ### 8.3 Test Data                                              │ │
│ │ [From QA Q11: realistic scenario with sample data]            │ │
│ │                                                                 │ │
│ │ ## 9. Future Enhancements                                      │ │
│ │ [From Critique + QA nice-to-haves]                             │ │
│ │                                                                 │ │
│ │ ## 10. References & Related Docs                               │ │
│ │ ### 10.1 Mother Document Sections                              │ │
│ │ - MotherHLD.md Section X.Y (lines A-B)                         │ │
│ │ ### 10.2 Mother Document Foundation                            │ │
│ │ - MotherHLD.md Part 1: Foundation                              │ │
│ │   - Section 2: Client Architecture (directory paths)           │ │
│ │   - Section 4: CLI Command Structure                           │ │
│ │ [From QA Q2 answers]                                           │ │
│ │                                                                 │ │
│ │ ## Appendix A: Glossary                                        │ │
│ │ [All technical terms defined]                                  │ │
│ │                                                                 │ │
│ │ ## Appendix B: Decision Log                                    │ │
│ │ **Decision 1**: [Key decision from Critique]                   │ │
│ │ - **Context**: [from Q&A]                                      │ │
│ │ - **Alternatives**: [from Critique concerns]                   │ │
│ │ - **Rationale**: [from Q&A answers]                            │ │
│ │                                                                 │ │
│ │ ---                                                             │ │
│ │ **Document Metadata**                                           │ │
│ │ - Version: 1.0                                                 │ │
│ │ - Last Updated: 2025-01-28                                     │ │
│ │ - Status: APPROVED ✓                                           │ │
│ │ - Perfect Draft: No TODOs, complete schemas, realistic tests   │ │
│ └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ USER REVIEWS → [APPROVE | REQUEST PHASE 4 REFINEMENTS]             │
└────────────────────────────────────────────────────────────────────┘
```

---

## Phase 4: Review & Refinement - Detailed Flow (Optional)

```
╔══════════════════════════════════════════════════════════════════════╗
║                   PHASE 4: REVIEW & REFINEMENT (Optional)            ║
║  Goal: Surgical updates to specific sections without regeneration    ║
║  Output: Updated ChildHLD_ComponentName.md                           ║
╚══════════════════════════════════════════════════════════════════════╝

USER INVOCATION (only if Phase 3 output needs changes)
┌────────────────────────────────────────────────────────────────────┐
│ Prompt: "Follow Phase4_ReviewRefinement.md instructions            │
│          for ChildHLD_ComponentName.md Section 5.1"                │
│                                                                     │
│ User Feedback: "Add 'confidence_score' column to Section 5.1"      │
│                                                                     │
│ Documents Attached:                                                 │
│  - Phase4_ReviewRefinement.md (instructions)                       │
│  - ChildHLD_ComponentName.md (Phase 3 output)                      │
│  [Optional: QA_ComponentName.md, Critique_ComponentName.md]        │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ LLM ANALYZES SCOPE OF CHANGE                                        │
│  - Change Type 1: Minor Correction (typo fix) → No validation      │
│  - Change Type 2: Schema/Data Change → Validate consistency        │
│  - Change Type 3: Architectural Change → Validate entire HLD       │
│                                                                     │
│  This request: Type 2 (Schema Change)                              │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ LLM USES DEPENDENCY MAP TO FIND AFFECTED SECTIONS                   │
│  Changing Section 5.1 (Input Schema) affects:                      │
│   - Section 3.1 (Input Dependencies)                               │
│   - Section 6.1 (Input Validation)                                 │
│   - Section 8.1 (Unit Tests)                                       │
│   - Section 8.3 (Test Data)                                        │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ LLM READS AFFECTED SECTIONS                                         │
│  - Read Section 5.1 (Input Schema)                                 │
│  - Read Section 6.1 (Input Validation)                             │
│  - Read Section 8.1 (Unit Tests)                                   │
│  - Read Section 8.3 (Test Data)                                    │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ LLM MAKES SURGICAL UPDATES                                          │
│  1. Section 5.1: Add row to Input Schema table                     │
│     | confidence_score | float | 0.0-1.0 | No | Model confidence | │
│  2. Section 6.1: Add validation rule                               │
│     "Reject if confidence_score < 0 or > 1"                        │
│  3. Section 8.1: Add unit test case                                │
│     "Test out-of-range confidence_score (expect rejection)"        │
│  4. Section 8.3: Add column to test data example                   │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ LLM RUNS VALIDATION CHECKLIST (for Type 2 changes)                 │
│  ✓ Schema consistency: confidence_score in 5.1 matches 6.1         │
│  ✓ Validation consistency: Range 0.0-1.0 enforced in 6.1           │
│  ✓ Test coverage: 8.1 has test for new validation rule             │
│  ✓ Example consistency: 8.3 test data includes new column          │
│  ✓ No TODOs introduced                                             │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ LLM UPDATES METADATA                                                │
│  - Version: 1.0 → 1.1                                              │
│  - Change Log:                                                      │
│    - v1.1 (2025-01-28): Added confidence_score to input schema     │
│    - v1.0 (2025-01-27): Initial version from Phase 3               │
│  - Status: APPROVED (unchanged)                                    │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ OUTPUT: Updated ChildHLD_ComponentName.md (v1.1)                   │
│                                                                     │
│ LLM Communicates:                                                   │
│ "Updated ChildHLD_ComponentName.md (v1.1):                         │
│                                                                     │
│  Changes Made:                                                      │
│   - Section 5.1: Added confidence_score column                     │
│   - Section 6.1: Added validation rule for confidence_score        │
│   - Section 8.1: Added unit test for out-of-range values           │
│   - Section 8.3: Updated test data with new column                 │
│                                                                     │
│  Affected Sections: 5.1, 6.1, 8.1, 8.3                             │
│                                                                     │
│  Validation: PASS                                                   │
│   - Schema consistency: ✓                                          │
│   - Validation consistency: ✓                                      │
│   - Test coverage: ✓"                                              │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ USER FINAL APPROVAL → CHILD HLD FINALIZED                          │
└────────────────────────────────────────────────────────────────────┘
```

---

## Document States Throughout Workflow

```
═══════════════════════════════════════════════════════════════════════
                    DOCUMENT STATES THROUGHOUT WORKFLOW
═══════════════════════════════════════════════════════════════════════

START STATE
├─ MotherHLD.md (exists, user-created)
│  ├─ Part 1: Foundation
│  └─ Parts 2-N: Project-specific sections
└─ Phase instruction files (exist, templates)
   ├─ Phase1_BusinessCritique.md
   ├─ Phase2_ClarificationQA.md
   ├─ Phase3_ChildHLDGeneration.md
   └─ Phase4_ReviewRefinement.md

                              ↓

AFTER PHASE 1 COMPLETE
├─ MotherHLD.md (unchanged)
├─ Critique_ComponentName.md (NEW - persistent Q&A)
│  ├─ Status: COMPLETE ✓
│  └─ Proceed to Phase 2: YES
└─ Phase instruction files

                              ↓

AFTER PHASE 2 COMPLETE
├─ MotherHLD.md (unchanged)
├─ Critique_ComponentName.md (complete, archived for reference)
├─ QA_ComponentName.md (NEW - persistent Q&A)
│  ├─ Status: COMPLETE ✓
│  └─ Proceed to Phase 3: YES
└─ Phase instruction files

                              ↓

AFTER PHASE 3 COMPLETE
├─ MotherHLD.md (unchanged)
├─ Critique_ComponentName.md (archived for reference)
├─ QA_ComponentName.md (archived for reference)
├─ ChildHLD_ComponentName.md (NEW - main output)
│  ├─ Version: 1.0
│  ├─ Status: APPROVED ✓
│  └─ Perfect draft: No TODOs, complete schemas
└─ Phase instruction files

                              ↓

AFTER PHASE 4 COMPLETE (if invoked)
├─ MotherHLD.md (unchanged)
├─ Critique_ComponentName.md (archived)
├─ QA_ComponentName.md (archived)
├─ ChildHLD_ComponentName.md (UPDATED)
│  ├─ Version: 1.1 (or higher)
│  ├─ Status: APPROVED ✓
│  └─ Change log updated
└─ Phase instruction files

                              ↓

READY FOR TI GENERATION
├─ MotherHLD.md Part 1 (Foundation)
├─ ChildHLD_ComponentName.md (Finalized)
└─ TI_Generation_Prompt.md (used to generate TI)
   → Output: ComponentName_TI.md (Technical Implementation)
```

---

## Documents Used Per Phase - Summary Table

```
┌─────────┬────────────────────────────────┬──────────────────────────┬───────────────────────────┬─────────────────────┐
│ Phase   │ User Attaches                  │ LLM Reads                │ LLM Writes                │ Output              │
├─────────┼────────────────────────────────┼──────────────────────────┼───────────────────────────┼─────────────────────┤
│ Phase 1 │ - Phase1_BusinessCritique.md   │ - Instructions           │ - Critique_ComponentName  │ Critique doc        │
│         │ - MotherHLD.md                 │ - Section X.Y            │   .md (iterative Q&A)     │ (COMPLETE)          │
│         │                                │ - Part 1                 │                           │                     │
├─────────┼────────────────────────────────┼──────────────────────────┼───────────────────────────┼─────────────────────┤
│ Phase 2 │ - Phase2_ClarificationQA.md    │ - Instructions           │ - QA_ComponentName.md     │ QA doc              │
│         │ - MotherHLD.md                 │ - Section X.Y            │   (iterative Q&A)         │ (COMPLETE)          │
│         │ - Critique_ComponentName.md    │ - Part 1                 │                           │                     │
│         │                                │ - Critique               │                           │                     │
├─────────┼────────────────────────────────┼──────────────────────────┼───────────────────────────┼─────────────────────┤
│ Phase 3 │ - Phase3_ChildHLDGeneration.md │ - Instructions           │ - ChildHLD_ComponentName  │ Child HLD           │
│         │ - ChildTemplate.md             │ - Template               │   .md (single output)     │ (COMPLETE)          │
│         │ - MotherHLD.md                 │ - Section X.Y            │                           │                     │
│         │ - Critique_ComponentName.md    │ - Part 1                 │                           │                     │
│         │ - QA_ComponentName.md          │ - Critique               │                           │                     │
│         │                                │ - QA                     │                           │                     │
├─────────┼────────────────────────────────┼──────────────────────────┼───────────────────────────┼─────────────────────┤
│ Phase 4 │ - Phase4_ReviewRefinement.md   │ - Instructions           │ - Updates to specific     │ Refined sections    │
│         │ - ChildHLD_ComponentName.md    │ - Child HLD              │   sections                │                     │
│         │ - User feedback (text)         │ - User feedback          │                           │                     │
└─────────┴────────────────────────────────┴──────────────────────────┴───────────────────────────┴─────────────────────┘
```

---

## Key Protocol Rules

### Iterative Q&A Protocol (Phases 1 & 2)
```
┌───────────────────────────────────────────────────────────────────┐
│                    ONE QUESTION AT A TIME                         │
│                                                                    │
│  1. LLM asks Question N                                           │
│  2. User answers Question N                                       │
│  3. LLM IMMEDIATELY updates persistent .md file                   │
│  4. LLM asks Question N+1                                         │
│  5. Repeat until all questions answered                           │
│                                                                    │
│  WHY: Prevents context loss during chat compaction                │
│       Each Q&A persists in file, survives summarization           │
└───────────────────────────────────────────────────────────────────┘
```

### Anti-Hallucination Rules (All Phases)
```
┌───────────────────────────────────────────────────────────────────┐
│                    ANTI-HALLUCINATION VALIDATION                  │
│                                                                    │
│  ✓ Trace all info to source (Critique, QA, Mother Doc)           │
│  ✓ Use EXACT field names from Q&A (not "field_1", "field_2")     │
│  ✓ Use EXACT error messages from Q&A (not invented)              │
│  ✓ Use EXACT performance targets from Q&A (not assumptions)      │
│  ✓ Reference Mother Part 1 sections correctly (verify exists)    │
│  ✓ Add source comments (# Source: Q&A Q5)                         │
│  ✓ No TODOs or placeholders in final output                      │
│                                                                    │
│  IF missing info → Ask user OR create Phase2_FollowUp.md         │
│  NEVER invent data to fill gaps                                  │
└───────────────────────────────────────────────────────────────────┘
```

### Perfect Draft Requirement (Phase 3)
```
┌───────────────────────────────────────────────────────────────────┐
│                      PERFECT DRAFT CHECKLIST                      │
│                                                                    │
│  Before outputting ChildHLD_ComponentName.md:                     │
│                                                                    │
│  □ Zero TODOs or placeholders?                                    │
│  □ All schema tables complete (all columns listed)?               │
│  □ All examples use realistic data (from Q&A)?                    │
│  □ All validation rules match Q&A answers?                        │
│  □ All error messages match Q&A answers?                          │
│  □ All Mother Part 1 references traced and verified?              │
│  □ All test scenarios realistic (from Q&A)?                       │
│  □ Decision log complete (key decisions documented)?              │
│                                                                    │
│  IF any check fails → DON'T output incomplete HLD                 │
│                     → Create Phase2_FollowUp.md instead           │
└───────────────────────────────────────────────────────────────────┘
```

---

## Exit Conditions

### When to Stay in Current Phase
- **Phase 1**: User answers questions, no fundamental design flaws found
- **Phase 2**: User provides complete answers, all [CRITICAL] gaps filled
- **Phase 3**: All validation checks pass, perfect draft achieved
- **Phase 4**: Changes are localized (<3 sections), validation passes

### When to Go Back to Previous Phase
- **Phase 2 → Phase 1**: Critique reveals fundamental design flaw
- **Phase 3 → Phase 2**: Missing critical info not covered in Q&A
- **Phase 4 → Phase 2**: Change requires new info not in original Q&A

### When to Skip to Next Phase
- **Phase 1 → Phase 2**: Final Decision = "Proceed to Phase 2: YES"
- **Phase 2 → Phase 3**: Completeness Check = "Proceed to Phase 3: YES"
- **Phase 3 → Phase 4**: User requests specific refinements
- **Phase 3 → TI Generation**: User approves Phase 3 output (skip Phase 4)

### When to Regenerate from Phase 3
- **Phase 4**: Changes affect >50% of sections (faster to regenerate)
- **Phase 4**: Validation reveals fundamental inconsistencies

---

## Quick Reference: User Invocation Commands

```
┌───────────────────────────────────────────────────────────────────┐
│                    PHASE INVOCATION QUICK REFERENCE               │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│ PHASE 1: Business Critique                                        │
│ ─────────────────────────────────────────────────────────────     │
│ Command:                                                           │
│   "Follow Phase1_BusinessCritique.md instructions                 │
│    for MotherHLD.md Section X.Y"                                  │
│                                                                    │
│ Attach:                                                            │
│   - Phase1_BusinessCritique.md                                    │
│   - MotherHLD.md                                                  │
│                                                                    │
│ Output: Critique_ComponentName.md                                 │
│                                                                    │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│ PHASE 2: Clarification Q&A                                        │
│ ─────────────────────────────────────────────────────────────     │
│ Command:                                                           │
│   "Follow Phase2_ClarificationQA.md instructions                  │
│    for MotherHLD.md Section X.Y"                                  │
│                                                                    │
│ Attach:                                                            │
│   - Phase2_ClarificationQA.md                                     │
│   - MotherHLD.md                                                  │
│   - Critique_ComponentName.md                                     │
│                                                                    │
│ Output: QA_ComponentName.md                                       │
│                                                                    │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│ PHASE 3: Child HLD Generation                                     │
│ ─────────────────────────────────────────────────────────────     │
│ Command:                                                           │
│   "Follow Phase3_ChildHLDGeneration.md instructions               │
│    for MotherHLD.md Section X.Y"                                  │
│                                                                    │
│ Attach:                                                            │
│   - Phase3_ChildHLDGeneration.md                                  │
│   - ChildTemplate.md                                              │
│   - MotherHLD.md                                                  │
│   - Critique_ComponentName.md                                     │
│   - QA_ComponentName.md                                           │
│                                                                    │
│ Output: ChildHLD_ComponentName.md                                 │
│                                                                    │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│ PHASE 4: Review & Refinement (Optional)                           │
│ ─────────────────────────────────────────────────────────────     │
│ Command:                                                           │
│   "Follow Phase4_ReviewRefinement.md instructions                 │
│    for ChildHLD_ComponentName.md Section X.Y"                     │
│                                                                    │
│ Feedback: "Add [specific change] to Section X.Y"                  │
│                                                                    │
│ Attach:                                                            │
│   - Phase4_ReviewRefinement.md                                    │
│   - ChildHLD_ComponentName.md                                     │
│                                                                    │
│ Output: Updated ChildHLD_ComponentName.md (v1.1+)                 │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Common Issues & Solutions

```
┌───────────────────────────────────────────────────────────────────┐
│                     COMMON ISSUES & SOLUTIONS                     │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│ ISSUE 1: Chat compaction loses Q&A context                       │
│ ───────────────────────────────────────────────────────────       │
│ Symptom: LLM re-asks questions already answered                   │
│ Cause: Questions Q1-Q10 asked before writing to file              │
│ Solution: ONE question at a time, write IMMEDIATELY after answer  │
│                                                                    │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│ ISSUE 2: Phase 3 output has TODOs or placeholders                │
│ ───────────────────────────────────────────────────────────       │
│ Symptom: "TODO: Add validation rules" in Section 6                │
│ Cause: Phase 2 didn't cover this info, Phase 3 can't fill gap    │
│ Solution: LLM should create Phase2_FollowUp.md instead of HLD     │
│           Ask missing questions, then re-run Phase 3              │
│                                                                    │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│ ISSUE 3: Generated HLD has invented field names                  │
│ ───────────────────────────────────────────────────────────       │
│ Symptom: Schema shows "field_1", "field_2" instead of real names  │
│ Cause: Phase 2 didn't ask for EXACT column names                 │
│ Solution: Phase 2 must ask: "What are EXACT column names?"        │
│           Phase 3 validation should catch this (fail checklist)   │
│                                                                    │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│ ISSUE 4: Phase 4 creates inconsistencies                         │
│ ───────────────────────────────────────────────────────────       │
│ Symptom: Section 5.1 has new column, but Section 6.1 unchanged   │
│ Cause: Didn't use dependency map to find affected sections        │
│ Solution: Always check dependency map, update ALL affected areas  │
│           Run validation checklist before outputting              │
│                                                                    │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│ ISSUE 5: Can't write Mother Part 1 references in Child HLD       │
│ ───────────────────────────────────────────────────────────       │
│ Symptom: Section 10.2 says "TODO: Which Mother sections used?"   │
│ Cause: Phase 2 didn't ask which Part 1 sections this uses         │
│ Solution: Phase 2 Category 2 MUST ask: "Which Mother Part 1       │
│           sections does this component depend on?"                │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

**Document Metadata**

- **Version**: 1.0
- **Created**: 2025-01-28
- **Status**: APPROVED
- **Purpose**: Blueprint for 4-phase Child HLD creation workflow
- **Related Files**:
  - DevSystem.md (overall documentation system)
  - Phase1_BusinessCritique.md (Phase 1 instructions)
  - Phase2_ClarificationQA.md (Phase 2 instructions)
  - Phase3_ChildHLDGeneration.md (Phase 3 instructions)
  - Phase4_ReviewRefinement.md (Phase 4 instructions)
  - ChildTemplate.md (10-section HLD template)

---
