# Documentation System - Developer Reference

> **Purpose**: Reusable documentation architecture for reducing LLM hallucination during implementation
> **Audience**: LLMs generating TI documents, developers needing quick reference
> **Status**: Production-ready template

---

## System Architecture (ASCII)

```
┌─────────────────────────────────────────────────────────┐
│ MOTHER DOC ({ProjectName}_Planning.md)                │
│ Human-readable overview, navigation, cross-component   │
│ Part 1: Foundation (→ FoundationCHILD.md)             │
│ Parts 2-N: Project-specific organization (stages,     │
│            layers, components, features)               │
└─────────────────────┬───────────────────────────────────┘
                      │
      ┌───────────────┼───────────────┐
      ↓               ↓               ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│FOUNDATION    │ │ CHILD DOC 1  │ │ CHILD DOC 2  │
│CHILD (Shared)│ │ (Component A)│ │ (Component B)│
│              │ │              │ │              │
│- System Goals│ │SELF-CONTAINED│ │SELF-CONTAINED│
│- Cross-Cutting│ │- Context    │ │- Context     │
│- Schemas     │ │- Design     │ │- Design      │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       │                ↓                ↓
       │         ┌──────────────┐ ┌──────────────┐
       │         │   TI DOC 1   │ │   TI DOC 2   │
       │         │              │ │              │
       └────────→│Built from:   │ │Built from:   │
                 │- Foundation  │ │- Foundation  │
                 │- Child 1     │ │- Child 2     │
                 └──────────────┘ └──────────────┘

KEY RULE: TI generation reads Foundation + Child ONLY (never Mother)
```

---

## 1. Document Tiers

### **Tier 1: Mother Document**

**File**: `{ProjectName}_Planning.md` (e.g., `MLPlanningv2.md`, `Loyalty.md`)

**Structure** (Required):
```markdown
Part 1: Foundation
  - System goals
  - Architecture overview
  - Cross-cutting concerns

Parts 2-N: Project-Specific
  - Organize by: stages, layers, components, features (project choice)
  - Each unit = future Child doc candidate
  - Minimal detail (2-3 sentences + 5-10 lines pseudocode per unit)
```

**Updates**: New components, architecture changes, high-level flow modifications

**Does NOT**: Feed TI generation directly (only via Child docs)

---

### **Tier 2a: Foundation Child (Shared)**

**File**: `FoundationCHILD.md` (one per project)

**Purpose**: Cross-cutting information all components depend on

**Required Categories** (organize as needed):

| Category | What It Contains | Examples |
|----------|------------------|----------|
| **System Context** | Goals, success criteria, constraints | Business objectives, performance targets, compliance requirements |
| **Architecture** | Directory structure, file organization, deployment topology | Folder templates, module layout, environment structure |
| **Configuration** | Environment setup, CLI parameters, settings schemas | ENV vars, command flags, config.json structures |
| **Cross-Cutting Schemas** | Data structures used by multiple components | API response formats, database connection specs, auth tokens |
| **Shared Resources** | External services, common dependencies, utilities | Third-party APIs, shared libraries, logging format |

**Extracted From**: Mother Doc Part 1 (Foundation)

**Used By**: ALL component Child docs and TI docs

---

### **Tier 2b: Component Child Documents**

**Files**: `{ComponentName}CHILD.md` (N per project, depends on granularity)

**Template**: `ChildTemplate.md` (10-section structure)

**Definition**: Any **implementable unit of work** that is:
- **Cohesive**: Single responsibility, clear boundaries
- **Self-contained**: Can generate TI without reading Mother doc
- **Hallucination-resistant**: All context needed for accurate implementation

**Granularity Examples**:

| Project Type | Possible Child Docs | Rationale |
|--------------|---------------------|-----------|
| **ML Pipeline** | VideoDiscoveryCHILD, FeatureAggregationCHILD | One per processing stage |
| **Web App** | AuthenticationCHILD, PaymentsCHILD, DashboardCHILD | One per backend feature or frontend component |
| **API Service** | UserEndpointsCHILD, ProductEndpointsCHILD | One per endpoint group |
| **Full-Stack Feature** | PointsRedemptionCHILD (backend + frontend) | One per complete user flow |

**Key Principle**: Project decides granularity based on implementation needs

**Structure** (10 Sections):
1. Context & Business Goal
2. Architecture & Design
3. Dependencies & Integration (references FoundationCHILD.md)
4. Configuration & Parameters
5. Data Schemas (component-specific)
6. Error Handling & Validation
7. Performance & Scalability
8. Testing Strategy
9. Future Enhancements
10. References (includes Foundation sections used)

---

### **Tier 3: Technical Implementation (TI) Documents**

**Files**: `{ComponentName}TI.md` (N+1 per project: 1 Foundation + N components)

**Template**: `A.HLD-TI.md` (12-section structure)

**Generation**: LLM reads `FoundationCHILD.md` + `{ComponentName}CHILD.md` → outputs TI

**Structure** (12 Sections):
1. Document Metadata (includes Foundation reference)
2. Stage Contract (includes Foundation config + paths)
3. Data Schemas (Foundation schemas + component schemas)
4. Algorithmic Specifications
5. Validation Rules
6. Error Handling
7. Complete Example Traces (uses Foundation paths)
8. File Structure & Integration (includes BASE_PATHS from Foundation)
9. Configuration & Environment (includes base config from Foundation)
10. Logging Specifications
11. Dependencies & Prerequisites (includes FoundationTI.md dependency)
12. HLD Traceability Matrix

---

## 2. Workflows

### **2.1: Creating a New Component**

**Step 1: Add to Mother Doc**
```markdown
1. Add component section to {ProjectName}_Planning.md (Parts 2-N)
2. Include: Purpose (1 line), Input/Output, Process (2-3 sentences + pseudocode)
3. Link to future Child doc
```

**Step 2: Create Component Child Doc**
```markdown
1. Copy ChildTemplate.md → {ComponentName}CHILD.md
2. Fill all 10 sections:
   - Section 3.1: Add FoundationCHILD.md as first input dependency
   - Section 10.2: List Foundation categories/sections referenced
3. Add realistic examples in Appendices
```

**Step 3: Generate TI Document**
```markdown
1. Provide LLM with:
   - FoundationCHILD.md
   - {ComponentName}CHILD.md
   - TI_Generation_Prompt.md
2. Prompt: "Generate TI from FoundationCHILD.md + {ComponentName}CHILD.md"
3. Validate using TI_Generation_Prompt.md checklist
```

**Step 4: Implement Code**
```markdown
1. Use TI as implementation spec
2. Follow exact schemas, algorithms, error handling from TI
3. Create tests from TI Section 8
```

---

### **2.2: Updating Existing Work**

**If Updating Foundation** (affects all components):
```markdown
1. Update FoundationCHILD.md (e.g., new ENV var, directory change)
2. Regenerate ALL TI docs: FoundationCHILD.md + each {ComponentName}CHILD.md
3. Update implementation code to match new TIs
```

**If Updating Component** (affects one component only):
```markdown
1. Update {ComponentName}CHILD.md (e.g., new algorithm, schema change)
2. Regenerate ONLY that TI: FoundationCHILD.md + {ComponentName}CHILD.md
3. Update implementation for that component only
```

---

### **2.3: TI Generation Process**

**Prompt Pattern**:
```
Generate TI from:
- FoundationCHILD.md
- {ComponentName}CHILD.md

Follow: TI_Generation_Prompt.md
Output: {ComponentName}TI.md
```

**LLM Steps**:
1. Read FoundationCHILD.md → Extract cross-cutting info (paths, config, schemas)
2. Read {ComponentName}CHILD.md → Extract component logic, algorithms, schemas
3. Apply mappings from TI_Generation_Prompt.md
4. Validate using checklist (30+ checks)
5. Output TI with all 12 sections

**Validation Checks**:
- ✅ All Foundation categories referenced
- ✅ All Child sections mapped to TI sections
- ✅ No hallucinated fields (all traced to Foundation or Child)
- ✅ File paths use templates from Foundation
- ✅ Config/ENV vars from Foundation included

---

## 3. Quick Reference

### **3.1: Document Inventory**

| Document | Type | Purpose | Updates When |
|----------|------|---------|--------------|
| `{ProjectName}_Planning.md` | Mother | High-level overview | New components, architecture changes |
| `FoundationCHILD.md` | Foundation Child | Cross-cutting concerns | Architecture/config changes affecting all components |
| `{ComponentName}CHILD.md` | Component Child | Component HLD | Component logic changes |
| `{ComponentName}TI.md` | TI | Implementation specs | Regenerate from Child docs |
| `ChildTemplate.md` | Template | Child doc format | Template improvements |
| `A.HLD-TI.md` | Template | TI format | Template improvements |
| `TI_Generation_Prompt.md` | Prompt | TI generation rules | Mapping logic changes |

---

### **3.2: Decision Tree**

**"Which document do I update?"**

```
What are you changing?

├─ Cross-cutting concerns (paths, config, shared schemas)?
│  └─ Update: FoundationCHILD.md → Regenerate ALL TIs
│
├─ Component-specific logic (algorithm, schema, error handling)?
│  └─ Update: {ComponentName}CHILD.md → Regenerate THAT TI only
│
├─ Adding new component?
│  └─ Update: {ProjectName}_Planning.md
│     → Create {ComponentName}CHILD.md
│     → Generate {ComponentName}TI.md
│
├─ Child doc structure or TI format?
│  └─ Update: ChildTemplate.md or A.HLD-TI.md
│
└─ TI generation logic?
   └─ Update: TI_Generation_Prompt.md
```

---

### **3.3: File Organization**

**Recommended Structure**:
```
/{ProjectRoot}/documentation/
├── {ProjectName}_Planning.md          # Mother doc
├── FoundationCHILD.md                 # Foundation (shared)
├── {Component1}CHILD.md               # Component 1 Child
├── {Component2}CHILD.md               # Component 2 Child
├── ...
├── FoundationTI.md                    # Foundation implementation (generated)
├── {Component1}TI.md                  # Component 1 TI (generated)
├── {Component2}TI.md                  # Component 2 TI (generated)
├── ...
├── ChildTemplate.md                   # Template for Children
├── A.HLD-TI.md                        # Template for TIs
└── TI_Generation_Prompt.md            # Generation instructions
```

---

### **3.4: Foundation Categories (Reference)**

When creating `FoundationCHILD.md`, include these categories (organize as appropriate for project):

**1. System Context**
- Business goals, success criteria
- Target users, use cases
- Performance/scalability targets
- Compliance requirements

**2. Architecture**
- Directory structure (file organization templates)
- Module/component topology
- Deployment architecture
- Technology stack overview

**3. Configuration**
- Environment variables (name, type, default, validation)
- CLI parameters (if applicable)
- Configuration file schemas (config.json, .env, etc.)
- Feature flags

**4. Cross-Cutting Schemas**
- API request/response formats (used by multiple components)
- Database connection specs
- Authentication/authorization structures
- Shared error response formats

**5. Shared Resources**
- External APIs (endpoints, auth, rate limits)
- Shared libraries/utilities
- Logging format/standards
- Monitoring/observability specs

---

### **3.5: Common Issues & Fixes**

| Issue | Cause | Fix |
|-------|-------|-----|
| TI hallucinating file paths | Foundation paths not used | Ensure Foundation includes path templates, TI Section 8 uses them |
| TI missing config/ENV vars | Foundation config not included | Check TI Section 9 includes Foundation config categories |
| Child doc not self-contained | Missing Foundation reference | Add Foundation dependency in Child Section 3.1, 10.2 |
| Duplicate info across Children | Should be in Foundation | Move to FoundationCHILD.md if used by 2+ components |
| TI doesn't match Child | Generation error | Re-run validation checklist from TI_Generation_Prompt.md |

---

## 4. Example Applications

### **Example 1: ML Pipeline Project**

**Mother Doc**: `MLPlanningv2.md`
- Part 1: Foundation → `FoundationCHILD.md`
- Part 2: Configuration (6 config dimensions)
- Part 3: Processing Pipeline (7 stages) → 7 Child docs
- Part 4: Future Enhancements

**Foundation Contains**:
- System goals (process 300 videos, bucket-specific ML)
- Directory structure (`/data/clients/{id}/...`)
- CLI parameters (`--client`, `--analysis-type`, `--video-count`)
- Config schemas (config.json, Apify metadata, checkpoints)
- Bucket definitions (8 duration buckets)

**Component Children**:
- `VideoDiscoveryCHILD.md` (Stage 1)
- `VideoProcessingCHILD.md` (Stage 2)
- `FeatureAggregationCHILD.md` (Stage 3)
- `FeatureTransformationCHILD.md` (Stage 4)
- ... (7 total)

---

### **Example 2: Loyalty System Project**

**Mother Doc**: `Loyalty.md`
- Part 1: Foundation → `FoundationCHILD.md`
- Part 2: Backend Components (Auth, Payments, Points, Rewards)
- Part 3: Frontend Components (Dashboard, User Portal, Admin Panel)
- Part 4: Database Design
- Part 5: Future Enhancements

**Foundation Contains**:
- System goals (multi-tenant loyalty platform, 1M users)
- Tech stack (Node.js, React, PostgreSQL)
- Environment variables (`DATABASE_URL`, `JWT_SECRET`, `STRIPE_KEY`)
- API structure (REST endpoints base URL, auth headers)
- Shared schemas (JWT payload, API error format, user object)

**Component Children** (examples):
- `AuthenticationCHILD.md` (Backend: login, JWT, password reset)
- `PointsEngineCHILD.md` (Backend: earn/redeem points)
- `DashboardCHILD.md` (Frontend: user analytics dashboard)
- `PaymentsCHILD.md` (Full-stack: Stripe integration)
- ... (10-15 total)

---

## 5. Templates Reference

### **5.1: Child Document Template** (ChildTemplate.md)

**10 Required Sections**:
1. Context & Business Goal
2. Architecture & Design (includes detailed pseudocode)
3. Dependencies & Integration (must reference FoundationCHILD.md)
4. Configuration & Parameters
5. Data Schemas (component-specific)
6. Error Handling & Validation
7. Performance & Scalability
8. Testing Strategy
9. Future Enhancements
10. References & Related Docs (must list Foundation categories used)

**Key Requirement**: Section 10.2 explicitly lists Foundation categories/sections referenced

---

### **5.2: TI Document Template** (A.HLD-TI.md)

**12 Required Sections**:
1. Document Metadata (includes `Foundation_HLD: FoundationCHILD.md`)
2. Stage Contract (includes Foundation config + paths)
3. Data Schemas (Foundation + component)
4. Algorithmic Specifications
5. Validation Rules
6. Error Handling
7. Complete Example Traces
8. File Structure & Integration (includes BASE_PATHS from Foundation)
9. Configuration & Environment (includes Foundation config)
10. Logging Specifications
11. Dependencies & Prerequisites (includes `FoundationTI.md` in `Depends_On`)
12. HLD Traceability Matrix

**Key Requirement**: Trace all info back to Foundation or Component Child (no hallucination)

---

## 6. Implementation Notes

### **6.1: For LLMs Generating TIs**

**Input Requirements**:
- MUST read both `FoundationCHILD.md` AND `{ComponentName}CHILD.md`
- MUST NOT read Mother doc (`{ProjectName}_Planning.md`)
- MUST follow `TI_Generation_Prompt.md` mappings

**Foundation Extraction Points**:
- TI Section 2 (Stage Contract): Extract config, paths from Foundation
- TI Section 3 (Data Schemas): Extract cross-cutting schemas from Foundation
- TI Section 8 (File Structure): Extract BASE_PATHS from Foundation
- TI Section 9 (Configuration): Extract ENV vars, CLI params from Foundation
- TI Section 11 (Dependencies): ALWAYS include `FoundationTI.md` in `Depends_On`

**Validation**:
- Run 30+ checklist items from `TI_Generation_Prompt.md`
- Verify all Foundation categories referenced in Child are included in TI
- Verify no invented fields (all traced to source)

---

### **6.2: For Developers**

**When Starting New Project**:
1. Create `{ProjectName}_Planning.md` with Part 1 (Foundation) + project-specific parts
2. Extract Part 1 → `FoundationCHILD.md` with required categories
3. Identify implementable units → Create `{ComponentName}CHILD.md` for each
4. Generate TIs → Start implementation

**When Maintaining Existing Project**:
- Cross-cutting change? Update `FoundationCHILD.md`, regenerate all TIs
- Component change? Update `{ComponentName}CHILD.md`, regenerate that TI only
- New component? Add to Mother → Create Child → Generate TI

---

## Appendix: Key Principles

1. **Self-Containment**: Child docs include ALL context for TI generation
2. **Shared Foundation**: FoundationCHILD.md eliminates duplication
3. **2-File Rule**: TI generation reads Foundation + Component Child (never Mother)
4. **Single Source of Truth**: Each piece of info lives in ONE place
5. **Project Flexibility**: Mother doc structure varies by project needs
6. **Granularity Choice**: Projects define component granularity based on implementation needs
7. **Foundation Categories**: Required categories, flexible organization

---

**Version**: 1.0
**Last Updated**: 2025-01-28
**Applies To**: All projects using this documentation architecture
**Examples**: MLPlanningv2 (ML pipeline), Loyalty (web app)
