# Metaprompt: Generate Production Flow Documentation for Complex Codebases

**Purpose**: This metaprompt guides LLM agents to create comprehensive production documentation (PRODUCTION_FLOW.md + STAGE_*_IMPL.md files) for any complex, multi-stage codebase.

**Use Case**: Software systems with sequential/parallel processing stages (ML pipelines, ETL workflows, data processing, build systems, etc.)

**Output**: Two-tier documentation structure that enables future LLM agents to understand and modify the codebase efficiently.

---

## Copy-Paste Instructions for User

```
I need you to create comprehensive production documentation for this codebase following a two-tier structure:

1. PRODUCTION_FLOW.md - Executive pipeline map (~500-800 lines)
2. STAGE_*_IMPL.md files - Implementation guides per stage (~200-400 lines each)

Follow the metaprompt in METAPROMPT_PRODUCTION_DOCS.md to:
1. Systematically read the codebase (Option A: Full systematic read)
2. Discover actual code patterns (not assumptions)
3. Create navigation-focused documentation

The goal is to enable future LLM agents to efficiently:
- Fix bugs in specific stages
- Add new stages
- Trace data flow
- Understand dependencies
- Modify the orchestrator

Start by reading METAPROMPT_PRODUCTION_DOCS.md for the complete workflow.
```

---

## Phase 1: Discovery Protocol (Mandatory)

### Objective
Understand the **actual codebase** through systematic file reading (NOT assumptions, NOT educated guesses).

### Step 1.1: Identify Orchestrator File

**Task**: Find the main orchestrator/entry point

**Discovery Commands**:
```bash
# Find main entry point
find . -name "main.py" -o -name "pipeline.py" -o -name "orchestrator.py" -o -name "runner.py" | head -10

# Check for CLI entry points
grep -r "if __name__ == '__main__'" --include="*.py" | head -10

# Look for package entry points
cat setup.py pyproject.toml 2>/dev/null | grep -E "entry_points|scripts"
```

**Identify**:
- Main orchestrator file path
- Total line count (`wc -l {file}`)
- Entry point function name

### Step 1.2: Map Directory Structure

**Task**: Discover stage/module organization

**Discovery Commands**:
```bash
# List directory structure (2 levels deep)
ls -R | head -100

# Find stage-specific directories
find . -type d -name "*stage*" -o -name "*step*" -o -name "*phase*"

# Find processor/handler directories
find . -type d -name "*processor*" -o -name "*handler*" -o -name "*pipeline*"
```

**Document**:
- Stage directories (e.g., `stage1_discovery/`, `stage2_processing/`)
- Shared modules (e.g., `foundation/`, `config/`, `utils/`)
- Output directories structure

### Step 1.3: Count Lines in All Stage Files

**Task**: Calculate reading strategy for each stage

**Discovery Commands**:
```bash
# Count lines per stage directory
wc -l stage1_*/*.py | sort -n
wc -l stage2_*/*.py | sort -n
# ... repeat for all stages

# Find all main entry points
find . -name "main.py" -o -name "__main__.py" | xargs wc -l
```

**Reading Strategy Decision**:
- **Files <2000 lines**: Read in one pass
- **Files >2000 lines**: Read in chunks (offset/limit)
- **Files >5000 lines**: Read key sections (imports, entry points, main functions)

### Step 1.4: Read Orchestrator File Completely

**Task**: Understand stage sequencing and dependencies

**Reading Protocol**:
```
1. Count total lines: wc -l {orchestrator_file}
2. If <2000 lines: Read entire file
3. If >2000 lines:
   - Read first 500 lines (imports, constants, initialization)
   - Read middle sections in 800-line chunks
   - Read last 500 lines (entry point, error handling)
4. Verify completion: Track last line number read
```

**Extract from orchestrator**:
- Stage execution order (sequencing logic)
- Checkpoint file paths
- Error handling strategy (skip vs exit)
- Function calls per stage (entry points)
- Input/output paths per stage

### Step 1.5: Read Foundation/Config Layer

**Task**: Understand shared infrastructure

**Files to Read**:
- `config/*.py` - Configuration, constants, bucket definitions
- `foundation/*.py` OR `core/*.py` - CLI, paths, schemas
- `utils/*.py` - Helper functions

**Extract**:
- Path building logic
- Configuration schemas
- Shared constants (thresholds, limits)
- CLI argument structure

### Step 1.6: Read All Stage Entry Points

**Task**: Understand stage implementation details

**Reading Protocol** (for EACH stage):
```
1. Find stage directory: stage_N_*/
2. Identify entry point file: main.py, {stage_name}.py, or function called by orchestrator
3. Count lines: wc -l {entry_point_file}
4. Read systematically:
   - Imports (what dependencies?)
   - Entry function (what signature?)
   - Input validation (what checks?)
   - Main processing logic
   - Output generation (what files created?)
   - Error handling (what exceptions?)
5. Verify completion: Last line number
```

**Extract per stage**:
- Entry function name + signature
- Input files/dependencies (actual paths from code)
- Output files (actual paths from code)
- Checkpoint logic (explicit or implicit?)
- Error handling (what exceptions, what actions?)
- External dependencies (APIs, services, env vars)

### Step 1.7: Trace File Paths from Code

**Task**: Map actual file creation/consumption (NOT assumptions)

**Discovery Commands**:
```bash
# Find all file writes
grep -r "\.to_csv\|\.to_json\|json\.dump\|open.*'w'" --include="*.py" | head -50

# Find checkpoint files
grep -r "checkpoint.*json" --include="*.py" | grep -E "write|save|load"

# Find all os.path.join or Path operations
grep -r "os\.path\.join\|Path\(" --include="*.py" | head -50
```

**Document**:
- Actual file paths (including hardcoded paths!)
- File creation patterns
- Directory structures
- Checkpoint locations

### Step 1.8: Map Checkpoint Strategies

**Task**: Understand resumability patterns

**Discovery Commands**:
```bash
# Find checkpoint creation
grep -r "checkpoint" --include="*.py" -A 3 -B 3 | grep -E "json\.dump|write"

# Find checkpoint validation
grep -r "checkpoint" --include="*.py" -A 5 | grep -E "exists|load|read"
```

**Classify checkpoints**:
- **Explicit**: Dedicated checkpoint files (e.g., `stage_3_checkpoint.json`)
- **Implicit**: Output file existence checks
- **State files**: Progress tracking (e.g., `.phase1_status.json`)

### Step 1.9: Document Function Call Chains

**Task**: Trace execution flow within each stage

**For each stage entry point**:
```
1. Identify main entry function (called by orchestrator)
2. Read function body
3. List all function calls in order
4. Note which are:
   - Validation functions
   - Processing functions
   - Output functions
   - Error handlers
```

**Format**:
```
Stage 3 Entry: aggregate_features(bucket_path)
  ├─> validate_input(df, bucket, expected_count)  [line 239]
  ├─> extract_window_features(temporal_windows)   [line 120]
  ├─> create_summary()                            [line 450]
  └─> write_csv(output_path, df)                  [line 550]
```

---

## Phase 2: PRODUCTION_FLOW.md Creation

### Objective
Create executive-level pipeline map for navigation and architecture understanding.

### Structure Template

```markdown
# {Project Name} Production Pipeline Flow

**Purpose**: Authoritative map of actual production code flow
**Source**: Generated from systematic code analysis
**Last Updated**: {date}

---

## Quick Navigation
- [Pipeline Overview](#pipeline-overview)
- [Stage Dependencies Graph](#stage-dependencies-graph)
- [Critical Path Analysis](#critical-path-analysis)
- [Stage Contracts](#stage-contracts)
- [File Lifecycle Map](#file-lifecycle-map)
- [Checkpoint Strategy](#checkpoint-strategy)
- [Error Propagation Matrix](#error-propagation-matrix)

---

## Pipeline Overview

### Execution Sequence
```
{ASCII diagram showing stage flow}
Stage 1 → Stage 2 → Stage 3 → ...
```

### Stage Count by Type
- Data Collection: X stages
- Processing: Y stages
- Analysis: Z stages

### Total Processing Time
- Full pipeline: ~X minutes
- Bottleneck: {slowest stage}
- Manual interventions: {any blocking stages}

---

## Stage Dependencies Graph

### Visual Dependency Map
```
{ASCII diagram with clear arrows showing dependencies}
```

---

## Critical Path Analysis

### Blocking Dependencies
| Stage | Blocks | Reason | Workaround |
|-------|--------|--------|------------|
| ... | ... | ... | ... |

### Parallel Processing Opportunities
- List stages that can run in parallel
- Note environmental variables for parallel mode

### Critical Timing Thresholds
- Stage X: <Ys per item
- Stage Y: <Zs total

---

## Stage Contracts

{For EACH stage, create a contract section}

### Stage N: {Stage Name}

**Implementation**: [`{relative_path}`]({relative_path})
**Entry Point**: `{function_name}()` (line {start}-{end})
**Orchestrator Call**: [`{orchestrator_file}:{line_number}`]({orchestrator_file}#L{line_number})

**Inputs**:
- {Dependency stage}: `{file_path}`
- {Environment vars}: `{ENV_VAR_NAME}`

**Outputs**:
```
{actual_base_path}/
├── {file1}               # Description
├── {file2}               # Description
└── {directory}/
    └── {file3}
```

**Key Functions**:
- `{function_name}()` - {purpose}
- `{helper_function}()` - {purpose}

**Checkpoint**: `{checkpoint_file_path}` OR "None (implicit via output files)"

**Depends On**: {List of stages}

**Consumed By**: {List of stages}

**Error Strategy**: {Skip bucket | Exit pipeline | Retry}

**Skip Logic**: {How checkpoint/skip detection works}

---

## File Lifecycle Map

| File | Created By | Consumed By | Lifespan | Location | Schema |
|------|------------|-------------|----------|----------|--------|
| {file1} | Stage X | Stage Y, Z | Pipeline/Persistent | {path} | {description} |
| ... | ... | ... | ... | ... | ... |

**Lifespan Types**:
- **Pipeline**: Temporary, deleted after completion
- **Persistent**: Kept for auditing/future use

### Critical Path Dependencies
```
{file1} (Stage X)
    ├─→ Stage Y
    ├─→ Stage Z
    └─→ Stage W
```

---

## Checkpoint Strategy

### Stages with Explicit Checkpoints
| Stage | Checkpoint File | Schema | Skip Logic |
|-------|----------------|--------|------------|
| ... | ... | ... | ... |

### Stages with Implicit Checkpoints
| Stage | Checkpoint Method | Resume Strategy |
|-------|-------------------|-----------------|
| ... | ... | ... |

---

## Error Propagation Matrix

### Error Handling Strategy by Exception Type
| Exception Type | Stage Action | Pipeline Action | Exit Code | Rationale |
|---------------|--------------|-----------------|-----------|-----------|
| ValueError | Skip item | Continue | 1 | Item-specific error |
| IOError | None | Exit pipeline | 4 | System failure |
| ... | ... | ... | ... | ... |

### Per-Stage Error Behavior
| Stage | Skip Item (Continue) | Exit Pipeline (Stop) |
|-------|---------------------|---------------------|
| Stage 1 | {conditions} | {conditions} |
| ... | ... | ... |

### Cross-Stage Impact Matrix
| Stage Modified | Impacts Downstream | Must Re-run | Auto-Detected? |
|---------------|-------------------|-------------|----------------|
| Stage 1 (re-run) | All stages | All downstream | No - manual cleanup |
| ... | ... | ... | ... |

---

## Implementation Documentation

For detailed implementation guides, see:
- **Stage 1**: [`docs/stages/STAGE_1_IMPL.md`](docs/stages/STAGE_1_IMPL.md)
- **Stage 2**: [`docs/stages/STAGE_2_IMPL.md`](docs/stages/STAGE_2_IMPL.md)
- ... (list all stages)

---

## Usage Examples

### For LLM Agents: Fix Bug in Specific Stage
{Example workflow}

### For LLM Agents: Add New Stage
{Example workflow}

### For LLM Agents: Trace Data Flow
{Example workflow}

---

## Maintenance Notes

### Updating This Document
**When to update**: {conditions}
**What NOT to update here**: {list}

### Document Hierarchy
```
PRODUCTION_FLOW.md (architecture)
    ↓
STAGE_*_IMPL.md (implementation)
    ↓
{technical_docs} (algorithms)
```
```

### Content Requirements

**Section 1: Pipeline Overview**
- ASCII diagram of stage sequence
- Total stage count
- Processing time estimates
- Bottleneck identification

**Section 2: Stage Dependencies Graph**
- Visual map showing all dependencies
- Identify blocking stages
- Show parallel opportunities

**Section 3: Critical Path Analysis**
- Table of blocking dependencies
- Parallel processing notes
- Timing thresholds per stage

**Section 4: Stage Contracts** (MOST IMPORTANT)
For each stage, document:
- Implementation file path (clickable link)
- Entry point function + line numbers
- Orchestrator call location + line number
- Input files (actual paths from code)
- Output files (actual directory structure)
- Key functions (name + line + purpose)
- Checkpoint file (if any)
- Dependencies (upstream stages)
- Consumers (downstream stages)
- Error strategy (skip vs exit)
- Skip logic (how resumability works)

**Section 5: File Lifecycle Map**
- Table of all critical files
- Creation stage
- Consumption stages
- Lifespan (temporary vs persistent)
- Actual file paths
- Schema description

**Section 6: Checkpoint Strategy**
- Explicit checkpoints (dedicated files)
- Implicit checkpoints (output existence)
- Resume strategies

**Section 7: Error Propagation Matrix**
- Exception types → actions
- Per-stage error behavior
- Cross-stage impact of changes

---

## Phase 3: STAGE_*_IMPL.md Creation

### Objective
Create implementation-focused guides for stage-specific work (bug fixes, feature additions).

### When to Create
Create one STAGE_*_IMPL.md for:
- Each processing stage (Stage 1, Stage 2, Stage 3, etc.)
- Complex substages (e.g., Stage 2.7 if significantly different from Stage 2)
- Optional: Combine simple stages (e.g., Stage 2.5 + 2.5.1 in one doc)

### Structure Template

```markdown
# Stage {N}: {Stage Name} - Implementation Guide

**Purpose**: {One-line description}
**Target Audience**: LLM agents fixing bugs or adding features to Stage {N}
**Related**: [PRODUCTION_FLOW.md Stage {N} Contract](../PRODUCTION_FLOW.md#stage-{n}-{stage-name})

---

## Quick Reference

- **Entry Point**: `{file_path}::{function_name}()` (line {line_number})
- **Orchestrator Call**: `{orchestrator_file}:{line_number}`
- **Checkpoint**: `{checkpoint_path}` OR "None (implicit)"
- **Average Duration**: ~{duration}s per {unit}
- **Bottleneck**: {if any}

---

## Input Contract

### Prerequisites
**Required Stages**: Stage X, Stage Y must complete first

**Input Files**:
```
{base_path}/
├── {file1}               # Created by Stage X
└── {file2}               # Created by Stage Y
```

**Validation Logic**:
```python
# {file_path}:{line_start}-{line_end}
{paste actual validation code from discovered file}
```

**Failure Modes**:
- Missing file: {what error, what action}
- Empty file: {what error, what action}
- Invalid schema: {what error, what action}

---

## Output Contract

### Files Created
```
{base_path}/
├── {output_file1}        # Format: {description}
├── {output_file2}        # Format: {description}
└── {checkpoint_file}     # Schema: {description}
```

### Output Schema

**{output_file1}** (`{file_path}`):
```json
{
  // Paste actual schema discovered from code
}
```

**Validation**: {How outputs are validated}

---

## Implementation Details

### Core Functions

| Function | File | Line | Purpose | Calls |
|----------|------|------|---------|-------|
| `{entry_function}()` | `{file}` | {line} | Main entry | {list functions it calls} |
| `{helper1}()` | `{file}` | {line} | {purpose} | {calls} |
| `{helper2}()` | `{file}` | {line} | {purpose} | {calls} |

### Data Flow

```
{input_file}
    ↓ [{function1}()]
{intermediate_structure}
    ↓ [{function2}()]
{output_file}
```

### Critical Logic

#### {Important Algorithm/Process}

**Location**: `{file}:{line_start}-{line_end}`

**Purpose**: {What it does}

**Code**:
```python
# Paste actual code snippet from discovered file
```

**Edge Cases**:
- {Condition 1}: {How handled}
- {Condition 2}: {How handled}

---

## Error Handling

### Stage {N} Errors

**From orchestrator** (`{orchestrator_file}:{line_start}-{line_end}`):

| Exception | Cause | Action | Exit Code |
|-----------|-------|--------|-----------|
| {ExceptionType} | {Description} | {Skip/Exit} | {code} |

### Common Failure Scenarios

**Scenario 1**: {Description}
- **Cause**: {Root cause from code}
- **Detection**: {How detected - line number}
- **Action**: {What happens}
- **Recovery**: {How to fix}

**Scenario 2**: {Description}
- ... (repeat)

---

## Modification Guide

### Adding a New {Feature Type}

**Scenario**: {Example modification task}

**Steps**:
1. **Update {function}** (`{file}:{line}`)
   ```python
   # Add this code at line {line}
   {example code}
   ```

2. **Update {schema}** ({documentation file})
   - Add field: `{field_name}`
   - Update expected count

3. **Test**: {How to test}
   ```bash
   {test command}
   ```

4. **Downstream impact**: {What stages affected}

---

## Debugging Checklist

**If Stage {N} fails with {error type}**:
- [ ] Check Stage {X} completed (`{checkpoint_file}` exists)
- [ ] Verify input files exist in `{directory}`
- [ ] Confirm checkpoint status (`{checkpoint_file}`)
- [ ] Review {log_file} for skip reasons
- [ ] Check {validation_file} for outliers

**Common Issues**:
- **Issue 1**: {Symptom} → {Fix}
- **Issue 2**: {Symptom} → {Fix}

---

## Dependencies

### Python Modules
- `{module1}` - {purpose}
- `{module2}` - {purpose}

### Internal Imports
- `{internal_module}` - {what it provides}

### External Services
- {API_NAME}: Requires `{ENV_VAR}` environment variable
- {Service}: Endpoint: `{url}`

---

## Testing

### Test Command
```bash
# Run Stage {N} only (assumes upstream stages complete)
{command to run stage standalone}
```

### Expected Output
- File: `{output_file}` with {X} rows/{Y} KB
- Checkpoint: `{checkpoint_file}` with status="completed"
- Duration: <{threshold}s

### Test Data
- **Minimum**: {requirements for minimal test}
- **Full**: {requirements for full test}

---

## Performance Characteristics

### Timing Breakdown
- {Operation 1}: ~{X}s ({Y}% of total)
- {Operation 2}: ~{Z}s ({W}% of total)

### Bottlenecks
- **Primary**: {Description of bottleneck}
- **Secondary**: {If any}

### Optimization Opportunities
- {Suggestion 1}
- {Suggestion 2}

---

## Related Documentation

- **PRODUCTION_FLOW.md**: [Stage {N} Contract](../PRODUCTION_FLOW.md#stage-{n})
- **Technical Spec**: [`{TI_doc_path}`](../../{TI_doc_path})
- **Upstream Stage**: [STAGE_{N-1}_IMPL.md](STAGE_{N-1}_IMPL.md)
- **Downstream Stage**: [STAGE_{N+1}_IMPL.md](STAGE_{N+1}_IMPL.md)

---

**Document Version**: 1.0
**Last Updated**: {date}
**Maintainer**: Update when Stage {N} implementation changes
```

### Content Requirements

**Quick Reference Section**:
- Entry point with line number
- Orchestrator call location
- Checkpoint path
- Performance metrics

**Input Contract**:
- Prerequisites (upstream stages)
- Input files with actual paths
- Validation logic (paste actual code)
- Failure modes with actions

**Output Contract**:
- All files created
- Actual schemas (from code)
- Validation logic

**Implementation Details**:
- Function table with line numbers
- Data flow diagram
- Critical logic with code snippets

**Error Handling**:
- Exception table
- Common failure scenarios
- Recovery procedures

**Modification Guide**:
- Step-by-step for common changes
- Code examples
- Downstream impact warnings

**Debugging Checklist**:
- Troubleshooting steps
- Common issues + fixes

**Testing**:
- How to test stage in isolation
- Expected outputs
- Test data requirements

---

## Phase 4: Quality Assurance

### Verification Checklist

**PRODUCTION_FLOW.md**:
- [ ] All stages have contracts
- [ ] File paths are actual (not assumed)
- [ ] Line numbers are accurate
- [ ] Links work (clickable to implementation files)
- [ ] ASCII diagrams render correctly
- [ ] Error matrix is complete
- [ ] Checkpoint strategies documented
- [ ] Cross-stage impacts mapped

**STAGE_*_IMPL.md files**:
- [ ] One file per major stage created
- [ ] Entry points have line numbers
- [ ] Code snippets are actual (copy-pasted from code)
- [ ] Function tables are complete
- [ ] Error scenarios documented
- [ ] Debugging checklists present
- [ ] Links to PRODUCTION_FLOW.md work
- [ ] Related documentation linked

**Accuracy Validation**:
- [ ] No assumptions - all info from actual code
- [ ] File paths verified to exist
- [ ] Line numbers checked
- [ ] Schemas match actual output
- [ ] Error codes match orchestrator

---

## Phase 5: Usage Testing

### Test Scenarios

**Scenario 1: Bug Fix in Specific Stage**
```
User: "Fix bug in Stage 3 - missing cross-window features"

Expected LLM Workflow:
1. Read PRODUCTION_FLOW.md → Find Stage 3 contract
2. Click link to docs/stages/STAGE_3_IMPL.md
3. Read "Critical Logic" section
4. Identify function + line number
5. Read actual file
6. Make fix
7. Follow "Testing" section to verify
```

**Test**: Can a fresh LLM agent follow this path without additional questions?

**Scenario 2: Trace Data Flow**
```
User: "Where does selection_manifest.json come from?"

Expected LLM Workflow:
1. Read PRODUCTION_FLOW.md → File Lifecycle Map
2. Find row: Created by Stage 2.5, Consumed by Stages 2.6, 2.7, 8
3. Click Stage 2.5 contract for creation details
4. Click Stage 2.7 contract for usage example
```

**Test**: Can a fresh LLM agent answer without reading any code?

**Scenario 3: Add New Stage**
```
User: "Add Stage 8 for PDF generation"

Expected LLM Workflow:
1. Read PRODUCTION_FLOW.md → Understand architecture
2. Identify dependencies (Stage 8 needs Stage 7 output)
3. Copy STAGE_7_IMPL.md as template
4. Update PRODUCTION_FLOW.md with Stage 8 contract
5. Implement stage
6. Update Error Propagation Matrix
```

**Test**: Can a fresh LLM agent create consistent documentation?

---

## Metaprompt Usage Instructions

### For Users (Copy-Paste to LLM)

```
# Discovery Phase
I need production documentation for this codebase.

Follow the metaprompt in METAPROMPT_PRODUCTION_DOCS.md to:

**Phase 1: Discovery (MANDATORY - Option A: Full Systematic Read)**
1. Identify orchestrator file
2. Map directory structure
3. Count lines in all stage files
4. Read orchestrator completely
5. Read foundation/config layer
6. Read ALL stage entry points systematically
7. Trace file paths from actual code (NOT assumptions)
8. Map checkpoint strategies
9. Document function call chains

**Phase 2: Create PRODUCTION_FLOW.md**
- Use discovered information ONLY
- Include all 7 sections (Overview, Dependencies, Critical Path, Contracts, File Lifecycle, Checkpoints, Errors)
- Ensure all file paths are actual (from code)
- Add line numbers to all function references

**Phase 3: Create STAGE_*_IMPL.md files**
- One file per major stage
- Use template from metaprompt
- Include actual code snippets (not pseudocode)
- Add debugging checklists

**Phase 4: Quality Assurance**
- Verify all line numbers
- Test all links
- Confirm no assumptions

Ask me before proceeding with Phase 2 if you have questions about the discovered architecture.
```

### For LLM Agents

When you receive this metaprompt:

1. **Acknowledge the task**:
   - Confirm you'll follow systematic discovery
   - Estimate file counts and reading strategy
   - Ask user to confirm approach

2. **Execute Phase 1 completely**:
   - Read ALL files systematically
   - Track line numbers read
   - Document actual paths (not assumptions)
   - Create discovery summary

3. **Present findings**:
   - Show stage sequence
   - Show file lifecycle map
   - Highlight critical discoveries (hardcoded paths, blocking stages, etc.)
   - Ask user if ready for Phase 2

4. **Create documentation**:
   - Generate PRODUCTION_FLOW.md first
   - Then generate STAGE_*_IMPL.md files
   - Use actual code snippets
   - Include line numbers everywhere

5. **Quality check**:
   - Verify links work
   - Confirm line numbers accurate
   - Test example scenarios

---

## Anti-Patterns to Avoid

### ❌ DON'T: Assume Based on Naming
```
Bad: "Stage 3 probably aggregates features based on the name"
Good: Read stage3_aggregation.py completely, extract actual logic
```

### ❌ DON'T: Use Pseudocode
```
Bad: "Function does something like: for each file, process it"
Good: Copy actual code snippet from line 120-145
```

### ❌ DON'T: Guess File Paths
```
Bad: "Output is probably in output/ or results/"
Good: Grep for "to_csv" and find actual path: /home/user/project/insights/
```

### ❌ DON'T: Skip Large Files
```
Bad: "File is 5000 lines, I'll summarize"
Good: Read in chunks (lines 1-800, 800-1600, etc.) until complete
```

### ❌ DON'T: Create Generic Docs
```
Bad: "This stage processes data and outputs results"
Good: "Processes 120 videos via extract_features() (line 342), outputs aggregated_features.csv (350 columns) to {bucket}/ml_analysis/"
```

---

## Success Criteria

### Documentation is Complete When:

1. **Future LLM Agent Can**:
   - Fix bug in any stage without reading full codebase
   - Trace any file from creation to consumption
   - Understand dependencies without code reading
   - Add new stage following existing patterns

2. **PRODUCTION_FLOW.md Contains**:
   - All stages with contracts
   - All critical files in lifecycle map
   - All error types documented
   - All checkpoints mapped
   - Accurate line numbers
   - Working links

3. **STAGE_*_IMPL.md Files Contain**:
   - Actual code snippets (not pseudocode)
   - Line numbers for all functions
   - Debugging checklists
   - Test commands
   - Common failure scenarios

4. **Accuracy Verified**:
   - All file paths exist
   - All line numbers correct
   - All links work
   - No assumptions documented

---

## Template Files Reference

### Minimal File Set
```
{project_root}/
├── PRODUCTION_FLOW.md              # Phase 2 output
├── METAPROMPT_PRODUCTION_DOCS.md   # This file (for future use)
└── docs/
    └── stages/
        ├── STAGE_1_IMPL.md         # Phase 3 output
        ├── STAGE_2_IMPL.md
        ├── STAGE_3_IMPL.md
        └── ... (one per major stage)
```

### Optional Additions
```
docs/
├── schemas/                         # File format specs
│   ├── checkpoint_format.md
│   ├── manifest_format.md
│   └── output_format.md
└── cross_cutting/                   # Cross-stage topics
    ├── CHECKPOINT_STRATEGY.md
    ├── ERROR_HANDLING.md
    └── TESTING_GUIDE.md
```

---

## Maintenance

### When to Update This Metaprompt

**Add new sections when**:
- New documentation patterns emerge
- New anti-patterns discovered
- New quality checks needed

**Update examples when**:
- Better templates found
- Clearer structures emerge

**Version History**:
- v1.0 (2025-01-28): Initial version from RumiAI pipeline documentation

---

**Metaprompt Version**: 1.0
**Created**: 2025-01-28
**Source**: RumiAI ML pipeline documentation project
**Use Case**: Complex multi-stage processing systems (ML pipelines, ETL, build systems)
