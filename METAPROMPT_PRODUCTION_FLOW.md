# Metaprompt: Generate PRODUCTION_FLOW.md for Complex Codebases

**Purpose**: Guide LLM agents to create an executive-level pipeline map (PRODUCTION_FLOW.md)
**Use Case**: Multi-stage processing systems (ML pipelines, ETL workflows, data processing, build systems)
**Output**: Single authoritative document (~500-800 lines) showing stage contracts, dependencies, and data flow

---

## Copy-Paste Instructions for User

```
I need you to create PRODUCTION_FLOW.md - an executive pipeline map for this codebase.

Follow METAPROMPT_PRODUCTION_FLOW.md to:
1. Discover the orchestrator and stage structure (systematic reading)
2. Map stage dependencies and data flow
3. Document stage contracts (inputs/outputs/checkpoints)
4. Create file lifecycle map
5. Document error handling strategies

Goal: Enable future LLM agents to quickly understand the architecture and navigate to specific stages.

Orchestrator file: {path_to_main_orchestrator}
```

---

## Phase 1: Discovery Protocol

### Step 1: Identify Orchestrator
```bash
# Find main orchestrator/entry point
find . -name "main.py" -o -name "pipeline.py" -o -name "orchestrator.py" | head -10

# Check for CLI entry points
grep -r "if __name__ == '__main__'" --include="*.py" | head -10
```

**Extract**:
- Orchestrator file path
- Total line count: `wc -l {orchestrator_file}`
- Entry point function

### Step 2: Map Stage Structure
```bash
# Find stage directories
find . -type d -name "*stage*" -o -name "*step*" -o -name "*phase*"

# Count lines per stage
wc -l stage1_*/*.py stage2_*/*.py | sort -n
```

**Document**:
- Stage directories (e.g., `stage1_discovery/`, `stage2_processing/`)
- Shared modules (e.g., `foundation/`, `config/`)
- File counts per stage

### Step 3: Read Orchestrator Completely

**Reading Strategy**:
- <2000 lines: Read entire file
- >2000 lines: Read in 800-line chunks with offset/limit
- Track last line read to verify completion

**Extract from orchestrator**:
```python
# What to look for:
1. Stage execution order (loop structure, function calls)
2. Checkpoint file paths (search for "checkpoint" in variable names)
3. Error handling (try/except blocks, exit codes)
4. Input validation per stage
5. Output file paths
```

### Step 4: Trace Data Flow

**For EACH stage mentioned in orchestrator**:
```bash
# Find where files are created
grep -rn "to_csv\|to_json\|json.dump" stage{N}_*/

# Find checkpoint files
grep -rn "checkpoint" stage{N}_*/ | grep "save\|write"

# Find file reads
grep -rn "load\|read_csv\|load_json" stage{N}_*/
```

**Document**:
- Input files (with actual paths from code)
- Output files (with actual paths from code)
- Checkpoint files (with schema if visible)

### Step 5: Map Dependencies

**Create dependency matrix**:
```
Stage 1 outputs: file_a.json, file_b.csv
Stage 2 reads: file_a.json (from Stage 1)
Stage 2 outputs: file_c.json
Stage 3 reads: file_a.json (from Stage 1), file_c.json (from Stage 2)
```

**Identify blocking dependencies**:
- Which stages cannot run in parallel?
- Which stages have manual intervention points?

---

## Phase 2: Create PRODUCTION_FLOW.md

### Document Structure

```markdown
# {Project Name} Production Pipeline Flow

**Purpose**: Authoritative map of actual production code flow
**Last Updated**: {date}

---

## Quick Navigation
- [Pipeline Overview](#pipeline-overview)
- [Stage Dependencies Graph](#stage-dependencies-graph)
- [Stage Contracts](#stage-contracts)
- [File Lifecycle Map](#file-lifecycle-map)
- [Checkpoint Strategy](#checkpoint-strategy)
- [Error Propagation Matrix](#error-propagation-matrix)

---

## Pipeline Overview

### Execution Sequence
```
Stage 1 → Stage 2 → Stage 3 → ...
```

### Total Processing Time
- Full pipeline: ~X minutes
- Bottleneck: {slowest stage}

---

## Stage Dependencies Graph

```
Stage 1
    ↓
Stage 2 ────→ Stage 3
    ↓           ↓
Stage 4 ←───────┘
```

---

## Critical Path Analysis

### Blocking Dependencies

| Stage | Blocks | Reason | Workaround |
|-------|--------|--------|------------|
| Stage X | Stage Y, Z | {reason} | {workaround if any} |

### Parallel Processing Opportunities

- Stage A and Stage B can run in parallel
- Stage C has optional parallel mode (env: `ENABLE_PARALLEL_MODE`)

### Critical Timing Thresholds

- Stage 1: ~{X}s per {unit}
- Stage 2: ~{Y}s per {unit} (bottleneck)
- Stage 3: <{Z}s per {unit}

---

## Stage Contracts

{For EACH stage discovered:}

### Stage {N}: {Stage Name}

**Implementation**: [`{relative_path}/`]({relative_path}/)
**Entry Point**: `{file}::{function}()` (line {line_number})
**Orchestrator Call**: [`{orchestrator_file}:{line}`]({orchestrator_file}#L{line})

**Inputs**:
- Stage {X}: `{file_path}`
- Environment: `{ENV_VAR_NAME}`

**Outputs**:
```
{base_path}/
├── {file1}               # Description
└── {file2}               # Description
```

**Key Functions**:
- `{function_name}()` - {purpose}
- `{helper_function}()` - {purpose}

**Checkpoint**: `{checkpoint_path}` OR "None (uses output file existence)"

**Depends On**: Stage {X}, Stage {Y}

**Consumed By**: Stage {Z}, Stage {W}

**Error Strategy**: {Skip item | Exit pipeline | Retry}

**Skip Logic**: {How checkpoint/skip detection works}

**Duration**: ~{X}s per {unit}

---

## File Lifecycle Map

| File | Created By | Consumed By | Lifespan | Location | Schema Doc |
|------|------------|-------------|----------|----------|------------|
| {file1} | Stage X | Stage Y, Z | Pipeline/Persistent | {path} | {description or link} |

**Lifespan Types**:
- **Pipeline**: Temporary (deleted after completion)
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

### Explicit Checkpoints
| Stage | Checkpoint File | Skip Logic |
|-------|----------------|------------|
| Stage X | `{path}` | Validates {criteria} |

### Implicit Checkpoints
| Stage | Checkpoint Method |
|-------|-------------------|
| Stage Y | Output file existence check |

---

## Error Propagation Matrix

### Error Handling Strategy by Exception Type

| Exception Type | Stage Action | Pipeline Action | Exit Code | Rationale |
|---------------|--------------|-----------------|-----------|-----------|
| ValueError | Skip item | Continue | 1 | Item-specific data issue |
| AssertionError | Skip item | Continue | 3 | Output validation failed |
| FileNotFoundError | Skip item | Continue | 1 | Missing input (upstream failure) |
| IOError / OSError | None | Exit pipeline | 4 | System-wide issue (disk full) |
| TimeoutError | None | Exit pipeline | 8 | System overload |
| RuntimeError | None | Exit pipeline | 99 | API authentication failure |

### Per-Stage Error Behavior

| Stage | Skip Item (Continue) | Exit Pipeline (Stop) |
|-------|---------------------|---------------------|
| Stage 1 | {conditions} | {conditions} |
| Stage 2 | {conditions} | {conditions} |

### Exit Codes Reference

```python
EXIT_CODES = {
    0: "Success (full pipeline completion)",
    1: "Error (validation failure, missing inputs)",
    2: "Paused for manual intervention",
    3: "Assertion error (output validation failed)",
    4: "I/O failure (disk full, permissions)",
    8: "Timeout (processing exceeded limits)",
    99: "Unexpected error",
    130: "User interrupt (Ctrl+C)"
}
```

### Cross-Stage Impact Matrix

| Stage Modified | Impacts Downstream | Must Re-run | Auto-Detected? |
|---------------|-------------------|-------------|----------------|
| Stage 1 (re-run) | All stages | All downstream | No - manual cleanup |
| Stage 2 (re-run) | Stage 3+ | All downstream | Yes - checkpoint tracks |

---

## Implementation Documentation

For detailed stage implementation, see:
- [`docs/stages/STAGE_1_IMPL.md`](docs/stages/STAGE_1_IMPL.md)
- [`docs/stages/STAGE_2_IMPL.md`](docs/stages/STAGE_2_IMPL.md)
- ... (list all stages)

---

## Usage Examples

### For LLM Agents: Navigate to Specific Stage
1. Read this file to find stage entry point
2. Click link to STAGE_*_IMPL.md for details
3. Read implementation file at line number

### For LLM Agents: Trace Data Flow
1. Find file in File Lifecycle Map
2. See which stages create/consume it
3. Follow dependency chain

---

## Maintenance Notes

**When to update**:
- Adding/removing stages
- Changing stage dependencies
- Modifying checkpoint strategies

**What NOT to update here**:
- Implementation details (goes in STAGE_*_IMPL.md)
- Function signatures (goes in STAGE_*_IMPL.md)
```

---

## Phase 3: Quality Assurance

### Verification Checklist

- [ ] All stages have contracts
- [ ] All file paths are actual (from code, not assumed)
- [ ] All line numbers are accurate
- [ ] All links work (clickable paths)
- [ ] ASCII diagrams render correctly
- [ ] Error matrix covers all exception types
- [ ] Checkpoint strategies documented for each stage
- [ ] Cross-stage impacts mapped

### Accuracy Tests

```bash
# Verify all file paths exist
grep -o '`[^`]*\.json`' PRODUCTION_FLOW.md | xargs -I {} test -f {}

# Verify all line numbers (spot check)
# Open files at specified line numbers and verify function names match
```

---

## Anti-Patterns to Avoid

### ❌ DON'T: Assume File Paths
```
Bad: "Output probably goes to output/ or results/"
Good: grep "to_csv" and find actual path: data/clients/test/output.csv
```

### ❌ DON'T: Guess Stage Order
```
Bad: "Probably Stage 1 → 2 → 3 based on naming"
Good: Read orchestrator loop, see actual execution: 1 → 2 → 2.5 → 3
```

### ❌ DON'T: Invent Exit Codes
```
Bad: "Probably uses standard exit codes 0, 1, 2"
Good: Search orchestrator for "sys.exit" and document actual codes
```

---

## Success Criteria

**PRODUCTION_FLOW.md is complete when**:

1. **LLM Agent Can**:
   - Understand pipeline architecture in <2 minutes
   - Find which stage creates/consumes any file
   - Navigate to specific stage implementation
   - Understand error handling without reading code

2. **Document Contains**:
   - All stages with complete contracts
   - All critical files in lifecycle map
   - Accurate line numbers for all entry points
   - Working links to implementation docs

3. **Zero Assumptions**:
   - All paths verified to exist
   - All line numbers verified
   - All dependencies traced from code

---

## Next Steps After Creating PRODUCTION_FLOW.md

Once PRODUCTION_FLOW.md is complete, use **METAPROMPT_STAGE_IMPL.md** to create detailed implementation guides for individual stages.

---

**Metaprompt Version**: 2.0 (Split from METAPROMPT_PRODUCTION_DOCS.md)
**Created**: 2025-01-28
**Use Case**: Multi-stage processing systems requiring executive-level architecture documentation
