# Metaprompt: Generate STAGE_*_IMPL.md for Individual Stages

**Purpose**: Guide LLM agents to create detailed implementation guides for specific pipeline stages
**Prerequisite**: PRODUCTION_FLOW.md must exist (created using METAPROMPT_PRODUCTION_FLOW.md)
**Use Case**: Debugging, feature additions, or modifications to a specific stage
**Output**: Focused implementation document (~300-600 lines per stage)

---

## Copy-Paste Instructions for User

```
I need you to create STAGE_{N}_IMPL.md for Stage {N}: {Stage Name}.

Prerequisites:
- PRODUCTION_FLOW.md exists (read it first to understand context)
- You know which stage to document: Stage {N}

Follow METAPROMPT_STAGE_IMPL.md to:
1. Read PRODUCTION_FLOW.md Stage {N} contract (context)
2. Discover all files for Stage {N} (complete reading)
3. Extract functions, schemas, error handling
4. Create focused implementation guide

Goal: Enable future LLM agents to fix bugs or add features to Stage {N} without reading unrelated code.
```

---

## Phase 1: Context from PRODUCTION_FLOW.md

### Step 1: Read Stage Contract

**Action**: Open `PRODUCTION_FLOW.md` and locate Stage {N} contract section

**Extract**:
```
Stage {N}: {Stage Name}
- Entry Point: {file}::{function}() (line {line})
- Orchestrator Call: {orchestrator_file}:{line}
- Inputs: {list of files}
- Outputs: {list of files}
- Depends On: {upstream stages}
- Consumed By: {downstream stages}
- Checkpoint: {path or "None"}
- Error Strategy: {strategy}
```

**This provides context**: What does this stage do in the pipeline?

---

## Phase 2: Stage-Specific Discovery

### Step 2.1: Identify Stage Files

**From PRODUCTION_FLOW.md contract**:
- Implementation path: `{directory}/`
- Entry point file: `{file}`

**Find all related files**:
```bash
# Find all Python files in stage directory
find {stage_directory} -name "*.py" -not -path "*/__pycache__/*" | xargs wc -l | sort -n

# Find helper modules (if stage uses shared modules)
grep -r "^from.*import" {stage_directory}/*.py | cut -d: -f2 | sort -u
```

**Document**:
- Entry point file + line count
- Helper files + line counts
- Shared modules used

### Step 2.2: Read All Stage Files Completely

**Reading Strategy**:
```
For each file in stage:
1. Count lines: wc -l {file}
2. If <2000 lines: Read entire file
3. If >2000 lines: Read in chunks (offset/limit 800 lines)
4. Track last line read to verify completion
```

**Extract from each file**:
- All function definitions (`grep -n "^def "`)
- All class definitions (`grep -n "^class "`)
- All imports (understand dependencies)
- All file I/O operations (understand inputs/outputs)
- All checkpoint operations
- All error handling (try/except blocks)

### Step 2.3: Find Actual Schemas

```bash
# Find output file writes
grep -rn "to_csv\|to_json\|json.dump" {stage_directory}/

# Find example outputs (if exist)
find . -path "*example*" -o -path "*test*" | grep -E "\.json|\.csv"

# Extract schema from code
# Look for dict/dataclass definitions that define output structure
```

**Document actual schemas** (not assumed):
- Input file formats (from code that reads them)
- Output file formats (from code that writes them)
- Checkpoint file formats

---

## Phase 3: Create STAGE_*_IMPL.md

### Document Structure

```markdown
# Stage {N}: {Stage Name} - Implementation Guide

**Purpose**: {One-line description from PRODUCTION_FLOW.md}
**Target Audience**: LLM agents fixing bugs or adding features to Stage {N}
**Related**: [PRODUCTION_FLOW.md Stage {N} Contract](../../PRODUCTION_FLOW.md#stage-{n}-{stage-name})

---

## Quick Reference

- **Entry Point**: `{file}::{function}()` (line {line})
- **Orchestrator Call**: [`{orchestrator}:{line}`](../../{orchestrator}#L{line})
- **Checkpoint**: `{path}` OR "None"
- **Duration**: ~{X}s per {unit}
- **Bottleneck**: {if any}

**Context**: Read [PRODUCTION_FLOW.md](../../PRODUCTION_FLOW.md) first for pipeline overview.

---

## Input Contract

### Prerequisites
**Required Stages**: {List upstream stages that must complete first}

**Input Files**:
```
{base_path}/
├── {file1}               # Created by Stage {X}
└── {file2}               # Created by Stage {Y}
```

### Validation
**File**: `{validation_file}::{validation_function}()` (line {line})

```python
# Paste actual validation code from file
{actual_validation_code}
```

**Failure Modes**:
- {Error type}: {Cause} → {Action}

---

## Output Contract

### Files Created
```
{base_path}/
├── {output1}             # Format: {description}
└── {output2}             # Format: {description}
```

### Output Schema

**{output_file}**:
```json
{paste actual schema from code or example file}
```

**Validation**: {How outputs are validated}

---

## Core Functions

### Function Call Chain

```
{entry_function}() [line {line}]
    ├─→ {helper1}() [line {line}]
    │   └─→ {helper1a}() [line {line}]
    ├─→ {helper2}() [line {line}]
    └─→ {helper3}() [line {line}]
```

### Function Reference Table

| Function | File | Line | Purpose | Calls |
|----------|------|------|---------|-------|
| `{entry}()` | `{file}` | {line} | {purpose} | {list} |
| `{helper1}()` | `{file}` | {line} | {purpose} | {list} |

### Critical Functions Detail

#### {function_name}()
**Location**: `{file}:{line_start}-{line_end}`
**Purpose**: {What it does}

```python
# Paste actual code from file (10-30 lines showing key logic)
{actual_code}
```

**Edge Cases**:
- {Condition}: {How handled}

---

## Data Flow

```
{input_file}
    ↓ [{function1}()]
{intermediate_data}
    ↓ [{function2}()]
{output_file}
```

---

## Error Handling

### Stage {N} Errors

**From orchestrator** (`{orchestrator}:{line_range}`):

| Exception | Cause | Action | Exit Code |
|-----------|-------|--------|-----------|
| {ExceptionType} | {Description} | {Skip/Exit} | {code} |

### Common Failure Scenarios

**Scenario 1**: {Description}
- **Cause**: {Root cause from code}
- **Detection**: `{file}:{line}`
- **Action**: {What happens}
- **Recovery**: {How to fix}

---

## Modification Guide

### Adding {Common Modification Type}

**Scenario**: {Example task}

**Steps**:

1. **Update {function}** (`{file}:{line}`)
   ```python
   # Add this code
   {example_code}
   ```

2. **Update validation** (`{validation_file}:{line}`)
   - Add check: {what to check}

3. **Test**:
   ```bash
   {test_command}
   ```

4. **Downstream impact**: {Which stages affected}

---

## Debugging Checklist

**If Stage {N} fails**:
- [ ] Check Stage {X} completed (`{prerequisite_file}` exists)
- [ ] Verify input files in `{directory}`
- [ ] Check checkpoint status
- [ ] Review logs for specific error
- [ ] {Stage-specific check}

**Common Issues**:
- **Issue 1**: {Symptom} → {Fix}
- **Issue 2**: {Symptom} → {Fix}

---

## Dependencies

### Python Modules
- `{module1}` - {purpose}

### Internal Imports
- `{internal_module}` - {what it provides}

### External Services
- {Service}: Requires `{ENV_VAR}`

---

## Testing

### Test Command
```bash
# Run Stage {N} in isolation (assumes prerequisites complete)
{command_to_run_stage_standalone}
```

### Expected Output
- File: `{output}` with {criteria}
- Duration: <{threshold}s
- Checkpoint: `{checkpoint_file}` with status="completed"

---

## Performance Characteristics

### Timing Breakdown
- {Operation1}: ~{X}s ({Y}% of total)
- {Operation2}: ~{Z}s ({W}% of total)

### Bottlenecks
- **Primary**: {Description}

### Optimization Opportunities
- {Suggestion}

---

## Related Documentation

- **PRODUCTION_FLOW.md**: [Stage {N} Contract](../../PRODUCTION_FLOW.md#stage-{n})
- **Upstream Stage**: [STAGE_{N-1}_IMPL.md](STAGE_{N-1}_IMPL.md)
- **Downstream Stage**: [STAGE_{N+1}_IMPL.md](STAGE_{N+1}_IMPL.md)

---

**Document Version**: 1.0
**Last Updated**: {date}
**Source**: 100% systematic code reading ({line_count} lines)
**Maintainer**: Update when Stage {N} implementation changes
```

---

## Phase 4: Quality Assurance

### Verification Checklist

- [ ] PRODUCTION_FLOW.md contract info matches
- [ ] All files read completely (tracked last line)
- [ ] All function line numbers accurate
- [ ] Code snippets are actual (not pseudocode)
- [ ] All schemas from actual code
- [ ] Links to PRODUCTION_FLOW.md work
- [ ] Links to related stages work
- [ ] Test command actually works

### Accuracy Tests

```bash
# Verify entry point line number
grep -n "def {entry_function}" {file}

# Verify function exists at line number
sed -n '{line}p' {file}

# Test standalone command
{test_command}
```

---

## Anti-Patterns to Avoid

### ❌ DON'T: Copy Content from PRODUCTION_FLOW.md
```
Bad: Repeat stage contract verbatim from PRODUCTION_FLOW.md
Good: Reference PRODUCTION_FLOW.md link, focus on implementation details
```

### ❌ DON'T: Use Pseudocode
```
Bad: "Function loops through files and processes them"
Good: Paste actual 10-30 line code snippet showing the loop
```

### ❌ DON'T: Guess Error Handling
```
Bad: "Probably raises ValueError on invalid input"
Good: Found try/except at line 245: raises FileNotFoundError if manifest missing
```

### ❌ DON'T: Assume Schemas
```
Bad: "Output is probably {field1, field2, field3}"
Good: Found json.dump at line 450, schema has {actual_fields_from_code}
```

---

## Success Criteria

**STAGE_*_IMPL.md is complete when**:

1. **LLM Agent Can**:
   - Fix bugs without reading other stages
   - Add features with clear modification guide
   - Understand error handling from scenarios
   - Test changes using provided commands

2. **Document Contains**:
   - Actual code snippets (not descriptions)
   - Line numbers for all functions
   - Complete function call chains
   - Actual schemas from code

3. **Zero Assumptions**:
   - All code copied from actual files
   - All line numbers verified
   - All schemas traced from code

---

## When to Create Combined Stage Docs

**Combine stages when**:
- They share 50%+ of modules (e.g., Stage 2.6 & 2.7 share validation, utils, checkpoint)
- Sequential dependency (e.g., 2.6 → manual curation → 2.7)
- Conceptually one unit (e.g., both are "Content Analysis")

**Example**: `STAGE_2.6_2.7_IMPL.md` instead of separate files

**Structure for combined docs**:
```markdown
# Stage {N} & {N+1}: {Combined Name} - Implementation Guide

## Quick Reference
- Stage {N} entry point
- Stage {N+1} entry point

## Part A: Stage {N}
{Stage N content}

## Part B: Stage {N+1}
{Stage N+1 content}

## Shared Modules
{Document shared modules once}
```

---

## Relationship to PRODUCTION_FLOW.md

**PRODUCTION_FLOW.md provides**:
- Pipeline architecture (what stages exist)
- Stage contracts (inputs/outputs/dependencies)
- Quick navigation (where to find details)

**STAGE_*_IMPL.md provides**:
- Implementation details (how stage works)
- Function call chains (code flow)
- Debugging guides (how to fix)
- Modification examples (how to extend)

**Think of it as**:
- PRODUCTION_FLOW.md = Table of contents
- STAGE_*_IMPL.md = Chapter with details

---

**Metaprompt Version**: 2.0 (Split from METAPROMPT_PRODUCTION_DOCS.md)
**Created**: 2025-01-28
**Use Case**: Creating detailed implementation guides for individual pipeline stages
