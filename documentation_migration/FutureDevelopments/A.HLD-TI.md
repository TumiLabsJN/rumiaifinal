  ### Section 1: Document Metadata
  ```yaml
  TI_Document: {TI_DOCUMENT_NAME}
  Parent_HLD: {HLD_DOCUMENT_NAME}
  Covers_HLD_Sections:
    - {LIST_ALL_SECTIONS_AND_LINE_NUMBERS}
  Related_TI_Docs:
    - Depends_On: [{UPSTREAM_TI_DOCS}]
    - Feeds_Into: [{DOWNSTREAM_TI_DOCS}]
  Implementation_Priority: {CRITICAL|HIGH|MEDIUM|LOW}

  ---
  Section 2: Stage Contract

  Define exact input and output contracts as Python dataclasses/TypedDicts:

  # INPUT CONTRACT
  class StageInput:
      """
      Exact structure this stage receives
      List every field with type, constraints, and examples
      """
      field_name: type  # Description, validation rule
      # Example: video_count: int  # Must be > 0, default 100

  # OUTPUT CONTRACT
  class StageOutput:
      """
      Exact structure this stage produces
      List every field with type, constraints, and examples
      """
      field_name: type  # Description, validation rule

      output_file_path: str  # Absolute path template

  Requirements:
  - Include ALL fields from HLD
  - Specify types (str, int, List, Dict, Optional)
  - Add inline validation rules
  - Provide example values
  - Note required vs optional fields

  ---
  Section 3: Data Schemas

  Define every data structure used in this stage with complete schemas:

  # Schema Name: {PURPOSE}
  SchemaName = {
      "field1": type,  # Required/Optional, constraints, example
      "field2": type,  # Required/Optional, constraints, example
      # ...
  }

  # Example:
  ApifyVideoMetadata = {
      "id": str,              # Required, unique, example: "7428596413707144481"
      "createTime": int,      # Required, Unix timestamp, example: 1704067200
      "playCount": int,       # Required, >= 0, example: 50000
      "duration": int,        # Required, > 0, seconds, example: 25
  }

  Include:
  - External API responses (Apify, Claude API)
  - Internal data structures
  - File formats (JSON, CSV column names)
  - Configuration objects
  - Database schemas (if applicable)
  - Computed fields with formulas

  Format: Every field must have:
  1. Type
  2. Required/Optional
  3. Constraints (ranges, regex, allowed values)
  4. Example value
  5. Default (if optional)

  ---
  Section 4: Algorithmic Specifications

  For each major function/process, provide:

  """
  Function: {function_name}({params})

  Purpose: {ONE_LINE_DESCRIPTION}

  Algorithm (Pseudocode):
  1. step_one = action
  2. step_two = action(step_one)
  3. if condition:
         step_three
     else:
         step_four
  4. return result

  Edge Cases (Exhaustive List):
  - Case 1: {condition} → {behavior}
  - Case 2: {condition} → {behavior}
  - Case 3: {condition} → {behavior}

  Validation Rules:
  - assert {condition}, "{error_message}"
  - assert {condition}, "{error_message}"

  Error Conditions:
  - {ErrorType}: {when_raised} → {recovery_action}

  Example Input:
  {complete_example_input}

  Example Output:
  {complete_example_output}

  Example Trace (Step-by-Step):
  Input: {input}
  Step 1: {intermediate_result_1}
  Step 2: {intermediate_result_2}
  ...
  Output: {final_output}
  """

  Requirements:
  - Pseudocode must be unambiguous
  - Enumerate ALL edge cases (not "etc.")
  - Provide complete examples (not "...")
  - Show intermediate states in traces
  - Include error conditions

  ---
  Section 5: Validation Rules

  Enumerate all validation rules as executable assertions:

  # Input Validation
  def validate_stage_input(input_data):
      # Rule 1
      assert condition, "error_message"
      # Rule 2
      assert condition, "error_message"
      # ...

  # Business Logic Validation
  def validate_business_rules(data):
      # Rule 1
      if not condition:
          logger.warning("warning_message")
      # ...

  # Output Validation
  def validate_stage_output(output_data):
      # Rule 1
      assert condition, "error_message"
      # ...

  Include:
  - Type checking
  - Range validation
  - Format validation (regex patterns)
  - Relationship validation (foreign keys, dependencies)
  - Business rule validation
  - Output contract validation

  ---
  Section 6: Error Handling

  Enumerate every error condition with exact handling:

  ERROR_CONDITIONS = {
      "error_name_1": {
          "condition": "When does this error occur?",
          "error_type": "ExceptionClassName",
          "action": "What to do? (retry, fallback, raise)",
          "retry_policy": "Max attempts, backoff strategy",
          "user_message": "Exact error message to display/log"
      },
      "error_name_2": {
          # ...
      }
  }

  Include:
  - Network errors (timeouts, connection failures)
  - Data errors (missing fields, invalid formats)
  - Business logic errors (insufficient data, conflicts)
  - External API errors (rate limits, authentication)
  - File system errors (permissions, disk space)

  ---
  Section 7: Complete Example Traces

  Provide 3-5 complete traces showing input → output for different scenarios:

  """
  TRACE 1: {SCENARIO_NAME} (Normal Happy Path)

  Input:
  {complete_input_object}

  Processing Steps:
  Step 1: {action} → {intermediate_result_1}
  Step 2: {action} → {intermediate_result_2}
  Step 3: {action} → {intermediate_result_3}
  ...

  Output:
  {complete_output_object}

  Files Created:
  - {file_path_1}: {brief_description}
  - {file_path_2}: {brief_description}

  Logs:
  - INFO: {log_message_1}
  - INFO: {log_message_2}
  """

  """
  TRACE 2: {EDGE_CASE_SCENARIO}
  [Same structure as above]
  """

  """
  TRACE 3: {ERROR_SCENARIO}
  [Same structure showing error handling]
  """

  Requirements:
  - Use realistic data (not placeholder values)
  - Show ALL intermediate states
  - Include file system side effects
  - Include log messages
  - Show error traces with recovery

  ---
  Section 8: File Structure & Integration

  # Module Location
  FILE_PATH = "{absolute_path_to_module}"

  # Imports (Exact)
  IMPORTS = [
      "from module import Class",
      "import library",
      # ...
  ]

  # Entry Point
  ENTRY_FUNCTION = "{main_function_name}"

  # Integration Points
  CALLS_TO_EXTERNAL_SYSTEMS = {
      "system_name": {
          "endpoint": "url_or_function",
          "auth": "how_to_authenticate",
          "timeout": "timeout_value",
          "retry": "retry_policy"
      }
  }

  # Output File Paths (Templates)
  OUTPUT_PATHS = {
      "primary_output": "{path_template}",
      "checkpoint": "{path_template}",
      "logs": "{path_template}"
  }

  ---
  Section 9: Configuration & Environment

  # Environment Variables Required
  ENV_VARS = {
      "VAR_NAME": {
          "required": True|False,
          "type": "str|int|bool",
          "default": "value or None",
          "example": "example_value",
          "validation": "validation_rule"
      }
  }

  # Configuration Object
  CONFIG_SCHEMA = {
      "section_name": {
          "field": "default_value"
      }
  }

  # Constants
  CONSTANTS = {
      "CONSTANT_NAME": "value",  # Purpose, where used
  }

  ---
  Section 10: Logging Specifications

  # Log Levels & Messages (Exhaustive)
  LOG_MESSAGES = {
      "stage_start": ("INFO", "Starting {stage_name} for {target}"),
      "data_loaded": ("INFO", "Loaded {count} videos from {source}"),
      "validation_warning": ("WARNING", "{warning_description}"),
      "error_occurred": ("ERROR", "{error_description}: {error_details}"),
      # ... (list ALL log messages this stage produces)
  }

  # Metrics to Track
  METRICS = {
      "metric_name": "description",
      # Example: "videos_scraped": "Total videos retrieved from Apify"
  }

  ---
  Section 11: Dependencies & Prerequisites

  # External Dependencies
  EXTERNAL_DEPS = {
      "library_name": {
          "version": ">=1.2.0",
          "purpose": "Why needed",
          "pip_install": "pip install library_name==1.2.0"
      }
  }

  # Upstream TI Requirements
  UPSTREAM_OUTPUTS_REQUIRED = {
      "{TI_Document_Name}": [
          "output_file_path_1",
          "output_file_path_2"
      ]
  }

  # System Prerequisites
  SYSTEM_REQUIREMENTS = {
      "disk_space": "5GB minimum",
      "memory": "2GB minimum",
      "api_keys": ["APIFY_API_KEY"],
      "network": "Internet access required"
  }

  ---
  Section 12: HLD Traceability Matrix

  Create a table mapping HLD specifications to TI sections:

  | HLD Section               | HLD Lines | TI Section                     | Implementation Status
   |
  |---------------------------|-----------|--------------------------------|----------------------
  -|
  | Stage 1.1: Apify Scraping | 537-549   | Section 4: scrape_videos()     | To Implement
   |
  | Stage 1.2: Date Filtering | 550-561   | Section 4: apply_date_filter() | To Implement
   |
  | ...                       | ...       | ...                            | ...
   |

  Purpose: Ensure every HLD requirement is covered in TI

  ---
  Output Requirements

  1. No Human Explanations: Remove "why" and "because" - only "what" and "how"
  2. No Placeholders: All examples must be complete and realistic
  3. No "etc.": Enumerate everything exhaustively
  4. No Ambiguity: Every decision point must be specified
  5. Copy-Paste Ready: Code snippets should be near-production ready
  6. Self-Contained: TI should be usable without referencing HLD again

  Quality Checklist

  Before submitting TI document, verify:
  - Every HLD requirement has a TI section
  - All schemas have complete field definitions
  - All functions have pseudocode + examples
  - All edge cases are enumerated
  - All error conditions have handling specs
  - At least 3 complete traces provided
  - All file paths are absolute and templated
  - All validation rules are executable assertions
  - No TODO/TBD/placeholder content
  - Traceability matrix is complete

  ---
  Generate the TI document now following this structure exactly.