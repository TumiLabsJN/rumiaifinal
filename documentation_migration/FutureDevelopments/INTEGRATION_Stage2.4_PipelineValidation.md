# Stage 2.4 Pipeline Validation - Integration Summary

**Date**: 2025-01-28
**Document**: MLPlanningv2.md
**New Feature**: Stage 2.4 - Pipeline Validation

---

## 1. Markdown Section for MLPlanningv2.md

✅ **COMPLETED** - Section added at lines 710-826

**Location**: Part 3: Processing Pipeline → Between Stage 2.3 and Stage 3

**Structure**:
```markdown
## Stage 2.4: Pipeline Validation
  ├── Purpose
  ├── Input
  ├── Process
  │   ├── 2.4.1: Rolling Statistics Tracking
  │   ├── 2.4.2: Anomaly Detection
  │   ├── 2.4.3: Investigation Package Creation
  │   └── 2.4.4: Notification System
  ├── Output
  ├── Child Documents
  ├── Future TI Document
  └── Related Future Features
```

---

## 2. Child Documents Needed

### **None Required**

**Rationale**:
- Stage 2.4 content fits within HLD guidelines (~115 lines)
- Pseudocode kept minimal (5-10 lines per subsection)
- Detection rules shown in table format (concise)
- All details can be deferred to TI document

**Future TI Document**:
- `PipelineValidationTI.md` (implementation details, edge cases, full detection algorithms)

---

## 3. Affected Sections

### **3.1: Pipeline Overview** (Line 504-523)
✅ **UPDATED**

**Changes**:
```markdown
OLD:
Stage 2: Video Processing (RumiAI Pipeline)
    ↓ temporal_windows_updated.json (N videos per qualified bucket)
Stage 3: Feature Aggregation

NEW:
Stage 2: Video Processing (RumiAI Pipeline)
    ↓ temporal_windows_updated.json (N videos per qualified bucket)
    ↓ Stage 2.4: Pipeline Validation
    ↓ rolling_stats.json + flagged_videos/ (if anomalies detected)
Stage 3: Feature Aggregation
```

---

### **3.2: Directory Structure** (Lines 107-151)
✅ **UPDATED**

**Changes**:
Added validation outputs to bucket directory structure:

```markdown
bucket_0-3s/
├── videos/
├── analysis/
│   ├── insights/
│   ├── unified/
│   └── service_debug/
├── validation/              # NEW - Stage 2.4 outputs
│   ├── rolling_stats.json
│   └── validation_summary.json
├── flagged_videos/          # NEW - Investigation packages
│   └── {video_id}/
│       ├── video.mp4
│       ├── temporal_windows_updated.json
│       ├── unified_analysis.json
│       ├── service_debug/
│       └── validation_report.json
├── ml_analysis/
├── models/
├── llm_reports/
├── reports/
├── checkpoints/
└── logs/
```

---

## 4. Key Design Decisions

### **4.1: Non-Blocking Validation**
- Pipeline continues processing even when anomalies detected
- Videos flagged for manual review, not rejected
- Rationale: Allows batch processing to complete, review happens after

### **4.2: Incremental Statistics**
- Rolling mean, std, quartiles updated after each video
- No need to reprocess all videos when adding new videos
- Efficient memory usage (only store aggregated stats, not all values)

### **4.3: Four Detection Rules**
1. **IQR Outlier**: Standard statistical outlier detection
2. **Extreme Outlier**: z-score > 3 for critical cases
3. **Suspicious Zero**: Domain-specific (count features shouldn't be 0)
4. **Invalid Range**: Type-specific (rates/ratios must be [0, 1])

### **4.4: Investigation Package**
- Centralized folder with all troubleshooting files
- No file hunting required (video + outputs + service debug in one place)
- Enables quick root cause analysis

---

## 5. Data Flow Integration

### **Before Stage 2.4**:
```
Stage 2.2: Sequential RumiAI Processing
    ↓
temporal_windows_updated.json (per video)
    ↓
Stage 3: Feature Aggregation
```

### **After Stage 2.4**:
```
Stage 2.2: Sequential RumiAI Processing
    ↓
temporal_windows_updated.json (per video)
    ↓
Stage 2.4: Pipeline Validation
    ├─ Update rolling_stats.json
    ├─ Detect anomalies
    ├─ Create investigation package (if anomalies)
    └─ Notify (if critical)
    ↓
Stage 3: Feature Aggregation (continues normally)
```

---

## 6. Output Files

### **Per Video (During Processing)**:
- `validation/rolling_stats.json` (updated incrementally)
- `flagged_videos/{video_id}/` (if anomalies detected)

### **Per Bucket (After All Videos Processed)**:
- `validation/validation_summary.json` (aggregated report)

### **Example File Sizes**:
| File | Size (Est.) | Count |
|------|-------------|-------|
| rolling_stats.json | ~50 KB | 1 per bucket |
| validation_summary.json | ~5-10 KB | 1 per bucket |
| validation_report.json | ~2-5 KB | 1 per flagged video |
| Investigation package | ~5-50 MB | 1 per flagged video (includes video.mp4) |

---

## 7. Notification Examples

### **Terminal - CRITICAL Anomaly**:
```
🚨 CRITICAL ANOMALY - Video 7428596413707144481 (Bucket: 18-33s)
Feature: middle_2_scene_count | Expected: 3.2±1.1 | Actual: 45 (z=38.2)
Investigation: /data/.../flagged_videos/7428596413707144481/
```

### **Terminal - ERROR Anomaly**:
```
⚠️  ERROR - Video 7428596413707144482 (Bucket: 18-33s)
Feature: hook_scene_count | Expected: 3.2±1.1 | Actual: 0 (suspicious zero)
Investigation: /data/.../flagged_videos/7428596413707144482/
```

### **Log Only - WARNING**:
```
[2025-01-28 14:32:15] WARNING - Video 7428596413707144483 (Bucket: 18-33s)
Feature: closing_word_count | Expected: 18.5±6.2 | Actual: 35 (IQR outlier)
Investigation: /data/.../flagged_videos/7428596413707144483/
```

---

## 8. Benefits

| Requirement | How Stage 2.4 Addresses It |
|-------------|---------------------------|
| **Early Detection** | Validates immediately after RumiAI processing |
| **Specific Identification** | Reports exact feature and video causing issue |
| **Centralized Troubleshooting** | Investigation package with all files in one folder |
| **Non-Blocking** | Flags issues but continues pipeline (review later) |
| **Granular Notifications** | Terminal alerts for CRITICAL/ERROR, log for WARNING |
| **Historical Tracking** | Validation summary shows patterns across all videos |

---

## 9. Implementation Priority

### **Phase 1 (MVP)**:
- ✅ Rolling statistics tracker (mean, std, Q1, Q3, IQR)
- ✅ Four anomaly detection rules (IQR, z-score, zero, range)
- ✅ Investigation package creation
- ✅ Terminal notifications (CRITICAL, ERROR, WARNING)
- ✅ Validation summary report

### **Phase 2 (Enhanced - Future)**:
- Email notifications (optional)
- Visual summary HTML reports
- ML-based anomaly detection (more sophisticated than statistical rules)
- Cross-bucket validation (compare feature distributions across buckets)

---

## 10. Testing Recommendations

### **Test Cases**:

1. **Normal Video** (No Anomalies):
   - All features within expected ranges
   - No investigation package created
   - rolling_stats.json updated
   - No terminal notifications

2. **IQR Outlier** (WARNING):
   - Feature value outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
   - Investigation package created
   - Log entry created
   - No terminal notification

3. **Extreme Outlier** (CRITICAL):
   - z-score > 3
   - Investigation package created
   - Terminal notification displayed
   - Log entry created

4. **Suspicious Zero** (ERROR):
   - Count feature = 0
   - Investigation package created
   - Terminal notification displayed
   - Log entry created

5. **Invalid Range** (CRITICAL):
   - Rate/ratio outside [0, 1]
   - Investigation package created
   - Terminal notification displayed
   - Log entry created

---

## 11. Cross-Document References

### **Updated Documents**:
1. ✅ MLPlanningv2.md - Stage 2.4 added
2. ✅ MLPlanningv2.md - Pipeline Overview updated
3. ✅ MLPlanningv2.md - Directory Structure updated

### **Documents That Reference Stage 2.4**:
- Stage 2.2 (Sequential RumiAI Processing) → Feeds into Stage 2.4
- Stage 3 (Feature Aggregation) → Receives validated data from Stage 2.4

### **Future Documents to Create**:
- PipelineValidationTI.md (technical implementation details)

---

## 12. Validation Report Schema

### **validation_report.json**:
```json
{
  "video_id": "7428596413707144481",
  "bucket": "18-33s",
  "timestamp": "2025-01-28T14:32:15Z",
  "status": "FLAGGED",
  "anomaly_count": 2,
  "anomalies": [
    {
      "type": "EXTREME_OUTLIER",
      "severity": "CRITICAL",
      "feature": "middle_2_scene_count",
      "value": 45,
      "z_score": 38.2,
      "mean": 3.2,
      "std": 1.1,
      "expected_range": [0, 7]
    },
    {
      "type": "IQR_OUTLIER",
      "severity": "WARNING",
      "feature": "closing_word_count",
      "value": 35,
      "mean": 18.5,
      "std": 6.2,
      "expected_range": [5, 40.5]
    }
  ]
}
```

### **rolling_stats.json**:
```json
{
  "bucket": "18-33s",
  "videos_processed": 45,
  "last_updated": "2025-01-28T14:32:15Z",
  "feature_statistics": {
    "hook_scene_count": {
      "mean": 3.2,
      "std": 1.1,
      "min": 1,
      "max": 8,
      "q1": 2,
      "q3": 4,
      "outlier_threshold_low": 0,
      "outlier_threshold_high": 7
    }
    // ... ~185 features for bucket 18-33s
  }
}
```

### **validation_summary.json**:
```json
{
  "bucket": "18-33s",
  "total_videos_processed": 100,
  "flagged_videos": 3,
  "flagged_percentage": 3.0,
  "anomalies_by_severity": {
    "CRITICAL": 1,
    "ERROR": 2,
    "WARNING": 8
  },
  "most_problematic_features": [
    {
      "feature": "middle_2_scene_count",
      "anomaly_count": 3,
      "videos_affected": ["123", "456", "789"]
    }
  ],
  "flagged_video_list": [
    {
      "video_id": "7428596413707144481",
      "anomaly_count": 2,
      "severities": ["CRITICAL", "WARNING"],
      "investigation_path": "bucket_18-33s/flagged_videos/7428596413707144481/"
    }
  ]
}
```

---

## 13. Summary

**What Was Added**:
- ✅ Stage 2.4: Pipeline Validation (115 lines)
- ✅ Pipeline Overview updated (3 lines)
- ✅ Directory Structure updated (11 lines)
- ✅ Total additions: ~130 lines

**Key Features**:
- Real-time anomaly detection (4 detection rules)
- Investigation packages for troubleshooting
- Non-blocking validation (pipeline continues)
- Granular notifications (CRITICAL, ERROR, WARNING)
- Rolling statistics (incremental updates)

**Benefits**:
- Early detection of ML service bugs
- Prevents bad data from contaminating training
- Centralized troubleshooting (all files in one place)
- Feature-level granularity (exact problem identification)

**Next Steps**:
- Create `PipelineValidationTI.md` when ready for implementation
- Define exact feature list for validation (which features to track)
- Implement incremental statistics algorithm
- Design notification system (terminal + log formatting)
