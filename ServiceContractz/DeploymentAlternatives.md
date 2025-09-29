# Output Validation Deployment Alternatives

**Version**: 1.0
**Last Updated**: January 2025
**Status**: Deployment Strategy Planning
**Scope**: Safe rollout strategies for Critical Output Validation Contract

## 1. Executive Summary

### 1.1 The Deployment Risk
Output validation can "break" currently functioning production code by catching previously ignored data quality issues. This document presents deployment strategies that minimize disruption while maximizing value.

### 1.2 Core Challenge
Your production pipeline might be:
- **Actually broken**: Running but producing corrupt data
- **Working around bugs**: Handling bad data downstream
- **Genuinely working**: Despite occasional spec violations

Without careful deployment, validation could stop all three scenarios, including the genuinely working one.

---

## 2. Understanding Production Breaking Risk

### 2.1 How Validation Can Break "Working" Code

#### Scenario 1: Hidden Bugs Become Visible
```python
# Current Production (appears to work)
yolo_result = {
    "objectAnnotations": [{
        "timestamp": 9999.99,  # Bug! Video is only 30s
        "confidence": 2.5      # Bug! Should be 0-1
    }]
}
# Today: Silently continues, corrupts downstream data
# With validation: Pipeline stops with clear error
```

#### Scenario 2: Downstream Workarounds Exist
```python
# Your code has adapted to bad data
def process_timestamps(annotations):
    for ann in annotations:
        # You already handle bad timestamps!
        if ann["timestamp"] > video_duration:
            ann["timestamp"] = video_duration
```
With validation, the bad timestamp never reaches your workaround, potentially changing behavior.

#### Scenario 3: Services Outside Your Control
- FEAT returns confidence values > 1 due to external bug
- You can't fix FEAT, only consume its output
- Strict validation would block all FEAT processing

### 2.2 Risk Matrix

| Risk Level | Scenario | Impact | Mitigation |
|------------|----------|--------|------------|
| **High** | Deploy strict validation directly to production | Full pipeline stoppage | Never do this |
| **Medium** | Deploy lenient mode without testing | Partial failures | Test in staging first |
| **Low** | Shadow mode logging | Zero impact | Recommended approach |
| **Very Low** | Gradual rollout with monitoring | Controlled impact | Best practice |

---

## 3. Deployment Strategy Options

### 3.1 Option A: Shadow Mode First (Recommended)

**Timeline: 3-4 weeks**

#### Phase 1: Shadow Mode (Week 1)
```python
class ShadowValidator:
    def validate(self, service, result, video_metadata):
        try:
            validation = self.validator.validate(service, result, video_metadata)
            if not validation.valid:
                # Log but don't fail
                logger.warning(f"SHADOW: {service} would fail validation", extra={
                    "service": service,
                    "error": validation.error_message,
                    "severity": validation.severity,
                    "video_id": video_metadata.get("id")
                })
                metrics.increment(f"validation.shadow.{severity}")
        except Exception as e:
            logger.error(f"SHADOW: Validation error", exc_info=e)

        # ALWAYS return original result
        return result
```

**What you learn:**
- Actual failure rate
- Common validation issues
- False positives to adjust

#### Phase 2: Analysis & Adjustment (Week 2)
```sql
-- Analyze shadow mode logs
SELECT
    service,
    error_message,
    COUNT(*) as frequency,
    COUNT(DISTINCT video_id) as affected_videos
FROM validation_logs
WHERE severity = 'CRITICAL'
GROUP BY service, error_message
ORDER BY frequency DESC;
```

**Decision points:**
- Is 5% failure rate acceptable?
- Which rules need adjustment?
- Which services need fixes?

#### Phase 3: Lenient Mode (Week 3)
```python
class LenientValidator:
    def validate(self, service, result, video_metadata):
        validation = self.validator.validate(service, result, video_metadata)

        if validation.severity == "CRITICAL":
            # Only fail on critical issues
            raise ValidationError(validation.error_message)
        elif validation.severity == "WARNING":
            # Log warnings but continue
            logger.warning(f"Validation warning: {validation.error_message}")
            metrics.increment(f"validation.warning.{service}")

        return result
```

#### Phase 4: Strict Mode (Week 4)
```python
class StrictValidator:
    def validate(self, service, result, video_metadata):
        validation = self.validator.validate(service, result, video_metadata)

        if not validation.valid:
            # Fail on any validation issue
            raise ValidationError(validation.error_message)

        return result
```

### 3.2 Option B: Service-by-Service Rollout

**Timeline: 4-6 weeks**

Instead of rolling out all validation at once, enable per-service:

```python
VALIDATION_ROLLOUT = {
    "yolo": "strict",      # Week 1: Start with most stable
    "whisper": "strict",   # Week 1: Also stable
    "mediapipe": "lenient", # Week 2: Known issues
    "feat": "shadow",      # Week 3: Has external bugs
    "deepface": "disabled" # Week 4: Not ready yet
}

def validate_with_rollout(service, result, metadata):
    mode = VALIDATION_ROLLOUT.get(service, "disabled")

    if mode == "disabled":
        return result
    elif mode == "shadow":
        return shadow_validate(service, result, metadata)
    elif mode == "lenient":
        return lenient_validate(service, result, metadata)
    elif mode == "strict":
        return strict_validate(service, result, metadata)
```

**Advantages:**
- Isolate problematic services
- Learn from each service
- Maintain partial protection

### 3.3 Option C: Percentage-Based Rollout

**Timeline: 2-3 weeks**

Gradually increase validation coverage:

```python
import random

class PercentageValidator:
    def __init__(self, validation_percentage=0):
        self.percentage = validation_percentage  # 0-100

    def should_validate(self, video_id):
        # Deterministic based on video_id
        hash_value = hash(video_id) % 100
        return hash_value < self.percentage

    def validate(self, service, result, metadata):
        if not self.should_validate(metadata["video_id"]):
            return result  # Skip validation

        return strict_validate(service, result, metadata)
```

**Rollout schedule:**
- Day 1-3: 1% validation
- Day 4-7: 10% validation
- Week 2: 50% validation
- Week 3: 100% validation

### 3.4 Option D: Feature Flag Control

**Timeline: Flexible**

Use feature flags for instant control:

```python
class FeatureFlagValidator:
    def validate(self, service, result, metadata):
        flags = get_feature_flags()

        # Global kill switch
        if not flags.get("validation.enabled", False):
            return result

        # Per-service control
        service_flag = f"validation.{service}.enabled"
        if not flags.get(service_flag, False):
            return result

        # Mode control
        mode = flags.get(f"validation.{service}.mode", "shadow")
        return self.validate_with_mode(mode, service, result, metadata)
```

**Advantages:**
- Instant rollback capability
- No deployment needed for changes
- A/B testing possible

---

## 4. Monitoring & Rollback Plan

### 4.1 Key Metrics to Track

```python
CRITICAL_METRICS = {
    "pipeline.success_rate": {
        "threshold": 0.95,
        "action": "rollback if below"
    },
    "validation.failure_rate": {
        "threshold": 0.10,
        "action": "investigate if above"
    },
    "processing.time_per_video": {
        "threshold": "baseline * 1.1",
        "action": "rollback if above"
    }
}
```

### 4.2 Rollback Triggers

Automatic rollback if:
1. Pipeline success rate drops below 95%
2. Validation failures exceed 10%
3. Processing time increases by >10%
4. Critical service errors spike

### 4.3 Rollback Implementation

```python
class ValidatorWithRollback:
    def __init__(self):
        self.enabled = True
        self.failure_count = 0
        self.total_count = 0

    def validate(self, service, result, metadata):
        if not self.enabled:
            return result  # Validation disabled

        try:
            self.total_count += 1
            validation = strict_validate(service, result, metadata)
            return validation
        except ValidationError as e:
            self.failure_count += 1

            # Auto-disable if failure rate too high
            failure_rate = self.failure_count / self.total_count
            if failure_rate > 0.10 and self.total_count > 100:
                logger.critical("Auto-disabling validation due to high failure rate")
                self.enabled = False
                alert_on_call_engineer("Validation auto-disabled")

            raise
```

---

## 5. Decision Framework

### 5.1 Choose Shadow Mode (Option A) If:
- ✅ First time implementing validation
- ✅ Uncertain about data quality
- ✅ No staging environment
- ✅ Risk-averse organization

### 5.2 Choose Service-by-Service (Option B) If:
- ✅ Services have varying quality
- ✅ Some services are external/uncontrolled
- ✅ Want to maintain partial protection
- ✅ Have service-level monitoring

### 5.3 Choose Percentage-Based (Option C) If:
- ✅ High confidence in validation rules
- ✅ Want fast rollout
- ✅ Good monitoring in place
- ✅ Can tolerate some failures

### 5.4 Choose Feature Flags (Option D) If:
- ✅ Have feature flag infrastructure
- ✅ Need instant rollback
- ✅ Want A/B testing capability
- ✅ Multiple environments

---

## 6. Recommended Approach for RumiAI

Given your context (300 video batches, ML pipeline, multiple services), I recommend:

### Hybrid Approach: Shadow + Service-by-Service

1. **Week 1**: Shadow mode on ALL services
   - Zero risk
   - Gather data on all services

2. **Week 2**: Analyze and categorize services
   - Stable services → Ready for strict
   - Problematic services → Need rule adjustments
   - External services → May stay lenient

3. **Week 3**: Service-by-service activation
   - YOLO, Whisper → Strict mode (usually stable)
   - MediaPipe, OCR → Lenient mode
   - FEAT, DeepFace → Shadow or lenient (external issues)

4. **Week 4**: Production deployment
   - Feature flag control for instant rollback
   - Monitoring dashboards active
   - On-call playbook ready

### Sample Implementation

```python
# config.py
VALIDATION_CONFIG = {
    "global_enabled": True,
    "rollout_phase": "shadow",  # shadow|lenient|strict
    "services": {
        "yolo": {
            "enabled": True,
            "mode": "shadow",
            "boundary_checks": True
        },
        "whisper": {
            "enabled": True,
            "mode": "shadow",
            "boundary_checks": True
        },
        # ... other services
    },
    "monitoring": {
        "log_failures": True,
        "metrics_enabled": True,
        "auto_rollback_threshold": 0.10
    }
}

# validator.py
class AdaptiveValidator:
    def __init__(self, config):
        self.config = config

    def validate(self, service, result, metadata):
        if not self.config["global_enabled"]:
            return result

        service_config = self.config["services"].get(service, {})
        if not service_config.get("enabled", False):
            return result

        mode = service_config.get("mode", "shadow")

        if mode == "shadow":
            return self.shadow_validate(service, result, metadata)
        elif mode == "lenient":
            return self.lenient_validate(service, result, metadata)
        elif mode == "strict":
            return self.strict_validate(service, result, metadata)
```

---

## 7. Communication Plan

### 7.1 Stakeholder Messaging

**For Engineering Team:**
"We're adding data quality validation to catch issues early. Shadow mode first means zero risk while we tune the rules."

**For Operations:**
"New validation layer will reduce debugging time by 50%. Gradual rollout ensures no surprises."

**For Management:**
"Investing 3 weeks in careful rollout prevents months of data quality issues and failed ML training."

### 7.2 Documentation Requirements

Before deployment, document:
1. Validation rules for each service
2. Common false positives and fixes
3. Rollback procedures
4. Monitoring dashboard location
5. On-call escalation path

---

## 8. Success Criteria

### Week 1 (Shadow Mode)
- ✅ Zero production impact
- ✅ Validation logs being collected
- ✅ Metrics dashboard operational

### Week 2 (Analysis)
- ✅ False positive rate < 5%
- ✅ All critical bugs identified
- ✅ Rule adjustments documented

### Week 3 (Lenient Mode)
- ✅ Pipeline success rate > 95%
- ✅ Critical errors caught
- ✅ Warnings properly logged

### Week 4 (Strict Mode)
- ✅ All validation active
- ✅ ML training data quality improved
- ✅ Rollback tested successfully

---

## 9. Conclusion

The key to safe deployment is **observability before enforcement**. Shadow mode gives you complete visibility into what would break without actually breaking anything. This data-driven approach ensures validation improves your pipeline rather than disrupting it.

Remember: The goal isn't to enforce perfect data from day one. It's to gradually improve data quality while maintaining pipeline stability.

---

## Document History
- v1.0 (2025-01-26): Initial deployment alternatives documented