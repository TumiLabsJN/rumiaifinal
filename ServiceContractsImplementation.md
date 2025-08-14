```python
class TimelineBuilderContract(BaseServiceContract):
    """
    ULTRA-STRICT contract for TimelineBuilder.build_timeline function.
    Validates ML results aggregation into unified timeline with fail-fast approach.
    """
    
    def __init__(self):
        super().__init__("TimelineBuilderContract")
    
    def validate_input(self, video_id: str, video_metadata: Dict[str, Any], 
                      ml_results: Dict[str, MLAnalysisResult]) -> None:
        """Validate input parameters with fail-fast approach"""
        
        # Validate video_id
        self.validate_or_fail(isinstance(video_id, str), 
                            "video_id must be string")
        self.validate_or_fail(len(video_id) > 0, 
                            "video_id must be non-empty")
        
        # Validate video_metadata
        self.validate_or_fail(isinstance(video_metadata, dict), 
                            "video_metadata must be dict")
        self.validate_or_fail('duration' in video_metadata, 
                            "video_metadata must contain 'duration'")
        self.validate_or_fail(isinstance(video_metadata['duration'], (int, float)), 
                            "video_metadata['duration'] must be numeric")
        self.validate_or_fail(video_metadata['duration'] > 0, 
                            f"video_metadata['duration'] must be positive, got {video_metadata['duration']}")
        
        # Validate ml_results
        self.validate_or_fail(isinstance(ml_results, dict), 
                            "ml_results must be dict")
        
        # Validate each MLAnalysisResult
        valid_models = {'yolo', 'whisper', 'mediapipe', 'ocr', 'scene_detection', 
                       'audio_energy', 'emotion_detection', 'feat'}
        
        for model_name, result in ml_results.items():
            # Check it's an MLAnalysisResult object or dict with required fields
            if hasattr(result, '__dict__'):
                # It's an object, check required attributes
                self.validate_or_fail(hasattr(result, 'model_name'), 
                                    f"{model_name} result missing 'model_name' attribute")
                self.validate_or_fail(hasattr(result, 'success'), 
                                    f"{model_name} result missing 'success' attribute")
                self.validate_or_fail(hasattr(result, 'data'), 
                                    f"{model_name} result missing 'data' attribute")
            else:
                # It's a dict, check required keys
                self.validate_or_fail(isinstance(result, dict), 
                                    f"{model_name} result must be MLAnalysisResult or dict")
                self.validate_or_fail('model_name' in result, 
                                    f"{model_name} result missing 'model_name'")
                self.validate_or_fail('success' in result, 
                                    f"{model_name} result missing 'success'")
                self.validate_or_fail('data' in result, 
                                    f"{model_name} result missing 'data'")
    
    # Define strict schemas for each entry type based on actual production data
    ENTRY_SCHEMAS = {
        'speech': {
            'required': ['text'],
            'optional': ['confidence', 'language', 'words'],
            'types': {
                'text': str,
                'confidence': (int, float),
                'language': str,
                'words': list
            },
            'validators': {
                'text': lambda x: len(x) > 0,
                'confidence': lambda x: UniversalValueValidator.validate_value('confidence', x)[0],
                'words': lambda x: isinstance(x, list)
            }
        },
        'text': {
            'required': ['text'],
            'optional': ['position', 'size', 'style'],
            'types': {
                'text': str,
                'position': str,
                'size': str,
                'style': str
            },
            'validators': {
                'text': lambda x: len(x) > 0,
                'position': lambda x: x in ['left', 'center', 'right', 'top-left', 'top-center', 'top-right', 'middle-left', 'middle-center', 'middle-right', 'bottom-left', 'bottom-center', 'bottom-right']
            }
        },
        'pose': {
            'required': ['landmarks'],
            'optional': ['action', 'confidence'],
            'types': {
                'landmarks': (int, list),
                'action': str,
                'confidence': (int, float)
            },
            'validators': {
                'landmarks': lambda x: UniversalValueValidator.validate_value('landmarks', x)[0],
                'confidence': lambda x: UniversalValueValidator.validate_value('confidence', x)[0]
            }
        },
        'face': {
            'required': ['landmarks'],
            'optional': ['emotion', 'gaze_direction'],
            'types': {
                'landmarks': list,
                'emotion': str,
                'gaze_direction': str
            },
            'validators': {
                'landmarks': lambda x: isinstance(x, list),
                'emotion': lambda x: x in ['neutral', 'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'unknown']
            }
        },
        'emotion': {
            'required': ['emotion'],
            'optional': ['confidence', 'all_scores', 'action_units', 'au_intensities', 'source', 'model', 'has_action_units'],
            'types': {
                'emotion': str,
                'confidence': (int, float),
                'all_scores': dict,
                'action_units': list,
                'au_intensities': dict,
                'source': str,
                'model': str,
                'has_action_units': bool
            },
            'validators': {
                'emotion': lambda x: x in ['neutral', 'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
                'confidence': lambda x: UniversalValueValidator.validate_value('confidence', x)[0],
                'source': lambda x: x in ['feat', 'mediapipe']
            }
        },
        'scene': {
            'required': ['scene_index'],
            'optional': ['duration'],
            'types': {
                'scene_index': int,
                'duration': (int, float)
            },
            'validators': {
                'scene_index': lambda x: UniversalValueValidator.validate_value('scene_number', x + 1)[0],  # Convert 0-based to 1-based
                'duration': lambda x: UniversalValueValidator.validate_value('duration', x)[0]
            }
        },
        'scene_change': {
            'required': ['transition_type', 'scene_index'],
            'optional': [],
            'types': {
                'transition_type': str,
                'scene_index': int
            },
            'validators': {
                'scene_index': lambda x: UniversalValueValidator.validate_value('scene_number', x + 1)[0]  # Convert 0-based to 1-based
            }
        },
        'object': {
            'required': ['class'],
            'optional': ['confidence', 'bbox', 'track_id'],
            'types': {
                'class': str,
                'confidence': (int, float),
                'bbox': list,
                'track_id': (str, int, type(None))
            },
            'validators': {
                'bbox': lambda x: UniversalValueValidator.validate_value('bbox', x)[0],
                'confidence': lambda x: UniversalValueValidator.validate_value('confidence', x)[0],
                'track_id': lambda x: UniversalValueValidator.validate_value('track_id', x)[0]
            }
        },
        'sticker': {
            'required': ['type', 'value'],
            'optional': ['position'],
            'types': {
                'type': str,
                'value': str,
                'position': str
            }
        },
        'gesture': {
            'required': ['type'],
            'optional': ['hand', 'confidence'],
            'types': {
                'type': str,
                'hand': str,
                'confidence': (int, float)
            },
            'validators': {
                'confidence': lambda x: UniversalValueValidator.validate_value('confidence', x)[0]
            }
        }
    }
    
    def validate_entry_data(self, entry_type: str, data: Dict[str, Any], entry_index: int) -> None:
        """Deep validation of entry data structure with universal value validation"""
        
        schema = self.ENTRY_SCHEMAS.get(entry_type)
        self.validate_or_fail(schema is not None,
                            f"Entry {entry_index}: Unknown entry type '{entry_type}'")
        
        # Check required fields
        for field in schema['required']:
            self.validate_or_fail(field in data,
                                f"Entry {entry_index} ({entry_type}): Missing required field '{field}'")
        
        # Check for unexpected fields (warn but don't fail for forward compatibility)
        allowed = set(schema['required'] + schema['optional'])
        unexpected = set(data.keys()) - allowed
        if unexpected:
            # Log warning but don't fail - allows forward compatibility
            import logging
            logging.warning(f"Entry {entry_index} ({entry_type}): Unexpected fields {unexpected}")
        
        # Type validation first
        for field, value in data.items():
            if field in schema['types']:
                expected_type = schema['types'][field]
                self.validate_or_fail(isinstance(value, expected_type),
                                    f"Entry {entry_index} ({entry_type}).{field}: Expected {expected_type}, got {type(value)}")
        
        # Universal value validation (catches range/content issues)
        is_valid, errors = UniversalValueValidator.validate_all_fields(data, f"Entry {entry_index} ({entry_type})")
        if not is_valid:
            for error in errors:
                self.validate_or_fail(False, error)
        
        # Custom validators (additional business logic)
        for field, value in data.items():
            if field in schema.get('validators', {}):
                validator = schema['validators'][field]
                try:
                    is_valid = validator(value)
                except Exception:
                    is_valid = False
                self.validate_or_fail(is_valid,
                                    f"Entry {entry_index} ({entry_type}).{field}: Invalid value '{value}'")
    
    def validate_output(self, result: Any) -> None:
        """Validate UnifiedAnalysis output with deep entry validation"""
        
        # Check it's a UnifiedAnalysis object or dict with required fields
        if hasattr(result, '__dict__'):
            # It's an object, validate attributes
            self.validate_or_fail(hasattr(result, 'video_id'), 
                                "UnifiedAnalysis missing 'video_id'")
            self.validate_or_fail(hasattr(result, 'video_metadata'), 
                                "UnifiedAnalysis missing 'video_metadata'")
            self.validate_or_fail(hasattr(result, 'timeline'), 
                                "UnifiedAnalysis missing 'timeline'")
            self.validate_or_fail(hasattr(result, 'ml_results'), 
                                "UnifiedAnalysis missing 'ml_results'")
            self.validate_or_fail(hasattr(result, 'processing_metadata'), 
                                "UnifiedAnalysis missing 'processing_metadata'")
            
            video_id = result.video_id
            video_metadata = result.video_metadata
            timeline = result.timeline
            ml_results = result.ml_results
            processing_metadata = result.processing_metadata
        else:
            # It's a dict
            self.validate_or_fail(isinstance(result, dict), 
                                "Output must be UnifiedAnalysis object or dict")
            self.validate_or_fail('video_id' in result, 
                                "Output missing 'video_id'")
            self.validate_or_fail('video_metadata' in result, 
                                "Output missing 'video_metadata'")
            self.validate_or_fail('timeline' in result, 
                                "Output missing 'timeline'")
            self.validate_or_fail('ml_results' in result, 
                                "Output missing 'ml_results'")
            self.validate_or_fail('processing_metadata' in result, 
                                "Output missing 'processing_metadata'")
            
            video_id = result['video_id']
            video_metadata = result['video_metadata']
            timeline = result['timeline']
            ml_results = result['ml_results']
            processing_metadata = result['processing_metadata']
        
        # Validate video_id matches
        self.validate_or_fail(isinstance(video_id, str) and len(video_id) > 0, 
                            "Output video_id must be non-empty string")
        
        # Validate timeline structure
        if hasattr(timeline, '__dict__'):
            # Timeline object
            self.validate_or_fail(hasattr(timeline, 'video_id'), 
                                "Timeline missing 'video_id'")
            self.validate_or_fail(hasattr(timeline, 'duration'), 
                                "Timeline missing 'duration'")
            self.validate_or_fail(hasattr(timeline, 'entries'), 
                                "Timeline missing 'entries'")
            
            timeline_duration = timeline.duration
            timeline_entries = timeline.entries
        else:
            # Timeline dict
            self.validate_or_fail(isinstance(timeline, dict), 
                                "Timeline must be Timeline object or dict")
            self.validate_or_fail('duration' in timeline, 
                                "Timeline missing 'duration'")
            self.validate_or_fail('entries' in timeline, 
                                "Timeline missing 'entries'")
            
            timeline_duration = timeline['duration']
            timeline_entries = timeline['entries']
        
        # Validate timeline duration
        self.validate_or_fail(isinstance(timeline_duration, (int, float)), 
                            "Timeline duration must be numeric")
        self.validate_or_fail(timeline_duration > 0, 
                            f"Timeline duration must be positive, got {timeline_duration}")
        
        # Validate timeline entries
        self.validate_or_fail(isinstance(timeline_entries, list), 
                            "Timeline entries must be list")
        
        # Valid entry types
        valid_entry_types = {
            'object', 'speech', 'text', 'sticker', 'pose', 'face', 
            'gesture', 'scene_change', 'scene', 'emotion'
        }
        
        # Validate each entry and check chronological order
        last_timestamp = -1
        entry_type_counts = {}
        timestamp_groups = {}  # Track overlapping entries
        
        for i, entry in enumerate(timeline_entries):
            # Get entry data and type
            if hasattr(entry, '__dict__'):
                entry_type = entry.entry_type
                entry_data = entry.data
                start = entry.start
                if hasattr(start, 'seconds'):
                    start_seconds = start.seconds
                else:
                    start_seconds = float(start)
            else:
                self.validate_or_fail(isinstance(entry, dict), 
                                    f"Entry {i} must be TimelineEntry or dict")
                self.validate_or_fail('entry_type' in entry, 
                                    f"Entry {i} missing 'entry_type'")
                self.validate_or_fail('start' in entry, 
                                    f"Entry {i} missing 'start'")
                self.validate_or_fail('data' in entry,
                                    f"Entry {i} missing 'data'")
                
                entry_type = entry['entry_type']
                entry_data = entry['data']
                start = entry['start']
                if isinstance(start, dict) and 'seconds' in start:
                    start_seconds = start['seconds']
                else:
                    start_seconds = float(start)
            
            # Validate entry type
            self.validate_or_fail(entry_type in valid_entry_types, 
                                f"Entry {i} has invalid type '{entry_type}'")
            
            # Deep validate entry data structure
            self.validate_entry_data(entry_type, entry_data, i)
            
            # Check chronological order
            self.validate_or_fail(start_seconds >= last_timestamp, 
                                f"Entry {i} breaks chronological order: {start_seconds} < {last_timestamp}")
            last_timestamp = start_seconds
            
            # Track overlapping entries at same timestamp
            if start_seconds in timestamp_groups:
                timestamp_groups[start_seconds].append((i, entry_type))
            else:
                timestamp_groups[start_seconds] = [(i, entry_type)]
            
            # Count entry types
            entry_type_counts[entry_type] = entry_type_counts.get(entry_type, 0) + 1
        
        # Validate no duplicate entry types at same timestamp
        for timestamp, entries_at_time in timestamp_groups.items():
            if len(entries_at_time) > 1:
                entry_types_at_time = [e[1] for e in entries_at_time]
                if len(entry_types_at_time) != len(set(entry_types_at_time)):
                    # Same type at same timestamp - could be error
                    duplicates = [t for t in entry_types_at_time if entry_types_at_time.count(t) > 1]
                    self.validate_or_fail(
                        False,
                        f"Duplicate entry types at timestamp {timestamp}s: {set(duplicates)}"
                    )
        
        # Validate processing_metadata
        self.validate_or_fail(isinstance(processing_metadata, dict), 
                            "processing_metadata must be dict")
        self.validate_or_fail('timeline_entries' in processing_metadata, 
                            "processing_metadata missing 'timeline_entries'")
        self.validate_or_fail('entry_types' in processing_metadata, 
                            "processing_metadata missing 'entry_types'")
        self.validate_or_fail('duration' in processing_metadata, 
                            "processing_metadata missing 'duration'")
        
        # Validate metadata accuracy
        self.validate_or_fail(
            processing_metadata['timeline_entries'] == len(timeline_entries),
            f"timeline_entries mismatch: metadata says {processing_metadata['timeline_entries']}, actual {len(timeline_entries)}"
        )
        
        # Validate entry_types counts match
        metadata_types = processing_metadata['entry_types']
        self.validate_or_fail(isinstance(metadata_types, dict), 
                            "entry_types must be dict")
        
        for entry_type, count in entry_type_counts.items():
            self.validate_or_fail(
                entry_type in metadata_types,
                f"Entry type '{entry_type}' in timeline but not in metadata"
            )
            self.validate_or_fail(
                metadata_types[entry_type] == count,
                f"Entry type '{entry_type}' count mismatch: metadata says {metadata_types.get(entry_type)}, actual {count}"
            )
        
        # Check for extra types in metadata
        for entry_type in metadata_types:
            if entry_type not in entry_type_counts:
                self.validate_or_fail(
                    metadata_types[entry_type] == 0,
                    f"Entry type '{entry_type}' in metadata with count {metadata_types[entry_type]} but not in timeline"
                )
```

### 5.2 DataMergerContract

```python
class DataMergerContract(BaseServiceContract):
    """
    ULTRA-STRICT contract for data merger operations with comprehensive conflict resolution.
    Validates merging strategies, conflict detection, and data integrity.
    """
    
    # Define valid merge strategies
    MERGE_STRATEGIES = {
        'FIRST_WINS': 'Keep first value encountered',
        'LAST_WINS': 'Keep last value encountered', 
        'HIGHEST_CONFIDENCE': 'Keep value with highest confidence',
        'UNION': 'Combine all unique values',
        'FAIL_ON_CONFLICT': 'Fail if values differ'
    }
    
    def __init__(self):
        super().__init__("DataMergerContract")
        self.conflict_fields = {}
    
    def validate_input(self, *data_sources: Dict[str, Any], merge_strategy: str = 'LAST_WINS') -> None:
        """Validate input data sources and merge strategy with conflict detection"""
        
        self.validate_or_fail(len(data_sources) > 0,
                            "Must provide at least one data source")
        
        self.validate_or_fail(merge_strategy in self.MERGE_STRATEGIES,
                            f"Invalid merge strategy '{merge_strategy}'. Must be one of: {list(self.MERGE_STRATEGIES.keys())}")
        
        # Track identifiers and common fields for conflict detection
        identifiers = []
        field_sources = {}
        
        for i, source in enumerate(data_sources):
            self.validate_or_fail(isinstance(source, dict),
                                f"Data source {i} must be dict")
            
            # Extract and validate identifier
            video_id = source.get('video_id') or source.get('id')
            self.validate_or_fail(video_id is not None,
                                f"Data source {i} missing identifier ('video_id' or 'id')")
            identifiers.append(video_id)
            
            # Track which sources have which fields for conflict detection
            for field in source:
                if field not in field_sources:
                    field_sources[field] = []
                field_sources[field].append(i)
            
            # Validate timeline if present
            if 'timeline' in source:
                self._validate_timeline_structure(source['timeline'], i)
            
            # Validate ml_results if present
            if 'ml_results' in source:
                self._validate_ml_results_structure(source['ml_results'], i)
        
        # All identifiers must match (can't merge different videos)
        unique_ids = set(identifiers)
        self.validate_or_fail(len(unique_ids) == 1,
                            f"Cannot merge different videos. Found IDs: {unique_ids}")
        
        # Detect potential conflicts (fields present in multiple sources)
        self.conflict_fields = {
            field: sources 
            for field, sources in field_sources.items() 
            if len(sources) > 1
        }
        
        # If using FAIL_ON_CONFLICT, check for actual conflicts
        if merge_strategy == 'FAIL_ON_CONFLICT' and self.conflict_fields:
            # Check if conflicting fields have different values
            for field, sources in self.conflict_fields.items():
                values = [data_sources[i].get(field) for i in sources]
                # Simple equality check (could be enhanced for deep comparison)
                if not all(v == values[0] for v in values):
                    self.validate_or_fail(False,
                                        f"Conflict in field '{field}' with FAIL_ON_CONFLICT strategy")
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """Validate merged output with comprehensive integrity checks"""
        
        self.validate_or_fail(isinstance(result, dict),
                            "Merged result must be dict")
        
        # Must have identifier
        self.validate_or_fail('video_id' in result or 'id' in result,
                            "Merged result missing identifier")
        
        # Must have merge metadata
        self.validate_or_fail('merge_metadata' in result or 'processing_metadata' in result,
                            "Missing merge metadata")
        
        # Extract metadata (support both field names)
        metadata = result.get('merge_metadata') or result.get('processing_metadata', {})
        
        # Validate merge metadata completeness
        if 'merge_metadata' in result:
            self.validate_or_fail('source_count' in metadata,
                                "Merge metadata missing 'source_count'")
            self.validate_or_fail(isinstance(metadata['source_count'], int) and metadata['source_count'] > 0,
                                f"Invalid source_count: {metadata.get('source_count')}")
            
            self.validate_or_fail('conflicts_resolved' in metadata,
                                "Merge metadata missing 'conflicts_resolved'")
            self.validate_or_fail('merge_strategy' in metadata,
                                "Merge metadata missing 'merge_strategy'")
            
            # Validate strategy is valid
            if 'merge_strategy' in metadata:
                self.validate_or_fail(metadata['merge_strategy'] in self.MERGE_STRATEGIES,
                                    f"Invalid merge strategy in metadata: {metadata['merge_strategy']}")
        
        # Validate timeline merging if present
        if 'timeline' in result:
            self._validate_merged_timeline(result['timeline'])
        
        # Validate ML results merging if present
        if 'ml_results' in result:
            self._validate_merged_ml_results(result['ml_results'])
    
    def _validate_timeline_structure(self, timeline: Any, source_index: int) -> None:
        """Validate timeline structure in input source"""
        
        if hasattr(timeline, '__dict__'):
            # Timeline object
            self.validate_or_fail(hasattr(timeline, 'entries'),
                                f"Source {source_index}: Timeline missing 'entries'")
            self.validate_or_fail(hasattr(timeline, 'duration'),
                                f"Source {source_index}: Timeline missing 'duration'")
        else:
            # Timeline dict
            self.validate_or_fail(isinstance(timeline, dict),
                                f"Source {source_index}: Timeline must be object or dict")
            self.validate_or_fail('entries' in timeline,
                                f"Source {source_index}: Timeline missing 'entries'")
            self.validate_or_fail('duration' in timeline,
                                f"Source {source_index}: Timeline missing 'duration'")
    
    def _validate_ml_results_structure(self, ml_results: Any, source_index: int) -> None:
        """Validate ML results structure in input source"""
        
        self.validate_or_fail(isinstance(ml_results, dict),
                            f"Source {source_index}: ml_results must be dict")
        
        for model_name, result in ml_results.items():
            # Each result should have basic structure
            if hasattr(result, '__dict__'):
                self.validate_or_fail(hasattr(result, 'success'),
                                    f"Source {source_index}: ML result '{model_name}' missing 'success'")
            else:
                self.validate_or_fail('success' in result,
                                    f"Source {source_index}: ML result '{model_name}' missing 'success'")
    
    def _validate_merged_timeline(self, timeline: Any) -> None:
        """Validate timeline was properly merged with chronological consistency"""
        
        # Extract entries
        if hasattr(timeline, 'entries'):
            entries = timeline.entries
        elif isinstance(timeline, dict) and 'entries' in timeline:
            entries = timeline['entries']
        else:
            self.validate_or_fail(False, "Merged timeline missing entries")
            return
        
        self.validate_or_fail(isinstance(entries, list),
                            "Timeline entries must be list")
        
        # Check chronological order maintained
        last_timestamp = -1
        duplicate_check = {}
        
        for i, entry in enumerate(entries):
            # Extract timestamp
            timestamp = self._extract_entry_timestamp(entry)
            self.validate_or_fail(timestamp is not None,
                                f"Entry {i} has invalid timestamp")
            
            # Check chronological order
            self.validate_or_fail(timestamp >= last_timestamp,
                                f"Timeline merge broke chronological order at entry {i}: {timestamp} < {last_timestamp}")
            last_timestamp = timestamp
            
            # Check for exact duplicates at same timestamp
            entry_type = self._extract_entry_type(entry)
            entry_key = f"{timestamp}:{entry_type}"
            
            if entry_key in duplicate_check:
                # Same type at exact same timestamp - potential duplicate
                import logging
                logging.warning(f"Potential duplicate entry: {entry_type} at {timestamp}s")
            duplicate_check[entry_key] = i
    
    def _validate_merged_ml_results(self, ml_results: Dict) -> None:
        """Validate ML results were properly merged"""
        
        self.validate_or_fail(isinstance(ml_results, dict),
                            "Merged ml_results must be dict")
        
        successful_models = []
        failed_models = []
        
        for model_name, result in ml_results.items():
            # Extract success flag
            if hasattr(result, 'success'):
                success = result.success
            elif isinstance(result, dict) and 'success' in result:
                success = result['success']
            else:
                self.validate_or_fail(False,
                                    f"ML result '{model_name}' missing success flag")
            
            if success:
                successful_models.append(model_name)
            else:
                failed_models.append(model_name)
            
            # If result has merge tracking, validate it
            if hasattr(result, 'merge_source') or (isinstance(result, dict) and 'merge_source' in result):
                merge_source = result.merge_source if hasattr(result, 'merge_source') else result['merge_source']
                self.validate_or_fail(isinstance(merge_source, (int, str)),
                                    f"Invalid merge_source for {model_name}: {merge_source}")
        
        # Log merge summary for debugging
        if successful_models or failed_models:
            import logging
            logging.info(f"ML merge summary - Success: {successful_models}, Failed: {failed_models}")
    
    def _extract_entry_timestamp(self, entry: Any) -> float:
        """Extract timestamp from timeline entry"""
        
        if hasattr(entry, 'start'):
            start = entry.start
            if hasattr(start, 'seconds'):
                return start.seconds
            return float(start)
        elif isinstance(entry, dict) and 'start' in entry:
            start = entry['start']
            if isinstance(start, dict) and 'seconds' in start:
                return start['seconds']
            return float(start)
        return None
    
    def _extract_entry_type(self, entry: Any) -> str:
        """Extract entry type from timeline entry"""
        
        if hasattr(entry, 'entry_type'):
            return entry.entry_type
        elif isinstance(entry, dict) and 'entry_type' in entry:
            return entry['entry_type']
        return 'unknown'
```

### 5.3 ResultValidatorContract

```python
class ResultValidatorContract(BaseServiceContract):
    """
    ULTRA-STRICT contract for result validation using actual JSON Schema validation.
    Actually validates data against schema instead of just checking schema exists!
    """
    
    def __init__(self):
        super().__init__("ResultValidatorContract")
    
    def validate_input(self, result: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Validate inputs with proper JSON Schema structure validation"""
        
        # Validate result is dict
        self.validate_or_fail(isinstance(result, dict),
                            "Result must be dict")
        
        # Validate schema is valid JSON Schema
        self.validate_or_fail(isinstance(schema, dict),
                            "Schema must be dict")
        
        # Check for valid JSON Schema structure
        valid_schema_keywords = {
            'type', 'properties', 'required', 'items', 'additionalProperties',
            'minimum', 'maximum', 'minLength', 'maxLength', 'pattern',
            'enum', 'const', 'allOf', 'anyOf', 'oneOf', 'not',
            '$schema', '$ref', 'definitions', 'title', 'description'
        }
        
        schema_keys = set(schema.keys())
        self.validate_or_fail(
            len(schema_keys & valid_schema_keywords) > 0,
            f"Schema has no valid JSON Schema keywords. Found: {schema_keys}"
        )
        
        # If type is specified, validate it
        if 'type' in schema:
            valid_types = ['object', 'array', 'string', 'number', 'integer', 'boolean', 'null']
            schema_type = schema['type']
            if isinstance(schema_type, str):
                self.validate_or_fail(schema_type in valid_types,
                                    f"Invalid schema type: {schema_type}")
        
        # For object schemas, validate properties structure
        if schema.get('type') == 'object' or 'properties' in schema:
            self.validate_or_fail('properties' in schema,
                                "Object schema must have 'properties'")
            self.validate_or_fail(isinstance(schema['properties'], dict),
                                "'properties' must be dict")
            
            # If required is specified, validate it
            if 'required' in schema:
                self.validate_or_fail(isinstance(schema['required'], list),
                                    "'required' must be list")
                for req in schema['required']:
                    self.validate_or_fail(req in schema['properties'],
                                        f"Required field '{req}' not in properties")
    
    def validate_output(self, validation_result: Dict[str, Any]) -> None:
        """Validate the validation result with actual schema compliance check"""
        
        self.validate_or_fail(isinstance(validation_result, dict),
                            "Validation result must be dict")
        
        # Must have validation outcome
        self.validate_or_fail('valid' in validation_result,
                            "Validation result missing 'valid' flag")
        self.validate_or_fail(isinstance(validation_result['valid'], bool),
                            "'valid' must be boolean")
        
        # Must show what was validated
        self.validate_or_fail('schema_used' in validation_result or 'validation_type' in validation_result,
                            "Must indicate what schema/validation was used")
        
        # If validation failed, must have detailed errors
        if not validation_result['valid']:
            self.validate_or_fail('errors' in validation_result,
                                "Invalid result must include errors")
            
            errors = validation_result['errors']
            self.validate_or_fail(isinstance(errors, (list, dict)),
                                "Errors must be list or dict")
            
            if isinstance(errors, list):
                self.validate_or_fail(len(errors) > 0,
                                    "Errors list cannot be empty")
                for error in errors:
                    if isinstance(error, dict):
                        # Each error should have details
                        self.validate_or_fail('message' in error or 'description' in error,
                                            "Error must have message or description")
                        self.validate_or_fail('path' in error or 'field' in error or 'location' in error,
                                            "Error must indicate location")
            else:
                # Dict of errors
                self.validate_or_fail(len(errors) > 0,
                                    "Errors dict cannot be empty")
        
        # If valid, optionally include validation details
        else:
            # Should indicate what passed
            if 'validation_details' in validation_result:
                details = validation_result['validation_details']
                self.validate_or_fail(isinstance(details, dict),
                                    "Validation details must be dict")
        
        # Must have ability to trace what was validated
        self.validate_or_fail('data_hash' in validation_result or 
                            'data_size' in validation_result or
                            'validated_at' in validation_result or
                            'fields_checked' in validation_result,
                            "Should include validation metadata")
    
    def validate_schema_against_data(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actually validate data against schema (what was missing!).
        Returns validation result that passes output validation.
        """
        
        validation_result = {
            'valid': True,
            'errors': [],
            'schema_used': schema.get('title', 'provided_schema'),
            'data_size': len(str(data)),
            'validation_type': 'json_schema',
            'fields_checked': []
        }
        
        # Basic type checking
        if 'type' in schema:
            expected_type = schema['type']
            actual_type = type(data).__name__.replace('dict', 'object').replace('list', 'array')
            
            if expected_type != actual_type:
                validation_result['valid'] = False
                validation_result['errors'].append({
                    'message': f"Type mismatch: expected {expected_type}, got {actual_type}",
                    'path': '$',
                    'rule': 'type'
                })
                return validation_result
        
        # Validate required fields
        if schema.get('type') == 'object' and 'required' in schema:
            for field in schema['required']:
                validation_result['fields_checked'].append(field)
                if field not in data:
                    validation_result['valid'] = False
                    validation_result['errors'].append({
                        'message': f"Missing required field: {field}",
                        'path': f"$.{field}",
                        'rule': 'required'
                    })
        
        # Validate properties
        if 'properties' in schema:
            for prop, prop_schema in schema['properties'].items():
                if prop in data:
                    validation_result['fields_checked'].append(prop)
                    # Recursively validate nested structures
                    if isinstance(prop_schema, dict) and 'type' in prop_schema:
                        if prop_schema['type'] == 'object' and isinstance(data[prop], dict):
                            nested_result = self.validate_schema_against_data(data[prop], prop_schema)
                            if not nested_result['valid']:
                                for error in nested_result['errors']:
                                    error['path'] = f"$.{prop}{error['path'][1:]}"
                                    validation_result['errors'].append(error)
                                validation_result['valid'] = False
                        elif prop_schema['type'] == 'array' and isinstance(data[prop], list):
                            # Validate array items if schema provided
                            if 'items' in prop_schema:
                                for i, item in enumerate(data[prop]):
                                    item_result = self.validate_schema_against_data(item, prop_schema['items'])
                                    if not item_result['valid']:
                                        for error in item_result['errors']:
                                            error['path'] = f"$.{prop}[{i}]{error['path'][1:]}"
                                            validation_result['errors'].append(error)
                                        validation_result['valid'] = False
        
        # Check additionalProperties
        if schema.get('additionalProperties') == False and 'properties' in schema:
            allowed = set(schema['properties'].keys())
            actual = set(data.keys())
            extra = actual - allowed
            if extra:
                validation_result['valid'] = False
                validation_result['errors'].append({
                    'message': f"Additional properties not allowed: {extra}",
                    'path': '$',
                    'rule': 'additionalProperties'
                })
        
        return validation_result
```

### 5.5 ErrorHandlerContract

```python
class UniversalValueValidator:
    """
    Universal value validation framework with explicit range/content rules.
    Addresses shared blindness in compute contracts - validates types AND ranges/content.
    """
    
    # Define value constraints for ALL fields across all contracts
    VALUE_CONSTRAINTS = {
        'confidence': {
            'type': (int, float),
            'range': (0.0, 1.0),
            'required': True,
            'description': 'Confidence scores must be 0.0-1.0'
        },
        'timestamp': {
            'type': (int, float),
            'range': (0.0, float('inf')),
            'required': True,
            'description': 'Timestamps must be non-negative seconds'
        },
        'duration': {
            'type': (int, float),
            'range': (0.0, float('inf')),
            'required': True,
            'description': 'Duration must be positive seconds'
        },
        'bbox': {
            'type': list,
            'length': 4,
            'element_type': (int, float),
            'element_range': (0.0, 10000.0),  # Reasonable pixel bounds
            'required': True,
            'description': 'Bounding box [x, y, width, height] with valid pixel coordinates'
        },
        'landmarks': {
            'type': (int, list),
            'int_exact': 33,  # For MediaPipe pose landmarks
            'list_min_length': 1,
            'list_max_length': 468,  # MediaPipe face landmarks
            'required': True,
            'description': 'Landmarks count (int) or coordinates (list)'
        },
        'energy_level': {
            'type': (int, float),
            'range': (0.0, 1.0),
            'required': True,
            'description': 'Audio energy levels 0.0-1.0'
        },
        'scene_number': {
            'type': int,
            'range': (1, 10000),  # Reasonable scene count
            'required': True,
            'description': 'Scene numbers start from 1'
        },
        'frame_number': {
            'type': int,
            'range': (0, 100000),  # Reasonable frame count
            'required': True,
            'description': 'Frame numbers start from 0'
        },
        'track_id': {
            'type': (str, int),
            'string_min_length': 1,
            'int_range': (0, 100000),
            'required': True,
            'description': 'Track IDs must be non-empty string or positive int'
        }
    }
    
    # Context-aware empty array validation rules
    EMPTY_ARRAY_RULES = {
        'objectAnnotations': {
            'empty_valid_when': ['static_scene', 'no_movement', 'blank_frame'],
            'suspicious_when': ['movement_detected', 'high_energy_audio'],
            'description': 'Objects may be absent in static scenes'
        },
        'poses': {
            'empty_valid_when': ['no_humans', 'crowd_scene', 'low_quality_frame'],
            'suspicious_when': ['faces_detected', 'speech_detected'],
            'description': 'Poses may be absent without humans'
        },
        'faces': {
            'empty_valid_when': ['no_humans', 'back_view', 'crowd_scene'],
            'suspicious_when': ['poses_detected', 'speech_detected'],
            'description': 'Faces may be absent in back views or crowds'
        },
        'hands': {
            'empty_valid_when': ['no_humans', 'hands_not_visible', 'low_quality'],
            'suspicious_when': ['poses_detected', 'gesture_activity'],
            'description': 'Hands may be absent when not visible'
        },
        'textAnnotations': {
            'empty_valid_when': ['no_text_content', 'poor_ocr_quality'],
            'suspicious_when': ['clear_text_visible', 'ui_elements_detected'],
            'description': 'Text may be absent in videos without overlays'
        }
    }
    
    @classmethod
    def validate_value(cls, field_name: str, value: Any, context: str = "") -> tuple[bool, str]:
        """Universal value validation with explicit range/content rules"""
        if field_name not in cls.VALUE_CONSTRAINTS:
            return True, f"No constraints for {field_name}"  # Allow unknown fields
        
        constraints = cls.VALUE_CONSTRAINTS[field_name]
        prefix = f"{context} {field_name}" if context else field_name
        
        # Type validation
        if not isinstance(value, constraints['type']):
            return False, f"{prefix}: must be {constraints['type']}, got {type(value).__name__}"
        
        # Range validation for numbers
        if 'range' in constraints and isinstance(value, (int, float)):
            min_val, max_val = constraints['range']
            if not (min_val <= value <= max_val):
                return False, f"{prefix}: {value} out of range [{min_val}, {max_val}]"
        
        # Exact value validation for ints
        if 'int_exact' in constraints and isinstance(value, int):
            if value != constraints['int_exact']:
                return False, f"{prefix}: must be exactly {constraints['int_exact']}, got {value}"
        
        # String length validation
        if isinstance(value, str):
            if 'string_min_length' in constraints and len(value) < constraints['string_min_length']:
                return False, f"{prefix}: string too short, got length {len(value)}"
        
        # Integer range for track_id
        if 'int_range' in constraints and isinstance(value, int):
            min_val, max_val = constraints['int_range']
            if not (min_val <= value <= max_val):
                return False, f"{prefix}: {value} out of range [{min_val}, {max_val}]"
        
        # Array length validation
        if isinstance(value, list):
            if 'length' in constraints and len(value) != constraints['length']:
                return False, f"{prefix}: must have {constraints['length']} elements, got {len(value)}"
            
            if 'list_min_length' in constraints and len(value) < constraints['list_min_length']:
                return False, f"{prefix}: must have at least {constraints['list_min_length']} elements, got {len(value)}"
            
            if 'list_max_length' in constraints and len(value) > constraints['list_max_length']:
                return False, f"{prefix}: must have at most {constraints['list_max_length']} elements, got {len(value)}"
        
        # Array element validation
        if 'element_type' in constraints and isinstance(value, list):
            for i, element in enumerate(value):
                if not isinstance(element, constraints['element_type']):
                    return False, f"{prefix}[{i}]: must be {constraints['element_type']}, got {type(element).__name__}"
                
                if 'element_range' in constraints and isinstance(element, (int, float)):
                    min_val, max_val = constraints['element_range']
                    if not (min_val <= element <= max_val):
                        return False, f"{prefix}[{i}]: {element} out of range [{min_val}, {max_val}]"
        
        # Special bbox validation - ensure width/height are positive
        if field_name == 'bbox' and isinstance(value, list) and len(value) == 4:
            x, y, w, h = value
            if w <= 0 or h <= 0:
                return False, f"{prefix}: invalid dimensions w={w}, h={h} (must be positive)"
        
        return True, "Valid"
    
    @classmethod
    def validate_array_emptiness(cls, array_name: str, array_data: list, 
                                context: Dict[str, Any] = None) -> tuple[bool, str]:
        """Context-aware empty array validation"""
        if array_name not in cls.EMPTY_ARRAY_RULES:
            return True, f"No emptiness rules for {array_name}"
        
        context = context or {}
        rules = cls.EMPTY_ARRAY_RULES[array_name]
        
        if len(array_data) == 0:
            # Check if empty is expected given context
            for valid_condition in rules['empty_valid_when']:
                if context.get(valid_condition, False):
                    return True, f"Empty {array_name} valid: {valid_condition}"
            
            # Check if empty is suspicious
            for suspicious_condition in rules['suspicious_when']:
                if context.get(suspicious_condition, False):
                    return False, f"Empty {array_name} suspicious: {suspicious_condition} detected"
            
            # Default: empty is valid but note it
            return True, f"Empty {array_name} - {rules['description']}"
        
        return True, f"Non-empty {array_name}"
    
    @classmethod
    def validate_all_fields(cls, data: Dict[str, Any], context: str = "") -> tuple[bool, List[str]]:
        """Validate all fields in a data structure"""
        errors = []
        
        for field_name, value in data.items():
            is_valid, message = cls.validate_value(field_name, value, context)
            if not is_valid:
                errors.append(message)
        
        return len(errors) == 0, errors


class ErrorHandlerContract(BaseServiceContract):
    """
    ULTRA-STRICT contract for error handling with severity levels and categorization.
    """
    
    # Error severity levels (aligned with logging standards)
    SEVERITY_LEVELS = {
        'CRITICAL': 50,  # System failure, must abort
        'ERROR': 40,     # Operation failed, recovery needed
        'WARNING': 30,   # Issue detected, can continue
        'INFO': 20,      # Informational
        'DEBUG': 10      # Debug details
    }
    
    # Error categories for proper routing
    ERROR_CATEGORIES = {
        'CONTRACT_VIOLATION': 'Service contract broken',
        'VALIDATION': 'Data validation failure',
        'ML_ANALYSIS': 'ML model failure',
        'API': 'External API error',
        'FILESYSTEM': 'File operation error',
        'CONFIGURATION': 'Config error',
        'TIMELINE': 'Timeline processing error',
        'PROMPT': 'Prompt generation error',
        'UNKNOWN': 'Uncategorized error'
    }
    
    def __init__(self):
        super().__init__("ErrorHandlerContract")
    
    def validate_input(self, error: Any, context: Dict[str, Any], 
                      severity: str = None, category: str = None) -> None:
        """Validate error handler input with flexible error types"""
        
        # Error can be Exception, string, or dict
        if isinstance(error, Exception):
            # Standard exception
            self.validate_or_fail(hasattr(error, '__class__'),
                                "Exception must have __class__ attribute")
        elif isinstance(error, str):
            # String error (common in JavaScript/API responses)
            self.validate_or_fail(len(error) > 0,
                                "String error must be non-empty")
        elif isinstance(error, dict):
            # Structured error (from APIs, JSON responses)
            self.validate_or_fail('message' in error or 'error' in error or 'detail' in error,
                                "Dict error must have message/error/detail field")
        else:
            self.validate_or_fail(False,
                                f"Error must be Exception, str, or dict, got {type(error)}")
        
        # Context validation - MUST have operation info
        self.validate_or_fail(isinstance(context, dict),
                            "Context must be dict")
        self.validate_or_fail('operation' in context or 'function' in context or 'service' in context,
                            "Context MUST identify the operation/function/service")
        self.validate_or_fail('timestamp' in context or 'time' in context or 'occurred_at' in context,
                            "Context MUST include timestamp")
        
        # Optional but recommended context fields
        recommended_fields = ['video_id', 'user_id', 'request_id', 'environment']
        missing_recommended = [f for f in recommended_fields if f not in context]
        if missing_recommended:
            import logging
            logging.warning(f"Context missing recommended fields: {missing_recommended}")
        
        # Severity validation if provided
        if severity is not None:
            self.validate_or_fail(severity in self.SEVERITY_LEVELS,
                                f"Invalid severity: {severity}. Must be one of {list(self.SEVERITY_LEVELS.keys())}")
        
        # Category validation if provided
        if category is not None:
            self.validate_or_fail(category in self.ERROR_CATEGORIES,
                                f"Invalid category: {category}. Must be one of {list(self.ERROR_CATEGORIES.keys())}")
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """Validate error handler output with comprehensive tracking"""
        
        self.validate_or_fail(isinstance(result, dict),
                            "Error handler result must be dict")
        
        # MUST indicate if error was handled
        self.validate_or_fail('handled' in result,
                            "Result MUST indicate if error was handled")
        self.validate_or_fail(isinstance(result['handled'], bool),
                            "'handled' must be boolean")
        
        # MUST have error identification
        self.validate_or_fail('error_id' in result or 'error_code' in result or 'trace_id' in result,
                            "Result MUST include error identifier")
        
        # MUST have severity level
        self.validate_or_fail('severity' in result,
                            "Result MUST include severity level")
        self.validate_or_fail(result['severity'] in self.SEVERITY_LEVELS,
                            f"Invalid severity in result: {result.get('severity')}")
        
        # MUST have error category
        self.validate_or_fail('category' in result,
                            "Result MUST include error category")
        self.validate_or_fail(result['category'] in self.ERROR_CATEGORIES,
                            f"Invalid category in result: {result.get('category')}")
        
        # If handled, MUST describe recovery
        if result.get('handled'):
            self.validate_or_fail('recovery_action' in result or 'resolution' in result,
                                "Handled errors MUST describe recovery action")
            self.validate_or_fail('can_retry' in result,
                                "Handled errors MUST indicate if retry is possible")
        
        # If not handled, MUST provide escalation info
        else:
            self.validate_or_fail('escalation' in result or 'next_steps' in result,
                                "Unhandled errors MUST provide escalation path")
            self.validate_or_fail('debug_info' in result or 'stack_trace' in result,
                                "Unhandled errors MUST include debug information")
        
        # MUST have logging/tracking info
        self.validate_or_fail('logged' in result,
                            "Result MUST indicate if error was logged")
        self.validate_or_fail('log_location' in result or 'log_id' in result,
                            "Result MUST indicate where error was logged")
        
        # Optional but valuable metrics
        if 'metrics' in result:
            metrics = result['metrics']
            self.validate_or_fail(isinstance(metrics, dict),
                                "Metrics must be dict")
            # Could track: error_count, retry_count, duration, impact_score
    
    def categorize_error(self, error: Any) -> str:
        """Determine error category from error type/content"""
        
        if isinstance(error, Exception):
            error_class = error.__class__.__name__
            
            # Map exception types to categories
            category_map = {
                'ValidationError': 'VALIDATION',
                'TimelineError': 'TIMELINE',
                'MLAnalysisError': 'ML_ANALYSIS',
                'APIError': 'API',
                'FileSystemError': 'FILESYSTEM',
                'ConfigurationError': 'CONFIGURATION',
                'ServiceContractViolation': 'CONTRACT_VIOLATION',
                'PromptError': 'PROMPT'
            }
            
            return category_map.get(error_class, 'UNKNOWN')
        
        elif isinstance(error, str):
            # Analyze string for keywords
            error_lower = error.lower()
            if 'validation' in error_lower or 'invalid' in error_lower:
                return 'VALIDATION'
            elif 'api' in error_lower or 'request' in error_lower:
                return 'API'
            elif 'file' in error_lower or 'path' in error_lower:
                return 'FILESYSTEM'
            elif 'config' in error_lower:
                return 'CONFIGURATION'
            else:
                return 'UNKNOWN'
        
        elif isinstance(error, dict):
            # Check error type field if present
            return error.get('category', error.get('type', 'UNKNOWN'))
        
        return 'UNKNOWN'
    
    def determine_severity(self, error: Any, context: Dict[str, Any]) -> str:
        """Determine error severity from error and context"""
        
        # Critical if system can't continue
        if isinstance(error, Exception):
            if any(critical in error.__class__.__name__ for critical in ['Critical', 'Fatal', 'Abort']):
                return 'CRITICAL'
        
        # Check context for severity hints
        if context.get('severity'):
            return context['severity']
        
        # Category-based severity defaults
        category = self.categorize_error(error)
        severity_defaults = {
            'CONTRACT_VIOLATION': 'ERROR',
            'VALIDATION': 'ERROR',
            'ML_ANALYSIS': 'ERROR',
            'API': 'WARNING',
            'FILESYSTEM': 'ERROR',
            'CONFIGURATION': 'CRITICAL',
            'TIMELINE': 'ERROR',
            'PROMPT': 'WARNING',
            'UNKNOWN': 'WARNING'
        }
        
        return severity_defaults.get(category, 'ERROR')
```

### 5.6 ProfessionalBlockCrossValidator

```python
class ProfessionalBlockCrossValidator(BaseServiceContract):
    """
    ULTRA-STRICT validation for 6-block professional outputs with semantic-aware
    dynamic confidence calculation and cross-block consistency enforcement.
    
    Addresses critical issues:
    - Replaces hardcoded confidence values with calculated ones
    - Enforces configurable tolerances per field
    - Validates confidence makes semantic sense relative to data
    - Ensures consistency across all 6 blocks
    """
    
    def __init__(self):
        super().__init__("ProfessionalBlockCrossValidator")
        
        # Define semantic rules and tolerances for each analysis type
        self.SEMANTIC_RULES = {
            'visualOverlay': {
                'min_data_points': 3,  # Need at least 3 overlays for 0.7+ confidence
                'required_blocks': [
                    'visualOverlayCoreMetrics',
                    'visualOverlayDynamics',
                    'visualOverlayInteractions',
                    'visualOverlayKeyEvents',
                    'visualOverlayPatterns',
                    'visualOverlayQuality'
                ],
                'confidence_factors': {
                    'data_completeness': 0.4,
                    'temporal_coverage': 0.3,
                    'cross_modal_alignment': 0.2,
                    'value_consistency': 0.1
                },
                'field_tolerances': {
                    'overlayDensity': 0.1,      # 10% tolerance
                    'rhythmConsistency': 0.15,   # 15% tolerance  
                    'crossModalCoherence': 0.2   # 20% tolerance
                },
                'confidence_bounds': (0.3, 0.95)
            },
            'emotionalJourney': {
                'min_data_points': 5,  # Need at least 5 emotion points
                'required_blocks': [
                    'emotionalJourneyCoreMetrics',
                    'emotionalJourneyDynamics',
                    'emotionalJourneyInteractions',
                    'emotionalJourneyKeyEvents',
                    'emotionalJourneyPatterns',
                    'emotionalJourneyQuality'
                ],
                'confidence_factors': {
                    'emotion_detection_quality': 0.5,
                    'temporal_coverage': 0.3,
                    'transition_logic': 0.2
                },
                'field_tolerances': {
                    'emotionalIntensity': 0.2,    # 20% tolerance
                    'stabilityScore': 0.1,         # 10% tolerance
                    'valenceShifts': 0.25          # 25% tolerance for emotion changes
                },
                'confidence_bounds': (0.4, 0.95)
            },
            'creativeDensity': {
                'min_data_points': 10,  # Need substantial data for density analysis
                'required_blocks': [
                    'densityCoreMetrics',
                    'densityDynamics',
                    'densityInteractions',
                    'densityKeyEvents',
                    'densityPatterns',
                    'densityQuality'
                ],
                'confidence_factors': {
                    'element_variety': 0.3,
                    'temporal_distribution': 0.3,
                    'peak_detection': 0.2,
                    'dead_zone_accuracy': 0.2
                },
                'field_tolerances': {
                    'avgDensity': 0.15,           # 15% tolerance
                    'peakDensity': 0.2,           # 20% tolerance
                    'densityVariance': 0.25       # 25% tolerance
                },
                'confidence_bounds': (0.5, 0.95)
            },
            'scenePacing': {
                'min_data_points': 2,  # At least 2 scenes
                'required_blocks': [
                    'scenePacingCoreMetrics',
                    'scenePacingDynamics',
                    'scenePacingInteractions',
                    'scenePacingKeyEvents',
                    'scenePacingPatterns',
                    'scenePacingQuality'
                ],
                'confidence_factors': {
                    'scene_detection_quality': 0.4,
                    'transition_detection': 0.3,
                    'rhythm_consistency': 0.3
                },
                'field_tolerances': {
                    'avgSceneDuration': 0.075,    # 7.5% tolerance (as originally mentioned)
                    'sceneChangeRate': 0.1,       # 10% tolerance
                    'rhythmConsistency': 0.15     # 15% tolerance
                },
                'confidence_bounds': (0.4, 0.95)
            }
            # Add more analysis types as needed
        }
        
        # Cross-block validation rules
        self.CROSS_BLOCK_RULES = {
            'max_confidence_deviation': 0.2,  # Max 20% deviation between blocks
            'required_confidence_field': True,  # All blocks must have confidence
            'min_confidence_threshold': 0.1,   # No confidence below 0.1
            'max_confidence_threshold': 0.95   # No confidence above 0.95
        }
    
    def validate_input(self, analysis_type: str, blocks: Dict[str, Any]) -> None:
        """Validate input parameters before processing"""
        
        self.validate_or_fail(analysis_type in self.SEMANTIC_RULES,
                            f"Unknown analysis type: {analysis_type}")
        
        self.validate_or_fail(isinstance(blocks, dict),
                            "Blocks must be a dictionary")
        
        self.validate_or_fail(len(blocks) > 0,
                            "Blocks dictionary cannot be empty")
    
    def validate_output(self, analysis_type: str, blocks: Dict[str, Any]) -> None:
        """
        Comprehensive validation of 6-block professional output with:
        1. Structure validation (all required blocks present)
        2. Confidence field validation (all blocks have valid confidence)
        3. Semantic confidence validation (confidence makes sense for data)
        4. Cross-block consistency (confidence values are consistent)
        5. Field tolerance validation (values within acceptable ranges)
        """
        
        rules = self.SEMANTIC_RULES.get(analysis_type)
        self.validate_or_fail(rules is not None,
                            f"No validation rules for analysis type: {analysis_type}")
        
        # 1. Validate all required blocks are present
        for required_block in rules['required_blocks']:
            self.validate_or_fail(required_block in blocks,
                                f"Missing required block: {required_block}")
        
        # 2. Validate all blocks have confidence fields
        for block_name, block_data in blocks.items():
            self.validate_or_fail('confidence' in block_data,
                                f"Block {block_name} missing required 'confidence' field")
            
            confidence = block_data['confidence']
            self.validate_or_fail(isinstance(confidence, (int, float)),
                                f"Block {block_name} confidence must be numeric, got {type(confidence)}")
            
            min_conf, max_conf = rules['confidence_bounds']
            self.validate_or_fail(min_conf <= confidence <= max_conf,
                                f"Block {block_name} confidence {confidence} outside bounds [{min_conf}, {max_conf}]")
        
        # 3. Validate semantic confidence (confidence matches data quality)
        self._validate_semantic_confidence(analysis_type, blocks, rules)
        
        # 4. Validate cross-block consistency
        self._validate_cross_block_consistency(blocks)
        
        # 5. Validate field tolerances
        self._validate_field_tolerances(blocks, rules)
    
    def _validate_semantic_confidence(self, analysis_type: str, blocks: Dict, rules: Dict) -> None:
        """
        Validate that confidence values make semantic sense relative to the data.
        High confidence requires sufficient data quality and completeness.
        """
        
        # Count actual data points across all blocks
        data_point_count = self._count_data_points(blocks)
        
        # Check minimum data requirements
        min_required = rules['min_data_points']
        if data_point_count < min_required:
            # With insufficient data, confidence should be limited
            max_allowed_confidence = 0.5 + (0.2 * (data_point_count / min_required))
            
            for block_name, block_data in blocks.items():
                confidence = block_data.get('confidence', 0)
                self.validate_or_fail(
                    confidence <= max_allowed_confidence,
                    f"Block {block_name} claims {confidence} confidence but only has "
                    f"{data_point_count} data points (need {min_required} for high confidence)"
                )
        
        # Validate confidence aligns with data completeness
        for block_name, block_data in blocks.items():
            confidence = block_data.get('confidence', 0)
            
            # Check for sparse data claiming high confidence
            if self._is_sparse_data(block_data):
                self.validate_or_fail(
                    confidence <= 0.7,
                    f"Block {block_name} has sparse data but claims {confidence} confidence"
                )
            
            # Check for contradictory data claiming high confidence
            if self._has_contradictions(block_data):
                self.validate_or_fail(
                    confidence <= 0.6,
                    f"Block {block_name} has contradictory values but claims {confidence} confidence"
                )
            
            # Validate confidence calculation factors
            calculated_confidence = self._calculate_expected_confidence(
                analysis_type, block_data, rules
            )
            
            # Allow 30% deviation from calculated confidence
            deviation = abs(confidence - calculated_confidence)
            self.validate_or_fail(
                deviation <= 0.3,
                f"Block {block_name} confidence {confidence} deviates {deviation:.2f} "
                f"from expected {calculated_confidence:.2f} based on data quality"
            )
    
    def _validate_cross_block_consistency(self, blocks: Dict) -> None:
        """
        Ensure confidence values are consistent across all blocks.
        All blocks should have similar confidence if analyzing the same video.
        """
        
        confidences = [block.get('confidence', 0) for block in blocks.values()]
        
        # Calculate statistics
        avg_confidence = mean(confidences)
        confidence_std = stdev(confidences) if len(confidences) > 1 else 0
        
        # Check maximum deviation
        max_deviation = self.CROSS_BLOCK_RULES['max_confidence_deviation']
        
        for block_name, block_data in blocks.items():
            confidence = block_data.get('confidence', 0)
            deviation = abs(confidence - avg_confidence)
            
            self.validate_or_fail(
                deviation <= max_deviation,
                f"Block {block_name} confidence {confidence} deviates {deviation:.2f} "
                f"from average {avg_confidence:.2f} (max allowed: {max_deviation})"
            )
        
        # Check for outliers using z-score
        if confidence_std > 0:
            for block_name, block_data in blocks.items():
                confidence = block_data.get('confidence', 0)
                z_score = abs((confidence - avg_confidence) / confidence_std)
                
                self.validate_or_fail(
                    z_score <= 2.5,  # 2.5 sigma rule
                    f"Block {block_name} confidence {confidence} is statistical outlier "
                    f"(z-score: {z_score:.2f})"
                )
    
    def _validate_field_tolerances(self, blocks: Dict, rules: Dict) -> None:
        """
        Validate that field values are within specified tolerances.
        This ensures consistency in metrics across blocks.
        """
        
        field_tolerances = rules.get('field_tolerances', {})
        
        for field_name, tolerance in field_tolerances.items():
            field_values = []
            
            # Collect field values from all blocks
            for block_data in blocks.values():
                if field_name in block_data:
                    value = block_data[field_name]
                    if isinstance(value, (int, float)):
                        field_values.append(value)
            
            if len(field_values) > 1:
                # Check that all values are within tolerance of the mean
                field_mean = mean(field_values)
                
                for block_name, block_data in blocks.items():
                    if field_name in block_data:
                        value = block_data[field_name]
                        if isinstance(value, (int, float)):
                            relative_deviation = abs(value - field_mean) / max(field_mean, 0.001)
                            
                            self.validate_or_fail(
                                relative_deviation <= tolerance,
                                f"Block {block_name} field '{field_name}' value {value} "
                                f"deviates {relative_deviation:.2%} from mean {field_mean:.2f} "
                                f"(tolerance: {tolerance:.1%})"
                            )
    
    def _count_data_points(self, blocks: Dict) -> int:
        """Count total meaningful data points across all blocks"""
        
        count = 0
        for block_data in blocks.values():
            # Count arrays/lists
            for key, value in block_data.items():
                if isinstance(value, list) and len(value) > 0:
                    count += len(value)
                elif isinstance(value, dict):
                    # Count nested data structures
                    count += len(value)
                elif isinstance(value, (int, float)) and key != 'confidence':
                    # Count numeric metrics
                    count += 1
        
        return count
    
    def _is_sparse_data(self, block_data: Dict) -> bool:
        """Check if block has sparse/minimal data"""
        
        # Count non-empty, non-metadata fields
        meaningful_fields = 0
        for key, value in block_data.items():
            if key in ['confidence', 'timestamp', 'metadata']:
                continue
            
            if isinstance(value, list) and len(value) > 0:
                meaningful_fields += 1
            elif isinstance(value, dict) and len(value) > 0:
                meaningful_fields += 1
            elif isinstance(value, (str, int, float)) and value:
                meaningful_fields += 1
        
        return meaningful_fields < 3
    
    def _has_contradictions(self, block_data: Dict) -> bool:
        """Check for contradictory values in block data"""
        
        contradictions = []
        
        # Example contradiction checks
        if 'totalCount' in block_data and 'items' in block_data:
            if isinstance(block_data['items'], list):
                if block_data['totalCount'] != len(block_data['items']):
                    contradictions.append("totalCount doesn't match items length")
        
        if 'average' in block_data and 'values' in block_data:
            if isinstance(block_data['values'], list) and block_data['values']:
                calculated_avg = mean(block_data['values'])
                if abs(block_data['average'] - calculated_avg) > 0.1:
                    contradictions.append("average doesn't match calculated mean")
        
        if 'min' in block_data and 'max' in block_data:
            if block_data['min'] > block_data['max']:
                contradictions.append("min is greater than max")
        
        return len(contradictions) > 0
    
    def _calculate_expected_confidence(self, analysis_type: str, block_data: Dict, rules: Dict) -> float:
        """
        Calculate expected confidence based on data quality factors.
        This provides a baseline for semantic validation.
        """
        
        confidence_scores = {}
        factors = rules['confidence_factors']
        
        # Evaluate each factor
        for factor, weight in factors.items():
            if factor == 'data_completeness':
                # Check how many fields have data
                filled_fields = sum(1 for v in block_data.values() 
                                  if v is not None and v != [] and v != {})
                total_fields = len(block_data)
                score = filled_fields / max(total_fields, 1)
                
            elif factor == 'temporal_coverage':
                # Check if data covers the full timeline
                if 'timelineEntries' in block_data:
                    score = min(1.0, len(block_data['timelineEntries']) / 10)
                else:
                    score = 0.5  # Default if no timeline data
                    
            elif factor == 'cross_modal_alignment':
                # Check multimodal coherence
                if 'crossModalCoherence' in block_data:
                    score = block_data['crossModalCoherence']
                else:
                    score = 0.5
                    
            elif factor == 'value_consistency':
                # Check if values are within expected ranges
                score = 0.8 if not self._has_contradictions(block_data) else 0.3
                
            elif factor == 'emotion_detection_quality':
                # For emotional journey
                if 'detectionConfidence' in block_data:
                    score = block_data['detectionConfidence']
                else:
                    score = 0.5
                    
            elif factor == 'transition_logic':
                # Check if transitions make sense
                if 'transitions' in block_data:
                    score = min(1.0, len(block_data['transitions']) / 5)
                else:
                    score = 0.5
                    
            else:
                # Default score for unknown factors
                score = 0.5
            
            confidence_scores[factor] = score * weight
        
        # Calculate weighted average
        base_confidence = sum(confidence_scores.values())
        
        # Apply penalties
        if self._is_sparse_data(block_data):
            base_confidence *= 0.7
        
        if self._has_contradictions(block_data):
            base_confidence *= 0.8
        
        # Apply bounds
        min_conf, max_conf = rules['confidence_bounds']
        return round(min(max_conf, max(min_conf, base_confidence)), 2)
    
    def calculate_dynamic_confidence(self, analysis_type: str, block_data: Dict) -> float:
        """
        Public method to calculate confidence for a block based on its data.
        Can be used by precompute functions to replace hardcoded confidence values.
        """
        
        rules = self.SEMANTIC_RULES.get(analysis_type)
        if not rules:
            return 0.5  # Default confidence if no rules defined
        
        return self._calculate_expected_confidence(analysis_type, block_data, rules)
```

### 5.7 TypeValidator

```python
from typing import Protocol, runtime_checkable, Any, Union, Tuple
import sys
import logging

logger = logging.getLogger(__name__)

@runtime_checkable
class ArrayLike(Protocol):
    """Protocol for array-like objects (list, numpy array, etc.)"""
    def __len__(self) -> int: ...
    def __getitem__(self, key: int) -> Any: ...

@runtime_checkable  
class DictLike(Protocol):
    """Protocol for dict-like objects"""
    def keys(self) -> Any: ...
    def values(self) -> Any: ...
    def __getitem__(self, key: str) -> Any: ...

class TypeValidator:
    """
    Advanced type validation beyond isinstance() for service contracts.
    
    Handles:
    - NumPy arrays and Pandas DataFrames
    - String encoding validation
    - Duck typing through protocols
    - ML-specific frame array validation
    """
    
    @staticmethod
    def is_string_like(value: Any, encoding: str = 'utf-8') -> Tuple[bool, str]:
        """
        Validate string-like objects with encoding check.
        
        Args:
            value: Value to validate
            encoding: Expected encoding (default 'utf-8')
            
        Returns:
            (is_valid, message) tuple
        """
        # Check basic string types
        if isinstance(value, str):
            try:
                # Verify encoding by encoding/decoding round-trip
                _ = value.encode(encoding).decode(encoding)
                return True, "Valid UTF-8 string"
            except UnicodeError as e:
                return False, f"String has invalid {encoding} encoding: {e}"
        
        # Check bytes that could be decoded
        if isinstance(value, bytes):
            try:
                decoded = value.decode(encoding)
                return True, f"Valid {encoding} bytes (decoded length: {len(decoded)})"
            except UnicodeDecodeError as e:
                return False, f"Bytes cannot be decoded as {encoding}: {e}"
        
        # Check for string subclasses
        if isinstance(value, str):
            return True, f"String subclass: {type(value).__name__}"
        
        return False, f"Not a string-like type: {type(value).__name__}"
    
    @staticmethod
    def is_array_like(value: Any, min_length: int = 0, max_length: int = None) -> Tuple[bool, str]:
        """
        Validate array-like objects (list, tuple, numpy array, pandas Series/DataFrame).
        
        Args:
            value: Value to validate
            min_length: Minimum required length
            max_length: Maximum allowed length
            
        Returns:
            (is_valid, message) tuple
        """
        # Fast path for common types
        if isinstance(value, (list, tuple)):
            length = len(value)
            if min_length > 0 and length < min_length:
                return False, f"{type(value).__name__} too short: {length} < {min_length}"
            if max_length and length > max_length:
                return False, f"{type(value).__name__} too long: {length} > {max_length}"
            return True, f"Standard {type(value).__name__} (length: {length})"
        
        # Check numpy arrays
        if 'numpy' in sys.modules:
            import numpy as np
            if isinstance(value, np.ndarray):
                shape_str = 'x'.join(str(d) for d in value.shape)
                if min_length > 0 and value.size < min_length:
                    return False, f"NumPy array too small: {value.size} < {min_length}"
                if max_length and value.size > max_length:
                    return False, f"NumPy array too large: {value.size} > {max_length}"
                return True, f"NumPy array (shape={shape_str}, dtype={value.dtype})"
        
        # Check pandas Series/DataFrame
        if 'pandas' in sys.modules:
            import pandas as pd
            if isinstance(value, pd.Series):
                if min_length > 0 and len(value) < min_length:
                    return False, f"Pandas Series too short: {len(value)} < {min_length}"
                if max_length and len(value) > max_length:
                    return False, f"Pandas Series too long: {len(value)} > {max_length}"
                return True, f"Pandas Series (length: {len(value)})"
            if isinstance(value, pd.DataFrame):
                rows, cols = value.shape
                if min_length > 0 and rows < min_length:
                    return False, f"DataFrame too few rows: {rows} < {min_length}"
                if max_length and rows > max_length:
                    return False, f"DataFrame too many rows: {rows} > {max_length}"
                return True, f"Pandas DataFrame (shape: {rows}x{cols})"
        
        # Duck typing - check if it behaves like an array
        if isinstance(value, ArrayLike):
            try:
                length = len(value)
                if min_length > 0 and length < min_length:
                    return False, f"Array-like too short: {length} < {min_length}"
                if max_length and length > max_length:
                    return False, f"Array-like too long: {length} > {max_length}"
                return True, f"Array-like object (duck typed, length: {length})"
            except:
                return True, "Array-like object (duck typed)"
        
        return False, f"Not array-like: {type(value).__name__}"
    
    @staticmethod
    def is_dict_like(value: Any, required_keys: list = None, strict: bool = False) -> Tuple[bool, str]:
        """
        Validate dict-like objects.
        
        Args:
            value: Value to validate
            required_keys: List of keys that must be present
            strict: If True, only accept actual dict instances
            
        Returns:
            (is_valid, message) tuple
        """
        if strict:
            if not isinstance(value, dict):
                return False, f"Not a dict (strict mode): {type(value).__name__}"
        
        # Accept dict subclasses and dict-like objects
        if isinstance(value, dict):
            dict_type = "Standard dict"
        elif isinstance(value, DictLike):
            dict_type = "Dict-like object (duck typed)"
        else:
            return False, f"Not dict-like: {type(value).__name__}"
        
        # Check required keys
        if required_keys:
            try:
                keys = set(value.keys())
                missing = set(required_keys) - keys
                if missing:
                    return False, f"{dict_type} missing required keys: {missing}"
            except Exception as e:
                return False, f"Cannot check keys: {e}"
        
        try:
            num_keys = len(value.keys()) if hasattr(value, 'keys') else len(value)
            return True, f"{dict_type} ({num_keys} keys)"
        except:
            return True, dict_type
    
    @staticmethod
    def validate_frame_array(frames: Any) -> Tuple[bool, str]:
        """
        Specific validation for frame arrays used in ML pipeline.
        
        Expects either:
        - numpy array of shape (batch, height, width, channels)
        - list of numpy arrays each of shape (height, width, channels)
        
        Returns:
            (is_valid, message) tuple
        """
        # Must be array-like
        is_array, msg = TypeValidator.is_array_like(frames)
        if not is_array:
            return False, f"Frames must be array-like: {msg}"
        
        # Empty frames array is invalid
        if hasattr(frames, '__len__') and len(frames) == 0:
            return False, "Empty frames array"
        
        # Check if numpy array with correct properties
        if 'numpy' in sys.modules:
            import numpy as np
            
            # Single numpy array (batch of frames)
            if isinstance(frames, np.ndarray):
                shape = frames.shape
                if len(shape) == 4:  # (batch, height, width, channels)
                    batch, h, w, c = shape
                    if c not in [1, 3, 4]:  # Grayscale, RGB, or RGBA
                        return False, f"Invalid channel count: {c} (expected 1, 3, or 4)"
                    if frames.dtype not in [np.uint8, np.float32, np.float64]:
                        return False, f"Invalid dtype: {frames.dtype} (expected uint8 or float32/64)"
                    return True, f"Valid frame batch array (shape: {batch}x{h}x{w}x{c})"
                else:
                    return False, f"Invalid frame array shape: {shape} (expected 4D)"
            
            # List of numpy arrays (individual frames)
            if isinstance(frames, (list, tuple)):
                if len(frames) > 0:
                    first = frames[0]
                    if isinstance(first, np.ndarray):
                        shape = first.shape
                        if len(shape) == 3:  # (height, width, channels)
                            h, w, c = shape
                            if c not in [1, 3, 4]:
                                return False, f"Invalid channel count: {c} (expected 1, 3, or 4)"
                            
                            # Check all frames have same shape
                            for i, frame in enumerate(frames[1:5], 1):  # Check first 5
                                if not isinstance(frame, np.ndarray):
                                    return False, f"Frame {i} is not numpy array"
                                if frame.shape != shape:
                                    return False, f"Frame {i} shape {frame.shape} != first frame {shape}"
                            
                            return True, f"Valid frame list ({len(frames)} frames, shape: {h}x{w}x{c})"
                        else:
                            return False, f"Invalid frame shape: {shape} (expected 3D)"
        
        # If numpy not available, accept any array-like
        return True, "Frame array accepted (numpy not available for validation)"
    
    @staticmethod
    def validate_numeric(value: Any, min_val: float = None, max_val: float = None, 
                        allow_nan: bool = False, allow_inf: bool = False) -> Tuple[bool, str]:
        """
        Validate numeric values with range and special value checks.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            allow_nan: Whether to allow NaN values
            allow_inf: Whether to allow infinite values
            
        Returns:
            (is_valid, message) tuple
        """
        # Check if numeric
        if 'numpy' in sys.modules:
            import numpy as np
            numeric_types = (int, float, np.integer, np.floating)
        else:
            numeric_types = (int, float)
        
        if not isinstance(value, numeric_types):
            return False, f"Not numeric: {type(value).__name__}"
        
        # Convert to float for checks
        try:
            num_val = float(value)
        except (ValueError, TypeError) as e:
            return False, f"Cannot convert to float: {e}"
        
        # Check for special values
        import math
        if math.isnan(num_val):
            if not allow_nan:
                return False, "NaN values not allowed"
            return True, "Valid NaN"
        
        if math.isinf(num_val):
            if not allow_inf:
                return False, "Infinite values not allowed"
            return True, f"Valid infinity ({'positive' if num_val > 0 else 'negative'})"
        
        # Check range
        if min_val is not None and num_val < min_val:
            return False, f"Value {num_val} below minimum {min_val}"
        if max_val is not None and num_val > max_val:
            return False, f"Value {num_val} above maximum {max_val}"
        
        return True, f"Valid numeric ({type(value).__name__}: {num_val})"


### 5.8 Enhanced BaseServiceContract with TypeValidator

```python
# This shows how BaseServiceContract in base.py should be enhanced
# to include TypeValidator capabilities for all existing contracts

class BaseServiceContract:
    """
    Base class for all service contracts with enhanced type validation.
    
    Now includes TypeValidator integration so ALL existing contracts
    automatically get advanced type checking capabilities without changes.
    """
    
    def __init__(self, contract_name: str):
        """
        Initialize base contract with type validator.
        """
        self.contract_name = contract_name
        # Add TypeValidator to base class so all contracts have it
        self.type_validator = TypeValidator()
    
    def validate_or_fail(self, condition: bool, error_message: str) -> None:
        """
        Core validation method used by ALL contracts.
        (Original method unchanged for compatibility)
        """
        if not condition:
            raise ServiceContractViolation(
                error_message,
                component=self.contract_name
            )
    
    def validate_input(self, *args, **kwargs) -> None:
        """
        Base validate_input - contracts override with specific signatures.
        """
        pass  # Subclasses implement their own
    
    def validate_output(self, *args, **kwargs) -> None:
        """
        Base validate_output - contracts override with specific signatures.
        """
        pass  # Subclasses implement their own
    
    # NEW METHODS ADDED TO BASE CLASS - All contracts get these automatically
    
    def validate_type(self, value: Any, expected: str, name: str, **kwargs) -> None:
        """
        Advanced type validation beyond isinstance.
        Available to all contracts that inherit from BaseServiceContract.
        
        Args:
            value: Value to check
            expected: Type category - 'string', 'array', 'dict', 'frames', 'numeric'
            name: Field name for error messages
            **kwargs: Additional validation parameters
        
        Example usage in a contract:
            # In TimelineBuilderContract
            self.validate_type(frames, 'frames', 'input_frames')
            self.validate_type(text, 'string', 'caption', encoding='utf-8')
            self.validate_type(data, 'array', 'timeline', min_length=1)
        """
        if expected == 'string':
            encoding = kwargs.get('encoding', 'utf-8')
            is_valid, msg = self.type_validator.is_string_like(value, encoding)
            
        elif expected == 'array':
            min_length = kwargs.get('min_length', 0)
            max_length = kwargs.get('max_length', None)
            is_valid, msg = self.type_validator.is_array_like(value, min_length, max_length)
            
        elif expected == 'dict':
            required_keys = kwargs.get('required_keys', None)
            strict = kwargs.get('strict', False)
            is_valid, msg = self.type_validator.is_dict_like(value, required_keys, strict)
            
        elif expected == 'frames':
            is_valid, msg = self.type_validator.validate_frame_array(value)
            
        elif expected == 'numeric':
            min_val = kwargs.get('min_val', None)
            max_val = kwargs.get('max_val', None)
            allow_nan = kwargs.get('allow_nan', False)
            allow_inf = kwargs.get('allow_inf', False)
            is_valid, msg = self.type_validator.validate_numeric(
                value, min_val, max_val, allow_nan, allow_inf
            )
        else:
            # Fallback to isinstance for other types
            try:
                expected_type = eval(expected) if isinstance(expected, str) else expected
                is_valid = isinstance(value, expected_type)
                msg = f"isinstance check for {expected}"
            except:
                is_valid = False
                msg = f"Unknown type: {expected}"
        
        self.validate_or_fail(
            is_valid,
            f"{name} type validation failed: {msg}"
        )
    
    def validate_encoding(self, text: str, name: str, encoding: str = 'utf-8') -> None:
        """
        Validate string encoding.
        Available to all contracts that inherit from BaseServiceContract.
        
        Args:
            text: String to validate
            name: Field name for error messages
            encoding: Expected encoding
            
        Example usage:
            self.validate_encoding(caption_text, 'caption', 'utf-8')
        """
        is_valid, msg = self.type_validator.is_string_like(text, encoding)
        self.validate_or_fail(
            is_valid,
            f"{name} encoding validation failed: {msg}"
        )
    
    def validate_numpy_array(self, array: Any, name: str, expected_shape: tuple = None,
                           expected_dtype: Any = None) -> None:
        """
        Validate numpy array with shape and dtype checks.
        Available to all contracts that inherit from BaseServiceContract.
        
        Args:
            array: Array to validate
            name: Field name for error messages
            expected_shape: Expected shape tuple (use -1 for any dimension)
            expected_dtype: Expected numpy dtype
            
        Example usage:
            # Validate frames array is 4D with uint8 dtype
            self.validate_numpy_array(
                frames, 'frames',
                expected_shape=(-1, -1, -1, 3),  # Any batch size, height, width, but 3 channels
                expected_dtype=np.uint8
            )
        """
        if 'numpy' not in sys.modules:
            self.validate_or_fail(
                False,
                f"{name}: NumPy not available for validation"
            )
            return
        
        import numpy as np
        
        self.validate_or_fail(
            isinstance(array, np.ndarray),
            f"{name} must be numpy array, got {type(array).__name__}"
        )
        
        if expected_shape:
            shape = array.shape
            if len(shape) != len(expected_shape):
                self.validate_or_fail(
                    False,
                    f"{name} wrong number of dimensions: {len(shape)} != {len(expected_shape)}"
                )
            
            for i, (actual, expected) in enumerate(zip(shape, expected_shape)):
                if expected != -1 and actual != expected:
                    self.validate_or_fail(
                        False,
                        f"{name} dimension {i} mismatch: {actual} != {expected}"
                    )
        
        if expected_dtype:
            self.validate_or_fail(
                array.dtype == expected_dtype,
                f"{name} wrong dtype: {array.dtype} != {expected_dtype}"
            )

# Note: The actual BaseServiceContract implementation should be in:
# rumiai_v2/contracts/base.py
# 
# This documentation shows how existing contracts can use the new methods
# without any changes required - they just inherit the new capabilities.
```