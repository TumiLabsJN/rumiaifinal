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
                'confidence': lambda x: 0 <= x <= 1,
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
                'landmarks': lambda x: (isinstance(x, int) and x == 33) or isinstance(x, list),
                'confidence': lambda x: 0 <= x <= 1
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
                'confidence': lambda x: 0 <= x <= 1,
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
                'scene_index': lambda x: x >= 0,
                'duration': lambda x: x > 0
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
                'scene_index': lambda x: x >= 0
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
                'bbox': lambda x: len(x) == 4 and all(isinstance(v, (int, float)) for v in x),
                'confidence': lambda x: 0 <= x <= 1
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
                'confidence': lambda x: 0 <= x <= 1
            }
        }
    }
    
    def validate_entry_data(self, entry_type: str, data: Dict[str, Any], entry_index: int) -> None:
        """Deep validation of entry data structure based on actual production schemas"""
        
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
        
        # Type and value validation
        for field, value in data.items():
            if field in schema['types']:
                expected_type = schema['types'][field]
                self.validate_or_fail(isinstance(value, expected_type),
                                    f"Entry {entry_index} ({entry_type}).{field}: Expected {expected_type}, got {type(value)}")
            
            # Custom validators
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