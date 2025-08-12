#!/usr/bin/env python3
"""
Compare actual test results against expected golden dataset
Usage: python3 scripts/compare_golden.py <video_id> <expected_json_path>
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

class GoldenComparator:
    def __init__(self, video_id: str, expected_path: str):
        self.video_id = video_id
        self.expected_path = Path(expected_path)
        self.actual_base = Path(f"insights/{video_id}")
        self.report = []
        self.stats = {
            'exact_matches': 0,
            'within_tolerance': 0,
            'outside_tolerance': 0,
            'missing_fields': 0,
            'extra_fields': 0
        }
        
    def load_expected(self) -> Dict:
        """Load expected values from JSON"""
        if not self.expected_path.exists():
            raise FileNotFoundError(f"Expected file not found: {self.expected_path}")
            
        with open(self.expected_path) as f:
            return json.load(f)
    
    def load_actual_analysis(self, analysis_type: str) -> Dict:
        """Load actual ML results for an analysis type"""
        # Find the most recent ML file
        ml_files = list(self.actual_base.glob(f"{analysis_type}/{analysis_type}_ml_*.json"))
        
        if not ml_files:
            return None
            
        # Get the most recent file
        ml_file = sorted(ml_files)[-1]
        
        with open(ml_file) as f:
            return json.load(f)
    
    def compare_value(self, expected: Any, actual: Any, field_path: str, tolerance: float = 0.1) -> str:
        """Compare two values with optional tolerance"""
        
        # Exact match
        if expected == actual:
            self.stats['exact_matches'] += 1
            return "✅ EXACT"
        
        # Numeric comparison with tolerance
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if expected == 0:
                diff_pct = 100 if actual != 0 else 0
            else:
                diff_pct = abs((actual - expected) / expected)
                
            if diff_pct <= tolerance:
                self.stats['within_tolerance'] += 1
                return f"⚠️ WITHIN TOLERANCE ({diff_pct:.1%})"
            else:
                self.stats['outside_tolerance'] += 1
                return f"❌ OUTSIDE TOLERANCE ({diff_pct:.1%})"
        
        # String comparison (case insensitive for some fields)
        if isinstance(expected, str) and isinstance(actual, str):
            if expected.lower() == actual.lower():
                self.stats['exact_matches'] += 1
                return "✅ EXACT (case-insensitive)"
        
        # List comparison (check if all expected items are present)
        if isinstance(expected, list) and isinstance(actual, list):
            expected_set = set(expected) if all(isinstance(x, (str, int, float)) for x in expected) else None
            actual_set = set(actual) if all(isinstance(x, (str, int, float)) for x in actual) else None
            
            if expected_set and actual_set:
                if expected_set == actual_set:
                    self.stats['exact_matches'] += 1
                    return "✅ EXACT (unordered)"
                elif expected_set.issubset(actual_set):
                    self.stats['within_tolerance'] += 1
                    return "⚠️ CONTAINS EXPECTED"
                else:
                    missing = expected_set - actual_set
                    self.stats['outside_tolerance'] += 1
                    return f"❌ MISSING: {missing}"
        
        self.stats['outside_tolerance'] += 1
        return "❌ MISMATCH"
    
    def compare_analysis(self, analysis_type: str, expected_data: Dict) -> List[str]:
        """Compare one analysis type"""
        results = []
        results.append(f"\n{'='*60}")
        results.append(f"📊 {analysis_type.upper()}")
        results.append('='*60)
        
        actual_data = self.load_actual_analysis(analysis_type)
        
        if actual_data is None:
            results.append(f"❌ No actual data found for {analysis_type}")
            self.stats['missing_fields'] += len(expected_data)
            return results
        
        # Compare each expected field
        for category, fields in expected_data.items():
            results.append(f"\n{category}:")
            
            actual_category = actual_data.get(category, {})
            if not actual_category:
                results.append(f"  ❌ Category missing in actual output")
                self.stats['missing_fields'] += len(fields) if isinstance(fields, dict) else 1
                continue
            
            if isinstance(fields, dict):
                for field, expected_value in fields.items():
                    actual_value = actual_category.get(field, "MISSING")
                    
                    if actual_value == "MISSING":
                        self.stats['missing_fields'] += 1
                        results.append(f"  ❌ {field}: NOT FOUND in actual")
                    else:
                        # Determine tolerance based on field type
                        tolerance = self.get_tolerance(analysis_type, field)
                        status = self.compare_value(expected_value, actual_value, f"{category}.{field}", tolerance)
                        
                        # Format output based on data type
                        if isinstance(expected_value, float):
                            results.append(f"  {status:<25} {field:30} Expected: {expected_value:8.2f}, Actual: {actual_value:8.2f}")
                        elif isinstance(expected_value, int):
                            results.append(f"  {status:<25} {field:30} Expected: {expected_value:8}, Actual: {actual_value:8}")
                        else:
                            results.append(f"  {status:<25} {field:30} Expected: {expected_value}, Actual: {actual_value}")
            else:
                # Direct value comparison
                status = self.compare_value(fields, actual_category, category)
                results.append(f"  {status:<25} {category}")
        
        return results
    
    def get_tolerance(self, analysis_type: str, field: str) -> float:
        """Get tolerance for specific field types"""
        
        # High tolerance fields (ML variance expected)
        high_tolerance = [
            'totalElements', 'avgDensity', 'overlayDensity',
            'emotionalIntensity', 'confidence', 'engagement_score'
        ]
        
        # Medium tolerance fields
        medium_tolerance = [
            'wordsPerMinute', 'speechDensity', 'timelineCoverage',
            'faceVisibilityRate', 'pacingScore'
        ]
        
        # Exact match fields
        exact_match = [
            'uniqueEmotions', 'totalWords', 'duration',
            'sceneChangeCount', 'hashtagCount'
        ]
        
        if field in exact_match:
            return 0.0
        elif field in high_tolerance:
            return 0.15  # 15% tolerance
        elif field in medium_tolerance:
            return 0.10  # 10% tolerance
        else:
            return 0.10  # Default 10%
    
    def generate_report(self) -> str:
        """Generate comparison report"""
        expected = self.load_expected()
        
        # Header
        self.report.append("="*70)
        self.report.append("🔍 GOLDEN DATASET COMPARISON REPORT")
        self.report.append("="*70)
        self.report.append(f"Video ID: {self.video_id}")
        self.report.append(f"Expected: {self.expected_path}")
        self.report.append(f"Actual: {self.actual_base}")
        self.report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Process each analysis type
        for analysis_type, expected_data in expected.items():
            if analysis_type.startswith('_'):  # Skip metadata fields
                continue
            results = self.compare_analysis(analysis_type, expected_data)
            self.report.extend(results)
        
        # Summary statistics
        self.report.append("\n" + "="*70)
        self.report.append("📈 SUMMARY STATISTICS")
        self.report.append("="*70)
        
        total_checks = sum(self.stats.values())
        
        self.report.append(f"Total Comparisons: {total_checks}")
        self.report.append(f"  ✅ Exact Matches:     {self.stats['exact_matches']:4} ({self.stats['exact_matches']/total_checks*100:5.1f}%)")
        self.report.append(f"  ⚠️ Within Tolerance:  {self.stats['within_tolerance']:4} ({self.stats['within_tolerance']/total_checks*100:5.1f}%)")
        self.report.append(f"  ❌ Outside Tolerance: {self.stats['outside_tolerance']:4} ({self.stats['outside_tolerance']/total_checks*100:5.1f}%)")
        self.report.append(f"  ❌ Missing Fields:    {self.stats['missing_fields']:4} ({self.stats['missing_fields']/total_checks*100:5.1f}%)")
        
        # Overall pass/fail
        pass_rate = (self.stats['exact_matches'] + self.stats['within_tolerance']) / total_checks
        
        self.report.append("\n" + "="*70)
        if pass_rate >= 0.90:
            self.report.append("✅ VALIDATION PASSED ({:.1f}% acceptance rate)".format(pass_rate * 100))
        elif pass_rate >= 0.75:
            self.report.append("⚠️ VALIDATION MARGINAL ({:.1f}% acceptance rate)".format(pass_rate * 100))
        else:
            self.report.append("❌ VALIDATION FAILED ({:.1f}% acceptance rate)".format(pass_rate * 100))
        self.report.append("="*70)
        
        return "\n".join(self.report)
    
    def save_report(self):
        """Save report to file"""
        report_dir = Path(f"validation_reports")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"{self.video_id}_validation_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(self.generate_report())
            
        return report_file

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/compare_golden.py <video_id> <expected_json_path>")
        print("Example: python3 scripts/compare_golden.py video_test_1 expected_outputs/video_test_1.json")
        sys.exit(1)
    
    video_id = sys.argv[1]
    expected_path = sys.argv[2]
    
    try:
        comparator = GoldenComparator(video_id, expected_path)
        report = comparator.generate_report()
        
        # Print to console
        print(report)
        
        # Save to file
        report_file = comparator.save_report()
        print(f"\n📄 Report saved to: {report_file}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()