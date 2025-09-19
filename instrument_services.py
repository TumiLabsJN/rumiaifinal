#!/usr/bin/env python3
"""Script to help instrument remaining service methods."""

import re

# Template for instrumented method
template = """    async def _run_{service}(self, video_id: str, video_path: Path) -> MLAnalysisResult:
        \"\"\"{description}\"\"\"
        start_time = time.time()

        with ThreadMonitor('{service}') as monitor:
            try:
{try_block}
                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='{service}',
                    model_version='{version}',
                    success=True,
                    data=data,
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'],
                    thread_flexibility=stats['thread_flexibility']
                )

            except Exception as e:
                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='{service}',
                    model_version='{version}',
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'] if stats['memory_delta_mb'] is not None else 0.0,
                    thread_flexibility=stats['thread_flexibility']
                )"""

# Services to instrument
services_to_instrument = [
    ('ocr', 'Run OCR text detection.', 'tesseract-5'),
    ('scene_detection', 'Run scene detection analysis.', 'scenedetect-0.6'),
    ('audio_energy', 'Run audio energy analysis.', '1.0'),
    ('emotion_detection', 'Run FEAT emotion detection.', 'feat-0.1'),
    ('deepface_gender', 'Run DeepFace gender detection.', 'deepface-0.0.75')
]

print("Services to instrument:")
for service, desc, version in services_to_instrument:
    print(f"  - {service}: {desc} (version: {version})")

print("\nYou'll need to extract the try block content for each service from the original code.")
print("Then use the template above to create the instrumented version.")