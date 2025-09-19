#!/usr/bin/env python3
"""Apply instrumentation to remaining service methods."""

import re

# Read the current file
with open('/home/jorge/rumiaifinal/rumiai_v2/processors/video_analyzer.py', 'r') as f:
    content = f.read()

# Services that still need instrumentation
services_to_instrument = [
    ('scene_detection', 'pyscenedetect-0.6'),
    ('audio_energy', '1.0'),
    ('emotion_detection', 'feat-0.1'),
    ('deepface_gender', 'deepface-0.0.75')
]

for service_name, version in services_to_instrument:
    print(f"Instrumenting {service_name}...")

    # Find the method
    pattern = rf'(    async def _run_{service_name}\(self, video_id: str, video_path: Path\) -> MLAnalysisResult:\n        """[^"]*"""\n)(        try:.*?)(            return MLAnalysisResult\(\n                model_name=\'{service_name}\',\n                model_version=\'[^\']*\',\n                success=True,\n                data=data,\n                processing_time=0\.0\n            \)\n                \n        except Exception as e:\n            return MLAnalysisResult\(\n                model_name=\'{service_name}\',\n                model_version=\'[^\']*\',\n                success=False,\n                error=str\(e\)\n            \))'

    # Create instrumented version
    def replacer(match):
        method_def = match.group(1)
        try_block = match.group(2)

        # Extract the actual model version from the existing code
        version_match = re.search(r"model_version='([^']*)'", match.group(3))
        actual_version = version_match.group(1) if version_match else version

        instrumented = f'''{method_def}        start_time = time.time()

        with ThreadMonitor('{service_name}') as monitor:
            {try_block}
                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='{service_name}',
                    model_version='{actual_version}',
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
                    model_name='{service_name}',
                    model_version='{actual_version}',
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'] if stats['memory_delta_mb'] is not None else 0.0,
                    thread_flexibility=stats['thread_flexibility']
                )'''

        return instrumented

    content = re.sub(pattern, replacer, content, flags=re.DOTALL)

# Write the updated file
with open('/home/jorge/rumiaifinal/rumiai_v2/processors/video_analyzer.py', 'w') as f:
    f.write(content)

print("Instrumentation complete!")