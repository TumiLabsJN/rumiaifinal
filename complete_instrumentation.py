#!/usr/bin/env python3
"""Complete instrumentation for all remaining services."""

# Read the file
with open('/home/jorge/rumiaifinal/rumiai_v2/processors/video_analyzer.py', 'r') as f:
    lines = f.readlines()

# Find and replace remaining un-instrumented methods
# We need to instrument: scene_detection, audio_energy, emotion_detection, deepface_gender

def instrument_method(lines, method_name, model_version):
    """Instrument a single method."""
    in_method = False
    method_start = -1
    indent = "    "

    for i, line in enumerate(lines):
        if f"async def _run_{method_name}" in line:
            in_method = True
            method_start = i
            # Check if already instrumented
            if i + 2 < len(lines) and "start_time = time.time()" in lines[i + 2]:
                print(f"  {method_name} already instrumented, skipping")
                return lines
            print(f"  Instrumenting {method_name} at line {i+1}")
            break

    if not in_method:
        print(f"  Method {method_name} not found!")
        return lines

    # Find the try block
    try_line = -1
    for i in range(method_start + 1, min(method_start + 5, len(lines))):
        if "try:" in lines[i]:
            try_line = i
            break

    if try_line == -1:
        print(f"  No try block found for {method_name}")
        return lines

    # Insert timing and monitor start
    insert_lines = [
        f"{indent}start_time = time.time()\n",
        f"\n",
        f"{indent}with ThreadMonitor('{method_name}') as monitor:\n",
    ]

    # Adjust indentation for try block and everything inside
    modified_lines = lines[:try_line]
    modified_lines.extend(insert_lines)

    # Process the rest of the method
    i = try_line
    while i < len(lines):
        line = lines[i]

        # Check if we're at a return statement with model_name matching our method
        if f"model_name='{method_name}'" in line and "return MLAnalysisResult" in lines[i-1]:
            # This is a return statement we need to modify

            # Check if it's already instrumented
            if "processing_time=time.time() - start_time" in ''.join(lines[i:i+10]):
                # Already instrumented, skip
                modified_lines.extend(lines[i:])
                break

            # Find where this return statement ends
            return_start = i - 1
            paren_count = 0
            return_end = return_start

            for j in range(return_start, min(return_start + 30, len(lines))):
                if "MLAnalysisResult(" in lines[j]:
                    paren_count = 1
                for char in lines[j]:
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            return_end = j
                            break
                if return_end != return_start:
                    break

            # Check if this is in try or except block
            in_try_block = True
            # Simple heuristic: if the previous 20 lines contain "except", we're in except block
            for j in range(max(0, i - 20), i):
                if "except Exception" in lines[j]:
                    in_try_block = False
                    break

            # Add stats = monitor.stop() before return
            if in_try_block:
                modified_lines.append(f"\n{indent*3}stats = monitor.stop()\n\n")
            else:
                modified_lines.append(f"{indent*3}stats = monitor.stop()\n\n")

            # Reconstruct return statement with instrumentation
            modified_lines.append(f"{indent*3}return MLAnalysisResult(\n")

            # Add fields from original return
            j = return_start + 1
            while j <= return_end:
                line = lines[j]
                if "processing_time=0.0" in line:
                    # Replace with actual timing
                    modified_lines.append(f"{indent*4}processing_time=time.time() - start_time,\n")
                    modified_lines.append(f"{indent*4}start_time=start_time,\n")
                    modified_lines.append(f"{indent*4}end_time=time.time(),\n")
                    modified_lines.append(f"{indent*4}threads_created=stats['threads_created'],\n")
                    if in_try_block:
                        modified_lines.append(f"{indent*4}memory_delta_mb=stats['memory_delta_mb'],\n")
                    else:
                        modified_lines.append(f"{indent*4}memory_delta_mb=stats['memory_delta_mb'] if stats['memory_delta_mb'] is not None else 0.0,\n")
                    modified_lines.append(f"{indent*4}thread_flexibility=stats['thread_flexibility']\n")
                elif ")" in line and j == return_end:
                    # Skip the closing paren, we already added fields
                    modified_lines.append(f"{indent*3})\n")
                elif "processing_time" not in line:
                    # Keep other fields
                    modified_lines.append(f"    {line}")
                j += 1

            i = return_end + 1
        else:
            # For try and except lines, add extra indentation
            if line.strip().startswith("try:") or line.strip().startswith("except"):
                modified_lines.append(f"    {line}")
            # For lines inside try/except, add extra indentation
            elif i > try_line and not line.strip().startswith("async def"):
                if line.strip():  # Non-empty lines
                    modified_lines.append(f"    {line}")
                else:
                    modified_lines.append(line)
            else:
                modified_lines.append(line)
            i += 1

            # Check if we've reached the next method
            if "async def _run_" in line and i > method_start + 1:
                # We've reached the next method, copy rest
                modified_lines.extend(lines[i:])
                break

    return modified_lines


# Services to instrument with their versions
services = [
    ('scene_detection', 'pyscenedetect-0.6'),
    ('audio_energy', 'librosa-0.11'),
    ('emotion_detection', 'feat-0.1'),
    ('deepface_gender', 'deepface-0.0.75')
]

print("Starting instrumentation...")
for service_name, version in services:
    lines = instrument_method(lines, service_name, version)

# Write back
with open('/home/jorge/rumiaifinal/rumiai_v2/processors/video_analyzer.py', 'w') as f:
    f.writelines(lines)

print("Instrumentation complete!")