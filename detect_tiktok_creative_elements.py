#!/usr/bin/env python3
"""
TikTok Creative Elements Detection
Combines YOLO with specialized detectors for TikTok-specific elements
"""

import os
import sys
import cv2
import json
import numpy as np
from datetime import datetime
import glob

# Check for PyTorch and CUDA availability
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# Install with: pip install easyocr
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️  EasyOCR not installed. Install with: pip install easyocr")

from ultralytics import YOLO

# Import temporal marker extractor
try:
    from python.temporal_marker_extractors import OCRTemporalExtractor
    TEMPORAL_MARKERS_AVAILABLE = True
except ImportError:
    TEMPORAL_MARKERS_AVAILABLE = False
    print("⚠️  Temporal markers module not available")

class TikTokCreativeDetector:
    def __init__(self):
        # Initialize YOLO for general objects
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Initialize text detector with GPU support if available
        if EASYOCR_AVAILABLE:
            # Check environment variable for GPU preference
            gpu_mode = os.environ.get('RUMIAI_USE_GPU', 'auto').lower()
            
            if gpu_mode == 'true':
                use_gpu = True
            elif gpu_mode == 'false':
                use_gpu = False
            else:  # auto mode
                use_gpu = CUDA_AVAILABLE
                
            # Initialize with proper GPU detection
            self.gpu_enabled = False
            if use_gpu and CUDA_AVAILABLE:
                try:
                    print("🚀 Initializing text detector with GPU acceleration...")
                    # Check available VRAM before initialization
                    if TORCH_AVAILABLE:
                        device_props = torch.cuda.get_device_properties(0)
                        vram_gb = device_props.total_memory / 1024**3
                        print(f"   GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f}GB VRAM)")
                        
                        # Check current VRAM usage
                        allocated_gb = torch.cuda.memory_allocated() / 1024**3
                        free_gb = (device_props.total_memory - torch.cuda.memory_allocated()) / 1024**3
                        print(f"   VRAM: {allocated_gb:.1f}GB used, {free_gb:.1f}GB free")
                        
                        # Warn if low on VRAM
                        if free_gb < 3.0:
                            print(f"   ⚠️  Low VRAM available ({free_gb:.1f}GB), may need smaller batch size")
                    
                    self.text_reader = easyocr.Reader(['en'], gpu=True)
                    # Pre-warm the model to avoid first-frame slowdown
                    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                    _ = self.text_reader.readtext(dummy_img)
                    self.gpu_enabled = True
                    print("   ✅ OCR using GPU acceleration")
                except Exception as e:
                    print(f"   ⚠️ GPU initialization failed: {e}")
                    print("   📝 Falling back to CPU mode...")
                    self.text_reader = easyocr.Reader(['en'], gpu=False)
                    self.gpu_enabled = False
            else:
                print("📝 Initializing text detector (CPU mode)...")
                if gpu_mode == 'auto' and not CUDA_AVAILABLE:
                    print("   ℹ️ No CUDA-capable GPU detected")
                self.text_reader = easyocr.Reader(['en'], gpu=False)
                self.gpu_enabled = False
        else:
            self.text_reader = None
            self.gpu_enabled = False
        
        # TikTok UI regions (right side buttons)
        self.ui_regions = {
            'like_button': {'x': 0.9, 'y': 0.4, 'w': 0.08, 'h': 0.08},
            'comment_button': {'x': 0.9, 'y': 0.5, 'w': 0.08, 'h': 0.08},
            'share_button': {'x': 0.9, 'y': 0.6, 'w': 0.08, 'h': 0.08},
            'profile_button': {'x': 0.9, 'y': 0.3, 'w': 0.08, 'h': 0.08}
        }
    
    def detect_text_regions(self, image_path):
        """Detect text using OCR"""
        if not self.text_reader:
            return []
        
        try:
            results = self.text_reader.readtext(image_path)
            text_detections = []
            
            for (bbox, text, prob) in results:
                if prob > 0.5:  # Confidence threshold
                    # Convert bbox points to x1,y1,x2,y2
                    points = np.array(bbox)
                    x1, y1 = points.min(axis=0)
                    x2, y2 = points.max(axis=0)
                    
                    text_detections.append({
                        'type': 'text',
                        'text': text,
                        'confidence': float(prob),
                        'bbox': {
                            'x1': float(x1),
                            'y1': float(y1),
                            'x2': float(x2),
                            'y2': float(y2)
                        },
                        'category': self.categorize_text(text, y1, image_path)
                    })
            
            return text_detections
        except Exception as e:
            print(f"Text detection error: {e}")
            return []
    
    def categorize_text(self, text, y_position, image_path):
        """Categorize text based on content and position"""
        text_lower = text.lower()
        
        # Get image height for position analysis
        img = cv2.imread(image_path)
        if img is None:
            return 'unknown'
        
        height = img.shape[0]
        relative_y = y_position / height
        
        # Categorization rules
        if any(cta in text_lower for cta in ['follow', 'like', 'comment', 'share', 'click', 'tap', 'swipe']):
            return 'call_to_action'
        elif relative_y < 0.2:  # Top 20% of frame
            return 'header_text'
        elif relative_y > 0.8:  # Bottom 20% of frame
            return 'caption'
        elif '@' in text:
            return 'username'
        elif '#' in text:
            return 'hashtag'
        else:
            return 'overlay_text'
    
    def detect_ui_elements(self, image_path):
        """Detect TikTok UI elements using template matching or region analysis"""
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        height, width = img.shape[:2]
        ui_detections = []
        
        # Check right side for UI buttons
        for element, region in self.ui_regions.items():
            x1 = int(region['x'] * width)
            y1 = int(region['y'] * height)
            x2 = int((region['x'] + region['w']) * width)
            y2 = int((region['y'] + region['h']) * height)
            
            # Extract region
            roi = img[y1:y2, x1:x2]
            
            # Simple detection: check if region has high contrast (likely UI element)
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                contrast = gray.std()
                
                if contrast > 30:  # Threshold for UI element presence
                    ui_detections.append({
                        'type': 'ui_element',
                        'element': element,
                        'confidence': min(contrast / 100, 1.0),
                        'bbox': {
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2
                        }
                    })
        
        return ui_detections
    
    def detect_colorful_regions(self, image_path):
        """Detect colorful regions that might be stickers, CTAs, or graphics"""
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Detect high saturation regions (likely graphics/stickers)
        saturation = hsv[:, :, 1]
        
        # Threshold for colorful regions
        _, binary = cv2.threshold(saturation, 180, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        colorful_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                # Classify based on aspect ratio and position
                aspect_ratio = w / h if h > 0 else 0
                region_type = 'graphic_overlay'
                
                if 0.8 < aspect_ratio < 1.2 and area < 5000:
                    region_type = 'sticker'
                elif w > img.shape[1] * 0.6 and h < img.shape[0] * 0.2:
                    region_type = 'banner'
                
                colorful_regions.append({
                    'type': 'creative_element',
                    'element': region_type,
                    'confidence': 0.7,
                    'bbox': {
                        'x1': x,
                        'y1': y,
                        'x2': x + w,
                        'y2': y + h
                    }
                })
        
        return colorful_regions[:5]  # Limit to top 5 regions
    
    def detect_all_elements(self, image_path):
        """Detect all creative elements in a frame"""
        print(f"🔍 Analyzing: {os.path.basename(image_path)}")
        
        all_detections = {
            'frame': os.path.basename(image_path),
            'yolo_objects': [],
            'text_elements': [],
            'ui_elements': [],
            'creative_elements': []
        }
        
        # 1. Run YOLO for general objects
        yolo_results = self.yolo_model(image_path, verbose=False)
        if yolo_results[0].boxes is not None:
            for box in yolo_results[0].boxes.data.tolist():
                all_detections['yolo_objects'].append({
                    'label': self.yolo_model.names[int(box[5])],
                    'confidence': float(box[4]),
                    'bbox': {
                        'x1': float(box[0]),
                        'y1': float(box[1]),
                        'x2': float(box[2]),
                        'y2': float(box[3])
                    }
                })
        
        # 2. Detect text
        all_detections['text_elements'] = self.detect_text_regions(image_path)
        
        # 3. Detect UI elements
        all_detections['ui_elements'] = self.detect_ui_elements(image_path)
        
        # 4. Detect creative elements (stickers, banners, etc.)
        all_detections['creative_elements'] = self.detect_colorful_regions(image_path)
        
        # Summary
        all_detections['summary'] = {
            'total_elements': (
                len(all_detections['yolo_objects']) +
                len(all_detections['text_elements']) +
                len(all_detections['ui_elements']) +
                len(all_detections['creative_elements'])
            ),
            'has_text': len(all_detections['text_elements']) > 0,
            'has_cta': any(t['category'] == 'call_to_action' for t in all_detections['text_elements']),
            'has_ui': len(all_detections['ui_elements']) > 0,
            'has_creative': len(all_detections['creative_elements']) > 0
        }
        
        return all_detections


def analyze_video_creative_elements(video_id, input_dir='frame_outputs', output_dir='creative_analysis_outputs'):
    """Analyze all frames of a video for creative elements"""
    detector = TikTokCreativeDetector()
    
    # Setup paths
    video_frame_dir = os.path.join(input_dir, video_id)
    video_output_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Get all frames
    frames = sorted(glob.glob(os.path.join(video_frame_dir, '*.jpg')))
    if not frames:
        print(f"⚠️  No frames found for {video_id}")
        return
    
    # Load video metadata to get duration
    metadata_path = os.path.join(video_frame_dir, 'metadata.json')
    video_duration = None
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                video_duration = metadata.get('duration', None)
        except:
            pass
    
    # If no metadata, estimate duration from frame count and typical FPS
    if video_duration is None:
        # Estimate based on frame count (assuming adaptive extraction was used)
        if len(frames) < 150:  # Likely < 30s at 5 FPS
            video_duration = len(frames) / 5.0
        elif len(frames) < 180:  # Likely 30-60s at 3 FPS
            video_duration = len(frames) / 3.0
        else:  # Likely > 60s at 2 FPS
            video_duration = len(frames) / 2.0
    
    print(f"\n🎬 Analyzing creative elements for: {video_id}")
    print(f"   Total frames: {len(frames)}")
    print(f"   Video duration: {video_duration:.1f}s")
    
    # Adaptive sampling based on video duration
    # Maintains ~2 FPS effective analysis rate
    sampled_frames = []
    if video_duration < 30:
        # Videos < 30s: Process every 2nd frame (2.5 FPS effective)
        print(f"   Strategy: Short video (<30s) - analyzing every 2nd frame")
        for i in range(0, len(frames), 2):
            sampled_frames.append(frames[i])
    elif video_duration < 60:
        # Videos 30-60s: Process first 2 of every 3 frames (2 FPS effective)
        print(f"   Strategy: Medium video (30-60s) - analyzing 2 of every 3 frames")
        for i in range(len(frames)):
            if i % 3 != 2:  # Skip every 3rd frame
                sampled_frames.append(frames[i])
    else:
        # Videos > 60s: Process every frame (2 FPS effective)
        print(f"   Strategy: Long video (>60s) - analyzing every frame")
        sampled_frames = frames
    
    print(f"   Analyzing {len(sampled_frames)} frames ({len(sampled_frames)/video_duration:.2f} effective FPS)")
    
    # Batch processing configuration for GPU memory management
    # Adjust batch size based on available VRAM
    if detector.gpu_enabled and CUDA_AVAILABLE and torch.cuda.is_available():
        free_vram_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
        if free_vram_gb < 2.0:
            BATCH_SIZE = 3  # Conservative for low VRAM
        elif free_vram_gb < 4.0:
            BATCH_SIZE = 5  # Medium batch size
        else:
            BATCH_SIZE = 10  # Full batch size for 4GB+ free
        print(f"   Using batch size: {BATCH_SIZE} (based on {free_vram_gb:.1f}GB free VRAM)")
    else:
        BATCH_SIZE = 1  # CPU mode - process one at a time
    
    # Process sampled frames
    all_frame_results = []
    creative_timeline = {
        'text_timeline': [],
        'cta_timeline': [],
        'ui_timeline': [],
        'creative_timeline': []
    }
    
    import time
    start_time = time.time()
    
    # Process frames in batches to manage VRAM
    for batch_idx in range(0, len(sampled_frames), BATCH_SIZE):
        batch_frames = sampled_frames[batch_idx:batch_idx + BATCH_SIZE]
        batch_start_time = time.time()
        
        for idx, frame_path in enumerate(batch_frames):
            global_idx = batch_idx + idx
            frame_start = time.time()
            
            # Extract actual frame number from filename (frame_XXXX_tY.YY.jpg)
            frame_filename = os.path.basename(frame_path)
            actual_frame_num = int(frame_filename.split('_')[1])
            
            # Detect elements
            frame_results = detector.detect_all_elements(frame_path)
            frame_results['actual_frame_number'] = actual_frame_num
            all_frame_results.append(frame_results)
            
            # Update timeline with actual frame numbers
            if frame_results['summary']['has_text']:
                creative_timeline['text_timeline'].append(actual_frame_num)
            if frame_results['summary']['has_cta']:
                creative_timeline['cta_timeline'].append(actual_frame_num)
            if frame_results['summary']['has_ui']:
                creative_timeline['ui_timeline'].append(actual_frame_num)
            if frame_results['summary']['has_creative']:
                creative_timeline['creative_timeline'].append(actual_frame_num)
                
            # Show progress with timing
            frame_time = time.time() - frame_start
            elapsed = time.time() - start_time
            avg_time = elapsed / (global_idx + 1) if global_idx > 0 else frame_time
            eta = avg_time * (len(sampled_frames) - global_idx - 1)
            
            print(f"   Processing frame {global_idx+1}/{len(sampled_frames)} "
                  f"(frame #{actual_frame_num}) - {frame_time:.1f}s/frame, "
                  f"ETA: {eta:.0f}s", end='\r')
        
        # Clear GPU memory between batches if using GPU
        if CUDA_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if batch_idx + BATCH_SIZE < len(sampled_frames):
                print(f"\n   Batch {batch_idx//BATCH_SIZE + 1} complete, clearing GPU memory...")
    
    print(f"\n   ✅ Analysis complete")
    
    # Generate insights
    insights = {
        'video_id': video_id,
        'video_duration': video_duration,
        'total_frames': len(frames),
        'analyzed_frames': len(sampled_frames),
        'effective_fps': len(sampled_frames) / video_duration,
        'sampling_strategy': 'every_2nd' if video_duration < 30 else ('2_of_3' if video_duration < 60 else 'all_frames'),
        'creative_density': sum(r['summary']['total_elements'] for r in all_frame_results) / len(sampled_frames),
        'text_coverage': len(creative_timeline['text_timeline']) / len(sampled_frames),
        'cta_frames': creative_timeline['cta_timeline'],
        'has_persistent_ui': len(creative_timeline['ui_timeline']) > len(sampled_frames) * 0.8,
        'creative_moments': creative_timeline,
        'processed_at': datetime.now().isoformat()
    }
    
    # Extract temporal markers if available
    temporal_markers = None
    if TEMPORAL_MARKERS_AVAILABLE and video_duration:
        try:
            print("   🎯 Extracting temporal markers...")
            video_metadata = {
                'fps': 30.0,  # Default assumption, could be read from metadata
                'extraction_fps': len(sampled_frames) / video_duration,
                'duration': video_duration,
                'frame_count': len(frames)
            }
            
            # Check if we have metadata file with actual FPS
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        video_metadata['fps'] = metadata.get('fps', 30.0)
                except:
                    pass
            
            extractor = OCRTemporalExtractor(video_metadata)
            temporal_markers = extractor.extract_temporal_markers(all_frame_results)
            
            # Add summary stats to insights
            if temporal_markers:
                first_5 = temporal_markers.get('first_5_seconds', {})
                cta = temporal_markers.get('cta_window', {})
                
                insights['temporal_markers_summary'] = {
                    'text_moments_first_5s': len(first_5.get('text_moments', [])),
                    'density_progression': first_5.get('density_progression', []),
                    'cta_appearances': len(cta.get('cta_appearances', [])),
                    'has_early_cta': any(t.get('is_cta', False) for t in first_5.get('text_moments', []))
                }
                
                print(f"      - Text moments (0-5s): {insights['temporal_markers_summary']['text_moments_first_5s']}")
                print(f"      - CTA appearances: {insights['temporal_markers_summary']['cta_appearances']}")
                
        except Exception as e:
            print(f"   ⚠️  Temporal marker extraction failed: {e}")
            temporal_markers = None
    
    # Save results
    output_data = {
        'insights': insights,
        'frame_details': all_frame_results
    }
    
    # Add temporal markers if extracted
    if temporal_markers:
        output_data['temporal_markers'] = temporal_markers
    
    output_file = os.path.join(video_output_dir, f'{video_id}_creative_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"   💾 Saved analysis: {output_file}")
    
    # Print summary
    print(f"\n   📊 Creative Elements Summary:")
    print(f"      - Average elements per frame: {insights['creative_density']:.1f}")
    print(f"      - Frames with text: {len(creative_timeline['text_timeline'])}/{len(frames)}")
    print(f"      - CTA frames: {len(creative_timeline['cta_timeline'])}")
    print(f"      - UI persistence: {'Yes' if insights['has_persistent_ui'] else 'No'}")
    
    # Performance summary
    total_time = time.time() - start_time
    fps_achieved = len(sampled_frames) / total_time
    mode = "GPU" if detector.gpu_enabled else "CPU"
    print(f"\n   ⚡ Performance: {fps_achieved:.1f} frames/sec ({mode} mode, {total_time:.1f}s total)")
    
    return insights


def main():
    """Main function to analyze all videos"""
    import sys
    
    print("🎨 TikTok Creative Elements Analyzer")
    print("=" * 50)
    
    if not EASYOCR_AVAILABLE:
        print("\n⚠️  EasyOCR not available. Install for text detection:")
        print("   pip install easyocr")
        print("\n   Continuing with limited functionality...\n")
    
    # Check for specific video or process all
    if len(sys.argv) > 1:
        video_id = sys.argv[1]
        analyze_video_creative_elements(video_id)
    else:
        # Process all videos
        frame_dirs = glob.glob('frame_outputs/*')
        for frame_dir in frame_dirs:
            if os.path.isdir(frame_dir):
                video_id = os.path.basename(frame_dir)
                analyze_video_creative_elements(video_id)


if __name__ == "__main__":
    main()