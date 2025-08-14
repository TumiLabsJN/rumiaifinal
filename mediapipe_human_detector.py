#!/usr/bin/env python3
"""
MediaPipe Human Detection for TikTok Videos
Detects face expressions, hand gestures, and body poses
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime
import glob

class MediaPipeHumanDetector:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize detectors
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=4,
            min_detection_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        
        # Gesture patterns
        self.gesture_patterns = {
            'thumbs_up': self._check_thumbs_up,
            'peace_sign': self._check_peace_sign,
            'pointing': self._check_pointing,
            'open_palm': self._check_open_palm,
            'fist': self._check_fist,
            'heart': self._check_heart_gesture
        }
        
        # Body pose patterns
        self.pose_patterns = {
            'standing': self._check_standing,
            'sitting': self._check_sitting,
            'dancing': self._check_dancing,
            'arms_up': self._check_arms_up,
            'arms_crossed': self._check_arms_crossed
        }
    
    def detect_gaze_only(self, image):
        """Extract iris landmarks for eye contact detection only"""
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return []
        
        gaze_data = []
        for face_landmarks in results.multi_face_landmarks:
            try:
                # Verify all required landmarks exist
                required_indices = [468, 473, 133, 33, 362, 263]  # Iris + eye corners
                if len(face_landmarks.landmark) < max(required_indices) + 1:
                    continue
                
                # Get iris centers for both eyes
                left_iris = face_landmarks.landmark[468]
                right_iris = face_landmarks.landmark[473]
                
                # Eye corners for normalization
                left_inner = face_landmarks.landmark[133]
                left_outer = face_landmarks.landmark[33]
                right_inner = face_landmarks.landmark[362]
                right_outer = face_landmarks.landmark[263]
                
                # Calculate normalized positions (0=inner, 1=outer)
                left_width = abs(left_outer.x - left_inner.x)
                left_pos = (left_iris.x - left_inner.x) / left_width if left_width > 0 else 0.5
                
                right_width = abs(right_outer.x - right_inner.x)
                right_pos = (right_iris.x - right_inner.x) / right_width if right_width > 0 else 0.5
                
                # Average both eyes
                avg_position = (left_pos + right_pos) / 2
                
                # Eye contact when iris is centered (0.35-0.65 range)
                eye_contact = 1.0 if 0.35 <= avg_position <= 0.65 else 0.0
                
                gaze_data.append({
                    'eye_contact': eye_contact,
                    'gaze_direction': 'camera' if eye_contact > 0.5 else 'away'
                })
            except (IndexError, AttributeError):
                # Skip this face if landmarks are missing or malformed
                continue
        
        return gaze_data
    
    
    def detect_hands_and_gestures(self, image):
        """Detect hands and recognize gestures"""
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        hand_data = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get hand bounding box
                h, w = image.shape[:2]
                x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                
                bbox = {
                    'x1': int(min(x_coords) - 20),
                    'y1': int(min(y_coords) - 20),
                    'x2': int(max(x_coords) + 20),
                    'y2': int(max(y_coords) + 20)
                }
                
                # Detect gesture
                gesture = self._detect_gesture(hand_landmarks.landmark)
                
                hand_data.append({
                    'type': 'hand',
                    'bbox': bbox,
                    'hand_type': handedness.classification[0].label,
                    'gesture': gesture,
                    'confidence': handedness.classification[0].score
                })
        
        return hand_data
    
    def _detect_gesture(self, landmarks):
        """Detect hand gesture from landmarks"""
        for gesture_name, check_func in self.gesture_patterns.items():
            if check_func(landmarks):
                return gesture_name
        return 'unknown'
    
    def _check_thumbs_up(self, landmarks):
        """Check if hand is making thumbs up gesture"""
        # Thumb tip higher than other fingers
        thumb_tip = landmarks[4].y
        index_tip = landmarks[8].y
        middle_tip = landmarks[12].y
        
        if thumb_tip < index_tip - 0.1 and thumb_tip < middle_tip - 0.1:
            # Check if other fingers are closed
            index_mcp = landmarks[5].y
            if index_tip > index_mcp:
                return True
        return False
    
    def _check_peace_sign(self, landmarks):
        """Check if hand is making peace sign"""
        # Index and middle fingers extended
        index_tip = landmarks[8].y
        middle_tip = landmarks[12].y
        index_mcp = landmarks[5].y
        middle_mcp = landmarks[9].y
        
        # Ring and pinky closed
        ring_tip = landmarks[16].y
        pinky_tip = landmarks[20].y
        ring_mcp = landmarks[13].y
        pinky_mcp = landmarks[17].y
        
        if (index_tip < index_mcp and middle_tip < middle_mcp and
            ring_tip > ring_mcp and pinky_tip > pinky_mcp):
            return True
        return False
    
    def _check_pointing(self, landmarks):
        """Check if hand is pointing"""
        # Index finger extended, others closed
        index_tip = landmarks[8].y
        index_mcp = landmarks[5].y
        middle_tip = landmarks[12].y
        middle_mcp = landmarks[9].y
        
        if index_tip < index_mcp - 0.05 and middle_tip > middle_mcp:
            return True
        return False
    
    def _check_open_palm(self, landmarks):
        """Check if hand is showing open palm"""
        # All fingers extended
        tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        mcps = [landmarks[2], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
        
        extended = sum(1 for tip, mcp in zip(tips, mcps) if tip.y < mcp.y)
        return extended >= 4
    
    def _check_fist(self, landmarks):
        """Check if hand is making a fist"""
        # All fingers closed
        tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        mcps = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
        
        closed = sum(1 for tip, mcp in zip(tips, mcps) if tip.y > mcp.y)
        return closed >= 4
    
    def _check_heart_gesture(self, landmarks):
        """Check if hands are making heart shape (requires both hands)"""
        # This is complex and requires both hands
        # Simplified: check if thumb and index are close
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        return distance < 0.05
    
    def detect_body_pose(self, image):
        """Detect body pose and posture"""
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pose_data = []
        
        if results.pose_landmarks:
            h, w = image.shape[:2]
            
            # Get pose bounding box
            x_coords = [landmark.x * w for landmark in results.pose_landmarks.landmark]
            y_coords = [landmark.y * h for landmark in results.pose_landmarks.landmark]
            
            bbox = {
                'x1': int(min(x_coords)),
                'y1': int(min(y_coords)),
                'x2': int(max(x_coords)),
                'y2': int(max(y_coords))
            }
            
            # Analyze pose
            pose_type = self._analyze_pose(results.pose_landmarks.landmark)
            
            # Check for specific actions
            actions = []
            if self._check_arms_up(results.pose_landmarks.landmark):
                actions.append('arms_raised')
            if self._check_dancing(results.pose_landmarks.landmark):
                actions.append('dancing')
            
            pose_data.append({
                'type': 'body_pose',
                'bbox': bbox,
                'pose': pose_type,
                'actions': actions,
                'confidence': 0.8
            })
        
        return pose_data
    
    def _analyze_pose(self, landmarks):
        """Analyze body pose from landmarks"""
        # Check each pose pattern
        for pose_name, check_func in self.pose_patterns.items():
            if check_func(landmarks):
                return pose_name
        return 'standing'
    
    def _check_standing(self, landmarks):
        """Check if person is standing"""
        # Hips above knees, knees above ankles
        left_hip = landmarks[23].y
        left_knee = landmarks[25].y
        left_ankle = landmarks[27].y
        
        return left_hip < left_knee < left_ankle
    
    def _check_sitting(self, landmarks):
        """Check if person is sitting"""
        # Hips at similar level to knees
        left_hip = landmarks[23].y
        left_knee = landmarks[25].y
        
        return abs(left_hip - left_knee) < 0.1
    
    def _check_dancing(self, landmarks):
        """Check if person might be dancing"""
        # Arms or legs in dynamic positions
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        # Check if arms are raised or extended
        shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
        arms_raised = left_wrist.y < shoulder_y or right_wrist.y < shoulder_y
        
        # Check if legs are apart
        legs_apart = abs(left_ankle.x - right_ankle.x) > 0.2
        
        return arms_raised or legs_apart
    
    def _check_arms_up(self, landmarks):
        """Check if arms are raised"""
        left_wrist = landmarks[15].y
        right_wrist = landmarks[16].y
        left_shoulder = landmarks[11].y
        right_shoulder = landmarks[12].y
        
        return left_wrist < left_shoulder or right_wrist < right_shoulder
    
    def _check_arms_crossed(self, landmarks):
        """Check if arms are crossed"""
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        
        # Wrists on opposite sides of body
        return left_wrist.x > 0.5 and right_wrist.x < 0.5
    
    def detect_all_human_elements(self, image_path):
        """Detect all human elements in a frame"""
        print(f"🔍 Analyzing human elements: {os.path.basename(image_path)}")
        
        image = cv2.imread(image_path)
        if image is None:
            return {}
        
        # Run all detections
        gaze = self.detect_gaze_only(image)
        hands = self.detect_hands_and_gestures(image)
        poses = self.detect_body_pose(image)
        
        # Compile results
        results = {
            'frame': os.path.basename(image_path),
            'gaze': gaze,
            'hands': hands,
            'poses': poses,
            'summary': {
                'gaze_count': len(gaze),
                'hand_count': len(hands),
                'body_count': len(poses),
                'eye_contact': [g['eye_contact'] for g in gaze],
                'gestures': [h['gesture'] for h in hands],
                'poses': [p['pose'] for p in poses],
                'has_interaction': len(hands) > 0 or any(g['eye_contact'] > 0.5 for g in gaze)
            }
        }
        
        return results
    
    def save_annotated_frame(self, image_path, detections, output_path):
        """Save frame with MediaPipe annotations"""
        image = cv2.imread(image_path)
        if image is None:
            return
        
        # Draw gaze indicators (eye contact)
        for gaze in detections.get('gaze', []):
            h, w = image.shape[:2]
            # Simple indicator at top of frame
            if gaze['eye_contact'] > 0.5:
                cv2.putText(image, "Eye Contact", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(image, "Looking Away", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw hand detections
        for hand in detections.get('hands', []):
            bbox = hand['bbox']
            cv2.rectangle(image, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (255, 0, 0), 2)
            cv2.putText(image, f"{hand['hand_type']}: {hand['gesture']}", (bbox['x1'], bbox['y1']-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw pose detections
        for pose in detections.get('poses', []):
            bbox = pose['bbox']
            cv2.rectangle(image, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 0, 255), 2)
            cv2.putText(image, f"Pose: {pose['pose']}", (bbox['x1'], bbox['y1']-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imwrite(output_path, image)


def analyze_video_human_elements(video_id, input_dir='frame_outputs', output_dir='human_analysis_outputs'):
    """Analyze all frames of a video for human elements"""
    detector = MediaPipeHumanDetector()
    
    # Setup paths
    video_frame_dir = os.path.join(input_dir, video_id)
    video_output_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Get all frames
    frames = sorted(glob.glob(os.path.join(video_frame_dir, '*.jpg')))
    if not frames:
        print(f"⚠️  No frames found for {video_id}")
        return
    
    print(f"\n🎬 Analyzing human elements for: {video_id}")
    print(f"   Frames: {len(frames)}")
    
    # Process each frame
    all_frame_results = []
    timeline = {
        'expressions': [],
        'gestures': [],
        'poses': [],
        'engagement_moments': []
    }
    
    for i, frame_path in enumerate(frames):
        print(f"   Processing frame {i+1}/{len(frames)}", end='\r')
        
        # Detect elements
        frame_results = detector.detect_all_human_elements(frame_path)
        all_frame_results.append(frame_results)
        
        # Update timeline
        frame_num = i + 1
        
        # Track expressions
        for face in frame_results['faces']:
            if face['expression'] != 'neutral':
                timeline['expressions'].append({
                    'frame': frame_num,
                    'expression': face['expression']
                })
        
        # Track gestures
        for hand in frame_results['hands']:
            if hand['gesture'] != 'unknown':
                timeline['gestures'].append({
                    'frame': frame_num,
                    'gesture': hand['gesture']
                })
        
        # Track poses
        for pose in frame_results['poses']:
            if pose['actions']:
                timeline['poses'].append({
                    'frame': frame_num,
                    'pose': pose['pose'],
                    'actions': pose['actions']
                })
        
        # Identify engagement moments
        if frame_results['summary']['has_interaction']:
            timeline['engagement_moments'].append(frame_num)
        
        # Save annotated frame
        if frame_results['summary']['face_count'] > 0 or frame_results['summary']['hand_count'] > 0:
            annotated_path = os.path.join(video_output_dir, f'annotated_{os.path.basename(frame_path)}')
            detector.save_annotated_frame(frame_path, frame_results, annotated_path)
    
    print(f"\n   ✅ Analysis complete")
    
    # Generate insights
    insights = {
        'video_id': video_id,
        'total_frames': len(frames),
        'human_presence': sum(1 for r in all_frame_results if r['summary']['face_count'] > 0) / len(frames),
        'average_faces': np.mean([r['summary']['face_count'] for r in all_frame_results]),
        'gesture_count': len(timeline['gestures']),
        'expression_variety': len(set(e['expression'] for e in timeline['expressions'])),
        'dominant_expressions': _get_dominant_items([e['expression'] for e in timeline['expressions']]),
        'dominant_gestures': _get_dominant_items([g['gesture'] for g in timeline['gestures']]),
        'engagement_rate': len(timeline['engagement_moments']) / len(frames),
        'timeline': timeline,
        'processed_at': datetime.now().isoformat()
    }
    
    # Save results
    output_file = os.path.join(video_output_dir, f'{video_id}_human_analysis.json')
    with open(output_file, 'w') as f:
        json.dump({
            'insights': insights,
            'frame_details': all_frame_results
        }, f, indent=2)
    
    print(f"   💾 Saved analysis: {output_file}")
    
    # Print summary
    print(f"\n   📊 Human Elements Summary:")
    print(f"      - Human presence: {insights['human_presence']*100:.1f}% of frames")
    print(f"      - Average faces per frame: {insights['average_faces']:.1f}")
    print(f"      - Unique expressions: {insights['expression_variety']}")
    print(f"      - Gesture count: {insights['gesture_count']}")
    print(f"      - Engagement rate: {insights['engagement_rate']*100:.1f}%")
    
    return insights


def _get_dominant_items(items, top_n=3):
    """Get most common items from a list"""
    if not items:
        return []
    
    from collections import Counter
    counter = Counter(items)
    return [item for item, _ in counter.most_common(top_n)]


def main():
    """Main function to analyze all videos"""
    import sys
    
    print("🎭 MediaPipe Human Elements Analyzer")
    print("=" * 50)
    
    # Check for specific video or process all
    if len(sys.argv) > 1:
        video_id = sys.argv[1]
        analyze_video_human_elements(video_id)
    else:
        # Process all videos
        frame_dirs = glob.glob('frame_outputs/*')
        for frame_dir in frame_dirs:
            if os.path.isdir(frame_dir):
                video_id = os.path.basename(frame_dir)
                analyze_video_human_elements(video_id)


if __name__ == "__main__":
    main()