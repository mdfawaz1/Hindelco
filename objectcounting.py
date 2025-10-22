#!/usr/bin/env python3
"""
Video Inference Script for YOLO Person Detection with Auto Camera Tracking
Processes video files or RTSP streams with hardware-accelerated decoding
Features automatic PTZ camera tracking when persons approach frame borders
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import argparse
from pathlib import Path
import time
import sys
from collections import deque
import queue
import threading

import gi
gi.require_version('GLib', '2.0')
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

# Import camera movement functions
from move import continuous_move, move_up, move_down, move_left, move_right, zoom_up, zoom_dowm
from onvif import ONVIFCamera
import zeep


class YOLOPersonDetector:
    """Person detection using YOLO model"""
    
    def __init__(self, model_path='yolov8n.pt', device='auto'):
        self.model_path = model_path
        
        # Model configuration
        self.classes = ['PERSON']  # YOLO COCO class 0 is 'person'
        self.colors = [(93, 173, 236)]  # BGR color for person
        self.confidence_threshold = 0.25
        self.person_class_id = 0  # COCO dataset person class ID
        
        # Device configuration
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"[INFO] Using device: {self.device}")
        
        # Load YOLO model
        print(f"[INFO] Loading YOLO model from {model_path}...")
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # Get model info
            print(f"[INFO] Model loaded successfully")
            print(f"[INFO] Model classes: {len(self.model.names)} classes")
            print(f"[INFO] Person class available: {'person' in self.model.names.values()}")
            
            # Find person class ID in the model
            for class_id, class_name in self.model.names.items():
                if class_name.lower() == 'person':
                    self.person_class_id = class_id
                    break
            
            print(f"[INFO] Person class ID: {self.person_class_id}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def infer(self, image):
        """Run YOLO inference"""
        try:
            # Run YOLO inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            
            # Convert YOLO results to our detection format
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates, confidence, and class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Only keep person detections
                        if class_id == self.person_class_id and conf >= self.confidence_threshold:
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class_id': 0,  # We map all persons to class 0
                                'class_name': 'PERSON'
                            })
            
            return detections
            
        except Exception as e:
            print(f"[ERROR] YOLO inference failed: {e}")
            return []
    
    def draw_detections(self, image, detections, show_fps=True, fps=0, 
                       camera_tracker=None, border_zone=None):
        """Draw bounding boxes and info overlay"""
        result = image.copy()
        
        # Draw border zones if camera tracker is provided
        if camera_tracker:
            result = camera_tracker.draw_border_zones(result)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_id = det['class_id']
            class_name = det['class_name']
            color = self.colors[class_id]
            
            # Highlight person in border zone
            if class_name == 'PERSON' and border_zone:
                color = (0, 0, 255)  # Red for person in border zone
                thickness = 3
            else:
                thickness = 2
            
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{class_name}: {conf:.2f}"
            if class_name == 'PERSON' and border_zone:
                label += f" [{border_zone.upper()}]"
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(result, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw info overlay
        if show_fps:
            info_y = 30
            overlay_height = 120 if border_zone else 90
            cv2.rectangle(result, (10, 10), (400, 10 + overlay_height), (0, 0, 0), -1)
            cv2.putText(result, f"FPS: {fps:.1f}", (20, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result, f"Detections: {len(detections)}", (20, info_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Count by class
            person_count = sum(1 for d in detections if d['class_name'] == 'PERSON')
            cv2.putText(result, f"PERSONS: {person_count}", 
                       (20, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show camera tracking status
            if border_zone:
                tracking_status = f"TRACKING: Person in {border_zone.upper()} zone"
                cv2.putText(result, tracking_status, (20, info_y + 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif camera_tracker and camera_tracker.camera_initialized:
                cv2.putText(result, "AUTO-TRACKING: Active", (20, info_y + 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return result
    
    def close(self):
        """Close YOLO model (cleanup if needed)"""
        # YOLO models are automatically cleaned up by garbage collection
        # but we can explicitly delete the model if needed
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


class FPSCounter:
    """Calculate FPS over a moving window"""
    def __init__(self, window_size=30):
        self.timestamps = deque(maxlen=window_size)
    
    def update(self):
        """Update with current timestamp"""
        self.timestamps.append(time.time())
    
    def get_fps(self):
        """Get current FPS"""
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return len(self.timestamps) / elapsed if elapsed > 0 else 0.0


class CameraAutoTracker:
    """Automatic camera tracking for person detection"""
    
    def __init__(self, camera_ip='192.168.11.206', camera_port=80, 
                 username='admin', password='admin@1234', 
                 border_padding=0.15, movement_cooldown=2.0, movement_duration=0.3):
        """
        Initialize camera auto-tracker
        
        Args:
            camera_ip: IP address of the PTZ camera
            camera_port: Port of the PTZ camera  
            username: Camera username
            password: Camera password
            border_padding: Border zone size as percentage of frame (0.0-0.5)
            movement_cooldown: Minimum seconds between camera movements
            movement_duration: Duration of each camera movement in seconds (0.1-2.0)
        """
        self.camera_ip = camera_ip
        self.camera_port = camera_port
        self.username = username
        self.password = password
        self.border_padding = max(0.05, min(0.4, border_padding))  # Clamp between 5-40%
        self.movement_cooldown = movement_cooldown
        self.movement_duration = max(0.1, min(2.0, movement_duration))  # Clamp between 0.1-2.0 seconds
        
        # Movement state
        self.last_movement_time = 0
        self.ptz = None
        self.request = None
        self.camera_initialized = False
        
        # Initialize camera connection
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize PTZ camera connection"""
        try:
            print(f"[INFO] Initializing PTZ camera at {self.camera_ip}:{self.camera_port}")
            
            # Set up zeep to handle responses properly
            def zeep_pythonvalue(self, xmlvalue):
                return xmlvalue
            zeep.xsd.simple.AnySimpleType.pythonvalue = zeep_pythonvalue
            
            # Create ONVIF camera object
            self.mycam = ONVIFCamera(self.camera_ip, self.camera_port, 
                                   self.username, self.password)
            
            # Create media and PTZ services
            media = self.mycam.create_media_service()
            self.ptz = self.mycam.create_ptz_service()
            
            # Get profile and PTZ configuration
            media_profile = media.GetProfiles()[0]
            
            # Get PTZ configuration options
            request = self.ptz.create_type('GetConfigurationOptions')
            request.ConfigurationToken = media_profile.PTZConfiguration.token
            ptz_configuration_options = self.ptz.GetConfigurationOptions(request)
            
            # Create move request template
            self.request = self.ptz.create_type('ContinuousMove')
            self.request.ProfileToken = media_profile.token
            self.ptz.Stop({'ProfileToken': media_profile.token})
            
            # Initialize velocity
            if self.request.Velocity is None:
                self.request.Velocity = self.ptz.GetStatus({'ProfileToken': media_profile.token}).Position
                self.request.Velocity.PanTilt.space = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].URI
                self.request.Velocity.Zoom.space = ptz_configuration_options.Spaces.ContinuousZoomVelocitySpace[0].URI
            
            self.camera_initialized = True
            print("[INFO] PTZ camera initialized successfully")
            
        except Exception as e:
            print(f"[WARNING] Failed to initialize PTZ camera: {e}")
            print("[INFO] Continuing without camera movement functionality")
            self.camera_initialized = False
    
    def get_border_zones(self, frame_width, frame_height):
        """Get border zone coordinates"""
        padding_x = int(frame_width * self.border_padding)
        padding_y = int(frame_height * self.border_padding)
        
        zones = {
            'left': (0, 0, padding_x, frame_height),
            'right': (frame_width - padding_x, 0, frame_width, frame_height),
            'top': (0, 0, frame_width, padding_y),
            'bottom': (0, frame_height - padding_y, frame_width, frame_height)
        }
        
        return zones
    
    def check_person_in_border(self, detections, frame_width, frame_height):
        """Check if any person is in border zones"""
        if not detections:
            return None
        
        border_zones = self.get_border_zones(frame_width, frame_height)
        
        # Check each person detection
        for detection in detections:
            if detection['class_name'] != 'PERSON':
                continue
                
            x1, y1, x2, y2 = detection['bbox']
            person_center_x = (x1 + x2) // 2
            person_center_y = (y1 + y2) // 2
            
            # Check which border zone the person is in
            for zone_name, (zx1, zy1, zx2, zy2) in border_zones.items():
                if zx1 <= person_center_x <= zx2 and zy1 <= person_center_y <= zy2:
                    return zone_name
        
        return None
    
    def move_camera(self, direction):
        """Move camera in specified direction"""
        if not self.camera_initialized or not self.ptz or not self.request:
            return False
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_movement_time < self.movement_cooldown:
            return False
        
        try:
            if direction == 'left':
                print(f"[CAMERA] Moving left ({self.movement_duration}s) to track person")
                move_left(self.ptz, self.request, self.movement_duration)
            elif direction == 'right':
                print(f"[CAMERA] Moving right ({self.movement_duration}s) to track person")
                move_right(self.ptz, self.request, self.movement_duration)
            elif direction == 'top':
                print(f"[CAMERA] Moving up ({self.movement_duration}s) to track person")
                move_up(self.ptz, self.request, self.movement_duration)
            elif direction == 'bottom':
                print(f"[CAMERA] Moving down ({self.movement_duration}s) to track person")
                move_down(self.ptz, self.request, self.movement_duration)
            
            self.last_movement_time = current_time
            return True
            
        except Exception as e:
            print(f"[ERROR] Camera movement failed: {e}")
            return False
    
    def draw_border_zones(self, image):
        """Draw border zones on image"""
        height, width = image.shape[:2]
        zones = self.get_border_zones(width, height)
        
        # Draw border rectangles
        border_color = (0, 255, 255)  # Yellow
        thickness = 2
        
        for zone_name, (x1, y1, x2, y2) in zones.items():
            cv2.rectangle(image, (x1, y1), (x2, y2), border_color, thickness)
            
            # Add zone label
            label_pos = (x1 + 10, y1 + 25)
            cv2.putText(image, zone_name.upper(), label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2)
        
        return image


class GStreamerCapture:
    """GStreamer-based video capture for RTSP and file sources"""
    
    def __init__(self, source, use_gstreamer=True):
        self.source = source
        self.use_gstreamer = use_gstreamer
        self.frame_queue = queue.Queue(maxsize=10)
        self.is_running_flag = False
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.total_frames = 0
        self.error = None
        
        # Determine if source is RTSP
        self.is_rtsp = source.startswith(('rtsp://', 'rtmp://'))
        
        if not self.use_gstreamer or not self.is_rtsp:
            # Fallback to OpenCV for non-RTSP or when GStreamer disabled
            self._init_opencv()
        else:
            # Use GStreamer for RTSP
            self._init_gstreamer()
    
    def _init_opencv(self):
        """Initialize OpenCV capture"""
        print("[INFO] Using OpenCV backend")
        if isinstance(self.source, int) or self.source.isdigit():
            self.cap = cv2.VideoCapture(int(self.source))
        else:
            self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.is_running_flag = True
    
    def _init_gstreamer(self):
        """Initialize GStreamer pipeline"""
        print("[INFO] Using GStreamer backend for RTSP")
        Gst.init(None)
        
        # Try different decoder options (in order of preference)
        decoders = [
            'decodebin',  # Auto-select best decoder
            'avdec_h264',  # Software decoder (libav)
            'openh264dec',  # Alternative software decoder
        ]
        
        decoder = None
        for dec in decoders:
            element = Gst.ElementFactory.make(dec.split('!')[0], None)
            if element is not None:
                decoder = dec
                print(f"[INFO] Using decoder: {decoder}")
                break
        
        if decoder is None:
            raise RuntimeError("No suitable H.264 decoder found. Install: sudo apt-get install gstreamer1.0-libav")
        
        # Build GStreamer pipeline with selected decoder
        if decoder == 'decodebin':
            pipeline_str = f"""
                rtspsrc location={self.source} latency=200 protocols=tcp !
                decodebin !
                videoconvert !
                video/x-raw,format=BGR !
                appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
            """
        else:
            pipeline_str = f"""
                rtspsrc location={self.source} latency=200 protocols=tcp !
                rtph264depay !
                h264parse !
                {decoder} !
                videoconvert !
                video/x-raw,format=BGR !
                appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
            """
        
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name('sink')
        
        # Get bus for messages
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)
        
        # Start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # Wait for stream info
        print("[INFO] Waiting for stream information...")
        timeout = time.time() + 10
        while time.time() < timeout:
            sample = self.appsink.emit('try-pull-sample', Gst.SECOND)
            if sample:
                caps = sample.get_caps()
                struct = caps.get_structure(0)
                self.width = struct.get_int('width')[1]
                self.height = struct.get_int('height')[1]
                fps_struct = struct.get_fraction('framerate')
                self.fps = fps_struct[1] / fps_struct[2] if fps_struct[0] else 25.0
                self.total_frames = 0  # Unknown for streams
                break
            time.sleep(0.1)
        
        if self.width == 0:
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Could not get stream information")
        
        self.is_running_flag = True
        
        # Start frame reading thread
        self.gst_thread = threading.Thread(target=self._gstreamer_loop, daemon=True)
        self.gst_thread.start()
    
    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages"""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.error = f"GStreamer Error: {err}"
            print(f"[ERROR] {self.error}")
            self.is_running_flag = False
        elif t == Gst.MessageType.EOS:
            print("[INFO] End of stream")
            self.is_running_flag = False
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"[WARNING] GStreamer: {warn}")
    
    def _gstreamer_loop(self):
        """Read frames from GStreamer in separate thread"""
        while self.is_running_flag:
            sample = self.appsink.emit('try-pull-sample', Gst.SECOND // 2)
            if sample:
                buffer = sample.get_buffer()
                caps = sample.get_caps()
                
                # Get frame dimensions
                struct = caps.get_structure(0)
                width = struct.get_int('width')[1]
                height = struct.get_int('height')[1]
                
                # Extract frame data
                success, map_info = buffer.map(Gst.MapFlags.READ)
                if success:
                    frame = np.ndarray(
                        shape=(height, width, 3),
                        dtype=np.uint8,
                        buffer=map_info.data
                    )
                    frame = frame.copy()  # Make a copy since buffer will be unmapped
                    buffer.unmap(map_info)
                    
                    # Put in queue (drop old frames if full)
                    try:
                        self.frame_queue.put(frame, block=False)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(frame, block=False)
                        except:
                            pass
    
    def read(self):
        """Read a frame"""
        if not self.use_gstreamer or not self.is_rtsp:
            return self.cap.read()
        else:
            try:
                frame = self.frame_queue.get(timeout=5)
                return True, frame
            except queue.Empty:
                if self.error:
                    print(f"[ERROR] Stream error: {self.error}")
                return False, None
    
    def isOpened(self):
        """Check if capture is open"""
        return self.is_running_flag
    
    def get(self, prop):
        """Get property (OpenCV compatibility)"""
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        elif prop == cv2.CAP_PROP_FPS:
            return self.fps
        elif prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.total_frames
        return 0
    
    def release(self):
        """Release capture"""
        self.is_running_flag = False
        if hasattr(self, 'cap'):
            self.cap.release()
        elif hasattr(self, 'pipeline'):
            self.pipeline.set_state(Gst.State.NULL)
            if hasattr(self, 'gst_thread'):
                self.gst_thread.join(timeout=2)


def process_video(detector, video_source, output_path=None, display=True, 
                 skip_frames=0, max_frames=None, use_gstreamer=True,
                 enable_tracking=True, camera_ip='192.168.11.206', 
                 camera_port=80, camera_username='admin', camera_password='admin@1234',
                 border_padding=0.15, movement_cooldown=2.0, movement_duration=0.3):
    """
    Process video file or stream with optional camera tracking
    
    Args:
        detector: YOLOPersonDetector instance
        video_source: Video file path, camera index, or RTSP URL
        output_path: Optional path to save output video
        display: Show video window
        skip_frames: Skip N frames between processing
        max_frames: Maximum frames to process
        use_gstreamer: Use GStreamer for RTSP (recommended)
        enable_tracking: Enable automatic camera tracking
        camera_ip: PTZ camera IP address
        camera_port: PTZ camera port
        camera_username: PTZ camera username
        camera_password: PTZ camera password
        border_padding: Border zone size as percentage (0.05-0.4)
        movement_cooldown: Minimum seconds between camera movements
        movement_duration: Duration of each camera movement (0.1-2.0 seconds)
    """
    
    # Initialize camera tracker if enabled
    camera_tracker = None
    if enable_tracking:
        try:
            camera_tracker = CameraAutoTracker(
                camera_ip=camera_ip,
                camera_port=camera_port,
                username=camera_username,
                password=camera_password,
                border_padding=border_padding,
                movement_cooldown=movement_cooldown,
                movement_duration=movement_duration
            )
            print(f"[INFO] Camera auto-tracking initialized")
        except Exception as e:
            print(f"[WARNING] Camera tracking disabled: {e}")
            camera_tracker = None
    else:
        print("[INFO] Camera auto-tracking disabled")
    
    # Open video
    print(f"[INFO] Opening video source: {video_source}")
    try:
        cap = GStreamerCapture(video_source, use_gstreamer=use_gstreamer)
    except Exception as e:
        print(f"[ERROR] Could not open video source: {e}")
        return
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video source")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[INFO] Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps_input:.2f}")
    if total_frames > 0:
        print(f"  Total frames: {total_frames}")
    else:
        print(f"  Total frames: Unknown (live stream)")
    
    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps_input, (width, height))
        print(f"[INFO] Writing output to: {output_path}")
    
    # FPS counter
    fps_counter = FPSCounter(window_size=30)
    
    # Processing stats
    frame_count = 0
    processed_count = 0
    total_detections = 0
    
    print(f"\n[INFO] Starting video processing...")
    print(f"[INFO] Press 'q' to quit, 'p' to pause")
    
    paused = False
    start_time = time.time()
    
    try:
        while cap.isOpened():
            if not paused:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] Stream ended or read failed")
                    break
                
                frame_count += 1
                
                # Skip frames if requested
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    continue
                
                # Process frame
                detections = detector.infer(frame)
                processed_count += 1
                total_detections += len(detections)
                
                # Check for person in border zones and move camera if needed
                border_zone = None
                if camera_tracker:
                    border_zone = camera_tracker.check_person_in_border(detections, width, height)
                    if border_zone:
                        # Move camera to follow person
                        moved = camera_tracker.move_camera(border_zone)
                        if moved:
                            print(f"[TRACKING] Camera moved {border_zone} to follow person")
                
                # Update FPS
                fps_counter.update()
                current_fps = fps_counter.get_fps()
                
                # Draw results with border zones and tracking info
                output_frame = detector.draw_detections(frame, detections, 
                                                       show_fps=True, fps=current_fps,
                                                       camera_tracker=camera_tracker,
                                                       border_zone=border_zone)
                
                # Write to file
                if writer:
                    writer.write(output_frame)
                
                # Display
                if display:
                    cv2.imshow('Triton Video Inference', output_frame)
                
                # Progress update
                if processed_count % 30 == 0:
                    if total_frames > 0:
                        progress = (frame_count / total_frames * 100)
                        print(f"[PROGRESS] Frames: {frame_count}/{total_frames} ({progress:.1f}%) | "
                              f"FPS: {current_fps:.1f} | Detections: {len(detections)}")
                    else:
                        print(f"[PROGRESS] Frames: {frame_count} | "
                              f"FPS: {current_fps:.1f} | Detections: {len(detections)}")
                
                # Check max frames
                if max_frames and processed_count >= max_frames:
                    print(f"[INFO] Reached max frames limit: {max_frames}")
                    break
            
            # Handle keyboard input
            if display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] Quit requested by user")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"[INFO] {'Paused' if paused else 'Resumed'}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Calculate final stats
        elapsed_time = time.time() - start_time
        avg_fps = processed_count / elapsed_time if elapsed_time > 0 else 0
        avg_detections = total_detections / processed_count if processed_count > 0 else 0
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total frames:         {frame_count}")
        print(f"Processed frames:     {processed_count}")
        print(f"Total time:           {elapsed_time:.2f}s")
        print(f"Average FPS:          {avg_fps:.2f}")
        print(f"Total detections:     {total_detections}")
        print(f"Avg detections/frame: {avg_detections:.2f}")
        if output_path:
            print(f"Output saved to:      {output_path}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Video Inference with YOLO Person Detection and Auto Camera Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process RTSP stream with auto-tracking (GStreamer - recommended)
  python objectcounting.py --video 'rtsp://user:pass@192.168.1.100/stream' --display
  
  # Process RTSP stream with custom YOLO model
  python objectcounting.py --video 'rtsp://...' --model yolov8s.pt --device cuda
  
  # Process video file with tracking disabled
  python objectcounting.py --video input.mp4 --output output.mp4 --disable-tracking
  
  # Use webcam with custom border padding
  python objectcounting.py --video 0 --display --border-padding 0.2
  
  # High sensitivity tracking (smaller border zones, faster movement)
  python objectcounting.py --video 'rtsp://...' --border-padding 0.1 --movement-cooldown 1.0 --movement-duration 0.2
  
  # Use custom YOLO model with CPU
  python objectcounting.py --video 'rtsp://...' --model custom_yolo.pt --device cpu
  
  # Disable GStreamer (use OpenCV for all sources)
  python objectcounting.py --video 'rtsp://...' --no-gstreamer
        """
    )
    
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device for YOLO inference: auto, cpu, cuda (default: auto)')
    parser.add_argument('--video', type=str, required=True,
                       help='Video file path, camera index (0 for webcam), or RTSP URL')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (optional)')
    parser.add_argument('--threshold', type=float, default=0.25,
                       help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--display', action='store_true', default=True,
                       help='Display video window (default: True)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Skip N frames between processing (for speed)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    parser.add_argument('--no-gstreamer', action='store_true',
                       help='Disable GStreamer, use OpenCV only')
    parser.add_argument('--enable-tracking', action='store_true', default=True,
                       help='Enable automatic camera tracking (default: True)')
    parser.add_argument('--disable-tracking', action='store_true',
                       help='Disable automatic camera tracking')
    parser.add_argument('--camera-ip', type=str, default='192.168.11.206',
                       help='PTZ camera IP address (default: 192.168.11.206)')
    parser.add_argument('--camera-port', type=int, default=80,
                       help='PTZ camera port (default: 80)')
    parser.add_argument('--camera-username', type=str, default='admin',
                       help='PTZ camera username (default: admin)')
    parser.add_argument('--camera-password', type=str, default='admin@1234',
                       help='PTZ camera password (default: admin@1234)')
    parser.add_argument('--border-padding', type=float, default=0.15,
                       help='Border zone size as percentage of frame (0.05-0.4, default: 0.15)')
    parser.add_argument('--movement-cooldown', type=float, default=2.0,
                       help='Minimum seconds between camera movements (default: 2.0)')
    parser.add_argument('--movement-duration', type=float, default=0.3,
                       help='Duration of each camera movement in seconds (0.1-2.0, default: 0.3)')
    
    args = parser.parse_args()
    
    # Handle display flag
    display = args.display and not args.no_display
    use_gstreamer = not args.no_gstreamer
    enable_tracking = args.enable_tracking and not args.disable_tracking
    
    # Validate border padding
    border_padding = max(0.05, min(0.4, args.border_padding))
    if border_padding != args.border_padding:
        print(f"[WARNING] Border padding clamped to {border_padding:.2f} (valid range: 0.05-0.4)")
    
    # Validate movement duration
    movement_duration = max(0.1, min(2.0, args.movement_duration))
    if movement_duration != args.movement_duration:
        print(f"[WARNING] Movement duration clamped to {movement_duration:.1f}s (valid range: 0.1-2.0)")
    
    # Validate video source (skip check for streams)
    if not args.video.isdigit() and not args.video.startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
        if not Path(args.video).exists():
            print(f"[ERROR] Video file not found: {args.video}")
            sys.exit(1)
    
    # Initialize detector
    try:
        detector = YOLOPersonDetector(
            model_path=args.model,
            device=args.device
        )
        detector.confidence_threshold = args.threshold
    except Exception as e:
        print(f"[ERROR] Failed to initialize YOLO detector: {e}")
        sys.exit(1)
    
    # Process video
    try:
        process_video(
            detector=detector,
            video_source=args.video,
            output_path=args.output,
            display=display,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames,
            use_gstreamer=use_gstreamer,
            enable_tracking=enable_tracking,
            camera_ip=args.camera_ip,
            camera_port=args.camera_port,
            camera_username=args.camera_username,
            camera_password=args.camera_password,
            border_padding=border_padding,
            movement_cooldown=args.movement_cooldown,
            movement_duration=movement_duration
        )
    except Exception as e:
        print(f"[ERROR] Video processing failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        detector.close()
    
    print("[INFO] Done!")


if __name__ == '__main__':
    main()