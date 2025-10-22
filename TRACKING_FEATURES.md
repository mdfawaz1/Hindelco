# Automatic Camera Tracking Features

## Overview

The `objectcounting.py` script has been enhanced with automatic PTZ camera tracking functionality that detects when a person approaches the edges of the camera view and automatically moves the camera to keep them in frame.

## How It Works

### 1. Person Detection
- Uses existing Triton inference server for person detection
- Identifies persons in the camera feed in real-time

### 2. Border Zone Detection
- Creates 4 border zones around the frame edges (left, right, top, bottom)
- Border zone size is configurable via `--border-padding` (default: 15% of frame)
- Yellow rectangles show the border zones visually

### 3. Automatic Camera Movement
- When a person's center point enters a border zone, camera moves in that direction
- Uses ONVIF protocol to control PTZ camera
- Integrates with existing `move.py` functions
- Movement cooldown prevents excessive camera movement (default: 2 seconds)

### 4. Visual Feedback
- **Green boxes**: Regular person detections
- **Red boxes**: Person in border zone (triggers camera movement)
- **Yellow rectangles**: Border zones
- **Status overlay**: Shows tracking status and detection counts

## Key Features

### CameraAutoTracker Class
- Handles PTZ camera connection and control
- Configurable border padding and movement cooldown
- Graceful error handling if camera is not available
- Visual border zone rendering

### Enhanced Detection Display
- Modified `draw_detections()` method to show border zones
- Color-coded bounding boxes based on tracking status
- Real-time tracking status display

### Integration with move.py
- Imports camera movement functions from `move.py`
- Uses existing ONVIF camera control logic
- Maintains original camera movement functionality

## Usage Examples

### Basic Usage with Tracking
```bash
python objectcounting.py --video "rtsp://admin:pass@192.168.1.100/stream" --display
```

### Custom Camera Settings
```bash
python objectcounting.py \
    --video "rtsp://camera_stream" \
    --camera-ip 192.168.1.200 \
    --camera-username admin \
    --camera-password mypass \
    --border-padding 0.2
```

### High Sensitivity Tracking
```bash
python objectcounting.py \
    --video "rtsp://camera_stream" \
    --border-padding 0.1 \
    --movement-cooldown 1.0
```

### Disable Tracking
```bash
python objectcounting.py --video input.mp4 --disable-tracking
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--enable-tracking` | True | Enable automatic camera tracking |
| `--disable-tracking` | False | Disable automatic camera tracking |
| `--camera-ip` | 192.168.11.206 | PTZ camera IP address |
| `--camera-port` | 80 | PTZ camera port |
| `--camera-username` | admin | PTZ camera username |
| `--camera-password` | admin@1234 | PTZ camera password |
| `--border-padding` | 0.15 | Border zone size (5%-40% of frame) |
| `--movement-cooldown` | 2.0 | Minimum seconds between movements |

## Technical Implementation

### Files Modified
- `objectcounting.py`: Main script with tracking integration
- `move.py`: Existing camera movement functions (unchanged)

### New Classes Added
- `CameraAutoTracker`: Handles camera tracking logic
  - `_initialize_camera()`: Sets up ONVIF connection
  - `get_border_zones()`: Calculates border zone coordinates
  - `check_person_in_border()`: Detects person in border zones
  - `move_camera()`: Triggers camera movement
  - `draw_border_zones()`: Renders border zones visually

### Integration Points
- Modified `draw_detections()` to include border zones and tracking status
- Enhanced `process_video()` function with tracking parameters
- Added command-line arguments for tracking configuration
- Integrated tracking logic into main processing loop

## Error Handling
- Graceful degradation if PTZ camera is not available
- Continues with detection-only mode if camera connection fails
- Movement cooldown prevents excessive camera commands
- Visual indicators show tracking status

## Dependencies
- Existing dependencies from original script
- `onvif` library for PTZ camera control
- `zeep` library for SOAP communication

## Benefits
1. **Automatic Tracking**: No manual camera control needed
2. **Configurable**: Adjustable sensitivity and timing
3. **Visual Feedback**: Clear indication of tracking status
4. **Robust**: Works with or without PTZ camera
5. **Integrated**: Seamlessly works with existing detection pipeline
