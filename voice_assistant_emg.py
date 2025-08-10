#!/usr/bin/env python3
"""
Simple AI Voice Assistant with EMG Processing and Visualization
Reads raw EMG data from Arduino, processes it in Python, and triggers voice assistant.
Uses Google's Gemini for speech-to-text, gTTS for text-to-speech, and Matplotlib for visualization.

Features:
- Two modes: Gesture Creation (default) and Gesture Execution
- Double-tap detection for starting recording in creation mode
- Single-tap detection for stopping recording in creation mode
- Direct gesture execution in execution mode (grip2open/grip2closed)
- Anti-double-detection debouncing (configurable via BURST_DEBOUNCE_TIME)
- Real-time EMG visualization
- Enhanced audio quality for better speech recognition

Environment Variables for Audio Quality:
- SAMPLE_RATE: Audio sample rate (default: 44100 Hz)
- CHUNK_SIZE: Audio chunk size (default: 512 samples)
- NOISE_REDUCTION_STRENGTH: Noise reduction intensity (default: 0.2)
- HIGH_PASS_FREQ: Low frequency cutoff (default: 60 Hz)
- LOW_PASS_FREQ: High frequency cutoff (default: 12000 Hz)
"""

import os
import time
import tempfile
import threading
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy import signal
from scipy.ndimage import uniform_filter1d
from dotenv import load_dotenv
import google.generativeai as genai
from gtts import gTTS
import pygame
import serial
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend which is more widely available
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json

# Porcupine imports for wake word detection
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    print("⚠️  Porcupine not available. Install with: pip install pvporcupine")
    PORCUPINE_AVAILABLE = False

# Load environment variables
load_dotenv()

def create_beep(filename="trigger.wav", freq=440, duration=0.5, sample_rate=44100):
    """Generate a beep sound if trigger.wav is missing"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    wav.write(filename, sample_rate, (audio * 32767).astype(np.int16))
    print(f"Created {filename}")

class SimpleVoiceAssistant:
    """EMG-Controlled Audio Recording System with Dual Mode Support
    
    This system has two modes:
    1. Gesture Creation Mode (default): Records audio when triggered by EMG muscle contractions
    2. Gesture Execution Mode: Executes predefined gestures based on EMG triggers
    
    Mode switching is done via wake word + voice command.
    """
    def __init__(self):
        """Initialize the voice assistant and EMG processing"""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Configure Google AI
        genai.configure(api_key=self.api_key)
        
        # Load gestures and servo mappings
        self.gestures = self.load_gestures()
        self.servo_limits = self.load_servo_limits()
        
        # Mode system
        self.current_mode = "gesture_creation"  # Default mode
        self.current_grip_state = "grip2open"  # Default grip state for execution mode
        self.mode_switch_commands = [
            "switch control mode", "switch to gesture creation", "gesture creation mode",
            "switch to gesture execution", "gesture execution mode", "execution mode"
        ]
        
        # Audio configuration - Improved for better speech recognition
        self.sample_rate = int(os.getenv('SAMPLE_RATE', 44100))  # Increased from 24kHz to 44.1kHz for better quality
        self.channels = int(os.getenv('CHANNELS', 1))
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 512))     # Reduced from 1024 for better responsiveness
        self.min_recording_duration = float(os.getenv('MIN_RECORDING_DURATION', 1.0))
        self.max_recording_duration = float(os.getenv('MAX_RECORDING_DURATION', 8.0))
        
        # Noise reduction configuration - Less aggressive for speech
        self.enable_noise_reduction = os.getenv('ENABLE_NOISE_REDUCTION', 'true').lower() == 'true'
        self.noise_reduction_strength = float(os.getenv('NOISE_REDUCTION_STRENGTH', 0.2))  # Reduced from 0.3 to 0.2
        self.noise_gate_threshold = float(os.getenv('NOISE_GATE_THRESHOLD', 0.005))  # Reduced from 0.01 to 0.005
        self.high_pass_freq = float(os.getenv('HIGH_PASS_FREQ', 60.0))  # Reduced from 80Hz to 60Hz (less aggressive)
        self.low_pass_freq = float(os.getenv('LOW_PASS_FREQ', 12000.0))  # Increased from 8kHz to 12kHz (preserve more speech)
        
        # Enhanced system prompt with gesture context
        self.system_prompt = self.generate_system_prompt()
        
        # Initialize models
        self.speech_model = genai.GenerativeModel('gemini-1.5-flash')
        self.llm_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Initialize Porcupine wake word detection
        self.porcupine = None
        if PORCUPINE_AVAILABLE:
            try:
                access_key = os.getenv('PORCUPINE_ACCESS_KEY')
                if not access_key:
                    print("⚠️  PORCUPINE_ACCESS_KEY not found in environment variables")
                else:
                    # Initialize Porcupine with HeyRonin.ppn
                    ppn_path = os.path.join(os.path.dirname(__file__), 'HeyRonin.ppn')
                    if os.path.exists(ppn_path):
                        self.porcupine = pvporcupine.create(
                            access_key=access_key,
                            keyword_paths=[ppn_path],
                            sensitivities=[0.9]
                        )
                        print("✅ Porcupine wake word detection initialized with HeyRonin.ppn")
                    else:
                        print(f"⚠️  HeyRonin.ppn not found at {ppn_path}")
            except Exception as e:
                print(f"❌ Failed to initialize Porcupine: {e}")
        
        # Load or create trigger sound
        self.trigger_sound_file = os.getenv('TRIGGER_SOUND_FILE', 'trigger.wav')
        if not os.path.exists(self.trigger_sound_file):
            print(f"⚠️ Warning: Trigger sound file '{self.trigger_sound_file}' not found. Generating default beep.")
            create_beep(self.trigger_sound_file)
        
        # Create tmp directory
        os.makedirs("tmp", exist_ok=True)
        
        # Serial port - try to auto-detect Arduino
        self.serial_port = self.detect_arduino_port()
        self.serial = None
        self.serial_lock = threading.Lock()
        
        # EMG processing parameters - Balanced sensitivity for reliable detection
        self.fs = 200.0
        self.sensitivity_multiplier = float(os.getenv('SENSITIVITY_MULTIPLIER', 0.2))  # Global sensitivity (0.5 = sensitive, 1.0 = normal, 2.0 = less sensitive)
        self.threshold_multiplier = float(os.getenv('THRESHOLD_MULTIPLIER', 1.2)) * self.sensitivity_multiplier
        self.burst_min_duration = float(os.getenv('BURST_MIN_DURATION', 0.02))  # Don't multiply timing by sensitivity
        self.burst_max_duration = float(os.getenv('BURST_MAX_DURATION', 0.5))   # Don't multiply timing by sensitivity
        self.double_tap_max_interval = float(os.getenv('DOUBLE_TAP_MAX_INTERVAL', 2.0))  # Don't multiply timing by sensitivity
        self.cooldown_time = float(os.getenv('COOLDOWN_TIME', 1.5))            # Don't multiply timing by sensitivity
        self.min_recording_time = float(os.getenv('MIN_RECORDING_TIME', 1.5))   # Don't multiply timing by sensitivity
        # Dynamic burst debounce time based on mode
        self.base_burst_debounce_time = float(os.getenv('BURST_DEBOUNCE_TIME', 0.08))  # Base debounce time
        self.execution_mode_debounce_time = float(os.getenv('EXECUTION_MODE_DEBOUNCE_TIME', 0.07))  # Faster for execution mode
        self.burst_debounce_time = self.base_burst_debounce_time  # Current debounce time (will be updated dynamically)
        
        # Filter coefficients
        self.b_bandpass = np.array([0.2929, 0, -0.2929])
        self.a_bandpass = np.array([1.0, -0.1716, 0.4142])
        self.bandpass_zi = signal.lfilter_zi(self.b_bandpass, self.a_bandpass)  # Shape (2,)
        self.baseline = 0.0
        self.burst_threshold = 1.0
        
        # EMG state - Enhanced for stability
        self.in_burst = False
        self.trigger = False
        self.burst_start = 0.0
        self.first_burst_end = 0.0
        self.last_detection_time = 0.0
        self.burst_count = 0
        self.last_trigger_time = 0.0  # Track last successful trigger
        self.in_cooldown = False      # Cooldown state
        self.hysteresis_threshold = 0.0  # Dynamic hysteresis threshold
        
        # Anti-double-detection mechanism
        self.last_burst_end_time = 0.0  # Track when last burst ended
        # Smart debouncing: only prevents rapid consecutive first taps
        # Allows second taps in double-tap sequences even if close together
        
        # New state management for the interaction model
        self.assistant_state = "IDLE"  # IDLE, RECORDING, PROCESSING
        self.recording_start_time = 0.0
        self.recording_thread = None
        self.recording_stop_event = threading.Event()
        self.audio_frames = []
        
        # EMG visualization data
        self.emg_data = []
        self.raw_data = []
        self.in_burst_data = []
        self.trigger_data = []
        self.time_data = []
        self.max_points = 200
        self.start_time = time.time()
        self.running = True
        
        # Audio calibration - Auto-adjust settings based on microphone
        self._calibrate_audio()
        self.buffer = ""
        
        # Dynamic recalibration parameters - DISABLED for stability
        self.auto_recalibrate = False  # No auto-calibration during operation
        self.recalibrate_interval = 30.0  # Not used when disabled
        self.last_calibration_time = time.time()
        self.signal_quality_threshold = 0.8  # Not used when disabled
        
        # Connect to serial and perform initial calibration
        self.connect_serial()
        if self.serial:
            self.perform_dynamic_calibration()
        else:
            print("⚠️ Warning: No serial connection available. EMG features will be disabled.")
        
        # Test microphone and audio system
        self.test_audio_system()
        
        # Print initial mode status
        print(f"🎯 Initialized in {self.current_mode} mode")
        print(f"   Use wake word + 'switch control mode' to change modes")
        print(f"   📝 Gesture Creation Mode (default): EMG triggers activate voice assistant for commands")
        print(f"   🎯 Gesture Execution Mode: EMG triggers execute grip2open/grip2closed gestures directly")
        print(f"   💡 Say 'HeyRonin, switch control mode' to toggle between modes")
    
    def _calibrate_audio(self):
        """Auto-calibrate audio settings based on microphone characteristics"""
        print("\n🎤 Audio Calibration...")
        print("=" * 40)
        
        try:
            # Test recording to measure microphone characteristics
            print("Recording test audio for calibration...")
            test_audio = sd.rec(
                int(2 * self.sample_rate),  # 2 seconds
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32
            )
            sd.wait()
            
            if test_audio is not None and len(test_audio) > 0:
                audio_level = np.abs(test_audio).mean()
                audio_max = np.abs(test_audio).max()
                
                print(f"📊 Audio Analysis:")
                print(f"   Average level: {audio_level:.6f}")
                print(f"   Peak level: {audio_max:.6f}")
                
                # Auto-adjust settings based on audio levels
                if audio_level < 0.001:
                    print("   ⚠️ Very low audio level detected")
                    print("   🔧 Adjusting settings for ultra-low-sensitivity microphone...")
                    # Disable noise reduction entirely for very low levels
                    self.noise_gate_threshold = 0.0001  # Extremely low threshold
                    self.noise_reduction_strength = 0.0  # No noise reduction
                    self.high_pass_freq = 30.0  # Very low high-pass to preserve all signal
                    print(f"   ✅ Adjusted: noise_gate={self.noise_gate_threshold}, noise_reduction={self.noise_reduction_strength}")
                    
                elif audio_level < 0.01:
                    print("   ⚠️ Low audio level detected")
                    print("   🔧 Adjusting settings for low-sensitivity microphone...")
                    # Minimal noise reduction for low levels
                    self.noise_gate_threshold = 0.0005  # Very low threshold
                    self.noise_reduction_strength = 0.05  # Minimal noise reduction
                    self.high_pass_freq = 40.0  # Low high-pass to preserve more signal
                    print(f"   ✅ Adjusted: noise_gate={self.noise_gate_threshold}, noise_reduction={self.noise_reduction_strength}")
                    
                elif audio_level < 0.1:
                    print("   ⚠️ Medium-low audio level detected")
                    print("   🔧 Adjusting settings for medium-sensitivity microphone...")
                    self.noise_gate_threshold = 0.002
                    self.noise_reduction_strength = 0.1
                    self.high_pass_freq = 50.0
                    print(f"   ✅ Adjusted: noise_gate={self.noise_gate_threshold}, noise_reduction={self.noise_reduction_strength}")
                    
                elif audio_level > 0.5:
                    print("   ⚠️ High audio level detected (possible clipping)")
                    print("   🔧 Adjusting settings for high-sensitivity microphone...")
                    self.noise_gate_threshold = 0.01
                    self.noise_reduction_strength = 0.3
                    self.high_pass_freq = 80.0
                    print(f"   ✅ Adjusted: noise_gate={self.noise_gate_threshold}, noise_reduction={self.noise_reduction_strength}")
                    
                else:
                    print("   ✅ Audio level is good - using default settings")
                
                print(f"   Final settings:")
                print(f"     Noise gate: {self.noise_gate_threshold}")
                print(f"     Noise reduction: {self.noise_reduction_strength}")
                print(f"     High-pass filter: {self.high_pass_freq} Hz")
                
            else:
                print("❌ Audio calibration failed - no test audio recorded")
                
        except Exception as e:
            print(f"❌ Audio calibration error: {e}")
            print("   Using default audio settings")
        
        print("=" * 40)
    
    def test_audio_system(self):
        """Test the microphone and audio recording system"""
        print("\n🎤 Testing Audio System...")
        print("=" * 40)
        
        try:
            # Test 1: Check audio devices
            print("Test 1: Audio Device Check")
            try:
                devices = sd.query_devices()
                input_devices = [d for d in devices if d['max_inputs'] > 0]
                if input_devices:
                    print(f"✅ Found {len(input_devices)} input device(s):")
                    for i, device in enumerate(input_devices):
                        print(f"   {i}: {device['name']} (channels: {device['max_inputs']})")
                    
                    # Use default input device
                    default_input = sd.query_devices(kind='input')
                    print(f"✅ Default input device: {default_input['name']}")
                else:
                    print("❌ No input devices found!")
                    return False
            except Exception as e:
                print(f"❌ Error checking audio devices: {e}")
                return False
            
            # Test 2: Quick audio recording test
            print("\nTest 2: Audio Recording Test")
            print("   Speak a few words when prompted...")
            
            try:
                # Record 3 seconds of audio
                test_audio = sd.rec(
                    int(3 * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=np.float32
                )
                sd.wait()
                
                if test_audio is not None and len(test_audio) > 0:
                    audio_level = np.abs(test_audio).mean()
                    print(f"✅ Audio recording successful!")
                    print(f"   Sample rate: {self.sample_rate} Hz")
                    print(f"   Channels: {self.channels}")
                    print(f"   Duration: {len(test_audio) / self.sample_rate:.2f}s")
                    print(f"   Average level: {audio_level:.6f}")
                    
                    if audio_level < 0.001:
                        print("   ⚠️ Warning: Audio level seems very low")
                        print("   💡 Try speaking louder or checking microphone settings")
                    elif audio_level > 0.9:
                        print("   ⚠️ Warning: Audio level seems very high (possible clipping)")
                        print("   💡 Try speaking quieter or reducing microphone gain")
                    else:
                        print("   ✅ Audio level looks good")
                    
                    return True
                else:
                    print("❌ Audio recording failed - no data received")
                    return False
                    
            except Exception as e:
                print(f"❌ Error during audio recording test: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Audio system test failed: {e}")
            return False
        
        print("=" * 40)
    
    def detect_arduino_port(self):
        """Auto-detect Arduino port from available serial ports"""
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            
            # First try to find Arduino by description
            for port in ports:
                if 'Arduino' in port.description:
                    print(f"✅ Auto-detected Arduino on {port.device}")
                    return port.device
            
            # Fallback to environment variable or default
            fallback_port = os.getenv('SERIAL_PORT', 'COM9')
            print(f"⚠️ No Arduino detected, using fallback port: {fallback_port}")
            return fallback_port
            
        except ImportError:
            fallback_port = os.getenv('SERIAL_PORT', 'COM9')
            print(f"⚠️ Could not auto-detect ports, using fallback: {fallback_port}")
            return fallback_port
    
    def perform_dynamic_calibration(self):
        """Perform comprehensive dynamic calibration with multiple phases"""
        print("\n🔄 Starting Dynamic EMG Calibration...")
        print("=" * 50)
        
        # Phase 1: Initial baseline calibration
        print("Phase 1: Baseline Calibration (Relax your muscle)")
        self.calibrate_emg()
        
        # Phase 2: Signal quality assessment
        print("\nPhase 2: Signal Quality Assessment")
        signal_quality = self.assess_signal_quality()
        
        # Phase 3: Adaptive threshold adjustment
        print("\nPhase 3: Adaptive Threshold Adjustment")
        self.adjust_threshold_adaptively(signal_quality)
        
        # Phase 4: Final validation
        print("\nPhase 4: Final Validation")
        self.validate_calibration()
        
        print("=" * 50)
        print("✅ Dynamic Calibration Complete!")
        print(f"   Final Threshold: {self.burst_threshold:.3f}")
        print(f"   Signal Quality: {signal_quality:.2f}")
        print("   Ready for use!")
    
    def assess_signal_quality(self):
        """Assess the quality of the EMG signal"""
        print("   Assessing signal quality... (3 seconds)")
        start_time = time.time()
        samples = []
        noise_samples = []
        
        while time.time() - start_time < 3.0:
            raw = self.read_emg()
            if raw is not None:
                samples.append(raw)
                # Calculate noise during quiet periods
                if len(samples) > 10:
                    recent_std = np.std(samples[-10:])
                    if recent_std < self.burst_threshold * 0.5:  # Quiet period
                        noise_samples.append(recent_std)
            time.sleep(0.01)
        
        if samples:
            signal_range = max(samples) - min(samples)
            signal_mean = np.mean(samples)
            signal_std = np.std(samples)
            noise_level = np.mean(noise_samples) if noise_samples else signal_std * 0.1
            
            # Calculate signal-to-noise ratio
            snr = signal_range / (noise_level + 1e-6)
            quality_score = min(1.0, snr / 10.0)  # Normalize to 0-1
            
            print(f"   Signal Range: {signal_range:.3f}")
            print(f"   Signal Std Dev: {signal_std:.3f}")
            print(f"   Noise Level: {noise_level:.3f}")
            print(f"   Signal-to-Noise Ratio: {snr:.2f}")
            print(f"   Quality Score: {quality_score:.2f}")
            
            return quality_score
        else:
            print("   ⚠️ No signal data received during assessment")
            return 0.0
    
    def adjust_threshold_adaptively(self, signal_quality):
        """Adjust threshold based on signal quality assessment"""
        print("   Adjusting threshold adaptively...")
        
        if signal_quality < 0.3:
            # Poor signal quality - use very sensitive threshold
            self.threshold_multiplier = 1.5
            print("   Signal quality is poor - using sensitive threshold")
        elif signal_quality < 0.6:
            # Medium signal quality - use moderate threshold
            self.threshold_multiplier = 2.0
            print("   Signal quality is medium - using moderate threshold")
        else:
            # Good signal quality - use standard threshold
            self.threshold_multiplier = 2.5
            print("   Signal quality is good - using standard threshold")
        
        # Recalculate threshold with new multiplier
        if hasattr(self, 'baseline') and hasattr(self, 'threshold_multiplier'):
            # Recalibrate with new parameters
            self.calibrate_emg()
    
    def validate_calibration(self):
        """Validate the calibration by testing with known muscle contractions"""
        print("   Validating calibration... (5 seconds)")
        print("   Please contract your muscle 2-3 times during this period")
        
        start_time = time.time()
        detections = 0
        false_positives = 0
        
        while time.time() - start_time < 5.0:
            raw = self.read_emg()
            if raw is not None:
                filtered, _ = signal.lfilter(self.b_bandpass, self.a_bandpass, [raw], zi=self.bandpass_zi)
                rectified = abs(filtered[0] - self.baseline)
                
                if rectified > self.burst_threshold:
                    detections += 1
                elif rectified > self.burst_threshold * 0.8:  # Near threshold
                    false_positives += 1
            time.sleep(0.01)
        
        print(f"   Detections: {detections}")
        print(f"   Near-threshold signals: {false_positives}")
        
        if detections > 0:
            print("   ✅ Calibration validation successful!")
        else:
            print("   ⚠️ No muscle contractions detected - consider recalibrating")
    
    def auto_recalibrate_if_needed(self):
        """Check if auto-recalibration is needed and perform it"""
        if not self.auto_recalibrate:
            return
            
        current_time = time.time()
        if current_time - self.last_calibration_time > self.recalibrate_interval:
            print(f"\n🔄 Auto-recalibration triggered after {self.recalibrate_interval:.0f} seconds")
            self.perform_dynamic_calibration()
            self.last_calibration_time = current_time
    
    def connect_serial(self):
        """Connect or reconnect to serial port"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                if self.serial and self.serial.is_open:
                    self.serial.close()
                
                print(f"Attempting to connect to {self.serial_port} (attempt {attempt + 1}/{max_retries})...")
                self.serial = serial.Serial(
                    port=self.serial_port,
                    baudrate=115200,
                    timeout=0.1,
                    write_timeout=1.0
                )
                
                # Wait for Arduino to stabilize
                time.sleep(2)
                
                # Test the connection by reading any available data
                if self.serial.in_waiting > 0:
                    test_data = self.serial.read(self.serial.in_waiting)
                    print(f"Received {len(test_data)} bytes during connection test")
                
                print(f"✅ Successfully connected to serial port: {self.serial_port}")
                return
                
            except serial.SerialException as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"   Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"❌ Failed to connect to serial port {self.serial_port} after {max_retries} attempts")
                    print("   Please check:")
                    print("   - Arduino is connected and powered")
                    print("   - Correct COM port is selected")
                    print("   - No other program is using the port")
                    print("   - Arduino code is uploaded and running")
                    self.serial = None
            except Exception as e:
                print(f"❌ Unexpected error during connection: {e}")
                self.serial = None
                break
    
    def calibrate_emg(self):
        """Calibrate EMG baseline and threshold with improved noise handling"""
        print("🎯 CALIBRATING EMG - Keep muscle completely relaxed for 3 seconds")
        start_time = time.time()
        calibration_samples = []
        zi = self.bandpass_zi.copy()  # Initialize filter state
        
        # Collect samples for 3 seconds with better timing
        while time.time() - start_time < 3.0:
            raw = self.read_emg()
            if raw is not None:
                filtered, zi = signal.lfilter(self.b_bandpass, self.a_bandpass, [raw], zi=zi)
                calibration_samples.append(filtered[0])
            time.sleep(0.005)  # 200Hz sampling
        
        if len(calibration_samples) >= 100:  # Need at least 100 samples
            # Remove outliers for more stable baseline
            samples_array = np.array(calibration_samples)
            q75, q25 = np.percentile(samples_array, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            # Filter out outliers
            clean_samples = samples_array[(samples_array >= lower_bound) & (samples_array <= upper_bound)]
            
            if len(clean_samples) > 0:
                self.baseline = np.mean(clean_samples)
                std_dev = np.std(clean_samples)
                self.burst_threshold = self.threshold_multiplier * std_dev
                
                # Ensure minimum threshold for stability
                min_threshold = 0.3  # Much lower for high sensitivity
                if self.burst_threshold < min_threshold:
                    self.burst_threshold = min_threshold
                    print(f"⚠️ Threshold too low, using minimum: {min_threshold}")
                
                print(f"📊 Calibration Results:")
                print(f"   Baseline: {self.baseline:.3f}")
                print(f"   Std Dev (noise): {std_dev:.3f}")
                print(f"   Dynamic Burst Threshold: {self.burst_threshold:.3f}")
                print(f"   Threshold Multiplier: {self.threshold_multiplier}")
                print(f"   Samples collected: {len(calibration_samples)}")
                print(f"   Clean samples (no outliers): {len(clean_samples)}")
            else:
                print("⚠️ All samples were outliers, using fallback values")
                self.baseline = 0.0
                self.burst_threshold = 0.5  # Much lower fallback
        else:
            print("⚠️ Insufficient calibration data. Using conservative fallback values.")
            self.baseline = 0.0
            self.burst_threshold = 0.5  # Much lower fallback
            print(f"Fallback Burst Threshold: {self.burst_threshold:.2f}")
        
        # Set hysteresis threshold for stable detection
        self.hysteresis_threshold = self.burst_threshold * 0.5  # More sensitive hysteresis
        print(f"   Hysteresis Threshold: {self.hysteresis_threshold:.3f}")
        print("✅ Calibration complete! Ready for double-tap detection.")
    
    def read_emg(self) -> Optional[float]:
        """Read raw EMG value from serial"""
        with self.serial_lock:
            if not self.serial or not self.serial.is_open:
                self.connect_serial()
                if not self.serial:
                    return None
            
            try:
                # Check if there's data available
                if self.serial.in_waiting == 0:
                    return None
                
                # Read available data
                data = self.serial.read(self.serial.in_waiting).decode('utf-8', errors='ignore')
                self.buffer += data
                
                # Process complete lines
                lines = self.buffer.split('\n')
                self.buffer = lines[-1]  # Keep incomplete line in buffer
                
                for line in lines[:-1]:
                    line = line.strip()
                    if line and line != "Starting EMG raw data transmission...":
                        # Try to parse as float, with better error handling
                        try:
                            value = float(line)
                            # Basic range validation for EMG values
                            if -1000 <= value <= 1000:  # Reasonable EMG range
                                return value
                            else:
                                if hasattr(self, '_invalid_count'):
                                    self._invalid_count += 1
                                else:
                                    self._invalid_count = 1
                                
                                # Only show warning every 10 invalid values to reduce spam
                                if self._invalid_count % 10 == 0:
                                    print(f"⚠️ Skipped out-of-range EMG value: {value} (count: {self._invalid_count})")
                        except ValueError:
                            if hasattr(self, '_parse_error_count'):
                                self._parse_error_count += 1
                            else:
                                self._parse_error_count = 1
                            
                            # Only show warning every 20 parse errors to reduce spam
                            if self._parse_error_count % 20 == 0:
                                print(f"⚠️ Skipped unparseable EMG data: {line} (count: {self._parse_error_count})")
                
                return None
                
            except serial.SerialException as e:
                print(f"❌ Serial error: {e}")
                self.serial = None
                return None
            except Exception as e:
                print(f"❌ Unexpected error reading EMG: {e}")
                return None
    
    def process_emg(self, raw: float) -> tuple[float, bool, bool]:
        """Process raw EMG with stable double-tap detection"""
        now = time.time()
        
        # Auto-recalibration disabled for stability
        # self.auto_recalibrate_if_needed()
        
        # Apply bandpass filter
        filtered, self.bandpass_zi = signal.lfilter(self.b_bandpass, self.a_bandpass, [raw], zi=self.bandpass_zi)
        
        # Additional noise reduction: simple moving average for stability
        if not hasattr(self, '_filter_buffer'):
            self._filter_buffer = []
        
        self._filter_buffer.append(filtered[0])
        if len(self._filter_buffer) > 3:  # Keep last 3 samples
            self._filter_buffer.pop(0)
        
        # Use averaged value for more stable detection
        filtered_avg = np.mean(self._filter_buffer)
        rectified = abs(filtered_avg - self.baseline)
        
        # Check cooldown first
        if self.in_cooldown:
            if now - self.last_trigger_time < self.cooldown_time:
                return rectified, False, False
            else:
                self.in_cooldown = False
                print("✅ Cooldown period ended")
        
        # Debug output every 200 samples (about every 1 second)
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 200 == 0:
            print(f"DEBUG: Raw={raw:.3f}, Rectified={rectified:.3f}, Threshold={self.burst_threshold:.3f}")
            print(f"       State: {self.assistant_state}, Burst Count: {self.burst_count}, Cooldown: {self.in_cooldown}")
            print(f"       Mode: {self.current_mode}")
            print(f"       Debounce: {self.burst_debounce_time:.3f}s, Last Burst: {now - self.last_burst_end_time:.3f}s ago")
            if self.burst_count == 1:
                time_since_first = now - self.first_burst_end
                print(f"       Waiting for second tap: {time_since_first:.2f}s since first tap (max: {self.double_tap_max_interval}s)")
        
        # Detect muscle burst with hysteresis
        if rectified > self.burst_threshold:
            if not self.in_burst:
                self.in_burst = True
                self.burst_start = now
                print(f"🟢 Burst START: {rectified:.3f} > {self.burst_threshold:.3f}")
        else:
            # Check if we're below hysteresis threshold to end burst
            if self.in_burst and rectified < (self.burst_threshold * 0.7):  # 30% hysteresis
                burst_duration = now - self.burst_start
                self.in_burst = False
                
                # Validate burst duration
                if self.burst_min_duration <= burst_duration <= self.burst_max_duration:
                    # Smart debouncing: only apply to non-double-tap sequences
                    # Allow second tap even if it's close to first tap
                    if (self.burst_count == 0 and 
                        now - self.last_burst_end_time < self.burst_debounce_time):
                        print(f"⏱️ First burst ignored (debouncing): {now - self.last_burst_end_time:.3f}s < {self.burst_debounce_time}s")
                        return rectified, False, False
                    
                    # Update last burst end time
                    self.last_burst_end_time = now
                    
                    print(f"✅ Valid burst: {burst_duration:.2f}s")
                    self.burst_count += 1
                    
                    # Handle EMG triggers based on current mode
                    if self.current_mode == "gesture_creation":
                        # Original behavior: Handle double-tap for starting recording
                        if self.assistant_state == "IDLE":
                            if self.burst_count == 1:
                                self.first_burst_end = now
                                print(f"🔄 First tap detected. Flex again within {self.double_tap_max_interval}s...")
                            elif self.burst_count == 2:
                                time_since_first = now - self.first_burst_end
                                print(f"🔍 Second tap detected! Time since first: {time_since_first:.3f}s (max: {self.double_tap_max_interval}s)")
                                if time_since_first <= self.double_tap_max_interval:
                                    print(f"🎯 DOUBLE-TAP DETECTED! ({time_since_first:.2f}s) Starting recording...")
                                    self.start_recording()
                                    self.burst_count = 0
                                    self.last_trigger_time = now
                                    self.in_cooldown = True
                                else:
                                    print(f"❌ Second tap too late ({time_since_first:.2f}s), resetting...")
                                    self.burst_count = 0
                        
                        # Handle single tap for stopping recording
                        elif self.assistant_state == "RECORDING":
                            recording_duration = now - self.recording_start_time
                            if recording_duration >= self.min_recording_time:
                                print(f"🎯 STOP TAP DETECTED! Stopping recording after {recording_duration:.1f}s...")
                                self.stop_recording()
                                self.burst_count = 0
                                self.last_trigger_time = now
                                self.in_cooldown = True
                            else:
                                remaining = self.min_recording_time - recording_duration
                                print(f"⚠️ Recording too short ({recording_duration:.1f}s), need {remaining:.1f}s more")
                                self.burst_count = 0
                        
                        # Ignore taps while processing
                        elif self.assistant_state == "PROCESSING":
                            print("⏳ Ignoring tap while processing...")
                            self.burst_count = 0
                    
                    elif self.current_mode == "gesture_execution":
                        # In execution mode, execute gestures directly based on trigger characteristics
                        if self.assistant_state == "IDLE":
                            print(f"🎯 Execution mode trigger: duration={burst_duration:.2f}s, count={self.burst_count}")
                            print(f"🔍 Current gestures available: {list(self.gestures.keys())}")
                            
                            # Initialize grip state if not exists
                            if not hasattr(self, 'current_grip_state'):
                                self.current_grip_state = "grip2open"  # Default state
                            
                            # Toggle between grip2open and grip2closed on each trigger
                            if self.current_grip_state == "grip2open":
                                gesture_name = "grip2closed"
                                self.current_grip_state = "grip2closed"
                                print(f"🎯 EMG trigger detected - switching to {gesture_name}")
                            else:
                                gesture_name = "grip2open"
                                self.current_grip_state = "grip2open"
                                print(f"🎯 EMG trigger detected - switching to {gesture_name}")
                            
                            # Execute the gesture using the FAST method for execution mode
                            if gesture_name in self.gestures:
                                print(f"🤖 Executing {gesture_name} gesture from gestures.json...")
                                print(f"📋 Gesture data: {self.gestures[gesture_name]}")
                                success = self.execute_roninhand_gesture_fast(gesture_name)  # Use fast execution for speed
                                if success:
                                    print(f"✅ Successfully executed {gesture_name} gesture")
                                else:
                                    print(f"❌ Failed to execute {gesture_name} gesture")
                                # Reset burst count after gesture execution
                                self.burst_count = 0
                                self.last_trigger_time = now
                                self.in_cooldown = True
                            else:
                                print(f"❌ Gesture '{gesture_name}' not found in gestures")
                                print(f"Available gestures: {list(self.gestures.keys())}")
                                self.burst_count = 0
                    
                else:
                    print(f"❌ Invalid burst duration: {burst_duration:.2f}s (need {self.burst_min_duration}-{self.burst_max_duration}s)")
                    self.burst_count = 0
        
        # Reset burst count if too much time has passed
        if self.burst_count == 1 and (now - self.first_burst_end > self.double_tap_max_interval):
            print("⏰ First tap expired, resetting...")
            self.burst_count = 0
        
        return rectified, self.in_burst, False
    
    def start_recording(self):
        """Start recording audio in a separate thread"""
        if self.assistant_state != "IDLE":
            print(f"⚠️ Cannot start recording: assistant is in {self.assistant_state} state")
            return
        
        print("🎤 Starting audio recording...")
        self.assistant_state = "RECORDING"
        self.recording_start_time = time.time()
        self.recording_stop_event.clear()
        self.audio_frames = []
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record_audio_thread, daemon=True)
        self.recording_thread.start()
        
        # Play trigger sound to indicate recording started
        self.play_trigger_sound()
    
    def stop_recording(self):
        """Stop recording and process the audio"""
        if self.assistant_state != "RECORDING":
            print(f"⚠️ Cannot stop recording: assistant is in {self.assistant_state} state")
            return
        
        print("⏹️ Stopping audio recording...")
        self.assistant_state = "PROCESSING"
        
        # Signal the recording thread to stop
        self.recording_stop_event.set()
        
        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        # Process the recorded audio
        if self.audio_frames:
            print(f"✅ Recording completed. Duration: {time.time() - self.recording_start_time:.1f}s")
            self.process_recorded_audio()
        else:
            print("❌ No audio recorded")
            self.assistant_state = "IDLE"
    
    def play_trigger_sound(self):
        """Play a sound to indicate the assistant is triggered"""
        if not self.trigger_sound_file:
            return
        try:
            print(f"🔊 Playing trigger sound: {self.trigger_sound_file}")
            pygame.mixer.music.load(self.trigger_sound_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"❌ Error playing trigger sound: {e}")
    
    def apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply comprehensive noise reduction to audio data"""
        if not self.enable_noise_reduction:
            return audio_data
        
        print("🔇 Applying noise reduction...")
        
        try:
            # Convert to float if needed
            if audio_data.dtype != np.float64:
                audio_data = audio_data.astype(np.float64)
            
            # Normalize audio to [-1, 1] range
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))
            
            # Check if audio data is long enough for filtering
            min_length_for_filter = 50  # Minimum samples needed for filters
            
            if len(audio_data) < min_length_for_filter:
                print(f"   ⚠️ Audio too short ({len(audio_data)} samples) for filtering, skipping filters")
                return audio_data
            
            # 1. High-pass filter to remove low frequency noise (rumble, wind, etc.)
            if self.high_pass_freq > 0:
                try:
                    b_highpass, a_highpass = signal.butter(4, self.high_pass_freq / (self.sample_rate / 2), btype='high')
                    # Check if filter order is appropriate for data length
                    if len(audio_data) > 3 * len(b_highpass):  # Need at least 3x filter order
                        audio_data = signal.filtfilt(b_highpass, a_highpass, audio_data)
                        print(f"   Applied high-pass filter at {self.high_pass_freq} Hz")
                    else:
                        print(f"   ⚠️ Audio too short for high-pass filter (need >{3 * len(b_highpass)} samples)")
                except Exception as e:
                    print(f"   ⚠️ High-pass filter failed: {e}")
            
            # 2. Low-pass filter to remove high frequency noise (hiss, static, etc.)
            if self.low_pass_freq < self.sample_rate / 2:
                try:
                    b_lowpass, a_lowpass = signal.butter(4, self.low_pass_freq / (self.sample_rate / 2), btype='low')
                    # Check if filter order is appropriate for data length
                    if len(audio_data) > 3 * len(b_lowpass):  # Need at least 3x filter order
                        audio_data = signal.filtfilt(b_lowpass, a_lowpass, audio_data)
                        print(f"   Applied low-pass filter at {self.low_pass_freq} Hz")
                    else:
                        print(f"   ⚠️ Audio too short for low-pass filter (need >{3 * len(b_lowpass)} samples)")
                except Exception as e:
                    print(f"   ⚠️ Low-pass filter failed: {e}")
            
            # 3. Noise gate to remove very quiet background noise
            if self.noise_gate_threshold > 0:
                noise_gate_mask = np.abs(audio_data) > self.noise_gate_threshold
                audio_data = audio_data * noise_gate_mask
                print(f"   Applied noise gate at threshold {self.noise_gate_threshold}")
            
            # 3.5. Audio amplification for very low levels
            if self.noise_reduction_strength == 0.0:  # Ultra-low sensitivity mode
                # Amplify audio significantly for very quiet microphones
                current_max = np.abs(audio_data).max()
                if current_max > 0:
                    target_max = 0.3  # Target 30% of full scale
                    amplification_factor = target_max / current_max
                    # Limit amplification to reasonable levels
                    amplification_factor = min(amplification_factor, 50.0)
                    audio_data = audio_data * amplification_factor
                    print(f"   Applied audio amplification: {amplification_factor:.1f}x (target: {target_max:.3f})")
            
            # 4. Spectral subtraction for broadband noise reduction
            if self.noise_reduction_strength > 0 and len(audio_data) > 100:
                try:
                    audio_data = self._spectral_subtraction(audio_data)
                    print(f"   Applied spectral subtraction with strength {self.noise_reduction_strength}")
                except Exception as e:
                    print(f"   ⚠️ Spectral subtraction failed: {e}")
            elif self.noise_reduction_strength > 0:
                print(f"   ⚠️ Audio too short for spectral subtraction (need >100 samples)")
            
            # 5. Smoothing filter to reduce artifacts
            if len(audio_data) > 100:
                try:
                    window_size = max(3, int(len(audio_data) * 0.001))  # Adaptive window size
                    # Ensure window size is not larger than data length
                    window_size = min(window_size, len(audio_data) // 2)
                    if window_size >= 3:
                        audio_data = uniform_filter1d(audio_data, size=window_size)
                        print(f"   Applied smoothing filter with window size {window_size}")
                    else:
                        print(f"   ⚠️ Audio too short for smoothing filter")
                except Exception as e:
                    print(f"   ⚠️ Smoothing filter failed: {e}")
            
            print("✅ Noise reduction completed")
            return audio_data
            
        except Exception as e:
            print(f"⚠️ Error during noise reduction: {e}")
            return audio_data
    
    def _spectral_subtraction(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction noise reduction"""
        try:
            # Estimate noise from first 0.5 seconds (assuming quiet start)
            noise_samples = int(0.5 * self.sample_rate)
            if len(audio_data) > noise_samples:
                noise_spectrum = np.mean(np.abs(np.fft.fft(audio_data[:noise_samples])), axis=0)
            else:
                noise_spectrum = np.mean(np.abs(np.fft.fft(audio_data)), axis=0)
            
            # Apply spectral subtraction
            fft_data = np.fft.fft(audio_data)
            magnitude = np.abs(fft_data)
            phase = np.angle(fft_data)
            
            # Subtract noise spectrum with adjustable strength
            cleaned_magnitude = np.maximum(magnitude - self.noise_reduction_strength * noise_spectrum, 
                                        magnitude * 0.1)  # Keep at least 10% of original
            
            # Reconstruct signal
            cleaned_fft = cleaned_magnitude * np.exp(1j * phase)
            cleaned_audio = np.real(np.fft.ifft(cleaned_fft))
            
            return cleaned_audio
            
        except Exception as e:
            print(f"⚠️ Error in spectral subtraction: {e}")
            return audio_data
    
    def _record_audio_thread(self):
        """Internal method to record audio in a separate thread"""
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio callback status: {status}")
            
            # Debug: Print audio data info for first few frames
            if len(self.audio_frames) < 3:
                print(f"   📊 Audio frame {len(self.audio_frames)}: shape={indata.shape}, dtype={indata.dtype}, range=[{indata.min():.3f}, {indata.max():.3f}]")
                print(f"   📊 Raw indata: min={indata.min():.6f}, max={indata.max():.6f}, mean={indata.mean():.6f}")
            
            # Ensure we get the right shape and format
            if indata.ndim == 2:  # Multi-channel, take first channel
                audio_frame = indata[:, 0].copy()
            else:  # Single channel
                audio_frame = indata.copy()
            
            # Additional debug info for first few frames
            if len(self.audio_frames) < 3:
                print(f"   📊 Processed frame: min={audio_frame.min():.6f}, max={audio_frame.max():.6f}, mean={audio_frame.mean():.6f}")
            
            # Check for audio quality issues
            if len(self.audio_frames) < 10:  # Only check first 10 frames
                audio_level = np.abs(audio_frame).mean()
                if audio_level < 0.001:  # Very low audio level
                    print(f"   ⚠️ Low audio level detected: {audio_level:.6f} (frame {len(self.audio_frames)})")
                    if len(self.audio_frames) == 0:  # First frame
                        print("   �� TIP: Check Windows microphone settings - volume too low!")
                        print("   💡 Right-click speaker → Sound settings → Sound Control Panel → Recording → Properties → Levels")
                elif audio_level > 0.9:  # Very high audio level (clipping)
                    print(f"   ⚠️ High audio level detected: {audio_level:.6f} (frame {len(self.audio_frames)})")
            
            self.audio_frames.append(audio_frame)
        
        try:
            # Debug: Check available audio devices
            print(f"🔍 Audio Recording Setup:")
            print(f"   Sample Rate: {self.sample_rate} Hz")
            print(f"   Channels: {self.channels}")
            print(f"   Chunk Size: {self.chunk_size}")
            print(f"   Data Type: float32")
            
            # List available input devices
            try:
                devices = sd.query_devices()
                input_devices = [d for d in devices if d.get('max_inputs', 0) > 0]
                print(f"   Available input devices: {len(input_devices)}")
                for i, device in enumerate(input_devices[:3]):  # Show first 3
                    print(f"     {i}: {device['name']} (inputs: {device.get('max_inputs', 'N/A')})")
            except Exception as e:
                print(f"   Could not list devices: {e}")
            
            with sd.InputStream(
                callback=audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=np.float32,
                device=None  # Use default input device
            ):
                start_time = time.time()
                last_progress_time = 0
                
                print("🎤 Recording started! Speak now...")
                print(f"💡 Tap your muscle again to stop recording (after at least {self.min_recording_time}s)")
                
                while not self.recording_stop_event.is_set():
                    time.sleep(0.1)  # Check more frequently for better responsiveness
                    current_time = time.time()
                    duration = current_time - start_time
                    
                    # Check if max duration reached
                    if duration > self.max_recording_duration:
                        print("⏱️ Max recording duration reached.")
                        break
                    
                    # Show recording progress every second for the first 10 seconds
                    if duration <= 10.0 and int(duration) > last_progress_time:
                        last_progress_time = int(duration)
                        remaining = self.max_recording_duration - duration
                        print(f"   Recording... {duration:.1f}s elapsed, {remaining:.1f}s remaining")
                    
                    # Show progress every 5 seconds after the first 10 seconds
                    elif duration > 10.0 and int(duration) % 5 == 0 and int(duration) > last_progress_time:
                        last_progress_time = int(duration)
                        remaining = self.max_recording_duration - duration
                        print(f"   Recording... {duration:.1f}s elapsed, {remaining:.1f}s remaining")
                
                print("🎤 Recording stopped.")
                print(f"   📊 Total audio frames captured: {len(self.audio_frames)}")
                
        except Exception as e:
            print(f"❌ Error during recording: {e}")
            self.assistant_state = "IDLE"
    
    def process_recorded_audio(self):
        """Process the recorded audio - convert to text and run full conversation cycle"""
        try:
            # Convert audio frames to bytes
            if not self.audio_frames:
                print("❌ No audio frames to process")
                self.assistant_state = "IDLE"
                return
            
            print(f"🔍 Processing {len(self.audio_frames)} audio frames...")
            
            # Concatenate all audio frames
            audio_data = np.concatenate(self.audio_frames, axis=0)
            print(f"   📊 Concatenated audio: shape={audio_data.shape}, dtype={audio_data.dtype}")
            print(f"   📊 Audio range: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
            
            # Ensure audio data is in the correct format for WAV files
            if audio_data.dtype != np.float32:
                print(f"   🔧 Converting audio data from {audio_data.dtype} to float32")
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio to [-1, 1] range if needed
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                max_val = max(abs(audio_data.max()), abs(audio_data.min()))
                if max_val > 0:
                    print(f"   🔧 Normalizing audio data (max_val: {max_val:.3f})")
                    audio_data = audio_data / max_val
                    print(f"   📊 After normalization: range=[{audio_data.min():.3f}, {audio_data.max():.3f}]")
            
            # Apply noise reduction
            audio_data = self.apply_noise_reduction(audio_data)
            print(f"   📊 After noise reduction: shape={audio_data.shape}, range=[{audio_data.min():.3f}, {audio_data.max():.3f}]")
            
            # Save USER'S RECORDING to tmp folder with descriptive name
            timestamp = int(time.time())
            user_recording_filename = f"user_recording_{timestamp}.wav"
            user_recording_path = os.path.join("tmp", user_recording_filename)
            
            try:
                # Save the user's recording with proper error handling
                print(f"   💾 Saving to: {user_recording_path}")
                wav.write(user_recording_path, self.sample_rate, audio_data)
                print(f"💾 Saved user recording: {user_recording_path}")
                print(f"   Duration: {len(audio_data) / self.sample_rate:.2f}s")
                print(f"   Size: {os.path.getsize(user_recording_path)} bytes")
                
                # Verify the file can be loaded back
                try:
                    test_sample_rate, test_audio = wav.read(user_recording_path)
                    print(f"   ✅ File verification: {test_sample_rate} Hz, {len(test_audio)} samples")
                except Exception as verify_error:
                    print(f"   ⚠️  File verification failed: {verify_error}")
                    # Try to fix the file by resaving with different format
                    print(f"   🔧 Attempting to fix audio format...")
                    # Convert to 16-bit integer format for better compatibility
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wav.write(user_recording_path, self.sample_rate, audio_int16)
                    print(f"   ✅ Audio format fixed (converted to 16-bit)")
                
            except Exception as save_error:
                print(f"   ❌ Error saving WAV file: {save_error}")
                # Try alternative format
                try:
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wav.write(user_recording_path, self.sample_rate, audio_int16)
                    print(f"   ✅ Saved with alternative format (16-bit integer)")
                except Exception as alt_error:
                    print(f"   ❌ Alternative format also failed: {alt_error}")
                    raise alt_error
            
            # Convert to bytes for processing
            with open(user_recording_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Run the FULL conversation cycle with speech-to-text and LLM processing
            self.run_conversation_cycle_with_audio(audio_bytes)
            
        except Exception as e:
            print(f"❌ Error processing recorded audio: {e}")
            print(f"   Audio data shape: {audio_data.shape if 'audio_data' in locals() else 'N/A'}")
            print(f"   Audio data type: {audio_data.dtype if 'audio_data' in locals() else 'N/A'}")
            if 'audio_data' in locals():
                print(f"   Audio data range: [{audio_data.min() if len(audio_data) > 0 else 'N/A'}, {audio_data.max() if len(audio_data) > 0 else 'N/A'}]")
            self.assistant_state = "IDLE"
    
    def run_conversation_cycle_with_audio(self, audio_bytes: bytes):
        """Run full conversation cycle with recorded audio - speech-to-text, LLM, and gesture execution"""
        try:
            print("🎯 Starting full conversation cycle with recorded audio...")
            self.assistant_state = "PROCESSING"
            
            # Step 1: Convert speech to text
            print("🔤 Converting speech to text...")
            transcribed_text = self.speech_to_text(audio_bytes)
            if not transcribed_text:
                print("❌ Failed to transcribe speech")
                self.assistant_state = "IDLE"
                return
            
            print(f"📝 Transcribed text: '{transcribed_text}'")
            
            # Step 2: Check for mode switch commands first
            print("🤖 Checking for mode switch commands...")
            mode_switched = False
            if self.check_mode_switch_command(transcribed_text):
                print("🔄 Mode switch command detected!")
                mode_response = self.handle_mode_switch(transcribed_text)
                print(f"🔄 Mode switch result: {mode_response}")
                
                # Use the mode switch response as the TTS message
                tts_message = mode_response
                mode_switched = True
                
                # Convert response to speech and play it
                print("🔊 Converting mode switch response to speech...")
                speech_result = self.text_to_speech(tts_message)
                if speech_result:
                    print("✅ Mode switch completed successfully!")
                else:
                    print("❌ Failed to convert mode switch response to speech")
                
                # Return to IDLE state
                self.assistant_state = "IDLE"
                print("🔄 Ready for next interaction...")
                return
            
            # Step 3: Try to execute RoninHand gesture if command is recognized (only in creation mode)
            if self.current_mode == "gesture_creation":
                print("🤖 Attempting to execute RoninHand gesture...")
                gesture_executed = self.execute_roninhand_gesture(transcribed_text)
            else:
                print("🤖 In execution mode - gestures executed via EMG triggers only")
                gesture_executed = False
            
            # Step 4: Get AI response from LLM
            print("🤖 Getting AI response...")
            # Ensure system prompt is current for this mode
            self.regenerate_system_prompt()
            llm_response = self.get_llm_response(transcribed_text)
            if not llm_response:
                print("❌ Failed to get LLM response")
                self.assistant_state = "IDLE"
                return
            
            print(f"🤖 LLM Response: {llm_response}")
            
            # Step 5: Check if LLM generated additional gestures and execute them (only in creation mode)
            if self.current_mode == "gesture_creation":
                print("🤖 Checking for LLM-generated gestures...")
                llm_gesture_executed = self.parse_and_execute_llm_gesture(llm_response, transcribed_text)
                
                if llm_gesture_executed:
                    if isinstance(llm_gesture_executed, str):
                        print(f"✅ LLM gesture '{llm_gesture_executed}' was created and executed successfully!")
                    else:
                        print("✅ LLM gesture was created and executed successfully!")
                else:
                    print("ℹ️  No LLM gesture was generated or executed")
            else:
                print("🤖 In execution mode - LLM gesture creation disabled")
                llm_gesture_executed = False
            
            # Step 6: Create a short success/failure message for TTS (not the full LLM response)
            tts_message = ""
            if self.current_mode == "gesture_creation":
                if gesture_executed and llm_gesture_executed:
                    tts_message = f"Successfully executed the {transcribed_text} gesture and created a new gesture based on your request."
                elif gesture_executed:
                    tts_message = f"Successfully executed the {transcribed_text} gesture on the RoninHand."
                elif llm_gesture_executed:
                    if isinstance(llm_gesture_executed, str):
                        tts_message = f"Successfully created and executed a new gesture called '{llm_gesture_executed}' based on your request."
                    else:
                        tts_message = f"Successfully created and executed a new gesture based on your request."
                else:
                    # If no gestures were executed, use a short version of the LLM response
                    # Remove any servo position details from the response
                    clean_response = llm_response
                    # Remove lines containing servo positions
                    lines = clean_response.split('\n')
                    filtered_lines = []
                    for line in lines:
                        if not any(word in line.lower() for word in ['servo_', 'position', 'angle', 'degrees']):
                            filtered_lines.append(line)
                    clean_response = '\n'.join(filtered_lines).strip()
                    
                    # Limit response length for TTS
                    if len(clean_response) > 200:
                        clean_response = clean_response[:200] + "..."
                    
                    tts_message = clean_response if clean_response else "I understand your request."
            else:
                # In execution mode, provide mode-specific response
                tts_message = f"In execution mode. Your request was processed. Use EMG triggers to execute grip2open/grip2closed gestures."
            
            # Step 7: Convert response to speech and play it
            print("🔊 Converting AI response to speech...")
            speech_result = self.text_to_speech(tts_message)
            if speech_result:
                print("✅ Full conversation cycle completed successfully!")
            else:
                print("❌ Failed to convert response to speech")
            
            # Return to IDLE state
            self.assistant_state = "IDLE"
            print("🔄 Ready for next interaction...")
            
        except Exception as e:
            print(f"❌ Error in conversation cycle: {e}")
            self.assistant_state = "IDLE"
            print("🔄 Ready for next interaction...")
    
    def record_audio(self) -> Optional[bytes]:
        """Record audio from microphone for a fixed duration or until manually stopped"""
        print("🎤 Recording started... (will stop after max duration or press 's' to stop)")
        print("   Press 's' to stop recording early")
        self.audio_frames = []
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio callback status: {status}")
            self.audio_frames.append(indata.copy())
        
        try:
            with sd.InputStream(
                callback=audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=np.float32,
                device=None  # Use default input device
            ):
                start_time = time.time()
                while True:
                    time.sleep(0.05)
                    current_time = time.time()
                    duration = current_time - start_time
                    
                    if duration > self.max_recording_duration:
                        print("⏱️ Max recording duration reached.")
                        break
                    
                    # Check if minimum duration is met and allow early stopping
                    if duration >= self.min_recording_duration:
                        # Show recording progress
                        if int(duration) % 5 == 0 and duration > 0:  # Every 5 seconds
                            remaining = self.max_recording_duration - duration
                            print(f"   Recording... {duration:.1f}s elapsed, {remaining:.1f}s remaining")
                        
                        # Check for manual stop (this would need to be implemented with keyboard input)
                        # For now, just continue recording
                        pass
            
            if not self.audio_frames:
                print("❌ No audio recorded")
                return None
            
            print(f"✅ Recording completed. Duration: {duration:.1f}s")
            audio_data = np.concatenate(self.audio_frames, axis=0)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file_path = temp_file.name
                wav.write(temp_file_path, self.sample_rate, audio_data)
            
            with open(temp_file_path, 'rb') as f:
                audio_bytes = f.read()
            
            os.unlink(temp_file_path)
            return audio_bytes
            
        except Exception as e:
            print(f"❌ Error recording audio: {e}")
            return None
    
    def detect_wake_word(self) -> bool:
        """Detect wake word using Porcupine"""
        if not self.porcupine:
            return False
        
        try:
            # Record a short audio sample for wake word detection
            audio_frames = []
            duration = 2.0  # 2 seconds should be enough for wake word
            
            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(f"Wake word audio callback status: {status}")
                audio_frames.append(indata.copy())
            
            # Ensure frame_length is a valid integer
            try:
                raw_frame_length = self.porcupine.frame_length
                print(f"🔍 Debug: Raw frame_length type={type(raw_frame_length)}, value={raw_frame_length}")
                
                if hasattr(raw_frame_length, 'item'):  # It's a numpy scalar
                    frame_length = int(raw_frame_length.item())
                else:
                    frame_length = int(raw_frame_length)
                
                print(f"🔍 Debug: Converted frame_length={frame_length}")
                
                if frame_length <= 0:
                    print("⚠️ Invalid frame length from Porcupine, using default")
                    frame_length = 512  # Default fallback
            except (ValueError, TypeError) as e:
                print(f"⚠️ Error getting frame length from Porcupine: {e}, using default")
                frame_length = 512  # Default fallback
            
            with sd.InputStream(
                callback=audio_callback,
                channels=1,  # Mono for wake word detection
                samplerate=self.porcupine.sample_rate,
                blocksize=frame_length,
                dtype=np.int16,  # Porcupine expects 16-bit PCM
                device=None
            ):
                start_time = time.time()
                while time.time() - start_time < duration:
                    time.sleep(0.01)
            
            if not audio_frames:
                return False
            
            # Concatenate audio frames
            audio_data = np.concatenate(audio_frames, axis=0)
            
            # Process audio in chunks for wake word detection
            for i in range(0, len(audio_data), frame_length):
                chunk = audio_data[i:i + frame_length]
                if len(chunk) == frame_length:
                    try:
                        # Ensure chunk is the right data type and shape
                        if chunk.dtype != np.int16:
                            chunk = chunk.astype(np.int16)
                        
                        # Ensure chunk is 1D array
                        if chunk.ndim > 1:
                            chunk = chunk.flatten()
                        
                        # Debug: Print chunk info on first iteration
                        if i == 0:
                            print(f"🔍 Debug: chunk shape={chunk.shape}, dtype={chunk.dtype}, range=[{chunk.min()}, {chunk.max()}]")
                        
                        keyword_index = self.porcupine.process(chunk)
                        if keyword_index >= 0:
                            print("🎯 Wake word 'HeyRonin' detected!")
                            return True
                    except Exception as process_error:
                        print(f"⚠️ Error processing audio chunk: {process_error}")
                        # Debug: Print more info about the error
                        if "scalar index" in str(process_error):
                            print(f"   🔍 Debug: chunk shape={chunk.shape}, dtype={chunk.dtype}, frame_length={frame_length}")
                        continue
            
            return False
            
        except Exception as e:
            print(f"❌ Error in wake word detection: {e}")
            return False
    
    def speech_to_text(self, audio_bytes: bytes) -> Optional[str]:
        """Convert speech to text using Gemini"""
        try:
            print("🔄 Converting speech to text...")
            
            # Validate audio data
            if len(audio_bytes) < 1000:  # Less than 1KB is suspicious
                print(f"⚠️ Audio data seems too small: {len(audio_bytes)} bytes")
                return None
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            # Enhanced prompt for better transcription - more lenient for low-quality audio
            prompt = """Transcribe the following audio to text. 
            - If you hear ANY speech-like sounds, words, or phrases, return the transcribed text
            - Even if the audio is quiet, muffled, or unclear, try to transcribe what you hear
            - Only return "NO_SPEECH_DETECTED" if you hear absolutely no human speech at all
            - If you hear partial words or unclear speech, transcribe what you can hear
            - Return only the transcribed text or "NO_SPEECH_DETECTED", nothing else"""
            
            # Try the newer API method first (direct audio data)
            try:
                response = self.speech_model.generate_content([
                    prompt,
                    {"mime_type": "audio/wav", "data": audio_bytes}
                ])
                print("✅ Used direct audio data method")
            except Exception as direct_error:
                print(f"   Direct method failed: {direct_error}")
                # Fallback to file upload method if available
                if hasattr(genai, 'upload_file'):
                    try:
                        uploaded_file = genai.upload_file(path=temp_file_path, mime_type="audio/wav")
                        response = self.speech_model.generate_content([prompt, uploaded_file])
                        genai.delete_file(uploaded_file.name)
                        print("✅ Used file upload method")
                    except Exception as upload_error:
                        print(f"   File upload method failed: {upload_error}")
                        # Final fallback: just send the prompt
                        response = self.speech_model.generate_content(prompt)
                        print("✅ Used prompt-only fallback method")
                else:
                    # No upload_file method available, use prompt-only
                    response = self.speech_model.generate_content(prompt)
                    print("✅ Used prompt-only method")
            
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
            
            transcribed_text = response.text.strip()
            if transcribed_text:
                print(f"📝 Transcribed: {transcribed_text}")
                
                # Check if transcription indicates no speech
                if "NO_SPEECH_DETECTED" in transcribed_text.upper():
                    print("⚠️ No clear speech detected in audio")
                    return None
                
                return transcribed_text
            else:
                print("❌ No text transcribed")
                return None
                
        except Exception as e:
            print(f"❌ Error in speech-to-text: {e}")
            return None
    
    def get_llm_response(self, text: str) -> Optional[str]:
        """Get response from LLM"""
        try:
            print("🤖 Getting AI response...")
            prompt = f"""
            {self.system_prompt}
            
            User: {text}
            Assistant:"""
            response = self.llm_model.generate_content(prompt)
            llm_response = response.text.strip()
            
            if llm_response:
                print(f"💬 AI Response: {llm_response}")
                return llm_response
            else:
                print("❌ No LLM response")
                return None
                
        except Exception as e:
            print(f"❌ Error getting LLM response: {e}")
            return None

    def get_llm_response_with_context(self, context: str) -> Optional[str]:
        """Get response from LLM with enhanced context"""
        try:
            print("🤖 Getting AI response with context...")
            prompt = f"""
            {self.system_prompt}
            
            {context}
            
            Assistant:"""
            response = self.llm_model.generate_content(prompt)
            llm_response = response.text.strip()
            
            if llm_response:
                print(f"💬 AI Response: {llm_response}")
                return llm_response
            else:
                print("❌ No LLM response")
                return None
                
        except Exception as e:
            print(f"❌ Error getting LLM response with context: {e}")
            return None
    
    def execute_roninhand_gesture(self, text: str) -> bool:
        """Intelligently execute gestures on RoninHand based on voice command"""
        try:
            import requests
            
            # First, try to find an exact match in available gestures
            text_lower = text.lower().strip()
            gesture_name = None
            
            # Check for exact gesture name matches
            for gesture in self.gestures.keys():
                if gesture.lower() == text_lower:
                    gesture_name = gesture
                    break
            
            # If no exact match, try partial matches
            if not gesture_name:
                for gesture in self.gestures.keys():
                    if gesture.lower() in text_lower or text_lower in gesture.lower():
                        gesture_name = gesture
                        break
            
            # If still no match, try common gesture mappings
            if not gesture_name:
                gesture_mappings = {
                    'fist': 'fist',
                    'make a fist': 'fist',
                    'close hand': 'fist',
                    'clench': 'fist',
                    'peace': 'peace',
                    'peace sign': 'peace',
                    'victory': 'peace',
                    'thumbs up': 'thumbs_up',
                    'thumb up': 'thumbs_up',
                    'open hand': 'open',
                    'open': 'open',
                    'spread': 'open',
                    'point': 'point',
                    'pointing': 'point',
                    'wave': 'wave',
                    'waving': 'wave',
                    'rock on': 'rock_on',
                    'handshake': 'handshake',
                    'grip': 'grip1closed',
                    'grip open': 'grip1open',
                    'grip closed': 'grip1closed',
                    'high five': 'High Five',
                    'rock sign': 'Rock Sign',
                    'ok sign': 'OK Sign'
                }
                
                for command, gesture in gesture_mappings.items():
                    if command in text_lower:
                        gesture_name = gesture
                        break
            
            if gesture_name:
                print(f"🤖 Executing RoninHand gesture: {gesture_name}")
                
                # Get the servo positions for this gesture
                if gesture_name in self.gestures:
                    servo_values = self.gestures[gesture_name]
                    
                    # Convert servo_1 format to 1 format for the server
                    positions = {}
                    for servo_key, value in servo_values.items():
                        if servo_key.startswith('servo_'):
                            servo_num = servo_key.split('_')[1]
                            positions[servo_num] = value
                    
                    print(f"📤 Sending positions to server: {positions}")
                    print(f"📤 JSON payload: {{'positions': {positions}}}")
                    
                    # Use the /update endpoint to set servo positions directly
                    response = requests.post('http://localhost:8000/update', 
                                           json={'positions': positions}, 
                                           timeout=2)
                    
                    if response.status_code == 200:
                        print(f"✅ Successfully executed {gesture_name} gesture!")
                        return True
                    else:
                        print(f"❌ Failed to execute {gesture_name} gesture - Status: {response.status_code}")
                        try:
                            error_detail = response.text
                            print(f"❌ Server response: {error_detail}")
                        except:
                            pass
                        return False
                else:
                    print(f"❌ Gesture '{gesture_name}' not found in gestures dictionary")
                    return False
            else:
                print(f"🤖 No recognized gesture found for: '{text}'")
                print(f"Available gestures: {', '.join(list(self.gestures.keys())[:10])}...")
                return False
                
        except Exception as e:
            print(f"⚠️  RoninHand gesture execution failed: {e}")
            return False

    def execute_roninhand_gesture_fast(self, gesture_name: str) -> bool:
        """Fast execution of gestures for execution mode - bypasses text parsing"""
        try:
            import requests
            
            if gesture_name not in self.gestures:
                print(f"❌ Gesture '{gesture_name}' not found in gestures")
                return False
            
            print(f"🚀 FAST EXECUTION: {gesture_name}")
            
            # Get the servo positions for this gesture
            servo_values = self.gestures[gesture_name]
            
            # Convert servo_1 format to 1 format for the server (optimized)
            positions = {}
            for servo_key, value in servo_values.items():
                if servo_key.startswith('servo_'):
                    servo_num = servo_key.split('_')[1]
                    positions[servo_num] = value
            
            # Use shorter timeout for faster execution mode
            response = requests.post('http://localhost:8000/update', 
                                   json={'positions': positions}, 
                                   timeout=1)  # Reduced timeout for speed
            
            if response.status_code == 200:
                print(f"✅ Fast execution successful: {gesture_name}")
                return True
            else:
                print(f"❌ Fast execution failed: {gesture_name} - Status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"⚠️  Fast gesture execution failed: {e}")
            return False

    def reload_gestures(self):
        """Reload gestures from the gestures.json file"""
        try:
            self.gestures = self.load_gestures()
            print(f"🔄 Reloaded {len(self.gestures)} gestures from gestures.json")
        except Exception as e:
            print(f"❌ Failed to reload gestures: {e}")

    def save_new_gesture(self, gesture_name: str, servo_values: dict) -> bool:
        """Save a new gesture to the gestures.json file"""
        try:
            print(f"💾 Saving new gesture '{gesture_name}' to gestures.json...")
            
            # Load current gestures from the main gestures.json file
            gestures_file = "hardware/roninhand/RHControl/gestures.json"
            if os.path.exists(gestures_file):
                with open(gestures_file, 'r') as f:
                    gestures_data = json.load(f)
            else:
                print(f"❌ Gestures file not found: {gestures_file}")
                return False
            
            # Add the new gesture to the gestures section
            gestures_data['gestures'][gesture_name] = servo_values
            
            # Save back to the main gestures.json file
            with open(gestures_file, 'w') as f:
                json.dump(gestures_data, f, indent=2)
            
            # Update local gestures dictionary
            self.gestures[gesture_name] = servo_values
            
            # Reload gestures to ensure system stays in sync
            self.reload_gestures()
            
            print(f"✅ Successfully saved new gesture '{gesture_name}' to {gestures_file}")
            print(f"📊 Gesture contains {len(servo_values)} servo positions")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save new gesture: {e}")
            return False

    def parse_and_execute_llm_gesture(self, llm_response: str, original_command: str = None) -> Union[str, bool]:
        """Parse LLM response for gesture generation and execute it"""
        try:
            # Look for servo values in the LLM response
            import re
            
            # Pattern to match servo values like "servo_1: 200" or "servo_1:200"
            servo_pattern = r'servo_(\d+)\s*:\s*(\d+)'
            matches = re.findall(servo_pattern, llm_response)
            
            if matches:
                # Generate a proper gesture name based on the user's original command
                gesture_name = None
                
                if original_command:
                    # Clean up the original command to create a meaningful gesture name
                    command_clean = original_command.strip().lower()
                    
                    # Remove common filler words and punctuation
                    filler_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'please', 'can', 'you', 'make', 'create', 'show', 'do', 'perform', 'execute']
                    for word in filler_words:
                        command_clean = command_clean.replace(word, ' ')
                    
                    # Extract meaningful words (3+ characters, alphanumeric)
                    words = [word.strip('.,!?-_') for word in command_clean.split() if len(word.strip('.,!?-_')) >= 3]
                    
                    # Filter out common non-descriptive words
                    descriptive_words = []
                    for word in words:
                        if word.isalnum() and word not in ['gesture', 'hand', 'finger', 'position', 'pose', 'move', 'set', 'put', 'make', 'create']:
                            descriptive_words.append(word)
                    
                    if descriptive_words:
                        # Use up to 3 most descriptive words
                        gesture_name = '_'.join(descriptive_words[:3])
                        print(f"🎯 Creating gesture from command: '{original_command}' -> '{gesture_name}'")
                
                # If still no meaningful name, try to extract from LLM response
                if not gesture_name:
                    lines = llm_response.split('\n')
                    for line in lines:
                        line_lower = line.lower()
                        # Look for common gesture-related words
                        if any(word in line_lower for word in ['finger', 'hand', 'gesture', 'pose', 'position']):
                            # Extract words that could be gesture names
                            words = line.split()
                            for word in words:
                                word_clean = word.strip('.,!?').lower()
                                if word_clean not in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
                                    if len(word_clean) > 2:  # Avoid very short words
                                        gesture_name = word_clean
                                        break
                            if gesture_name:
                                break
                
                # If still no meaningful name, create a descriptive timestamp-based name
                if not gesture_name:
                    # Create a descriptive name with timestamp
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    gesture_name = f"custom_gesture_{timestamp}"
                    print(f"⚠️  No descriptive name found, using timestamp: {gesture_name}")
                
                # Clean up the gesture name for safe file naming
                gesture_name = gesture_name.replace(' ', '_').replace('-', '_').replace('.', '').replace(',', '')
                gesture_name = ''.join(c for c in gesture_name if c.isalnum() or c == '_')
                
                # Ensure the name starts with a letter (not a number or underscore)
                if gesture_name and not gesture_name[0].isalpha():
                    gesture_name = 'gesture_' + gesture_name
                
                # Ensure the name is not too long
                if len(gesture_name) > 50:
                    gesture_name = gesture_name[:50]
                
                print(f"🎯 Final gesture name: '{gesture_name}'")
                
                # Check if gesture name already exists and suggest a variation
                original_name = gesture_name
                counter = 1
                while gesture_name in self.gestures:
                    gesture_name = f"{original_name}_{counter}"
                    counter += 1
                    if counter > 10:  # Prevent infinite loop
                        break
                
                if gesture_name != original_name:
                    print(f"⚠️  Gesture name '{original_name}' already exists, using '{gesture_name}' instead")
                
                # Build servo values dictionary
                servo_values = {}
                for servo_num, value in matches:
                    servo_key = f"servo_{servo_num}"
                    try:
                        servo_value = int(value)
                        
                        # Validate servo value against limits
                        if servo_key in self.servo_limits:
                            min_val = self.servo_limits[servo_key]['min']
                            max_val = self.servo_limits[servo_key]['max']
                            servo_value = max(min_val, min(max_val, servo_value))
                        
                        servo_values[servo_key] = servo_value
                    except ValueError:
                        print(f"⚠️  Invalid servo value for {servo_key}: {value}")
                        continue
                
                # Ensure ALL required servos are present (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)
                required_servos = ['servo_1', 'servo_2', 'servo_3', 'servo_4', 'servo_5', 'servo_6', 'servo_7', 'servo_8', 'servo_9', 'servo_10', 'servo_12']
                
                # Fill in missing servos with default values from existing gestures
                for servo_key in required_servos:
                    if servo_key not in servo_values:
                        if self.gestures:
                            # Use the first gesture as a template for defaults
                            first_gesture = list(self.gestures.values())[0]
                            default_value = first_gesture.get(servo_key, 20)
                        else:
                            # Fallback to basic defaults
                            default_value = 20
                        
                        servo_values[servo_key] = default_value
                        print(f"🤖 Added default value for {servo_key}: {default_value}")
                
                print(f"🤖 Generated gesture '{gesture_name}' with servo values:")
                for servo, value in servo_values.items():
                    print(f"   {servo}: {value}")
                
                # Validate that we have all required servos
                missing_servos = [servo for servo in required_servos if servo not in servo_values]
                if missing_servos:
                    print(f"❌ Missing required servos: {missing_servos}")
                    return False
                
                print(f"✅ All required servos present: {len(servo_values)}/11")
                
                # Show the full gesture JSON that will be saved
                gesture_json = {
                    gesture_name: servo_values
                }
                print(f"📝 Gesture JSON to save:")
                print(json.dumps(gesture_json, indent=2))
                
                # Save the new gesture
                if self.save_new_gesture(gesture_name, servo_values):
                    # Execute the gesture
                    if self.execute_custom_gesture(gesture_name, servo_values):
                        return gesture_name  # Return the gesture name for success message
                    else:
                        return False
                
            return False
            
        except Exception as e:
            print(f"❌ Error parsing LLM gesture: {e}")
            return False

    def execute_custom_gesture(self, gesture_name: str, servo_values: dict) -> bool:
        """Execute a custom gesture with servo values"""
        try:
            import requests
            
            print(f"🤖 Executing custom gesture: {gesture_name}")
            
            # Use the /update endpoint to set servo positions directly
            # Convert servo_1 format to 1 format for the server
            positions = {}
            for servo_key, value in servo_values.items():
                if servo_key.startswith('servo_'):
                    servo_num = servo_key.split('_')[1]
                    positions[servo_num] = value
            
            print(f"📤 Sending positions to server: {positions}")
            print(f"📤 JSON payload: {{'positions': {positions}}}")
            
            response = requests.post('http://localhost:8000/update', 
                                   json={'positions': positions}, 
                                   timeout=2)
            
            if response.status_code == 200:
                print(f"✅ Successfully executed custom gesture '{gesture_name}'!")
                return True
            else:
                print(f"❌ Failed to execute custom gesture '{gesture_name}' - Status: {response.status_code}")
                try:
                    error_detail = response.text
                    print(f"❌ Server response: {error_detail}")
                except:
                    pass
                return False
                
        except Exception as e:
            print(f"⚠️  Custom gesture execution failed: {e}")
            return False

    def list_available_gestures(self) -> str:
        """Return a formatted string of available gestures"""
        if not self.gestures:
            return "No gestures are currently available."
        
        gesture_list = []
        for i, gesture_name in enumerate(self.gestures.keys(), 1):
            gesture_list.append(f"{i}. {gesture_name}")
        
        return "Available gestures:\n" + "\n".join(gesture_list[:20]) + ("\n... and more" if len(gesture_list) > 20 else "")

    def get_gesture_info(self, gesture_name: str) -> str:
        """Get information about a specific gesture"""
        if gesture_name not in self.gestures:
            return f"Gesture '{gesture_name}' not found."
        
        gesture = self.gestures[gesture_name]
        servo_info = []
        
        for servo, value in gesture.items():
            if servo.startswith('servo_'):
                servo_info.append(f"{servo}: {value}")
        
        return f"Gesture '{gesture_name}' uses: {', '.join(servo_info)}"
    
    def text_to_speech(self, text: str) -> Optional[str]:
        """Convert text to speech using gTTS and play directly without saving to tmp"""
        try:
            print("🔊 Converting text to speech...")
            
            # Create a temporary file in memory (not in tmp folder)
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file_path = temp_file.name
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(temp_file_path)
                print(f"Generated AI response audio (size: {os.path.getsize(temp_file_path)} bytes)")
            
            if os.path.exists(temp_file_path):
                print(f"Playing AI response audio...")
                pygame.mixer.music.load(temp_file_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            else:
                print(f"❌ AI response audio file not found: {temp_file_path}")
                return None
            
            print(f"🔊 Spoken: {text}")
            time.sleep(0.5)
            
            # Clean up the temporary AI response file (not saved to tmp)
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                print(f"Cleaned up AI response audio file")
            
            return text
                
        except Exception as e:
            print(f"❌ Error in text-to-speech: {e}")
            return None
    
    def run_conversation_cycle(self):
        """Run one complete conversation cycle with RoninHand gesture support"""
        audio_bytes = self.record_audio()
        if not audio_bytes:
            print("❌ Failed to record audio")
            return
        
        transcribed_text = self.speech_to_text(audio_bytes)
        if not transcribed_text:
            print("❌ Failed to transcribe speech")
            return
        
        print(f"🎤 User said: '{transcribed_text}'")
        
        # Try to execute RoninHand gesture if command is recognized
        gesture_executed = self.execute_roninhand_gesture(transcribed_text)
        
        # Get AI response with enhanced context
        enhanced_prompt = f"""
        User request: {transcribed_text}
        
        Available gestures: {', '.join(list(self.gestures.keys())[:15])}...
        
        If the user requested a gesture:
        1. Confirm what gesture you're executing
        2. Explain what the gesture does
        3. Provide any relevant information about the gesture
        
        If the user asked a question or made a request unrelated to gestures:
        1. Answer their question or fulfill their request
        2. Keep responses concise and friendly
        
        Respond naturally as a helpful assistant."""
        
        llm_response = self.get_llm_response_with_context(enhanced_prompt)
        if not llm_response:
            print("❌ Failed to get LLM response")
            return
        
        # If gesture was executed, ensure the response acknowledges it
        if gesture_executed and not any(word in llm_response.lower() for word in ['executed', 'gesture', 'roninhand']):
            llm_response = f"I've executed the requested gesture on the RoninHand. {llm_response}"
        
        # Check if LLM generated a new gesture
        llm_gesture_executed = self.parse_and_execute_llm_gesture(llm_response)
        
        # If both gestures were executed, update the response
        if llm_gesture_executed:
            if not any(word in llm_response.lower() for word in ['executed', 'gesture', 'roninhand']):
                llm_response = f"I've also generated and executed a new gesture based on your request. {llm_response}"
        
        speech_result = self.text_to_speech(llm_response)
        if speech_result:
            print("✅ Conversation cycle completed!")
        else:
            print("❌ Failed to convert response to speech")
    
    def handle_emg_and_triggers(self):
        """Process EMG data and handle triggers in a separate thread"""
        while self.running:
            raw = self.read_emg()
            if raw is not None:
                rectified, in_burst, trigger = self.process_emg(raw)
                current_time = time.time() - self.start_time
                
                with self.serial_lock:
                    self.raw_data.append(raw)
                    self.emg_data.append(rectified)
                    self.in_burst_data.append(1.0 if in_burst else 0.0)
                    self.trigger_data.append(1.0 if trigger else 0.0)
                    self.time_data.append(current_time)
                    
                    if len(self.emg_data) > self.max_points:
                        self.raw_data.pop(0)
                        self.emg_data.pop(0)
                        self.in_burst_data.pop(0)
                        self.trigger_data.pop(0)
                        self.time_data.pop(0)
                
                # Note: No more trigger handling here - it's now done in process_emg
                # based on the assistant state
            
            time.sleep(0.005)
    
    def handle_wake_word_detection(self):
        """Handle wake word detection in a separate thread"""
        while self.running:
            try:
                # Only check for wake word when in IDLE state
                if self.assistant_state == "IDLE":
                    if self.detect_wake_word():
                        print("🎯 Wake word triggered - starting recording...")
                        self.start_recording()
                        # Add a small delay to prevent multiple detections
                        time.sleep(1.0)
                
                # Check every 100ms to balance responsiveness and CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error in wake word detection: {e}")
                time.sleep(0.1)
    
    def update_plot(self, frame):
        """Update the EMG visualization"""
        try:
            # Check if matplotlib window is still valid
            if not hasattr(self, 'fig') or not self.fig or not self.fig.canvas:
                return None,
            
            with self.serial_lock:
                if not self.emg_data:
                    return self.ax,
                
                # Clear the plot safely
                try:
                    self.ax.clear()
                except Exception as clear_error:
                    print(f"⚠️ Error clearing plot: {clear_error}")
                    return self.ax,
                
                # Plot data with error handling
                try:
                    self.ax.plot(self.time_data, self.emg_data, label="Rectified EMG", color='blue', linewidth=1.5)
                    self.ax.plot(self.time_data, self.raw_data, label="Raw EMG", color='gray', alpha=0.5)
                    self.ax.axhline(y=self.burst_threshold, color='red', linestyle='--', linewidth=2, label=f"Burst Threshold: {self.burst_threshold:.3f}")
                    
                    # Add threshold zone for better visibility
                    if self.emg_data:
                        max_val = max(self.emg_data) if self.emg_data else 1
                        self.ax.axhspan(0, self.burst_threshold, alpha=0.1, color='green', label="Below Threshold")
                        self.ax.axhspan(self.burst_threshold, max_val, alpha=0.1, color='red', label="Above Threshold")
                    
                    # Add state indicator background
                    if self.time_data:
                        max_time = max(self.time_data) if self.time_data else 1
                        if self.assistant_state == "RECORDING":
                            self.ax.axvspan(0, max_time, alpha=0.1, color='red', label="Recording State")
                        elif self.assistant_state == "PROCESSING":
                            self.ax.axvspan(0, max_time, alpha=0.1, color='orange', label="Processing State")
                    
                    # Plot burst regions
                    burst_starts = []
                    burst_ends = []
                    in_burst = False
                    for i, burst in enumerate(self.in_burst_data):
                        if burst == 1.0 and not in_burst:
                            in_burst = True
                            burst_starts.append(self.time_data[i])
                        elif burst == 0.0 and in_burst:
                            in_burst = False
                            burst_ends.append(self.time_data[i])
                    if in_burst and self.in_burst_data and self.in_burst_data[-1] == 1.0:
                        burst_ends.append(self.time_data[-1])
                    
                    for start, end in zip(burst_starts, burst_ends):
                        self.ax.axvspan(start, end, color='yellow', alpha=0.3, label="Burst" if start == burst_starts[0] else "")
                    
                    # Plot trigger points
                    for i, trig in enumerate(self.trigger_data):
                        if trig == 1.0:
                            self.ax.axvline(x=self.time_data[i], color='red', linestyle='-', label="Trigger" if i == 0 else "")
                    
                    self.ax.set_xlabel("Time (s)")
                    self.ax.set_ylabel("EMG Signal")
                    
                    # Add statistics to title
                    if self.emg_data:
                        current_max = max(self.emg_data[-50:]) if len(self.emg_data) >= 50 else max(self.emg_data)
                        current_avg = np.mean(self.emg_data[-50:]) if len(self.emg_data) >= 50 else np.mean(self.emg_data)
                        if self.current_mode == "gesture_execution":
                            title = f"Real-Time EMG Signal | Mode: {self.current_mode.upper()} | Grip: {self.current_grip_state} | State: {self.assistant_state} | Current Max: {current_max:.3f} | Current Avg: {current_avg:.3f} | Threshold: {self.burst_threshold:.3f}"
                        else:
                            title = f"Real-Time EMG Signal | Mode: {self.current_mode.upper()} | State: {self.assistant_state} | Current Max: {current_max:.3f} | Current Avg: {current_avg:.3f} | Threshold: {self.burst_threshold:.3f}"
                    else:
                        if self.current_mode == "gesture_execution":
                            title = f"Real-Time EMG Signal | Mode: {self.current_mode.upper()} | Grip: {self.current_grip_state} | State: {self.assistant_state} | Threshold: {self.burst_threshold:.3f}"
                        else:
                            title = f"Real-Time EMG Signal | Mode: {self.current_mode.upper()} | State: {self.assistant_state} | Threshold: {self.burst_threshold:.3f}"
                    self.ax.set_title(title)
                    
                    self.ax.legend(loc="upper right")
                    self.ax.grid(True)
                    
                except Exception as plot_error:
                    print(f"⚠️ Error plotting data: {plot_error}")
                    # Try to show a simple error message on the plot
                    try:
                        self.ax.text(0.5, 0.5, f"Plot Error: {plot_error}", 
                                   transform=self.ax.transAxes, ha='center', va='center')
                    except:
                        pass
                
                return self.ax,
                
        except Exception as e:
            print(f"❌ Error in update_plot: {e}")
            # Don't crash the application, just return empty plot
            try:
                if hasattr(self, 'ax') and self.ax:
                    return self.ax,
            except:
                pass
            return None,
    
    def run_interactive(self):
        """Run the voice assistant with EMG visualization"""
        print("🎤 Starting Simple Voice Assistant with EMG Processing and Wake Word Detection...")
        print("🔄 DUAL TRIGGER INTERACTION MODEL:")
        print("   🎯 DOUBLE-TAP (2 muscle twitches) = Start recording")
        print("   🎯 SAY 'HeyRonin' = Start recording")
        print("   ⏹️  SINGLE-TAP (1 muscle twitch) = Stop recording & process")
        print("   📱 Current state will be shown in the interface")
        print("")
        print("🎯 TWO-MODE SYSTEM:")
        print("📝 GESTURE CREATION MODE (default):")
        print("   • EMG triggers activate voice assistant for commands")
        print("   • Wake word 'HeyRonin' activates voice assistant")
        print("   • Create and execute gestures through voice commands")
        print()
        print("🎯 GESTURE EXECUTION MODE:")
        print("   • EMG triggers toggle between grip2open and grip2closed")
        print("   • Default state: grip2open")
        print("   • Each flex switches to opposite grip state")
        print("   • Wake word 'HeyRonin' activates voice assistant for mode switching only")
        print("   • Say 'switch control mode' to return to creation mode")
        print()
        print("🎯 RONINHAND VOICE COMMANDS:")
        print("• 'Make a fist' or 'Close hand'")
        print("• 'Peace sign' or 'Victory'")
        print("• 'Thumbs up'")
        print("• 'Open hand'")
        print("• 'Point' or 'Pointing'")
        print("• 'Wave' or 'Waving'")
        print()
        print("💡 TIP: Contract your muscles to trigger voice recording, then speak your command!")
        print("🔧 IMPROVED NOISE HANDLING: Initial calibration only, no auto-recalibration")
        print("🔄 MODE SWITCHING: Say 'HeyRonin, switch control mode' to toggle between modes")
        print("=" * 50)
        
        print("💡 Tips:")
        print("   - Make sure your muscle is relaxed during calibration")
        print("   - Contract your muscle firmly for 0.1-0.5 seconds")
        print("   - Double-tap within 2 seconds to start recording")
        print("   - Single-tap to stop recording and process your request")
        print(f"   - Recording will continue until you tap again or max duration ({self.max_recording_duration}s) reached")
        print("   - Press 't' to manually adjust threshold")
        print("   - Press 'c' to trigger manual recalibration")
        print("   - Press 'a' to toggle auto-recalibration")
        print("   - Press 'r' to manually start/stop recording")
        print("   - Press 'n' to toggle noise reduction")
        print("   - Press 's' to adjust noise reduction strength")
        print("   - Press 'i' to show current EMG status")
        print("   - Press 'l' to list user recordings")
        print("   - Press 'x' to clean up old recordings")
        print("   - Press 'm' to show current mode status")
        print("   - Press 'd' to debug gesture state")
        print("   - Press 'e' to test gesture execution mode")
        print("   - Press 'R' to reload gestures")
        print("   - Press 'h' to show this help menu")
        print("   - Press 'q' to quit")
        print("🔄 Auto-recalibration is DISABLED for stability - only initial calibration performed")
        print("=" * 50)
        
        # MANDATORY INITIAL CALIBRATION FIRST
        print("\n🎯 INITIAL EMG CALIBRATION REQUIRED")
        print("Please relax your muscle completely for 3 seconds...")
        time.sleep(1)
        print("Starting calibration in 2...")
        time.sleep(1)
        print("Starting calibration in 1...")
        time.sleep(1)
        print("🎯 CALIBRATING NOW - KEEP MUSCLE RELAXED!")
        self.calibrate_emg()
        print("✅ Initial calibration complete! System ready.")
        print("=" * 50)
        
        emg_thread = threading.Thread(target=self.handle_emg_and_triggers, daemon=True)
        emg_thread.start()
        
        # Start wake word detection thread if Porcupine is available
        wake_word_thread = None
        if self.porcupine:
            wake_word_thread = threading.Thread(target=self.handle_wake_word_detection, daemon=True)
            wake_word_thread.start()
            print("🎯 Wake word detection thread started")
        
        # Set up matplotlib with error handling
        # Check if user wants to disable visualization
        disable_viz = os.getenv('DISABLE_EMG_VISUALIZATION', 'false').lower() == 'true'
        
        if disable_viz:
            print("🖥️ EMG visualization disabled by environment variable")
            self.fig = None
            self.ax = None
            ani = None
        else:
            try:
                # Use a more robust backend
                import matplotlib
                matplotlib.use('TkAgg', force=True)
                
                self.fig, self.ax = plt.subplots(figsize=(12, 8))
                self.fig.canvas.manager.set_window_title('RoninHand EMG Voice Control')
                
                # Create animation with error handling
                try:
                    ani = FuncAnimation(self.fig, self.update_plot, interval=50, cache_frame_data=False, blit=False)
                    print("✅ EMG visualization initialized successfully")
                except Exception as anim_error:
                    print(f"⚠️ Animation setup error: {anim_error}")
                    # Continue without animation if it fails
                    ani = None
                    
            except Exception as plot_setup_error:
                print(f"❌ Error setting up matplotlib: {plot_setup_error}")
                print("⚠️ Continuing without EMG visualization...")
                self.fig = None
                self.ax = None
                ani = None
        
        # Add keyboard event handling
        def on_key(event):
            if event.key == 't':
                try:
                    new_threshold = float(input("Enter new threshold value (current: {:.3f}): ".format(self.burst_threshold)))
                    self.burst_threshold = new_threshold
                    print(f"Threshold updated to: {self.burst_threshold:.3f}")
                except ValueError:
                    print("Invalid input. Threshold unchanged.")
            elif event.key == 'c':
                print("🔄 Manual recalibration triggered...")
                self.perform_dynamic_calibration()
            elif event.key == 'a':
                # Toggle auto-recalibration (currently disabled for stability)
                print("⚠️ Auto-recalibration is disabled for stability")
                print("   Use 'c' for manual recalibration when needed")
            elif event.key == 'r':
                # Manual recording control for testing
                if self.assistant_state == "IDLE":
                    print("🎤 Manual recording start...")
                    self.start_recording()
                elif self.assistant_state == "RECORDING":
                    print("⏹️ Manual recording stop...")
                    self.stop_recording()
                else:
                    print(f"⚠️ Cannot control recording while in {self.assistant_state} state")
            elif event.key == 'n':
                # Toggle noise reduction
                self.enable_noise_reduction = not self.enable_noise_reduction
                status = "enabled" if self.enable_noise_reduction else "disabled"
                print(f"Noise reduction {status}")
            elif event.key == 's':
                # Adjust noise reduction strength
                try:
                    new_strength = float(input(f"Enter noise reduction strength (0.0-1.0, current: {self.noise_reduction_strength:.2f}): "))
                    if 0.0 <= new_strength <= 1.0:
                        self.noise_reduction_strength = new_strength
                        print(f"Noise reduction strength updated to: {self.noise_reduction_strength:.2f}")
                    else:
                        print("Strength must be between 0.0 and 1.0")
                except ValueError:
                    print("Invalid input. Strength unchanged.")
            elif event.key == 'i':
                # Show current EMG status
                print(f"\n📊 Current EMG Status:")
                print(f"   Assistant State: {self.assistant_state}")
                if self.assistant_state == "RECORDING":
                    duration = time.time() - self.recording_start_time
                    print(f"   Recording duration: {duration:.1f}s")
                print(f"   EMG Threshold: {self.burst_threshold:.3f}")
                print(f"   EMG Baseline: {self.baseline:.3f}")
                print(f"   Burst Count: {self.burst_count}")
                print(f"   Noise Reduction: {'Enabled' if self.enable_noise_reduction else 'Disabled'}")
                if self.enable_noise_reduction:
                    print(f"   Noise Reduction Strength: {self.noise_reduction_strength:.2f}")
                    print(f"   High-Pass Filter: {self.high_pass_freq} Hz")
                    print(f"   Low-Pass Filter: {self.low_pass_freq} Hz")
                    print(f"   Noise Gate Threshold: {self.noise_gate_threshold:.3f}")
                print(f"   Wake Word Detection: {'Enabled' if self.porcupine else 'Disabled'}")
                if self.porcupine:
                    print(f"   Wake Word: HeyRonin")
            elif event.key == 'l':
                # List user recordings
                print("\n📁 User Recordings:")
                self.list_user_recordings()
            elif event.key == 'g':
                # List custom gestures created by user
                print("\n🤖 Custom Gestures Created:")
                self.list_custom_gestures()
            elif event.key == 'x':
                # Clean up old recordings
                try:
                    hours = int(input("Clean up recordings older than how many hours? (default: 24): ") or "24")
                    self.cleanup_old_recordings(hours)
                except ValueError:
                    print("Invalid input. Using default 24 hours.")
                    self.cleanup_old_recordings(24)
            elif event.key == 'm':
                # Show current mode status
                if self.current_mode == "gesture_execution":
                    print(f"\n🎯 Current Mode: {self.current_mode.upper()}")
                    print(f"🎯 Current Grip State: {self.current_grip_state}")
                else:
                    print(f"\n🎯 Current Mode: {self.current_mode.upper()}")
            elif event.key == 'd':
                # Debug gesture state
                self.debug_gesture_state()
            elif event.key == 'e':
                # Test gesture execution mode
                if self.current_mode == "gesture_execution":
                    print(f"\n🧪 Testing Gesture Execution Mode")
                    print(f"🎯 Current grip state: {self.current_grip_state}")
                    
                    # Toggle grip state and execute gesture
                    if self.current_grip_state == "grip2open":
                        gesture_name = "grip2closed"
                        self.current_grip_state = "grip2closed"
                    else:
                        gesture_name = "grip2open"
                        self.current_grip_state = "grip2open"
                    
                    print(f"🤖 Testing execution of {gesture_name} gesture...")
                    if gesture_name in self.gestures:
                        success = self.execute_roninhand_gesture(gesture_name)
                        if success:
                            print(f"✅ Successfully executed {gesture_name} gesture")
                        else:
                            print(f"❌ Failed to execute {gesture_name} gesture")
                    else:
                        print(f"❌ Gesture '{gesture_name}' not found in gestures")
                        print(f"Available gestures: {list(self.gestures.keys())}")
                else:
                    print(f"\n⚠️ Not in gesture execution mode. Current mode: {self.current_mode}")
                    print("Use voice command 'switch to gesture execution' or 'gesture execution mode' to switch")
            elif event.key == 'R':
                # Reload gestures (capital R to avoid conflict with 'r' for recording)
                print("\n🔄 Reloading gestures...")
                self.reload_gestures()
                self.debug_gesture_state()
            elif event.key == 'h':
                print("\n💡 Keyboard Controls:")
                print("   't' - Adjust threshold manually")
                print("   'c' - Manual recalibration")
                print("   'a' - Toggle auto-recalibration")
                print("   'r' - Manual start/stop recording")
                print("   'n' - Toggle noise reduction")
                print("   's' - Adjust noise reduction strength")
                print("   'i' - Show EMG status")
                print("   'l' - List user recordings")
                print("   'g' - List custom gestures")
                print("   'x' - Clean up old recordings")
                print("   'm' - Show current mode status")
                print("   'd' - Debug gesture state")
                print("   'e' - Test gesture execution mode")
                print("   'R' - Reload gestures")
                print("   'h' - Show this help menu")
                print("   'q' - Quit")
        
        # Connect keyboard events only if matplotlib is working
        if self.fig and self.fig.canvas:
            try:
                self.fig.canvas.mpl_connect('key_press_event', on_key)
                print("✅ Keyboard controls enabled")
            except Exception as key_error:
                print(f"⚠️ Keyboard controls disabled: {key_error}")
        
        # Main event loop with robust error handling
        try:
            if self.fig:
                print("🖥️ Starting EMG visualization...")
                plt.show()
            else:
                print("🖥️ Running without visualization (console mode)...")
                # Console mode - keep running until user quits
                last_status_time = time.time()
                while self.running:
                    try:
                        time.sleep(1)
                        # Show status every 10 seconds
                        current_time = time.time()
                        if current_time - last_status_time >= 10:
                            if self.current_mode == "gesture_execution":
                                print(f"📊 Status: Mode={self.current_mode.upper()}, Grip State={self.current_grip_state}, State={self.assistant_state}, EMG Active={'Yes' if self.emg_active else 'No'}")
                            else:
                                print(f"📊 Status: Mode={self.current_mode.upper()}, State={self.assistant_state}, EMG Active={'Yes' if self.emg_active else 'No'}")
                            last_status_time = current_time
                    except KeyboardInterrupt:
                        break
                        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as show_error:
            print(f"❌ Error in matplotlib display: {show_error}")
            print("🔄 Continuing in console mode...")
            # Fallback to console mode
            last_status_time = time.time()
            while self.running:
                try:
                    time.sleep(1)
                    # Show status every 10 seconds
                    current_time = time.time()
                    if current_time - last_status_time >= 10:
                        if self.current_mode == "gesture_execution":
                            print(f"📊 Status: Mode={self.current_mode.upper()}, Grip State={self.current_grip_state}, State={self.assistant_state}, EMG Active={'Yes' if self.emg_active else 'No'}")
                        else:
                            print(f"📊 Status: Mode={self.current_mode.upper()}, State={self.assistant_state}, EMG Active={'Yes' if self.emg_active else 'No'}")
                        last_status_time = current_time
                except KeyboardInterrupt:
                    break
        finally:
            print("🧹 Cleaning up...")
            self.running = False
            if self.serial and self.serial.is_open:
                self.serial.close()
            if self.porcupine:
                self.porcupine.delete()
                print("🧹 Porcupine cleanup completed")
            try:
                plt.close('all')
            except:
                pass

    def list_custom_gestures(self):
        """List all custom gestures created by the user (not pre-built ones)"""
        try:
            if not self.gestures:
                print("📁 No gestures found")
                return
            
            # Filter out pre-built gestures (those that don't start with common prefixes)
            custom_gestures = []
            prebuilt_prefixes = ['fist', 'open', 'point', 'peace', 'thumbs', 'okay', 'rock', 'paper', 'scissors']
            
            for gesture_name in self.gestures.keys():
                is_custom = True
                for prefix in prebuilt_prefixes:
                    if gesture_name.lower().startswith(prefix):
                        is_custom = False
                        break
                
                if is_custom:
                    custom_gestures.append(gesture_name)
            
            if custom_gestures:
                print(f"🤖 Found {len(custom_gestures)} custom gestures:")
                for i, gesture_name in enumerate(sorted(custom_gestures), 1):
                    print(f"   {i}. {gesture_name}")
                    # Show a few servo values as preview
                    gesture = self.gestures[gesture_name]
                    servo_preview = []
                    for servo_key in ['servo_1', 'servo_2', 'servo_3']:
                        if servo_key in gesture:
                            servo_preview.append(f"{servo_key}: {gesture[servo_key]}")
                    if servo_preview:
                        print(f"      Preview: {', '.join(servo_preview)}...")
            else:
                print("🤖 No custom gestures found yet. Create some by asking the AI to make new hand positions!")
            
        except Exception as e:
            print(f"❌ Error listing custom gestures: {e}")

    def list_user_recordings(self):
        """List all user recordings in the tmp folder"""
        try:
            if not os.path.exists("tmp"):
                print("📁 No tmp folder found")
                return []
            
            recordings = []
            for filename in os.listdir("tmp"):
                if filename.startswith("user_recording_") and filename.endswith(".wav"):
                    filepath = os.path.join("tmp", filename)
                    file_size = os.path.getsize(filepath)
                    file_time = os.path.getctime(filepath)
                    recordings.append({
                        'filename': filename,
                        'filepath': filepath,
                        'size': file_size,
                        'created': file_time
                    })
            
            if recordings:
                print(f"📁 Found {len(recordings)} user recordings:")
                for i, rec in enumerate(sorted(recordings, key=lambda x: x['created'], reverse=True)):
                    created_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec['created']))
                    size_kb = rec['size'] / 1024
                    print(f"   {i+1}. {rec['filename']}")
                    print(f"      Created: {created_str}")
                    print(f"      Size: {size_kb:.1f} KB")
            else:
                print("📁 No user recordings found in tmp folder")
            
            return recordings
            
        except Exception as e:
            print(f"❌ Error listing recordings: {e}")
            return []
    
    def cleanup_old_recordings(self, max_age_hours=24):
        """Clean up old user recordings (older than max_age_hours)"""
        try:
            recordings = self.list_user_recordings()
            if not recordings:
                return
            
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            cleaned_count = 0
            
            for rec in recordings:
                age = current_time - rec['created']
                if age > max_age_seconds:
                    try:
                        os.unlink(rec['filepath'])
                        print(f"🗑️  Cleaned up old recording: {rec['filename']}")
                        cleaned_count += 1
                    except Exception as e:
                        print(f"❌ Failed to clean up {rec['filename']}: {e}")
            
            if cleaned_count > 0:
                print(f"✅ Cleaned up {cleaned_count} old recordings")
            else:
                print("✅ No old recordings to clean up")
                
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")

    def load_gestures(self):
        """Load gestures from gesture files"""
        gestures = {}
        
        # Try to load from multiple possible locations
        gesture_files = [
            "hardware/roninhand/RHControl/gestures.json",
            "data/gestures/custom_gestures.json",
            "var/custom_gestures.json"
        ]
        
        for file_path in gesture_files:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if 'gestures' in data:
                            gestures.update(data['gestures'])
                        else:
                            # If it's a custom gestures file with direct gesture definitions
                            gestures.update(data)
                    print(f"✅ Loaded gestures from {file_path}")
                    
                    # Debug: Check if grip gestures are loaded
                    if 'grip2open' in gestures:
                        print(f"✅ Found grip2open gesture: {gestures['grip2open']}")
                    if 'grip2closed' in gestures:
                        print(f"✅ Found grip2closed gesture: {gestures['grip2closed']}")
                        
            except Exception as e:
                print(f"⚠️ Failed to load gestures from {file_path}: {e}")
        
        if not gestures:
            print("⚠️ No gestures loaded, using fallback gestures")
            gestures = {
                'fist': {'servo_1': 440, 'servo_2': 440, 'servo_3': 440, 'servo_4': 440, 'servo_5': 440, 'servo_6': 440, 'servo_7': 440, 'servo_8': 440, 'servo_9': 440, 'servo_10': 440, 'servo_12': 20},
                'open': {'servo_1': 20, 'servo_2': 20, 'servo_3': 20, 'servo_4': 20, 'servo_5': 20, 'servo_6': 20, 'servo_7': 20, 'servo_8': 20, 'servo_9': 20, 'servo_10': 20, 'servo_12': 20},
                'point': {'servo_1': 440, 'servo_2': 440, 'servo_3': 440, 'servo_4': 440, 'servo_5': 440, 'servo_6': 20, 'servo_7': 440, 'servo_8': 20, 'servo_9': 440, 'servo_10': 440, 'servo_12': 20},
                'peace': {'servo_1': 440, 'servo_2': 440, 'servo_3': 440, 'servo_4': 440, 'servo_5': 440, 'servo_6': 20, 'servo_7': 440, 'servo_8': 20, 'servo_9': 440, 'servo_10': 440, 'servo_12': 20},
                'thumbs_up': {'servo_1': 440, 'servo_2': 440, 'servo_3': 440, 'servo_4': 440, 'servo_5': 440, 'servo_6': 440, 'servo_7': 440, 'servo_8': 440, 'servo_9': 440, 'servo_10': 20, 'servo_12': 20}
            }
        
        # Final debug output
        print(f"📊 Total gestures loaded: {len(gestures)}")
        print(f"📋 Available gestures: {list(gestures.keys())}")
        
        return gestures

    def load_servo_limits(self):
        """Load servo limits from gesture files"""
        servo_limits = {}
        
        # Try to load from multiple possible locations
        gesture_files = [
            "hardware/roninhand/RHControl/gestures.json",
            "data/gestures/custom_gestures.json",
            "var/custom_gestures.json"
        ]
        
        for file_path in gesture_files:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if 'servo_limits' in data:
                            servo_limits.update(data['servo_limits'])
                            break
            except Exception as e:
                print(f"⚠️ Failed to load servo limits from {file_path}: {e}")
        
        if not servo_limits:
            print("⚠️ No servo limits loaded, using default limits")
            servo_limits = {
                'servo_1': {'min': 20, 'max': 440},
                'servo_2': {'min': 20, 'max': 440},
                'servo_3': {'min': 20, 'max': 440},
                'servo_4': {'min': 20, 'max': 440},
                'servo_5': {'min': 20, 'max': 440},
                'servo_6': {'min': 20, 'max': 440},
                'servo_7': {'min': 20, 'max': 440},
                'servo_8': {'min': 20, 'max': 440},
                'servo_9': {'min': 20, 'max': 440},
                'servo_10': {'min': 20, 'max': 440},
                'servo_12': {'min': 20, 'max': 800}
            }
        
        return servo_limits

    def generate_system_prompt(self):
        """Generate a context-aware system prompt that adapts to the current mode"""
        available_gestures = list(self.gestures.keys())
        gesture_list = ", ".join(available_gestures)
        
        if self.current_mode == "gesture_creation":
            prompt = f"""You are a robotic hand gesture controller in GESTURE CREATION mode. 

AVAILABLE GESTURES: {gesture_list}

When the user requests a new gesture, create it with ALL 11 servo values.
When the user asks questions or makes other requests, respond naturally as a helpful assistant.

REQUIREMENTS FOR GESTURES:
- Generate servo values for ALL servos: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12
- Use format "servo_X: Y" for each servo
- Base values on existing gestures
- Keep responses short - servo values only

Example gesture format:
servo_1: 200
servo_2: 150
servo_3: 180
servo_4: 160
servo_5: 140
servo_6: 120
servo_7: 200
servo_8: 100
servo_9: 220
servo_10: 180
servo_12: 400"""
        else:
            # In execution mode, don't instruct gesture creation
            prompt = f"""You are a robotic hand gesture controller in GESTURE EXECUTION mode.

AVAILABLE GESTURES: {gesture_list}

You are in execution mode where EMG triggers control gestures directly.
Respond naturally to user questions and requests as a helpful assistant.
Do NOT create new gestures in this mode."""
        
        return prompt

    def switch_mode(self, new_mode: str) -> str:
        """Switch between gesture creation and execution modes"""
        if new_mode == "gesture_creation":
            if self.current_mode != "gesture_creation":
                self.current_mode = "gesture_creation"
                # Use base debounce time for creation mode (more precise control needed)
                self.burst_debounce_time = self.base_burst_debounce_time
                # Regenerate system prompt for new mode
                self.system_prompt = self.generate_system_prompt()
                print(f"🔄 Switched to {self.current_mode} mode (debounce: {self.burst_debounce_time:.3f}s)")
                return f"Switched to gesture creation mode. EMG triggers activate voice assistant for commands and gesture creation."
            else:
                return "Already in gesture creation mode."
        elif new_mode == "gesture_execution":
            if self.current_mode != "gesture_execution":
                self.current_mode = "gesture_execution"
                self.current_grip_state = "grip2open"  # Reset to default state
                # Use faster debounce time for execution mode (faster response needed)
                self.burst_debounce_time = self.execution_mode_debounce_time
                # Regenerate system prompt for new mode
                self.system_prompt = self.generate_system_prompt()
                print(f"🔄 Switched to {self.current_mode} mode (debounce: {self.burst_debounce_time:.3f}s)")
                return f"Switched to gesture execution mode. EMG triggers now toggle between grip2open and grip2closed. Starting with grip2open."
            else:
                return "Already in gesture execution mode."
        else:
            return f"Unknown mode: {new_mode}. Available modes: gesture_creation, gesture_execution"
    
    def get_current_mode(self) -> str:
        """Get current mode status"""
        return f"Current mode: {self.current_mode}"
    
    def regenerate_system_prompt(self):
        """Regenerate the system prompt based on current mode"""
        self.system_prompt = self.generate_system_prompt()
        print(f"🔄 Regenerated system prompt for {self.current_mode} mode")
    
    def execute_gesture_execution_mode(self, burst_duration: float, burst_count: int) -> bool:
        """Execute gestures in execution mode based on EMG trigger characteristics"""
        try:
            if self.current_mode != "gesture_execution":
                return False
            
            # Initialize grip state if not exists
            if not hasattr(self, 'current_grip_state'):
                self.current_grip_state = "grip2open"  # Default state
            
            # Toggle between grip2open and grip2closed on each trigger
            if self.current_grip_state == "grip2open":
                gesture_name = "grip2closed"
                self.current_grip_state = "grip2closed"
                print(f"🎯 EMG trigger detected - switching to {gesture_name}")
            else:
                gesture_name = "grip2open"
                self.current_grip_state = "grip2open"
                print(f"🎯 EMG trigger detected - switching to {gesture_name}")
            
            # Execute the gesture using the FAST method for execution mode
            if gesture_name in self.gestures:
                print(f"🤖 Executing {gesture_name} gesture from gestures.json...")
                return self.execute_roninhand_gesture_fast(gesture_name)  # Use fast execution for speed
            else:
                print(f"❌ Gesture '{gesture_name}' not found in gestures")
                print(f"Available gestures: {list(self.gestures.keys())}")
                return False
                
        except Exception as e:
            print(f"❌ Error in gesture execution mode: {e}")
            return False
    
    def check_mode_switch_command(self, text: str) -> bool:
        """Check if the text contains a mode switch command"""
        text_lower = text.lower().strip()
        
        for command in self.mode_switch_commands:
            if command in text_lower:
                return True
        
        return False
    
    def handle_mode_switch(self, text: str) -> str:
        """Handle mode switching based on voice command"""
        text_lower = text.lower().strip()
        
        if any(cmd in text_lower for cmd in ["gesture creation", "creation mode"]):
            if self.current_mode == "gesture_creation":
                return "Already in gesture creation mode. EMG triggers activate LLM for voice commands and gesture creation."
            else:
                return self.switch_mode("gesture_creation")
        elif any(cmd in text_lower for cmd in ["gesture execution", "execution mode"]):
            if self.current_mode == "gesture_execution":
                return "Already in gesture execution mode. EMG triggers execute grip2open/grip2closed gestures directly."
            else:
                return self.switch_mode("gesture_execution")
        elif "switch control mode" in text_lower:
            # Toggle between modes
            if self.current_mode == "gesture_creation":
                return self.switch_mode("gesture_execution")
            else:
                return self.switch_mode("gesture_creation")
        else:
            return "Mode switch command not recognized. Use 'switch control mode' or specify the mode."

    def debug_gesture_state(self):
        """Debug method to show current gesture state and available gestures"""
        print("🔍 DEBUG: Current Gesture State")
        print(f"📊 Current mode: {self.current_mode}")
        print(f"🎯 Current grip state: {getattr(self, 'current_grip_state', 'Not set')}")
        print(f"⏱️ Current burst debounce time: {self.burst_debounce_time:.3f}s")
        print(f"⏱️ Base debounce time: {self.base_burst_debounce_time:.3f}s")
        print(f"⏱️ Execution mode debounce time: {self.execution_mode_debounce_time:.3f}s")
        print(f"🤖 Current system prompt mode: {'GESTURE CREATION' if 'GESTURE CREATION mode' in self.system_prompt else 'GESTURE EXECUTION' if 'GESTURE EXECUTION mode' in self.system_prompt else 'UNKNOWN'}")
        print(f"📝 System prompt preview: {self.system_prompt[:100]}...")
        print(f"📋 Total gestures loaded: {len(self.gestures)}")
        print(f"📋 Available gestures: {list(self.gestures.keys())}")
        
        # Test mode switching functionality
        print("\n🧪 Testing Mode Switching:")
        original_mode = self.current_mode
        print(f"   Original mode: {original_mode}")
        
        # Test switching to opposite mode
        if original_mode == "gesture_creation":
            test_mode = "gesture_execution"
        else:
            test_mode = "gesture_creation"
        
        print(f"   Testing switch to: {test_mode}")
        result = self.switch_mode(test_mode)
        print(f"   Switch result: {result}")
        print(f"   New mode: {self.current_mode}")
        print(f"   New debounce time: {self.burst_debounce_time:.3f}s")
        print(f"   New system prompt mode: {'GESTURE CREATION' if 'GESTURE CREATION mode' in self.system_prompt else 'GESTURE EXECUTION' if 'GESTURE EXECUTION mode' in self.system_prompt else 'UNKNOWN'}")
        
        # Switch back to original mode
        print(f"   Switching back to: {original_mode}")
        self.switch_mode(original_mode)
        print(f"   Restored mode: {self.current_mode}")
        print(f"   Restored debounce time: {self.burst_debounce_time:.3f}s")
        
        # Check for specific grip gestures
        if 'grip2open' in self.gestures:
            print(f"✅ grip2open found: {self.gestures['grip2open']}")
        else:
            print("❌ grip2open NOT found in gestures")
            
        if 'grip2closed' in self.gestures:
            print(f"✅ grip2closed found: {self.gestures['grip2closed']}")
        else:
            print("❌ grip2closed NOT found in gestures")
        
        # Show gesture file locations
        gesture_files = [
            "hardware/roninhand/RHControl/gestures.json",
            "data/gestures/custom_gestures.json",
            "var/custom_gestures.json"
        ]
        
        for file_path in gesture_files:
            if os.path.exists(file_path):
                print(f"📁 {file_path}: EXISTS")
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if 'gestures' in data:
                            grip_gestures = [g for g in data['gestures'].keys() if 'grip' in g.lower()]
                            print(f"   📋 Grip gestures in file: {grip_gestures}")
                        else:
                            grip_gestures = [g for g in data.keys() if 'grip' in g.lower()]
                            print(f"   📋 Grip gestures in file: {grip_gestures}")
                except Exception as e:
                    print(f"   ❌ Error reading file: {e}")
            else:
                print(f"📁 {file_path}: NOT FOUND")

def main():
    """Main function - Enhanced with RoninHand simulation startup"""
    import time  # Add missing import
    print("🚀 Starting RoninHand EMG Voice Control System...")
    print("=" * 50)
    
    # Check if simulation is already running
    try:
        import requests
        response = requests.get('http://localhost:8000', timeout=2)
        print("✅ Simulation server already running")
    except:
        print("🔄 Starting simulation server...")
        # Start the simulation server
        import subprocess
        import sys
        from pathlib import Path
        server_path = Path("hardware/roninhand/RHControl/server.py")
        if server_path.exists():
            print(f"📍 Found server at: {server_path.absolute()}")
            # Start server in its own directory so it can find gestures.json
            subprocess.Popen([sys.executable, "server.py"], 
                           cwd=str(server_path.parent))
            print("⏳ Waiting for simulation to start...")
            time.sleep(5)
        else:
            print(f"❌ Simulation server not found at: {server_path.absolute()}")
            print("Continuing with EMG voice control only...")
    
    # Test simulation if available
    try:
        import requests
        response = requests.post('http://localhost:8000/execute', 
                               json={'gesture': 'peace'}, 
                               timeout=1)
        if response.status_code == 200:
            print("✅ Simulation working - showing peace gesture!")
            time.sleep(2)
        else:
            print("❌ Simulation not responding properly")
    except Exception as e:
        print(f"⚠️  Simulation not available: {e}")
        print("Continuing with EMG voice control only...")
    
    print("🎤 Starting EMG-Controlled Voice Assistant...")
    print("=" * 50)
    print("This system records audio when you contract your muscles (EMG trigger)")
    print("Recordings are saved to the tmp/ folder with timestamps")
    print("Voice commands can control the RoninHand if simulation is running")
    print()
    print("🎯 TWO-MODE SYSTEM:")
    print("📝 GESTURE CREATION MODE (default):")
    print("   • EMG triggers activate voice assistant for commands")
    print("   • Wake word 'HeyRonin' activates voice assistant")
    print("   • Create and execute gestures through voice commands")
    print()
    print("🎯 GESTURE EXECUTION MODE:")
    print("   • EMG triggers toggle between grip2open and grip2closed")
    print("   • Default state: grip2open")
    print("   • Each flex switches to opposite grip state")
    print("   • Wake word 'HeyRonin' activates voice assistant for mode switching only")
    print("   • Say 'switch control mode' to return to creation mode")
    print()
    print("🎯 RONINHAND VOICE COMMANDS:")
    print("• 'Make a fist' or 'Close hand'")
    print("• 'Peace sign' or 'Victory'")
    print("• 'Thumbs up'")
    print("• 'Open hand'")
    print("• 'Point' or 'Pointing'")
    print("• 'Wave' or 'Waving'")
    print()
    print("💡 TIP: Contract your muscles to trigger voice recording, then speak your command!")
    print("🔧 IMPROVED NOISE HANDLING: Initial calibration only, no auto-recalibration")
    print("🔄 MODE SWITCHING: Say 'HeyRonin, switch control mode' to toggle between modes")
    print("=" * 50)
    
    # Create tmp directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)
    
    # Add EMG test mode option
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test-emg":
        print("🧪 EMG TEST MODE: Testing EMG detection without voice processing...")
        print("Flex your muscle twice quickly to test double-tap detection")
        print("Press Ctrl+C to exit test mode")
        print("=" * 50)
        
        assistant = SimpleVoiceAssistant()
        try:
            # Simple EMG test loop
            while True:
                raw = assistant.read_emg()
                if raw is not None:
                    rectified, in_burst, trigger = assistant.process_emg(raw)
                    time.sleep(0.005)  # 200Hz sampling
        except KeyboardInterrupt:
            print("\n🧪 Test mode ended")
        finally:
            assistant.running = False
            if assistant.serial and assistant.serial.is_open:
                assistant.serial.close()
    else:
        # Normal voice assistant mode
        assistant = SimpleVoiceAssistant()
        try:
            assistant.run_interactive()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        finally:
            assistant.running = False
            if assistant.serial and assistant.serial.is_open:
                assistant.serial.close()

if __name__ == "__main__":
    main()