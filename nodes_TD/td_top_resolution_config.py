"""
TouchDesigner Resolution Configuration - Merged Version
Combines manual presets with custom input for maximum flexibility
Script TOP node for configuring YOLO resolution
"""

import json
import time
import numpy as np

# Configuration file
CONFIG_FILE = "/tmp/yolo_td_config.json"

# Common resolutions
PRESETS = {
    "640x640": (640, 640, "Square - Best for pose"),
    "1280x720": (1280, 720, "HD 720p"),
    "1920x1080": (1920, 1080, "Full HD 1080p"),
    "960x540": (960, 540, "Quarter HD"),
    "1024x1024": (1024, 1024, "Square - Large"),
    "1280x960": (1280, 960, "4:3 Aspect"),
    "640x480": (640, 480, "VGA"),
}

# Global state
_current_config = None
_last_save_time = 0


def onSetupParameters(scriptOp):
    """Setup resolution configuration UI"""
    print("\n=== YOLO Resolution Configuration (Merged) ===")

    # Create page
    page = scriptOp.appendCustomPage("Resolution Config")

    # Info display
    p = page.appendStr("Info", label="Current Config")
    p.val = "Not saved yet"
    p.readOnly = True

    # Section 1: Quick Presets
    # Use string parameter as visual separator
    p = page.appendStr("Sep1", label="═══ Quick Presets ═══")
    p.val = ""
    p.readOnly = True

    # Create preset buttons with TD-compliant names
    preset_buttons = {
        "Presetsquare": ("640x640", "Square - Best for pose"),
        "Presethd": ("1280x720", "HD 720p"),
        "Presetfhd": ("1920x1080", "Full HD 1080p"),
        "Presetqhd": ("960x540", "Quarter HD"),
        "Presetlgsquare": ("1024x1024", "Square - Large"),
        "Presetfourthree": ("1280x960", "4:3 Aspect"),
        "Presetvga": ("640x480", "VGA"),
    }

    for button_name, (res_key, desc) in preset_buttons.items():
        w, h, _ = PRESETS[res_key]
        p = page.appendPulse(button_name, label=f"{res_key} - {desc}")
        p.help = f"Set resolution to {w}x{h}"

    # Section 2: Custom Resolution
    # Use string parameter as visual separator
    p = page.appendStr("Sep2", label="═══ Custom Resolution ═══")
    p.val = ""
    p.readOnly = True

    # Width parameter
    p = page.appendInt("Reswidth", label="Custom Width")
    p.default = 1280
    p.min = 320
    p.max = 3840
    p.val = 1280
    p.help = "Set custom width (320-3840)"

    # Height parameter
    p = page.appendInt("Resheight", label="Custom Height")
    p.default = 720
    p.min = 240
    p.max = 2160
    p.val = 720
    p.help = "Set custom height (240-2160)"

    # Apply custom button
    p = page.appendPulse("Applycustom", label="Apply Custom Resolution")
    p.help = "Apply the custom width/height values"

    # Section 3: Actions
    # Use string parameter as visual separator
    p = page.appendStr("Sep3", label="═══ Actions ═══")
    p.val = ""
    p.readOnly = True

    # Save button
    p = page.appendPulse("Save", label="💾 Save Configuration")
    p.help = "Save current resolution to config file"

    # Load button
    p = page.appendPulse("Load", label="📂 Load Existing Config")
    p.help = "Load previously saved configuration"

    # Status display
    p = page.appendStr("Status", label="Status")
    p.val = "Ready"
    p.readOnly = True

    print("[OK] Select a preset or set custom resolution, then click Save")
    print(f"   Config file: {CONFIG_FILE}")

    # Try to load existing config
    load_existing_config(scriptOp)

    return


def onPulse(par):
    """Handle button presses"""
    scriptOp = par.owner

    # Handle preset buttons - map button names to resolutions
    preset_mapping = {
        "Presetsquare": "640x640",
        "Presethd": "1280x720",
        "Presetfhd": "1920x1080",
        "Presetqhd": "960x540",
        "Presetlgsquare": "1024x1024",
        "Presetfourthree": "1280x960",
        "Presetvga": "640x480",
    }

    if par.name in preset_mapping:
        preset_key = preset_mapping[par.name]
        if preset_key in PRESETS:
            width, height, desc = PRESETS[preset_key]
            set_resolution(scriptOp, width, height)
            update_status(scriptOp, f"Set to {preset_key} - {desc}")

    # Handle custom resolution
    elif par.name == "Applycustom":
        try:
            width = int(scriptOp.par.Reswidth.eval())
            height = int(scriptOp.par.Resheight.eval())
            set_resolution(scriptOp, width, height)
            update_status(scriptOp, f"Set to custom {width}x{height}")
        except Exception as e:
            update_status(scriptOp, f"Error: {e}", error=True)

    # Handle save
    elif par.name == "Save":
        save_configuration(scriptOp)

    # Handle load
    elif par.name == "Load":
        load_existing_config(scriptOp)


def set_resolution(scriptOp, width, height):
    """Update the resolution parameters"""
    global _current_config

    try:
        # Update custom parameters to reflect current values
        scriptOp.par.Reswidth.val = width
        scriptOp.par.Resheight.val = height

        # Update current config
        _current_config = {"width": width, "height": height, "timestamp": time.time()}

        # Update info display
        scriptOp.par.Info.val = f"{width}x{height} (not saved)"

    except Exception as e:
        print(f"[ERROR] Failed to set resolution: {e}")


def save_configuration(scriptOp):
    """Save current configuration to file"""
    global _current_config, _last_save_time

    if _current_config is None:
        update_status(scriptOp, "No resolution set!", error=True)
        return

    try:
        # Update timestamp
        _current_config["timestamp"] = time.time()

        # Save to file
        with open(CONFIG_FILE, "w") as f:
            json.dump(_current_config, f, indent=2)

        _last_save_time = time.time()

        # Update displays
        width = _current_config["width"]
        height = _current_config["height"]
        scriptOp.par.Info.val = f"{width}x{height} (saved)"
        update_status(scriptOp, f"✓ Saved to {CONFIG_FILE}")

        print(f"\n[OK] Configuration saved: {width}x{height}")
        print(f"   File: {CONFIG_FILE}")
        print("\n[NEXT STEPS]:")
        print("   1. Start YOLO server: python setup_all.py -w {width} -h {height}")
        print("   2. Use td_top_yolo.py to connect")

    except Exception as e:
        update_status(scriptOp, f"Save failed: {e}", error=True)
        print(f"[ERROR] Failed to save: {e}")


def load_existing_config(scriptOp):
    """Load existing configuration if available"""
    global _current_config

    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)

        width = config.get("width", 1280)
        height = config.get("height", 720)
        timestamp = config.get("timestamp", 0)

        # Update parameters
        set_resolution(scriptOp, width, height)

        # Mark as saved
        _current_config = config
        scriptOp.par.Info.val = f"{width}x{height} (loaded)"

        # Show age of config
        age = time.time() - timestamp
        if age < 60:
            age_str = f"{int(age)}s ago"
        elif age < 3600:
            age_str = f"{int(age/60)}m ago"
        else:
            age_str = f"{int(age/3600)}h ago"

        update_status(scriptOp, f"Loaded {width}x{height} (saved {age_str})")

    except FileNotFoundError:
        update_status(scriptOp, "No saved config found")
    except Exception as e:
        update_status(scriptOp, f"Load error: {e}", error=True)


def update_status(scriptOp, message, error=False):
    """Update status display"""
    try:
        scriptOp.par.Status.val = message
        if error:
            print(f"[ERROR] {message}")
        else:
            print(f"[INFO] {message}")
    except:
        print(f"[STATUS] {message}")


def onCook(scriptOp):
    """Required for Script TOP - just pass through or generate info frame"""
    # For a config node, we can just generate a simple info display
    # or pass through input if available

    if scriptOp.inputs and len(scriptOp.inputs) > 0:
        # Pass through input
        input_top = scriptOp.inputs[0].numpyArray(delayed=True, writable=False)
        if input_top is not None:
            scriptOp.copyNumpyArray(input_top)
            return

    # Otherwise create an info display
    global _current_config

    # Create a small info image
    width = 640
    height = 360
    info_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Add some text info (this is just a placeholder - TouchDesigner would need Text TOP for real text)
    if _current_config:
        # Create a simple pattern to indicate config is loaded
        info_frame[10:50, 10:50] = [0, 255, 0]  # Green square = configured
        res_w = _current_config["width"]
        res_h = _current_config["height"]
        # Create resolution indicator bars
        bar_w = min(width - 20, int((res_w / 1920) * (width - 20)))
        bar_h = min(height - 70, int((res_h / 1080) * (height - 70)))
        info_frame[60:65, 10 : 10 + bar_w] = [255, 255, 255]  # Width bar
        info_frame[70 : 70 + bar_h, 10:15] = [255, 255, 255]  # Height bar
    else:
        # Red square = not configured
        info_frame[10:50, 10:50] = [255, 0, 0]

    scriptOp.copyNumpyArray(info_frame)
    return


# Make sure time is imported at module level
if "time" not in globals():
    import time
