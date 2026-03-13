import socket
import getpass
import platform
import numpy as np

def get_system_metadata():
    """Returns a dictionary of environment metadata for the audit trail."""
    return {
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "os": platform.system(),
        "version": "1.0.0" # Hardcoded or pulled from a version file
    }

def format_corners(corners: np.ndarray) -> str:
    """Converts [[x,y], [x,y]...] to a flat string for CSV storage."""
    if corners is None:
        return "N/A"
    # Format as UL(0,0)|UR(100,0)|LR(100,100)|LL(0,100)
    labels = ["UL", "UR", "LR", "LL"]
    pairs = [f"{labels[i]}({int(c[0])},{int(c[1])})" for i, c in enumerate(corners)]
    return "|".join(pairs)