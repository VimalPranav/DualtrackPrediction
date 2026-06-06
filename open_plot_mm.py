import webbrowser
import os

# Path to the interactive HTML plot
html_path = "/home/trizzz/DualTrack/results/visuals/sub050__LH_Par_C_DtP/trajectory_3d_interactive_mm.html"

if os.path.exists(html_path):
    print(f"Opening {html_path} in your default web browser...")
    webbrowser.open('file://' + os.path.realpath(html_path))
else:
    print("Error: The interactive plot could not be found. Please generate it first.")
