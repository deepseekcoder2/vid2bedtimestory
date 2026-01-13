#!/usr/bin/env python3
import subprocess
import sys
import os
import shutil
import json
from pathlib import Path

def find_gs():
    """Attempts to find the Ghostscript binary in various locations."""
    # 1. Check PATH
    gs_path = shutil.which("gs")
    if gs_path:
        return ["gs"]

    # 2. Check common macOS Homebrew locations
    brew_locations = ["/opt/homebrew/bin/gs", "/usr/local/bin/gs"]
    for loc in brew_locations:
        if os.path.exists(loc):
            return [loc]

    # 3. Check for dedicated conda environment 'pdf-compress'
    conda_path = shutil.which("conda")
    if conda_path:
        try:
            # Check if environment exists
            result = subprocess.run(
                ["conda", "env", "list", "--json"], 
                capture_output=True, text=True, check=True
            )
            envs = json.loads(result.stdout).get("envs", [])
            if any("pdf-compress" in env for env in envs):
                return ["conda", "run", "-n", "pdf-compress", "gs"]
        except:
            pass

    return None

def compress_pdf(input_path, output_path, power=2):
    """
    Compresses a PDF using Ghostscript.
    
    Power levels:
    0: default    (Low compression, high quality)
    1: prepress   (High quality, 300 dpi)
    2: printer    (Good quality, 300 dpi) - RECOMMENDED
    3: ebook      (Medium quality, 150 dpi)
    4: screen     (Low quality, 72 dpi)    - SMALLEST
    """
    
    quality = {
        0: '/default',
        1: '/prepress',
        2: '/printer',
        3: '/ebook',
        4: '/screen'
    }
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    # Find GS binary or conda wrapper
    gs_base_cmd = find_gs()
    if not gs_base_cmd:
        print("\nError: Ghostscript ('gs') not found.")
        print("Installation options:")
        print("  1. brew install ghostscript")
        print("  2. conda create -n pdf-compress -c conda-forge ghostscript")
        return

    gs_command = gs_base_cmd + [
        '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
        f'-dPDFSETTINGS={quality.get(power, "/printer")}',
        '-dNOPAUSE', '-dQUIET', '-dBATCH',
        f'-sOutputFile={output_path}', input_path
    ]
    
    print(f"--- PDF Compression ---")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Setting: {quality.get(power, '/printer')} (Level {power})")
    if len(gs_base_cmd) > 1:
        print(f"Using environment: {' '.join(gs_base_cmd[:4])}")
    
    try:
        subprocess.run(gs_command, check=True)
        
        initial_size = os.path.getsize(input_path)
        final_size = os.path.getsize(output_path)
        
        if final_size == 0:
            print("\nError: Output file is empty. Compression failed.")
            return

        ratio = 1 - (final_size / initial_size)
        
        print(f"\nSuccess!")
        print(f"Initial size: {initial_size / (1024*1024):.2f} MB")
        print(f"Final size:   {final_size / (1024*1024):.2f} MB")
        print(f"Reduction:    {ratio*100:.1f}%")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during compression: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compress_pdf.py <input.pdf> [output.pdf] [power 0-4]")
        print("\nPower levels:")
        print("  0: default  (lowest compression)")
        print("  1: prepress (300dpi)")
        print("  2: printer  (300dpi, recommended)")
        print("  3: ebook    (150dpi)")
        print("  4: screen   (72dpi, smallest)")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    # Handle output path default
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        # If the user just gave a number as the second arg, they meant power
        if output_file.isdigit():
            power_level = int(output_file)
            output_file = str(Path(input_file).with_name("compressed_" + Path(input_file).name))
        else:
            power_level = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    else:
        output_file = str(Path(input_file).with_name("compressed_" + Path(input_file).name))
        power_level = 2
    
    compress_pdf(input_file, output_file, power_level)
