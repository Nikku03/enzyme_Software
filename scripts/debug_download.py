"""
Debug script to check downloaded CYP450 files
"""
import os
import zipfile
import urllib.request

# Check file sizes
files = [
    "/content/enzyme_Software/data/figshare_cyp450/CYP3A4_trainingset.csv",
    "/content/enzyme_Software/data/figshare_cyp450/CYP3A4_testingset.csv"
]

for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"{f}: {size} bytes")
        
        # Show first 500 chars
        with open(f, 'r', errors='ignore') as fp:
            content = fp.read(500)
            print(f"Content preview:\n{content}\n")
    else:
        print(f"{f}: NOT FOUND")

# Try alternative: direct Zenodo download
print("\n" + "="*70)
print("Trying Zenodo download (the actual data repository)")
print("="*70)

import urllib.request

# The GitHub repo for this dataset
zenodo_url = "https://zenodo.org/records/13364709/files/CYP450.zip?download=1"

try:
    print(f"Downloading from Zenodo: {zenodo_url}")
    urllib.request.urlretrieve(zenodo_url, "/content/cyp450_data.zip")
    print("Downloaded CYP450.zip")
    
    # Unzip
    import zipfile
    with zipfile.ZipFile("/content/cyp450_data.zip", 'r') as z:
        z.extractall("/content/cyp450_extracted")
        print(f"Extracted to /content/cyp450_extracted")
        
    # List contents
    for root, dirs, files in os.walk("/content/cyp450_extracted"):
        for f in files:
            print(f"  {os.path.join(root, f)}")
            
except Exception as e:
    print(f"Zenodo download failed: {e}")
    
    # Try GitHub
    print("\nTrying GitHub repository...")
    github_url = "https://github.com/cmdm-Lab/CYP450/archive/refs/heads/main.zip"
    
    try:
        urllib.request.urlretrieve(github_url, "/content/cyp450_github.zip")
        print("Downloaded from GitHub")
        
        with zipfile.ZipFile("/content/cyp450_github.zip", 'r') as z:
            z.extractall("/content/cyp450_github")
            print("Extracted")
            
        # List
        for root, dirs, files in os.walk("/content/cyp450_github"):
            level = root.replace("/content/cyp450_github", '').count(os.sep)
            if level < 3:
                for f in files[:10]:
                    print(f"  {os.path.join(root, f)}")
                    
    except Exception as e2:
        print(f"GitHub download also failed: {e2}")
