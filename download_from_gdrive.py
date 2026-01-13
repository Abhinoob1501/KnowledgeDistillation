import re
import gdown

# Put your Google Drive link or file ID here
GDRIVE_LINK = "https://drive.google.com/drive/folders/10kXJKlgy2PVnyUQ80FQ7JEqSeqJkVIUp?usp=drive_link"
OUTPUT_FILENAME = "i3_slices.h5"  # Optional: specify output name or set to None

# Extract file ID
match = re.search(r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', GDRIVE_LINK)
file_id = match.group(1) if match else GDRIVE_LINK

# Download
url = f"https://drive.google.com/uc?id={file_id}"
downloaded = gdown.download(url, OUTPUT_FILENAME, quiet=False, fuzzy=True)

if downloaded:
    print(f"✓ Downloaded: {downloaded}")
else:
    print("✗ Failed. Check file permissions.")
