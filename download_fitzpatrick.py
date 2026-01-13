import pandas as pd
import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from config import Config

def download_image(url, save_path, retries=3):
    """
    Downloads an image from a URL to a specific path with retry logic.
    """
    if os.path.exists(save_path):
        # Check if file is valid (not empty)
        if os.path.getsize(save_path) > 0:
            return "skipped" 
    
    # Headers to mimic a browser to avoid 403 Forbidden on some servers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return "success"
        except Exception as e:
            if attempt == retries - 1:
                return f"failed: {str(e)}"
            # Exponential backoff
            time.sleep(1 * (attempt + 1))

def main():
    # 1. Determine Data Path
    data_path = Config.data_path
    # If configured path doesn't exist (e.g. running locally vs cloud), fallback to current directory
    if not os.path.exists(data_path):
        print(f"Configured data_path '{data_path}' does not exist. Using current directory '.'")
        data_path = '.'
    
    # 2. Setup Paths
    csv_path = os.path.join(data_path, 'fitzpatrick17k_metadata.csv')
    images_dir = os.path.join(data_path, 'images')
    
    # 3. Check Metadata
    if not os.path.exists(csv_path):
        # Try looking in current directory if not found in data_path
        if os.path.exists('fitzpatrick17k_metadata.csv'):
            csv_path = 'fitzpatrick17k_metadata.csv'
        else:
            print(f"Error: Metadata file not found at {csv_path}")
            print("Please ensure 'fitzpatrick17k_metadata.csv' is in the data directory.")
            return

    print(f"Reading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 4. Create Images Directory
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"Created images directory: {images_dir}")
    else:
        print(f"Images directory exists: {images_dir}")
    
    print(f"Found {len(df)} images in metadata.")
    
    # 5. Prepare Download Tasks
    tasks = []
    for _, row in df.iterrows():
        url = row['url']
        md5 = row['md5hash']
        
        # Ensure we have a URL
        if pd.isna(url) or url == '':
            continue
            
        # Save as {md5hash}.jpg
        save_path = os.path.join(images_dir, f"{md5}.jpg")
        tasks.append((url, save_path))
    
    print(f"Prepared {len(tasks)} download tasks.")
    
    # 6. Execute Downloads (Parallel)
    # Adjust max_workers based on your internet connection and CPU
    max_workers = 16 
    print(f"Starting download with {max_workers} workers...")
    
    results = {'success': 0, 'skipped': 0, 'failed': 0}
    failed_log = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map future to url for error reporting
        future_to_url = {executor.submit(download_image, url, path): url for url, path in tasks}
        
        # Use tqdm for progress bar
        for future in tqdm(as_completed(future_to_url), total=len(tasks), unit='img'):
            url = future_to_url[future]
            try:
                status = future.result()
                if status == "success":
                    results['success'] += 1
                elif status == "skipped":
                    results['skipped'] += 1
                else:
                    results['failed'] += 1
                    failed_log.append(f"{url} | {status}")
            except Exception as exc:
                results['failed'] += 1
                failed_log.append(f"{url} | Exception: {exc}")

    # 7. Summary
    print("\n" + "="*30)
    print("DOWNLOAD SUMMARY")
    print("="*30)
    print(f"Success: {results['success']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Failed:  {results['failed']}")
    print("="*30)
    
    if failed_log:
        log_file = os.path.join(data_path, 'failed_downloads.txt')
        print(f"\nSaving list of failed downloads to {log_file}...")
        with open(log_file, 'w') as f:
            for line in failed_log:
                f.write(line + "\n")

if __name__ == "__main__":
    main()
