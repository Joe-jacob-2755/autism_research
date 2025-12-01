
import os
import requests
import zipfile
import shutil
from io import BytesIO
import torch
from torch.utils.data import Dataset, DataLoader

# --- 1. Kaggle Downloader Function (Requires API Key Setup) ---

def download_kaggle_dataset(dataset_id: str, download_path: str):
    """
    Downloads a specified Kaggle dataset using direct API download (ZIP format) 
    and Python's zipfile module.

    NOTE: This requires the user to have Kaggle API credentials configured 
    (KAGGLE_USERNAME and KAGGLE_KEY environment variables are typically needed 
    for the requests library to handle authentication for API endpoints).

    Args:
        dataset_id (str): The ID of the dataset on Kaggle (e.g., 'owner/dataset-name').
        download_path (str): The local folder path where the dataset should be extracted.
    
    Returns:
        bool: True if download/extraction was successful, False otherwise.
    """
    if not dataset_id:
        print("Error: Kaggle Dataset ID cannot be empty.")
        return False

    print(f"\n--- Starting Kaggle Download for '{dataset_id}' ---")
    
    # 2. Define the download URL
    download_url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_id}"
    print(f"Attempting download from API endpoint: {download_url}")

    # 3. Execute the download (Authentication needs to be handled by user environment)
    try:
        # Authentication often works if the environment variables KAGGLE_USERNAME 
        # and KAGGLE_KEY are set, but for simplicity, we use a basic request here.
        response = requests.get(download_url, stream=True)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        # 4. Unzip the content directly from memory
        print("Download successful. Unzipping files...")
        
        # Read content into a BytesIO buffer
        zip_buffer = BytesIO(response.content)
        
        # Extract files
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            zip_ref.extractall(download_path)

        print(f"✅ Kaggle download and extraction successful to {download_path}!")
        return True

    except requests.exceptions.HTTPError as e:
        print(f"\n❌ Download Failed due to HTTP Error: {e}")
        if response.status_code == 401:
            print("Authentication failed. Check your Kaggle API key/credentials.")
        elif response.status_code == 404:
            print(f"Dataset '{dataset_id}' not found.")
        return False
        
    except zipfile.BadZipFile:
        print("\n❌ Extraction Failed: Downloaded content is not a valid ZIP file.")
        print("This may be an HTML error page (e.g., login prompt).")
        return False

    except requests.exceptions.RequestException as e:
        print(f"\n❌ An error occurred during the request: {e}")
        return False
    
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        return False

# --- 2. GitHub/Direct URL Downloader Function ---

def download_from_url(url: str, download_path: str):
    """
    Downloads a file from a direct URL and saves it to the specified path.
    Handles extraction if the file is a ZIP archive.
    
    Args:
        url (str): The direct URL to the data file (e.g., CSV, ZIP, TXT).
        download_path (str): The local folder path where the data should be saved/extracted.
    
    Returns:
        bool: True if download/extraction was successful, False otherwise.
    """
    print(f"\n--- Starting URL Download from: {url} ---")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        file_name = url.split('/')[-1]
        local_file_path = os.path.join(download_path, file_name)

        if file_name.endswith('.zip'):
            # Handle ZIP file extraction
            print("Detected ZIP file. Unzipping contents...")
            zip_buffer = BytesIO(response.content)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            print(f"✅ ZIP file extracted to {download_path}")
            return True
            
        else:
            # Handle direct file save (CSV, TXT, etc.)
            with open(local_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ File saved successfully to {local_file_path}")
            return True

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Download failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        return False

# --- 3. PyTorch Dataset Placeholder ---

class CustomPyTorchDataset(Dataset):
    """
    A minimal PyTorch Dataset class to show how the downloaded data 
    would be loaded and accessed after the download script runs.
    """
    def __init__(self, data_root_dir):
        # In a real application, you would load metadata/file paths here
        self.data_root = data_root_dir
        
        # Example: look for files in the downloaded directory
        self.files = [f for f in os.listdir(data_root_dir) if not f.startswith('.')]
        print(f"\nPyTorch Dataset initialized, found {len(self.files)} items in {data_root_dir}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        
        # In a real application, load and preprocess data (e.g., torch.load, cv2.imread, pd.read_csv)
        sample = torch.tensor([idx, 0.5])  # Dummy tensor
        label = torch.tensor([1])           # Dummy label

        return sample, label

# --- 4. Main Execution Block ---

if __name__ == '__main__':
    # --- Setup ---
    print("\n--- PyTorch Data Downloader ---")
    
    # 1. Get source choice from user
    while True:
        source_choice = input("Choose source (Kaggle/GitHub/URL): ").strip().lower()
        if source_choice in ['kaggle', 'github', 'url']:
            break
        print("Invalid choice. Please enter 'Kaggle' or 'GitHub'/'URL'.")

    # 2. Get destination folder name from user
    data_folder_name = input("Enter the local folder name for the data (e.g., 'my_data'): ").strip()
    
    if not data_folder_name:
        print("Folder name cannot be empty. Exiting.")
        exit()
        
    # Ensure the directory is created
    try:
        os.makedirs(data_folder_name, exist_ok=True)
        print(f"Target folder ensured: ./{data_folder_name}")
    except OSError as e:
        print(f"Error creating directory {data_folder_name}: {e}")
        exit()

    success = False
    
    # --- Download Logic ---
    if source_choice == 'kaggle':
        dataset_id = input("Enter the Kaggle Dataset ID (e.g., owner/dataset-name): ").strip()
        success = download_kaggle_dataset(dataset_id, data_folder_name)
        
    elif source_choice in ['github', 'url']:
        # Note: Use a direct link to the raw file or a ZIP/TAR archive on GitHub
        url = input("Enter the direct GitHub/URL link to the file or ZIP archive: ").strip()
        success = download_from_url(url, data_folder_name)

    # --- PyTorch Data Integration Example ---
    if success:
        print("\n--- Data Pipeline Integration ---")
        try:
            # Instantiate the dataset using the newly downloaded data path
            dataset = CustomPyTorchDataset(data_folder_name)
            
            # Create a DataLoader
            # NOTE: Batch size is set small for the example
            data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
            
            print(f"PyTorch DataLoader created with batch size 4.")
            
            # Show a sample batch
            first_batch = next(iter(data_loader))
            print("\nExample batch from DataLoader:")
            print(f"Sample data shape: {first_batch[0].shape}")
            print(f"Sample label shape: {first_batch[1].shape}")
            print("\nSetup complete. You can now use the 'dataset' object in your training loop.")
            
        except Exception as e:
            print(f"Error during PyTorch setup: {e}")
    else:
        print("\nDownload failed. Cannot proceed with PyTorch integration.")
