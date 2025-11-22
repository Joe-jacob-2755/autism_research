import requests
import os
import time
def download_github_file(raw_url, local_filename=None):
    """
    Downloads a single file from a GitHub repository's raw URL.

    The GitHub raw URL typically looks like:
    https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path/to/file}

    Args:
        raw_url (str): The raw GitHub URL of the file to download.
        local_filename (str, optional): The name to save the file as locally.
                                        If None, the original filename from the URL is used.
    """
    # Defensive check to ensure the URL points to raw content
    if not raw_url.startswith("https://raw.githubusercontent.com/"):
        print("❌ Error: Please ensure the URL is the 'raw' link from GitHub.")
        print("Example: https://raw.githubusercontent.com/username/repo/main/data/data.csv")
        return

    # Determine the local filename
    if local_filename is None:
        # Extract filename from the URL path
        local_filename = raw_url.split('/')[-1]

    print(f"--- Attempting to download: '{local_filename}' ---")
    print(f"Source URL: {raw_url}")

    try:
        # Use a user-agent header to ensure compatibility and politeness
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Stream the request to handle potentially large files
        response = requests.get(raw_url, headers=headers, stream=True)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

        # Write the content to a local file in binary mode ('wb')
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

        print(f"✅ Success: File saved as '{os.path.abspath(local_filename)}'")

    except requests.exceptions.HTTPError as errh:
        print(f"❌ HTTP Error: {errh} (Check if the file path or branch name is correct)")
    except requests.exceptions.ConnectionError as errc:
        print(f"❌ Connection Error: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"❌ Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"❌ An unexpected error occurred: {err}")

# --- Interactive Usage ---
if __name__ == "__main__":
    print("--- GitHub Data Downloader ---")
    print("Please enter the RAW URL of the file you want to download from GitHub.")
    print("Example: https://raw.githubusercontent.com/username/repo/main/data/data.csv")

    # Get the raw URL from the user
    raw_url = input("\nEnter the RAW GitHub URL: ")

    # Prompt for local filename, making it optional (pass None if empty)
    local_target_name_input = input("Enter the local filename to save as (or leave blank to use the original filename): ")

    # Use the input if it's not just whitespace, otherwise use None
    local_target_name = local_target_name_input.strip() if local_target_name_input.strip() else None

    # Perform the download
    download_github_file(raw_url, local_target_name)
