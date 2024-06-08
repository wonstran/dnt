import requests

def download_file(url, destination, verbose=True):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check if the request was successful
    
    with open(destination, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    
    if verbose:
        print(f"Downloaded {url} to {destination}")