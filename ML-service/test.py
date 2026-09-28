import kagglehub

# Download latest version
path = kagglehub.dataset_download("azathoth42/myanimelist")

print("Path to dataset files:", path)