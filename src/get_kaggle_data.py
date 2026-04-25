import kagglehub

# Download latest version
path = kagglehub.competition_download('kaggle-pog-series-s01e03')

print("Path to competition files:", path)