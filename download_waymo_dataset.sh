#!/bin/bash
# download_waymo.sh
# Script to download the full Waymo Open Motion Dataset (uncompressed scenario)

# Exit immediately if a command exits with a non-zero status
set -e

# Make sure gsutil is installed
if ! command -v gsutil &> /dev/null
then
    echo "Error: gsutil could not be found. Please install Google Cloud SDK."
    exit 1
fi

# Run the download command
echo "Starting download of Waymo Open Motion Dataset..."
gsutil -m cp -r "gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario" .

echo "Download completed successfully."