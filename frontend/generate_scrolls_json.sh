#!/bin/bash

echo '{ "scrolls": [' > frontend/scrolls.json
first=1

for dir in frontend/scrolls/*; do
    if [ -d "$dir" ]; then
        folder=$(basename "$dir")
        images=()

        # Store find results in a temp file to avoid process substitution issues
        find "$dir/img" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) > temp_images.txt

        while IFS= read -r img; do
            images+=("\"$(basename "$img")\"")
        done < temp_images.txt

        if [ ${#images[@]} -gt 0 ]; then
            if [ $first -eq 0 ]; then echo ',' >> frontend/scrolls.json; fi
            first=0
            echo "  { \"$folder\": { \"img\": [$(IFS=,; echo "${images[*]}")] } }" >> frontend/scrolls.json
        fi
    fi
done

echo ']}' >> frontend/scrolls.json
rm -f temp_images.txt  # Clean up temp file
