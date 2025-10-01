#!/bin/bash

set -e

for file in "$@"; do
    if [[ "$file" != metalign/*.py ]]; then
        continue 
    fi
    module_base=$(echo "$file" | sed -E 's/^metalign\/(.*)\.py$/\1/')
    
    module_name="metalign.${module_base}"
    
    echo "Running metalign module: $module_name for file $file"
    uv run -m "$module_name"
done