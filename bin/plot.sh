#!/bin/bash

for file in "bin"/plot*.py; do
  uv run "$file"
done

