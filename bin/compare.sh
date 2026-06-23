#!/bin/bash

for file in "bin"/compare*.py; do
  uv run "$file"
done
