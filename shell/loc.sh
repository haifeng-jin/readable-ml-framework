#!/bin/bash

directory="framework"
python_lines=$(find "$directory" -type f -name "*.py" -print0 | xargs -0 wc -l | tail -n 1 | awk '{print $1}')
cpp_lines=$(find "$directory" -type f \( -name "*.cpp" -o -name "*.h" \) -print0 | xargs -0 wc -l | tail -n 1 | awk '{print $1}')

echo "Lines of Python code in '$directory': ${python_lines:-0}"
echo "Lines of C/C++ code in '$directory': ${cpp_lines:-0}"

exit 0
