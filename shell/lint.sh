isort -c .
flake8 .
black --check .
# Format all .h and .cpp files in the current directory and subdirectories
find . -name "*.h" -o -name "*.cpp" -exec clang-format -n {} \;
