#!/bin/bash

# Code formatting script using black

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if we're in check mode or format mode
CHECK_MODE=false
if [ "$1" = "--check" ] || [ "$1" = "-c" ]; then
    CHECK_MODE=true
fi

echo -e "${YELLOW}Running code quality checks...${NC}"
echo ""

if [ "$CHECK_MODE" = true ]; then
    echo -e "${YELLOW}Checking code formatting with black...${NC}"
    if uv run black --check .; then
        echo -e "${GREEN}All files are properly formatted!${NC}"
    else
        echo -e "${RED}Some files need formatting. Run './scripts/format.sh' to fix.${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Formatting code with black...${NC}"
    uv run black .
    echo -e "${GREEN}Code formatting complete!${NC}"
fi
