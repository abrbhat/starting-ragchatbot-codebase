@echo off
setlocal enabledelayedexpansion

:: Code formatting script using black

:: Get the project root directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."

:: Check if we're in check mode or format mode
set "CHECK_MODE=false"
if "%1"=="--check" set "CHECK_MODE=true"
if "%1"=="-c" set "CHECK_MODE=true"

echo Running code quality checks...
echo.

if "%CHECK_MODE%"=="true" (
    echo Checking code formatting with black...
    uv run black --check .
    if !errorlevel! equ 0 (
        echo All files are properly formatted!
    ) else (
        echo Some files need formatting. Run 'scripts\format.bat' to fix.
        exit /b 1
    )
) else (
    echo Formatting code with black...
    uv run black .
    echo Code formatting complete!
)
