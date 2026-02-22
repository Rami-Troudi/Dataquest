@echo off
REM MLOps Docker Deployment Script for Windows
REM Quick deployment to Docker

setlocal enabledelayedexpansion

echo ================================
echo MLOps Docker Deployment Script
echo ================================

set ACTION=%1
if "%ACTION%"=="" set ACTION=up
set COMPOSE_FILE=docker-compose.yml

REM Check Docker installation
docker --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Docker is not installed. Please install Docker Desktop.
    exit /b 1
)
echo [SUCCESS] Docker is installed

REM Check Docker Compose installation
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Docker Compose is not installed.
    exit /b 1
)
echo [SUCCESS] Docker Compose is installed

if "%ACTION%"=="up" (
    echo [INFO] Building Docker images...
    docker-compose -f %COMPOSE_FILE% build
    echo [INFO] Starting services...
    docker-compose -f %COMPOSE_FILE% up -d
    echo [SUCCESS] Services started
    timeout /t 5 /nobreak
    echo [INFO] Opening API documentation...
    start http://localhost/docs
) else if "%ACTION%"=="down" (
    echo [INFO] Stopping services...
    docker-compose -f %COMPOSE_FILE% down
    echo [SUCCESS] Services stopped
) else if "%ACTION%"=="build" (
    echo [INFO] Building Docker images...
    docker-compose -f %COMPOSE_FILE% build
    echo [SUCCESS] Build completed
) else if "%ACTION%"=="logs" (
    echo [INFO] Showing logs ^(Ctrl+C to exit^)...
    docker-compose -f %COMPOSE_FILE% logs -f mlops-api
) else if "%ACTION%"=="status" (
    echo [INFO] Showing service status...
    docker-compose -f %COMPOSE_FILE% ps
) else if "%ACTION%"=="test" (
    echo [INFO] Testing API...
    powershell -Command "Invoke-RestMethod -Uri 'http://localhost/health' -Method Get | ConvertTo-Json"
) else if "%ACTION%"=="clean" (
    echo [INFO] Cleaning up Docker resources...
    docker-compose -f %COMPOSE_FILE% down -v
    echo [SUCCESS] Cleanup completed
) else if "%ACTION%"=="help" (
    echo.
    echo Usage: deploy.bat [COMMAND]
    echo.
    echo Commands:
    echo   up        Start services (default)
    echo   down      Stop services
    echo   build     Build Docker images
    echo   logs      Show live logs
    echo   status    Show service status
    echo   test      Test API endpoints
    echo   clean     Clean up all Docker resources
    echo   help      Show this help message
    echo.
) else (
    echo [WARNING] Unknown command: %ACTION%
    call :help
)

endlocal
