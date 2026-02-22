#!/bin/bash
# MLOps Deployment Script
# Quick deployment to Docker

set -e

echo "================================"
echo "MLOps Docker Deployment Script"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ACTION=${1:-"up"}
COMPOSE_FILE="docker-compose.yml"

# Functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check Docker installation
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed"
}

# Check Docker Compose installation
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_warning "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose is installed"
}

# Build images
build_images() {
    print_info "Building Docker images..."
    docker-compose -f $COMPOSE_FILE build
    print_success "Images built successfully"
}

# Start services
start_services() {
    print_info "Starting services..."
    docker-compose -f $COMPOSE_FILE up -d
    print_success "Services started"
    
    # Wait for services to be ready
    print_info "Waiting for services to be ready..."
    sleep 5
    
    # Check health
    if curl -s http://localhost/health > /dev/null 2>&1; then
        print_success "API is healthy and ready"
    else
        print_warning "API is not responding yet, it may still be starting up"
    fi
}

# Stop services
stop_services() {
    print_info "Stopping services..."
    docker-compose -f $COMPOSE_FILE down
    print_success "Services stopped"
}

# Show status
show_status() {
    print_info "Showing service status..."
    docker-compose -f $COMPOSE_FILE ps
}

# Show logs
show_logs() {
    print_info "Showing logs (Ctrl+C to exit)..."
    docker-compose -f $COMPOSE_FILE logs -f mlops-api
}

# Clean up
cleanup() {
    print_info "Cleaning up Docker resources..."
    docker-compose -f $COMPOSE_FILE down -v
    print_success "Cleanup completed"
}

# Test API
test_api() {
    print_info "Testing API endpoints..."
    
    # Test health
    print_info "Testing /health endpoint..."
    curl -s http://localhost/health | python -m json.tool
    
    # Test docs
    print_info "API docs available at: http://localhost/docs"
}

# Display help
show_help() {
    cat << EOF
Usage: $0 [COMMAND]

Commands:
    up          Start services (default)
    down        Stop services
    build       Build Docker images
    logs        Show live logs
    status      Show service status
    test        Test API endpoints
    clean       Clean up all Docker resources
    help        Show this help message

Examples:
    $0 up           # Start services
    $0 down         # Stop services
    $0 logs         # View logs
    $0 test         # Test API

EOF
}

# Main execution
check_docker
check_docker_compose

case $ACTION in
    up)
        build_images
        start_services
        ;;
    down)
        stop_services
        ;;
    build)
        build_images
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    test)
        test_api
        ;;
    clean)
        cleanup
        ;;
    help)
        show_help
        ;;
    *)
        print_warning "Unknown command: $ACTION"
        show_help
        exit 1
        ;;
esac

print_success "Operation completed successfully!"
