#!/bin/bash

# Database management script for AGI Memory System
# Usage: ./db-manage.sh [command]

set -e  # Exit on any error

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default values
POSTGRES_USER=${POSTGRES_USER:-postgres}
POSTGRES_DB=${POSTGRES_DB:-memory_db}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-password}

show_help() {
    echo "AGI Memory Database Management Script"
    echo ""
    echo "Usage: ./db-manage.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start           Start the database container"
    echo "  stop            Stop the database container"
    echo "  restart         Restart the database container"
    echo "  reset           Delete and recreate the database (DESTRUCTIVE)"
    echo "  status          Show database container status"
    echo "  logs            Show database logs"
    echo "  wait            Wait for database to be ready"
    echo "  shell           Open psql shell to database"
    echo "  backup          Create a backup of the database"
    echo "  test            Run database tests"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./db-manage.sh reset     # Delete and recreate database"
    echo "  ./db-manage.sh start     # Start database and wait for ready"
    echo "  ./db-manage.sh shell     # Open database shell"
}

wait_for_db() {
    echo "Waiting for database to be ready..."
    
    # Wait for container to be running
    until docker compose ps db | grep -q "Up"; do
        echo "Container is not running yet, waiting..."
        sleep 2
    done
    
    # Wait for PostgreSQL to be ready
    until docker compose exec db pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" > /dev/null 2>&1; do
        echo "Database is not ready yet, waiting..."
        sleep 2
    done
    
    echo "Database is ready!"
    
    # Test connection
    if docker compose exec db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1;" > /dev/null 2>&1; then
        echo "Database connection verified!"
        return 0
    else
        echo "Database connection test failed"
        return 1
    fi
}

start_db() {
    echo "Starting database..."
    docker compose up -d db
    wait_for_db
    echo "Database is fully operational!"
}

stop_db() {
    echo "Stopping database..."
    docker compose stop db
    echo "Database stopped."
}

restart_db() {
    echo "Restarting database..."
    docker compose restart db
    wait_for_db
    echo "Database restarted and ready!"
}

reset_db() {
    echo "⚠️  WARNING: This will DELETE ALL DATA in the database!"
    echo "Are you sure you want to continue? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Resetting database..."
        
        # Stop and remove containers and volumes
        docker compose down -v
        
        # Remove any orphaned containers
        docker compose rm -f
        
        # Start fresh
        echo "Starting fresh database..."
        docker compose up -d db
        
        # Wait for it to be ready
        wait_for_db
        
        echo "✅ Database has been reset and reinitialized!"
        echo "Schema has been automatically loaded from schema.sql"
    else
        echo "Database reset cancelled."
    fi
}

show_status() {
    echo "Database container status:"
    docker compose ps db
    echo ""
    
    if docker compose ps db | grep -q "Up"; then
        echo "Database health check:"
        if docker compose exec db pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" > /dev/null 2>&1; then
            echo "✅ Database is healthy and accepting connections"
        else
            echo "❌ Database container is up but not accepting connections"
        fi
    else
        echo "❌ Database container is not running"
    fi
}

show_logs() {
    echo "Database logs (last 50 lines):"
    docker compose logs --tail=50 db
}

open_shell() {
    echo "Opening database shell..."
    echo "Type \\q to exit"
    docker compose exec db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"
}

backup_db() {
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_file="backup_${timestamp}.sql"
    
    echo "Creating backup: $backup_file"
    docker compose exec db pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" > "$backup_file"
    
    if [ $? -eq 0 ]; then
        echo "✅ Backup created successfully: $backup_file"
    else
        echo "❌ Backup failed"
        return 1
    fi
}

run_tests() {
    echo "Running database tests..."
    if [ -f "test.py" ]; then
        python test.py
    else
        echo "No test.py file found"
        return 1
    fi
}

# Main command handling
case "${1:-help}" in
    start)
        start_db
        ;;
    stop)
        stop_db
        ;;
    restart)
        restart_db
        ;;
    reset)
        reset_db
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    wait)
        wait_for_db
        ;;
    shell)
        open_shell
        ;;
    backup)
        backup_db
        ;;
    test)
        run_tests
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
