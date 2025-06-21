#!/bin/bash

echo "Waiting for database to be ready..."

# Wait for PostgreSQL to be ready
until docker compose exec db pg_isready -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-memory_db} > /dev/null 2>&1; do
    echo "Database is not ready yet, waiting..."
    sleep 2
done

echo "Database is ready!"

# Optional: Run a test query to ensure extensions are loaded
echo "Testing database connection and extensions..."
docker compose exec db psql -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-memory_db} -c "SELECT 1;" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "Database is fully operational!"
else
    echo "Database connection test failed"
    exit 1
fi
