# AGI Memory System - MCP Server Implementation Plan

## 1. Overview

This document outlines the plan to implement a self-contained, Dockerized MCP (Model Context Protocol) server for the AGI Memory System. The primary goal is to provide a standardized interface for interacting with the memory system while maintaining backward compatibility with existing clients that directly access the database.

## 2. Architecture

The system will be deployed as a single Docker container containing:

*   **PostgreSQL Database:**  With the existing `schema.sql` and extensions (pgvector, Apache AGE).
*   **MCP Server:** A Python script (`mcp_server.py`) implementing the MCP protocol and handling requests.
*   **API Layer:** A Python module (`api.py`) to encapsulate database interactions, providing a clean separation between the MCP server and the database.

Communication with the system will be via MCP (using JSON-RPC 2.0) over an exposed port.

## 3. Backward Compatibility Strategy

To ensure a smooth transition and avoid breaking existing clients, the following strategy will be employed:

*   **Dual Interface:** The system will support *both* direct database access (for existing clients) *and* the new MCP interface (for new clients). This means any existing code directly interacting with the database will continue to function without modification.
*   **No Schema Changes:** The existing database schema (`schema.sql`) will *not* be modified. The MCP server will interact with the database using the existing schema.
*   **Phased Rollout:** The MCP interface can be rolled out gradually. Existing clients can continue to use direct database access, while new clients can be developed to use the MCP interface. Over time, existing clients can be migrated to the MCP interface.

## 4. Implementation Steps

The following high-level steps will be taken:

1.  **Create `api.py`:** Develop a Python module to encapsulate all database interactions (CRUD operations, graph queries, etc.). This will provide a clean API for the MCP server to use and will make the code more maintainable.
2.  **Create `mcp_server.py`:** Develop the core MCP server logic. This script will:
    *   Implement the MCP protocol (using a JSON-RPC library).
    *   Handle incoming requests.
    *   Use `api.py` to interact with the database.
    *   Send responses.
    *   Handle notifications.
3.  **Create `Dockerfile`:** Create a `Dockerfile` to build the self-contained Docker image. This will include:
    *   Installing PostgreSQL and extensions.
    *   Copying the code (`mcp_server.py`, `api.py`, `schema.sql`, etc.).
    *   Setting up the database.
    *   Exposing the MCP server port.
4.  **Update `docker-compose.yml` (Optional):**  Modify `docker-compose.yml` to simplify building and running the container (although a single `Dockerfile` is sufficient).
5.  **Update Tests:**
    *   Adapt existing tests in `test.py` to work with the API layer (`api.py`).
    *   Add new tests specifically for the MCP server (`mcp_server.py`).
6.  **Documentation:** Update `README.md` and `README-agi.md` to describe the new MCP server, its interface, and the deployment process.

## 5. Detailed Steps and File References

The following steps provide more detail and link to the specific files involved. These steps correspond to the tickets in the appendix (`implementation_plan_appendix.md`).

1.  **Create `api.py` and Unit Tests:**
    *   Create the file `agi_memory/api.py`.
    *   Implement functions for all database interactions (see tickets AGI-MEM-1 to AGI-MEM-7 in the appendix).
    *   Create the file `agi_memory/test_api.py` and add corresponding unit tests.

2.  **Create MCP Server Skeleton (`mcp_server.py`) and Unit Tests:**
    *   Create the file `agi_memory/mcp_server.py`.
    *   Set up basic JSON-RPC server structure (see ticket AGI-MEM-8).
    *   Create/Update unit tests in `agi_memory/test_mcp.py` (or `agi_memory/test.py`)

3.  **Implement MCP Request Handlers and Unit Tests:**
    *   Implement handlers for each MCP request type (see tickets AGI-MEM-9 to AGI-MEM-15).
    *   These handlers will use `api.py` to interact with the database.
    *   Create/Update unit tests in `agi_memory/test_mcp.py` (or `agi_memory/test.py`)

4.  **Create `Dockerfile`:**
    *   Create the file `agi_memory/Dockerfile`.
    *   Define the steps to build the Docker image (see ticket AGI-MEM-16).

5.  **Update `docker-compose.yml` (Optional):**
    *   Update `agi_memory/docker-compose.yml` (see ticket AGI-MEM-17).

6.  **Adapt Existing Tests:**
    *   Modify `agi_memory/test.py` to use `api.py` (see ticket AGI-MEM-18).

7.  **Add MCP Server Integration Tests:**
    *   Create new tests (or add to `test.py`) to specifically test the MCP server functionality (see ticket AGI-MEM-19).  These will be *integration* tests.

8.  **Update Documentation:**
    *   Update `agi_memory/README.md` and `agi_memory/README-agi.md` (see ticket AGI-MEM-20).

## 6. Testing

A comprehensive testing strategy is crucial for ensuring the quality and reliability of the AGI Memory System. We will employ the following:

*   **Unit Tests:** Each function in `api.py` and `mcp_server.py` will have corresponding unit tests to verify its behavior in isolation.
*   **Integration Tests:** Integration tests will verify the interactions between the MCP server and the database (using the API layer).
*   **Test-Driven Development (TDD):** We strongly recommend following a Test-Driven Development approach: write tests *before* implementing the functionality. This helps clarify requirements and ensures test coverage from the start.
*   **Pytest:** We will use `pytest` as the testing framework, as it's already used in the existing `test.py`.
*   **Mocking:** For unit tests, we will use mocking (e.g., with `unittest.mock` or `pytest-mock`) to isolate the code being tested, particularly for `mcp_server.py`, avoiding the need for a live database connection for every unit test.
