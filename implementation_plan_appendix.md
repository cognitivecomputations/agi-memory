# Implementation Plan Appendix: Ticket Plan

This appendix provides a detailed ticket plan for implementing the AGI Memory System MCP Server. Each ticket is designed for a junior developer and is estimated as a 1-point task on the Fibonacci scale.

**Ticket ID:** AGI-MEM-1
**Title:** Create `api.py` - Database Connection and Setup
**Description:** Create the `api.py` file and implement the basic database connection setup using `asyncpg`. Include a function to initialize the connection pool.
**Files to Create:** `agi_memory/api.py`
**Files to Update:** None
**Files for Context:** `agi_memory/test.py`, `agi_memory/schema.sql`
**Tests to Create/Update:** `agi_memory/test_api.py` (create this file and add tests for `init_db` and `get_db`)
**Estimate:** 1
**Dependencies:** None

**Ticket ID:** AGI-MEM-2
**Title:** Implement `api.py` - Create Memory Function
**Description:** Implement a function in `api.py` to create a new memory in the database. This function should take the memory type, content, and embedding as input and insert the data into the `memories` table and the appropriate type-specific table (e.g., `episodic_memories`).
**Files to Create:** None
**Files to Update:** `agi_memory/api.py`
**Files for Context:** `agi_memory/schema.sql`, `agi_memory/test.py`
**Tests to Create/Update:** `agi_memory/test_api.py` (add tests for the `create_memory` function)
**Estimate:** 1
**Dependencies:** AGI-MEM-1

**Ticket ID:** AGI-MEM-3
**Title:** Implement `api.py` - Get Memory Function
**Description:** Implement a function in `api.py` to retrieve a memory from the database by its ID. This function should return the memory data, including data from both the `memories` table and the relevant type-specific table.
**Files to Create:** None
**Files to Update:** `agi_memory/api.py`
**Files for Context:** `agi_memory/schema.sql`, `agi_memory/test.py`
**Tests to Create/Update:** `agi_memory/test_api.py` (add tests for the `get_memory` function)
**Estimate:** 1
**Dependencies:** AGI-MEM-1

**Ticket ID:** AGI-MEM-4
**Title:** Implement `api.py` - Update Memory Function
**Description:** Implement a function in `api.py` to update an existing memory in the database. This function should allow updating the content, embedding, and type-specific data.
**Files to Create:** None
**Files to Update:** `agi_memory/api.py`
**Files for Context:** `agi_memory/schema.sql`, `agi_memory/test.py`
**Tests to Create/Update:** `agi_memory/test_api.py` (add tests for the `update_memory` function)
**Estimate:** 1
**Dependencies:** AGI-MEM-1

**Ticket ID:** AGI-MEM-5
**Title:** Implement `api.py` - Delete Memory Function
**Description:** Implement a function in `api.py` to delete a memory from the database by its ID. This should delete the record from the `memories` table and the corresponding type-specific table.
**Files to Create:** None
**Files to Update:** `agi_memory/api.py`
**Files for Context:** `agi_memory/schema.sql`, `agi_memory/test.py`
**Tests to Create/Update:** `agi_memory/test_api.py` (add tests for the `delete_memory` function)
**Estimate:** 1
**Dependencies:** AGI-MEM-1

**Ticket ID:** AGI-MEM-6
**Title:** Implement `api.py` - Vector Search Function
**Description:** Implement a function in `api.py` to perform vector similarity searches using the `embedding` column. This function should take a query embedding and a threshold as input and return memories within the specified similarity threshold.
**Files to Create:** None
**Files to Update:** `agi_memory/api.py`
**Files for Context:** `agi_memory/schema.sql`, `agi_memory/test.py`
**Tests to Create/Update:** `agi_memory/test_api.py` (add tests for the `vector_search` function)
**Estimate:** 1
**Dependencies:** AGI-MEM-1

**Ticket ID:** AGI-MEM-7
**Title:** Implement `api.py` - Graph Query Function
**Description:** Implement a function in `api.py` to execute Cypher queries against the Apache AGE graph database. This function should take a Cypher query string as input and return the results.
**Files to Create:** None
**Files to Update:** `agi_memory/api.py`
**Files for Context:** `agi_memory/schema.sql`, `agi_memory/test.py`, `agi_memory/README.md`
**Tests to Create/Update:** `agi_memory/test_api.py` (add tests for the `graph_query` function)
**Estimate:** 1
**Dependencies:** AGI-MEM-1

**Ticket ID:** AGI-MEM-8
**Title:** Create `mcp_server.py` - Basic Server Setup
**Description:** Create the `mcp_server.py` file and set up the basic structure for a JSON-RPC server using a suitable library (e.g., `jsonrpcserver` or `aiohttp-json-rpc`).  This includes setting up the server to listen for connections and defining a basic "ping" method for testing.
**Files to Create:** `agi_memory/mcp_server.py`
**Files to Update:** None
**Files for Context:** `agi_memory/reference/mcp_schema.json`
**Tests to Create/Update:** `agi_memory/test_mcp.py` (create this file and add a test for basic server connection)
**Estimate:** 1
**Dependencies:** None

**Ticket ID:** AGI-MEM-9
**Title:** Implement `mcp_server.py` - `memory/create` Handler
**Description:** Implement the request handler for the `memory/create` method in `mcp_server.py`. This handler should use the `api.py` function to create a new memory.
**Files to Create:** None
**Files to Update:** `agi_memory/mcp_server.py`
**Files for Context:** `agi_memory/reference/mcp_schema.json`, `agi_memory/api.py`
**Tests to Create/Update:** `agi_memory/test_mcp.py` (add tests for the `memory/create` handler)
**Estimate:** 1
**Dependencies:** AGI-MEM-2, AGI-MEM-8

**Ticket ID:** AGI-MEM-10
**Title:** Implement `mcp_server.py` - `memory/query` Handler (by ID)
**Description:** Implement the request handler for the `memory/query` method (retrieving by ID) in `mcp_server.py`. This handler should use the `api.py` function to retrieve a memory by its ID.
**Files to Create:** None
**Files to Update:** `agi_memory/mcp_server.py`
**Files for Context:** `agi_memory/reference/mcp_schema.json`, `agi_memory/api.py`
**Tests to Create/Update:** `agi_memory/test_mcp.py` (add tests for the `memory/query` handler - by ID)
**Estimate:** 1
**Dependencies:** AGI-MEM-3, AGI-MEM-8

**Ticket ID:** AGI-MEM-11
**Title:** Implement `mcp_server.py` - `memory/query` Handler (vector search)
**Description:** Implement the request handler for the `memory/query` method (vector search) in `mcp_server.py`. This handler should use the `api.py` function to perform vector similarity searches.
**Files to Create:** None
**Files to Update:** `agi_memory/mcp_server.py`
**Files for Context:** `agi_memory/reference/mcp_schema.json`, `agi_memory/api.py`
**Tests to Create/Update:** `agi_memory/test_mcp.py` (add tests for the `memory/query` handler - vector search)
**Estimate:** 1
**Dependencies:** AGI-MEM-6, AGI-MEM-8

**Ticket ID:** AGI-MEM-12
**Title:** Implement `mcp_server.py` - `memory/update` Handler
**Description:** Implement the request handler for the `memory/update` method in `mcp_server.py`. This handler should use the `api.py` function to update an existing memory.
**Files to Create:** None
**Files to Update:** `agi_memory/mcp_server.py`
**Files for Context:** `agi_memory/reference/mcp_schema.json`, `agi_memory/api.py`
**Tests to Create/Update:** `agi_memory/test_mcp.py` (add tests for the `memory/update` handler)
**Estimate:** 1
**Dependencies:** AGI-MEM-4, AGI-MEM-8

**Ticket ID:** AGI-MEM-13
**Title:** Implement `mcp_server.py` - `memory/delete` Handler
**Description:** Implement the request handler for the `memory/delete` method in `mcp_server.py`. This handler should use the `api.py` function to delete a memory.
**Files to Create:** None
**Files to Update:** `agi_memory/mcp_server.py`
**Files for Context:** `agi_memory/reference/mcp_schema.json`, `agi_memory/api.py`
**Tests to Create/Update:** `agi_memory/test_mcp.py` (add tests for the `memory/delete` handler)
**Estimate:** 1
**Dependencies:** AGI-MEM-5, AGI-MEM-8

**Ticket ID:** AGI-MEM-14
**Title:** Implement `mcp_server.py` - `graph/query` Handler
**Description:** Implement the request handler for the `graph/query` method in `mcp_server.py`. This handler should use the `api.py` function to execute Cypher queries.
**Files to Create:** None
**Files to Update:** `agi_memory/mcp_server.py`
**Files for Context:** `agi_memory/reference/mcp_schema.json`, `agi_memory/api.py`
**Tests to Create/Update:** `agi_memory/test_mcp.py` (add tests for the `graph/query` handler)
**Estimate:** 1
**Dependencies:** AGI-MEM-7, AGI-MEM-8

**Ticket ID:** AGI-MEM-15
**Title:** Implement `mcp_server.py` - Initialization and Capabilities
**Description:** Implement the `initialize` request handler in `mcp_server.py`. This handler should return the server's capabilities (e.g., supported methods).
**Files to Create:** None
**Files to Update:** `agi_memory/mcp_server.py`
**Files for Context:** `agi_memory/reference/mcp_schema.json`
**Estimate:** 1
**Dependencies:** AGI-MEM-8

**Ticket ID:** AGI-MEM-16
**Title:** Create `Dockerfile`
**Description:** Create a `Dockerfile` to build the self-contained Docker image for the AGI Memory System. This file should include instructions for installing PostgreSQL, the required extensions, the Python dependencies, and the AGI Memory System code. It should also set up the database schema and expose the necessary port.
**Files to Create:** `agi_memory/Dockerfile`
**Files to Update:** None
**Files for Context:** `agi_memory/schema.sql`, `agi_memory/requirements.txt`, `agi_memory/implementation_plan.md`
**Estimate:** 1
**Dependencies:** None

**Ticket ID:** AGI-MEM-17
**Title:** Update `docker-compose.yml` (Optional)
**Description:** Update the `docker-compose.yml` file to simplify building and running the container. This step is optional if you are using the `Dockerfile` directly.
**Files to Create:** None
**Files to Update:** `agi_memory/docker-compose.yml`
**Files for Context:** `agi_memory/Dockerfile`
**Estimate:** 1
**Dependencies:** AGI-MEM-16

**Ticket ID:** AGI-MEM-18
**Title:** Adapt Existing Tests
**Description:** Modify the existing tests in `test.py` to use the `api.py` module for database interactions instead of direct database connections.
**Files to Create:** None
**Files to Update:** `agi_memory/test.py`
**Files for Context:** `agi_memory/api.py`, `agi_memory/test.py`
**Estimate:** 1
**Dependencies:** AGI-MEM-1 to AGI-MEM-7

**Ticket ID:** AGI-MEM-19
**Title:** Add MCP Server Tests
**Description:** Add new tests (either to `test.py` or a new test file) to specifically test the MCP server functionality. This should include sending MCP requests and verifying the responses.
**Files to Create:** Potentially a new test file (e.g., `agi_memory/test_mcp.py`)
**Files to Update:** `agi_memory/test.py` (if adding to existing file)
**Files for Context:** `agi_memory/mcp_server.py`, `agi_memory/reference/mcp_schema.json`
**Estimate:** 1
**Dependencies:** AGI-MEM-8, AGI-MEM-9 to AGI-MEM-15

**Ticket ID:** AGI-MEM-20
**Title:** Update Documentation
**Description:** Update the `README.md` and `README-agi.md` files to describe the new MCP server, its interface, how to deploy it using Docker, and the backward compatibility strategy.
**Files to Create:** None
**Files to Update:** `agi_memory/README.md`, `agi_memory/README-agi.md`
**Files for Context:** `agi_memory/implementation_plan.md`, `agi_memory/mcp_server.py`, `agi_memory/Dockerfile`
**Estimate:** 1
**Dependencies:** All other tickets