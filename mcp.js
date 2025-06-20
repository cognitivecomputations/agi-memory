#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import pg from 'pg';
const { Pool } = pg;

// Database connection pool
const pool = new Pool({
  user: process.env.POSTGRES_USER || 'postgres',
  host: process.env.POSTGRES_HOST || 'localhost',
  database: process.env.POSTGRES_DB || 'memory_db',
  password: process.env.POSTGRES_PASSWORD || 'password',
  port: parseInt(process.env.POSTGRES_PORT || '5432'),
});

// Memory management class
class MemoryManager {
  
  async createMemory(type, content, embedding, importance = 0.0, metadata = {}) {
    const client = await pool.connect();
    try {
      await client.query('BEGIN');
      
      // Insert base memory
      const memoryResult = await client.query(`
        INSERT INTO memories (type, content, embedding, importance)
        VALUES ($1, $2, $3, $4)
        RETURNING id, created_at
      `, [type, content, `[${embedding.join(',')}]`, importance]);
      
      const memoryId = memoryResult.rows[0].id;
      
      // Insert type-specific details
      switch (type) {
        case 'episodic':
          await client.query(`
            INSERT INTO episodic_memories (memory_id, action_taken, context, result, emotional_valence, event_time)
            VALUES ($1, $2, $3, $4, $5, $6)
          `, [
            memoryId, 
            metadata.action_taken || null,
            metadata.context || null, 
            metadata.result || null,
            metadata.emotional_valence || 0.0,
            metadata.event_time || new Date()
          ]);
          break;
          
        case 'semantic':
          await client.query(`
            INSERT INTO semantic_memories (memory_id, confidence, category, related_concepts)
            VALUES ($1, $2, $3, $4)
          `, [
            memoryId,
            metadata.confidence || 0.8,
            metadata.category || [],
            metadata.related_concepts || []
          ]);
          break;
          
        case 'procedural':
          await client.query(`
            INSERT INTO procedural_memories (memory_id, steps, prerequisites)
            VALUES ($1, $2, $3)
          `, [
            memoryId,
            metadata.steps || {},
            metadata.prerequisites || {}
          ]);
          break;
          
        case 'strategic':
          await client.query(`
            INSERT INTO strategic_memories (memory_id, pattern_description, confidence_score)
            VALUES ($1, $2, $3)
          `, [
            memoryId,
            metadata.pattern_description || content,
            metadata.confidence_score || 0.7
          ]);
          break;
      }
      
      await client.query('COMMIT');
      
      // Auto-assign to clusters
      await this.assignMemoryToClusters(memoryId);
      
      return memoryResult.rows[0];
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  async searchMemoriesBySimilarity(queryEmbedding, limit = 10, threshold = 0.7) {
    const client = await pool.connect();
    try {
      const result = await client.query(`
        SELECT 
          m.id, m.type, m.content, m.importance, m.access_count, m.created_at,
          m.relevance_score,
          1 - (m.embedding <=> $1) as similarity
        FROM memories m
        WHERE m.status = 'active' 
          AND 1 - (m.embedding <=> $1) >= $2
        ORDER BY m.embedding <=> $1
        LIMIT $3
      `, [`[${queryEmbedding.join(',')}]`, threshold, limit]);
      
      return result.rows;
    } finally {
      client.release();
    }
  }

  async getMemoryById(memoryId) {
    const client = await pool.connect();
    try {
      const result = await client.query(`
        SELECT m.*, 
          CASE m.type
            WHEN 'episodic' THEN row_to_json(em.*)
            WHEN 'semantic' THEN row_to_json(sm.*)
            WHEN 'procedural' THEN row_to_json(pm.*)
            WHEN 'strategic' THEN row_to_json(stm.*)
          END as type_specific_data
        FROM memories m
        LEFT JOIN episodic_memories em ON m.id = em.memory_id
        LEFT JOIN semantic_memories sm ON m.id = sm.memory_id  
        LEFT JOIN procedural_memories pm ON m.id = pm.memory_id
        LEFT JOIN strategic_memories stm ON m.id = stm.memory_id
        WHERE m.id = $1
      `, [memoryId]);
      
      return result.rows[0];
    } finally {
      client.release();
    }
  }

  async accessMemory(memoryId) {
    const client = await pool.connect();
    try {
      await client.query(`
        UPDATE memories 
        SET access_count = access_count + 1,
            last_accessed = CURRENT_TIMESTAMP
        WHERE id = $1
      `, [memoryId]);
      
      return await this.getMemoryById(memoryId);
    } finally {
      client.release();
    }
  }

  async createMemoryCluster(name, clusterType, description, keywords = []) {
    const client = await pool.connect();
    try {
      const result = await client.query(`
        INSERT INTO memory_clusters (name, cluster_type, description, keywords)
        VALUES ($1, $2, $3, $4)
        RETURNING id, created_at
      `, [name, clusterType, description, keywords]);
      
      return result.rows[0];
    } finally {
      client.release();
    }
  }

  async getMemoryClusters(limit = 20) {
    const client = await pool.connect();
    try {
      const result = await client.query(`
        SELECT 
          mc.*,
          count(mcm.memory_id) as memory_count,
          array_agg(mcm.memory_id) FILTER (WHERE mcm.memory_id IS NOT NULL) as memory_ids
        FROM memory_clusters mc
        LEFT JOIN memory_cluster_members mcm ON mc.id = mcm.cluster_id
        GROUP BY mc.id
        ORDER BY mc.importance_score DESC, mc.last_activated DESC NULLS LAST
        LIMIT $1
      `, [limit]);
      
      return result.rows;
    } finally {
      client.release();
    }
  }

  async getClusterMemories(clusterId, limit = 10) {
    const client = await pool.connect();
    try {
      const result = await client.query(`
        SELECT 
          m.*,
          mcm.membership_strength
        FROM memories m
        JOIN memory_cluster_members mcm ON m.id = mcm.memory_id
        WHERE mcm.cluster_id = $1 AND m.status = 'active'
        ORDER BY mcm.membership_strength DESC, m.relevance_score DESC
        LIMIT $2
      `, [clusterId, limit]);
      
      return result.rows;
    } finally {
      client.release();
    }
  }

  async assignMemoryToClusters(memoryId) {
    const client = await pool.connect();
    try {
      await client.query('SELECT assign_memory_to_clusters($1)', [memoryId]);
    } finally {
      client.release();
    }
  }

  async activateCluster(clusterId, context = null) {
    const client = await pool.connect();
    try {
      await client.query('BEGIN');
      
      // Update cluster activation
      await client.query(`
        UPDATE memory_clusters 
        SET activation_count = activation_count + 1
        WHERE id = $1
      `, [clusterId]);
      
      // Record activation
      await client.query(`
        INSERT INTO cluster_activation_history (cluster_id, activation_context, activation_strength)
        VALUES ($1, $2, $3)
      `, [clusterId, context, 1.0]);
      
      await client.query('COMMIT');
      
      // Return cluster with recent memories
      return await this.getClusterMemories(clusterId);
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  async searchMemoriesByText(query, limit = 10) {
    const client = await pool.connect();
    try {
      const result = await client.query(`
        SELECT 
          m.*,
          ts_rank(to_tsvector('english', m.content), plainto_tsquery('english', $1)) as text_rank
        FROM memories m
        WHERE m.status = 'active' 
          AND to_tsvector('english', m.content) @@ plainto_tsquery('english', $1)
        ORDER BY text_rank DESC, m.relevance_score DESC
        LIMIT $2
      `, [query, limit]);
      
      return result.rows;
    } finally {
      client.release();
    }
  }

  async getIdentityCore() {
    const client = await pool.connect();
    try {
      const result = await client.query(`
        SELECT 
          im.*,
          array_agg(mc.name) as core_cluster_names
        FROM identity_model im
        LEFT JOIN memory_clusters mc ON mc.id = ANY(im.core_memory_clusters)
        GROUP BY im.id
        ORDER BY im.id DESC
        LIMIT 1
      `);
      
      return result.rows[0];
    } finally {
      client.release();
    }
  }

  async getWorldviewPrimitives() {
    const client = await pool.connect();
    try {
      const result = await client.query(`
        SELECT * FROM worldview_primitives
        ORDER BY confidence DESC, stability_score DESC
      `);
      
      return result.rows;
    } finally {
      client.release();
    }
  }

  async getMemoryHealth() {
    const client = await pool.connect();
    try {
      const result = await client.query(`
        SELECT * FROM memory_health
      `);
      
      return result.rows;
    } finally {
      client.release();
    }
  }

  async getActiveThemes(days = 7) {
    const client = await pool.connect();
    try {
      const result = await client.query(`
        SELECT 
          mc.name as theme,
          mc.emotional_signature,
          mc.keywords,
          count(DISTINCT mch.id) as recent_activations,
          mc.importance_score
        FROM memory_clusters mc
        JOIN cluster_activation_history mch ON mc.id = mch.cluster_id
        WHERE mch.activated_at > CURRENT_TIMESTAMP - INTERVAL '$1 days'
        GROUP BY mc.id, mc.name, mc.emotional_signature, mc.keywords, mc.importance_score
        ORDER BY count(DISTINCT mch.id) DESC, mc.importance_score DESC
      `, [days]);
      
      return result.rows;
    } finally {
      client.release();
    }
  }
}

const memoryManager = new MemoryManager();

// MCP Server setup
const server = new Server({
  name: "memory-server",
  version: "1.0.0",
}, {
  capabilities: {
    tools: {},
  },
});

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "create_memory",
        description: "Create a new memory with optional type-specific metadata",
        inputSchema: {
          type: "object",
          properties: {
            type: {
              type: "string",
              enum: ["episodic", "semantic", "procedural", "strategic"],
              description: "Type of memory to create"
            },
            content: {
              type: "string",
              description: "The main content/text of the memory"
            },
            embedding: {
              type: "array",
              items: { type: "number" },
              description: "Vector embedding for the memory content"
            },
            importance: {
              type: "number",
              description: "Importance score (0.0 to 1.0)",
              default: 0.0
            },
            metadata: {
              type: "object",
              description: "Type-specific metadata (action_taken, context, confidence, etc.)",
              default: {}
            }
          },
          required: ["type", "content", "embedding"]
        }
      },
      {
        name: "search_memories_similarity",
        description: "Search memories by vector similarity",
        inputSchema: {
          type: "object",
          properties: {
            embedding: {
              type: "array",
              items: { type: "number" },
              description: "Query embedding vector"
            },
            limit: {
              type: "integer",
              description: "Maximum number of results",
              default: 10
            },
            threshold: {
              type: "number",
              description: "Minimum similarity threshold",
              default: 0.7
            }
          },
          required: ["embedding"]
        }
      },
      {
        name: "search_memories_text",
        description: "Search memories by text content using full-text search",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Text query to search for"
            },
            limit: {
              type: "integer",
              description: "Maximum number of results",
              default: 10
            }
          },
          required: ["query"]
        }
      },
      {
        name: "get_memory",
        description: "Retrieve a specific memory by ID and mark it as accessed",
        inputSchema: {
          type: "object",
          properties: {
            memory_id: {
              type: "string",
              description: "UUID of the memory to retrieve"
            }
          },
          required: ["memory_id"]
        }
      },
      {
        name: "get_memory_clusters",
        description: "Retrieve memory clusters ordered by importance/activity",
        inputSchema: {
          type: "object",
          properties: {
            limit: {
              type: "integer",
              description: "Maximum number of clusters to return",
              default: 20
            }
          }
        }
      },
      {
        name: "activate_cluster",
        description: "Activate a memory cluster and get its associated memories",
        inputSchema: {
          type: "object",
          properties: {
            cluster_id: {
              type: "string",
              description: "UUID of the cluster to activate"
            },
            context: {
              type: "string",
              description: "Context description for this activation",
              default: null
            }
          },
          required: ["cluster_id"]
        }
      },
      {
        name: "create_memory_cluster",
        description: "Create a new memory cluster",
        inputSchema: {
          type: "object",
          properties: {
            name: {
              type: "string",
              description: "Name of the cluster"
            },
            cluster_type: {
              type: "string",
              enum: ["theme", "emotion", "temporal", "person", "pattern", "mixed"],
              description: "Type of cluster"
            },
            description: {
              type: "string",
              description: "Description of the cluster"
            },
            keywords: {
              type: "array",
              items: { type: "string" },
              description: "Keywords associated with this cluster",
              default: []
            }
          },
          required: ["name", "cluster_type"]
        }
      },
      {
        name: "get_identity_core",
        description: "Retrieve the current identity model and core memory clusters",
        inputSchema: {
          type: "object",
          properties: {}
        }
      },
      {
        name: "get_worldview",
        description: "Retrieve current worldview primitives and beliefs",
        inputSchema: {
          type: "object",
          properties: {}
        }
      },
      {
        name: "get_memory_health",
        description: "Get overall statistics about memory system health",
        inputSchema: {
          type: "object",
          properties: {}
        }
      },
      {
        name: "get_active_themes",
        description: "Get recently activated memory themes and patterns",
        inputSchema: {
          type: "object",
          properties: {
            days: {
              type: "integer",
              description: "Number of days to look back",
              default: 7
            }
          }
        }
      }
    ]
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  if (!args) {
    throw new Error(`No arguments provided for tool: ${name}`);
  }

  try {
    switch (name) {
      case "create_memory":
        const memory = await memoryManager.createMemory(
          args.type,
          args.content,
          args.embedding,
          args.importance || 0.0,
          args.metadata || {}
        );
        return { content: [{ type: "text", text: JSON.stringify(memory, null, 2) }] };

      case "search_memories_similarity":
        const similarMemories = await memoryManager.searchMemoriesBySimilarity(
          args.embedding,
          args.limit || 10,
          args.threshold || 0.7
        );
        return { content: [{ type: "text", text: JSON.stringify(similarMemories, null, 2) }] };

      case "search_memories_text":
        const textResults = await memoryManager.searchMemoriesByText(
          args.query,
          args.limit || 10
        );
        return { content: [{ type: "text", text: JSON.stringify(textResults, null, 2) }] };

      case "get_memory":
        const retrievedMemory = await memoryManager.accessMemory(args.memory_id);
        return { content: [{ type: "text", text: JSON.stringify(retrievedMemory, null, 2) }] };

      case "get_memory_clusters":
        const clusters = await memoryManager.getMemoryClusters(args.limit || 20);
        return { content: [{ type: "text", text: JSON.stringify(clusters, null, 2) }] };

      case "activate_cluster":
        const clusterMemories = await memoryManager.activateCluster(
          args.cluster_id,
          args.context || null
        );
        return { content: [{ type: "text", text: JSON.stringify(clusterMemories, null, 2) }] };

      case "create_memory_cluster":
        const newCluster = await memoryManager.createMemoryCluster(
          args.name,
          args.cluster_type,
          args.description,
          args.keywords || []
        );
        return { content: [{ type: "text", text: JSON.stringify(newCluster, null, 2) }] };

      case "get_identity_core":
        const identity = await memoryManager.getIdentityCore();
        return { content: [{ type: "text", text: JSON.stringify(identity, null, 2) }] };

      case "get_worldview":
        const worldview = await memoryManager.getWorldviewPrimitives();
        return { content: [{ type: "text", text: JSON.stringify(worldview, null, 2) }] };

      case "get_memory_health":
        const health = await memoryManager.getMemoryHealth();
        return { content: [{ type: "text", text: JSON.stringify(health, null, 2) }] };

      case "get_active_themes":
        const themes = await memoryManager.getActiveThemes(args.days || 7);
        return { content: [{ type: "text", text: JSON.stringify(themes, null, 2) }] };

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    return { 
      content: [{ 
        type: "text", 
        text: `Error executing ${name}: ${error.message}` 
      }],
      isError: true
    };
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Memory MCP Server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});
