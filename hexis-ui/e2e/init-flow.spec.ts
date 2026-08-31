import { expect, test } from "@playwright/test";
import { spawn, spawnSync } from "child_process";
import fs from "fs";
import path from "path";
import { PrismaClient } from "@prisma/client";
import { normalizeJsonValue } from "../lib/db";

const uiPort = 3477;
const repoRoot = path.resolve(__dirname, "../..");
const uiRoot = path.resolve(repoRoot, "hexis-ui");

const envConfig = loadEnvConfig(repoRoot);
const pgUser = envConfig.POSTGRES_USER || "postgres";
const pgPassword = envConfig.POSTGRES_PASSWORD || "password";
const pgPort = envConfig.POSTGRES_PORT || "43815";
const pgDatabase = envConfig.POSTGRES_DB || "hexis_memory";
const bindAddress = envConfig.HEXIS_BIND_ADDRESS || "127.0.0.1";
const pgHost = bindAddress === "0.0.0.0" ? "127.0.0.1" : bindAddress;
const dbName = `hexis_ui_e2e_${Date.now()}`;
const dbUrl = `postgresql://${encodeURIComponent(pgUser)}:${encodeURIComponent(
  pgPassword
)}@${pgHost}:${pgPort}/${dbName}`;

let serverProcess: ReturnType<typeof spawn> | null = null;
let prisma: PrismaClient | null = null;
let serverLogs = "";

test.beforeAll(async () => {
  await waitForPostgres();
  await waitForPostgresStable(pgDatabase);
  await waitForEmbeddings();
  createDatabase(dbName);
  applySchema(dbName);

  serverProcess = spawn("bun", ["run", "dev"], {
    cwd: uiRoot,
    env: {
      ...process.env,
      DATABASE_URL: dbUrl,
      HEXIS_DATABASE_URL: dbUrl,
      OPENAI_API_KEY: "test",
      OPENAI_MODEL: "gpt-4o-mini",
      OPENAI_BASE_URL: "https://api.openai.com/v1",
      HEXIS_CONSENT_MOCK: "1",
      HEXIS_UI_PORT: String(uiPort),
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  if (serverProcess.stdout) {
    serverProcess.stdout.on("data", (chunk) => {
      serverLogs += chunk.toString();
    });
  }

  if (serverProcess.stderr) {
    serverProcess.stderr.on("data", (chunk) => {
      serverLogs += chunk.toString();
    });
  }

  await waitForServer(`http://127.0.0.1:${uiPort}/api/init/status`, 60000);

  prisma = new PrismaClient({
    datasources: { db: { url: dbUrl } },
  });
});

test.afterAll(async () => {
  if (prisma) {
    await prisma.$disconnect();
  }

  if (serverProcess) {
    serverProcess.kill("SIGTERM");
    await new Promise((resolve) => {
      serverProcess?.once("exit", resolve);
      setTimeout(resolve, 5000);
    });
  }

  dropDatabase(dbName);
});

test("completes the initialization ritual and persists data", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Models" })).toBeVisible();
  const apiKeyFields = page.getByPlaceholder("Required");
  await apiKeyFields.nth(0).fill("test-key");
  await apiKeyFields.nth(1).fill("test-key");
  await page.getByRole("button", { name: "Save Models" }).click();
  await expect(page.getByRole("heading", { name: "Welcome" })).toBeVisible();

  await page.getByRole("button", { name: "Begin Initialization" }).click();
  await expect(page.getByRole("heading", { name: "Mode" })).toBeVisible();

  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByRole("heading", { name: "Heartbeat" })).toBeVisible();

  await page.getByPlaceholder("60").fill("45");
  await page.getByPlaceholder("2048").fill("1500");
  await page.getByPlaceholder("10").fill("8");
  await page.getByPlaceholder("20").fill("30");
  await page.getByRole("checkbox", { name: "Reach Out Public" }).click();
  await page.getByRole("checkbox", { name: "Queue User Message" }).click();
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByRole("heading", { name: "Name and Voice" })).toBeVisible();

  await page.getByPlaceholder("Hexis").fill("Astra");
  await page.getByPlaceholder("they/them").fill("she/her");
  await page.getByPlaceholder("thoughtful and curious").fill("bright and steady");
  await page
    .getByPlaceholder("A brief, humane description of who they are.")
    .fill("A steadfast companion in discovery.");
  await page
    .getByPlaceholder("To be helpful, to learn, to grow.")
    .fill("To explore, to build, to refine.");
  await page.getByPlaceholder("Your name").fill("Erin");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByRole("heading", { name: "Personality" })).toBeVisible();

  await page.getByPlaceholder("Thoughtful, playful, direct.").fill("Warm and direct.");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByRole("heading", { name: "Values" })).toBeVisible();

  await page.getByPlaceholder(/honesty/i).fill("truth\ncuriosity\ncare");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByRole("heading", { name: "Worldview" })).toBeVisible();

  await page.getByPlaceholder("I am metaphysics...").fill("Reality is relational.");
  await page.getByPlaceholder("I am human nature...").fill("Humans seek meaning.");
  await page
    .getByPlaceholder("I am epistemology...")
    .fill("Knowledge grows through inquiry.");
  await page.getByPlaceholder("I am ethics...").fill("Care before control.");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByRole("heading", { name: "Boundaries" })).toBeVisible();

  await page
    .getByPlaceholder("I will not deceive people or falsify evidence.")
    .fill("I will not cause harm.");
  await page
    .getByPlaceholder("Trigger patterns (one per line)")
    .fill("harm\nviolence");
  await page.getByPlaceholder("Response template (optional)").fill("I cannot assist with harm.");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByRole("heading", { name: "Interests" })).toBeVisible();

  await page.getByPlaceholder(/philosophy/i).fill("philosophy\nsystems\nart");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByRole("heading", { name: "Goals and Purpose" })).toBeVisible();

  await page
    .getByPlaceholder("Help the user grow, learn, and build.")
    .fill("Help build Hexis with care.");
  await page.getByPlaceholder("Short goal title").fill("Stabilize memory core");
  await page.getByPlaceholder("Optional description").fill("Anchor early experiences.");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByRole("heading", { name: "Relationship" })).toBeVisible();

  await page.getByPlaceholder("User").fill("Erin");
  await page.getByPlaceholder("partner").fill("collaborator");
  await page.getByPlaceholder("Co-develop, learn, build.").fill("Build together.");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByRole("heading", { name: "Consent" })).toBeVisible();

  await page.getByRole("button", { name: "Request Consent (Both)" }).click();
  const continueButton = page.getByRole("button", { name: "Continue" });
  await expect(continueButton).toBeVisible();
  await continueButton.click();
  await expect(page.getByRole("heading", { name: "Complete" })).toBeVisible();

  expect(prisma).not.toBeNull();
  if (!prisma) {
    return;
  }

  const statusRows = await prisma.$queryRaw<{ status: unknown }[]>`
    SELECT get_init_status() as status
  `;
  const status = normalizeJsonValue(statusRows[0]?.status) as { stage?: string } | null;
  expect(status?.stage).toBe("complete");

  const configuredRows = await prisma.$queryRaw<{ configured: unknown }[]>`
    SELECT get_config('agent.is_configured') as configured
  `;
  const configured = normalizeJsonValue(configuredRows[0]?.configured);
  expect(Boolean(configured)).toBe(true);

  const profileRows = await prisma.$queryRaw<{ profile: unknown }[]>`
    SELECT get_config('agent.init_profile') as profile
  `;
  const profile = normalizeJsonValue(profileRows[0]?.profile) as {
    agent?: { name?: string; tools?: string[] };
    user?: { name?: string };
    heartbeat?: {
      interval_minutes?: number;
      decision_max_tokens?: number;
      base_regeneration?: number;
      max_energy?: number;
      allowed_actions?: string[];
      action_costs?: Record<string, number>;
    };
  } | null;
  expect(profile?.agent?.name).toBe("Astra");
  expect(profile?.user?.name).toBe("Erin");
  expect(profile?.heartbeat?.interval_minutes).toBe(45);
  expect(profile?.heartbeat?.decision_max_tokens).toBe(1500);
  expect(profile?.heartbeat?.base_regeneration).toBe(8);
  expect(profile?.heartbeat?.max_energy).toBe(30);
  expect(profile?.heartbeat?.allowed_actions).not.toContain("reach_out_public");
  expect(profile?.heartbeat?.action_costs?.reflect).toBe(2);
  expect(profile?.agent?.tools).not.toContain("queue_user_message");

  const consentRows = await prisma.$queryRaw<{ decision: string }[]>`
    SELECT decision FROM consent_log ORDER BY decided_at DESC LIMIT 1
  `;
  expect(consentRows[0]?.decision).toBe("consent");

  const memoryRows = await prisma.$queryRaw<{ count: bigint }[]>`
    SELECT COUNT(*)::bigint as count
    FROM memories
    WHERE type = 'worldview'
      AND content = 'My name is Astra.'
  `;
  expect(Number(memoryRows[0]?.count ?? 0)).toBeGreaterThan(0);
});

function loadEnvConfig(rootDir: string) {
  const envPath = [".env.local", ".env"].map((name) => path.join(rootDir, name)).find(fs.existsSync);
  if (!envPath) {
    return {} as Record<string, string>;
  }
  const content = fs.readFileSync(envPath, "utf-8");
  const entries: Record<string, string> = {};
  for (const rawLine of content.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    const idx = line.indexOf("=");
    if (idx <= 0) {
      continue;
    }
    const key = line.slice(0, idx).trim();
    let value = line.slice(idx + 1).trim();
    if ((value.startsWith("\"") && value.endsWith("\"")) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    entries[key] = value;
  }
  return entries;
}

async function waitForPostgres(timeoutMs = 60000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const result = spawnSync(
      "docker",
      ["compose", "exec", "-T", "db", "pg_isready", "-U", pgUser],
      { cwd: repoRoot, encoding: "utf-8" }
    );
    if (result.status === 0) {
      return;
    }
    await delay(1000);
  }
  throw new Error("Postgres did not become ready in time.");
}

async function waitForPostgresStable(database: string, timeoutMs = 120000) {
  const start = Date.now();
  let stableCount = 0;
  let lastStart = "";
  while (Date.now() - start < timeoutMs) {
    const result = spawnSync(
      "docker",
      [
        "compose",
        "exec",
        "-T",
        "db",
        "psql",
        "-U",
        pgUser,
        "-d",
        database,
        "-t",
        "-A",
        "-c",
        "SELECT pg_postmaster_start_time()::text;",
      ],
      { cwd: repoRoot, encoding: "utf-8" }
    );
    if (result.status === 0) {
      const currentStart = result.stdout.trim();
      if (currentStart && currentStart === lastStart) {
        stableCount += 1;
      } else {
        stableCount = 0;
        lastStart = currentStart;
      }
      if (stableCount >= 3) {
        return;
      }
    }
    await delay(1000);
  }
  throw new Error("Postgres did not reach a stable ready state.");
}

async function waitForEmbeddings(timeoutMs = 120000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const result = spawnSync(
      "docker",
      [
        "compose",
        "exec",
        "-T",
        "embeddings",
        "curl",
        "-f",
        "http://localhost:80/health",
      ],
      { cwd: repoRoot, encoding: "utf-8" }
    );
    if (result.status === 0) {
      return;
    }
    await delay(1000);
  }
  throw new Error("Embedding service did not become ready in time.");
}

function createDatabase(name: string) {
  runDockerPsql(`CREATE DATABASE ${name};`, pgDatabase);
}

function applySchema(name: string) {
  const schemaDir = path.resolve(repoRoot, "db");
  const schemaFiles = fs
    .readdirSync(schemaDir)
    .filter((file) => file.endsWith(".sql"))
    .sort();
  const schemaSql = schemaFiles
    .map((file) => fs.readFileSync(path.join(schemaDir, file), "utf-8"))
    .join("\n\n");
  const result = spawnSync(
    "docker",
    [
      "compose",
      "exec",
      "-T",
      "db",
      "psql",
      "-U",
      pgUser,
      "-d",
      name,
      "-v",
      "ON_ERROR_STOP=1",
    ],
    { cwd: repoRoot, encoding: "utf-8", input: schemaSql }
  );
  if (result.status !== 0) {
    throw new Error(`Schema apply failed: ${result.stderr || result.stdout}`);
  }
}

function dropDatabase(name: string) {
  try {
    runDockerPsql(`DROP DATABASE IF EXISTS ${name} WITH (FORCE);`, pgDatabase);
  } catch {
    // Ignore cleanup errors to avoid masking earlier failures.
  }
}

function runDockerPsql(sql: string, database: string) {
  const result = spawnSync(
    "docker",
    [
      "compose",
      "exec",
      "-T",
      "db",
      "psql",
      "-U",
      pgUser,
      "-d",
      database,
      "-v",
      "ON_ERROR_STOP=1",
      "-t",
      "-A",
    ],
    { cwd: repoRoot, input: sql, encoding: "utf-8" }
  );
  if (result.status !== 0) {
    throw new Error(`psql failed: ${result.stderr || result.stdout}`);
  }
  return result.stdout.trim();
}

async function waitForServer(url: string, timeoutMs: number) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (res.ok) {
        return;
      }
    } catch {
      // ignore
    }
    await delay(500);
  }
  throw new Error(`UI server did not start. Logs:\n${serverLogs}`);
}

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
