import { defineConfig } from "@playwright/test";

const port = Number(process.env.PORT ?? 8765);

export default defineConfig({
  testDir: "./tests",
  timeout: 120_000,
  // Retries hide non-determinism. LLM tests measure it with --repeat-each instead.
  retries: 0,
  reporter: [["list"], ["html", { open: "never" }]],
  use: { baseURL: `http://127.0.0.1:${port}`, trace: "retain-on-failure" },
  projects: [
    { name: "ui", testMatch: /chat-ui\.spec\.ts/ },
    { name: "llm", testMatch: /chat-llm\.spec\.ts/ },
  ],
  webServer: {
    command: "uv run python -m ragapp.web",
    cwd: "..",
    url: `http://127.0.0.1:${port}/health`,
    reuseExistingServer: !process.env.CI,
    // UI tests run against the deterministic fake model unless you override EVAL_LLM_MODE.
    env: { EVAL_LLM_MODE: process.env.EVAL_LLM_MODE ?? "fake", PORT: String(port) },
  },
});
