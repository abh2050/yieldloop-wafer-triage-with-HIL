import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end configuration.
 *
 * These run against the real API and the real Postgres holding the real WM811K
 * wafers. There is no mock server and no stubbed fetch layer: a test that passes
 * against a stub tells you the component renders, not that the system works.
 *
 * The dev server is started here, but the API is not -- it must already be
 * running with a migrated, seeded database. Failing loudly when it is not is
 * better than silently testing against an empty one.
 */
export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? [["github"], ["list"]] : [["list"]],
  timeout: 30_000,
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:5173",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run dev",
    url: "http://localhost:5173",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
