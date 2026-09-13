import { expect, test } from "@playwright/test";

/**
 * The triage queue and the anchoring rule.
 *
 * The property under test is not that the UI hides low-confidence predictions,
 * it is that it never receives them. A test that only checked the rendered DOM
 * would still pass if the API started leaking them.
 */

test("triage queue renders with the configured routing bands", async ({ page }) => {
  await page.goto("/triage");
  await expect(page.getByTestId("triage-queue")).toBeVisible();
  await expect(page.getByText(/Auto-commit at/)).toBeVisible();
});

test("predictions below the floor are withheld, not merely hidden", async ({ request }) => {
  const response = await request.get("http://localhost:8000/triage/queue?limit=50", {
    headers: { "X-Reviewer-Id": "e2e-reviewer" },
  });
  expect(response.ok()).toBeTruthy();
  const body = await response.json();

  for (const item of body.items) {
    if (!item.show_prediction) {
      // The API must omit them entirely. Hiding in the client is not enough:
      // anything present in the payload is one devtools tab away from the
      // reviewer, and that is exactly the anchoring this rule prevents.
      expect(item.predicted_label ?? null).toBeNull();
      expect(item.confidence ?? null).toBeNull();
    } else {
      expect(item.confidence).toBeGreaterThanOrEqual(body.confidence_floor);
    }
  }
});

test("withheld rows say so explicitly", async ({ page }) => {
  await page.goto("/triage");
  const rows = page.getByTestId("triage-row");
  test.skip((await rows.count()) === 0, "triage queue is empty");

  const withheld = page.getByTestId("prediction-withheld");
  if ((await withheld.count()) > 0) {
    // Explicit rather than blank: the reviewer should know the model has an
    // opinion being deliberately withheld, not assume it had none.
    await expect(withheld.first()).toContainText("withheld");
  }
});

test("wafer detail withholds the prediction below the floor", async ({ page, request }) => {
  const response = await request.get("http://localhost:8000/triage/wafer/lot1-1", {
    headers: { "X-Reviewer-Id": "e2e-reviewer" },
  });
  expect(response.ok()).toBeTruthy();
  const wafer = await response.json();

  await page.goto(`/wafer/${wafer.wafer_id}`);
  await expect(page.getByTestId("detail-wafer-id")).toHaveText(wafer.wafer_id);
  await expect(page.getByTestId("wafer-map")).toBeVisible();

  if (!wafer.show_prediction) {
    await expect(page.getByTestId("detail-prediction-withheld")).toBeVisible();
    await expect(page.getByTestId("detail-prediction")).toHaveCount(0);
  }
});

test("model health reports override rate both ways", async ({ page }) => {
  await page.goto("/health");
  await expect(page.getByTestId("model-health")).toBeVisible();
  // The gap between these two is the anchoring effect; reporting only the first
  // would hide it.
  await expect(page.getByText("Override rate")).toBeVisible();
  await expect(page.getByText("Override (shown)")).toBeVisible();
});
