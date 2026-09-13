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

  // Wait for the fetch to resolve before counting. Counting immediately after
  // navigation returns zero while the request is still in flight, which made
  // this skip itself as "queue is empty" even when the queue was full -- a skip
  // that looks like a pass and hides the assertion entirely.
  const rows = page.getByTestId("triage-row");
  const empty = page.getByTestId("empty-triage");
  await expect(rows.first().or(empty)).toBeVisible({ timeout: 10_000 });

  test.skip(await empty.isVisible(), "triage queue is empty; run route_predictions.py");

  const total = await rows.count();
  expect(total).toBeGreaterThan(0);

  // Every row either shows a prediction or says it is withheld. A row that did
  // neither would leave the reviewer to assume the model had no opinion, when in
  // fact it has one that policy is deliberately holding back.
  const shown = await page.getByTestId("prediction-shown").count();
  const withheld = await page.getByTestId("prediction-withheld").count();
  expect(shown + withheld).toBe(total);

  if (withheld > 0) {
    await expect(page.getByTestId("prediction-withheld").first()).toContainText("withheld");
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
