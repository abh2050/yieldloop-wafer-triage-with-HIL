import { expect, test } from "@playwright/test";

/**
 * LabelGrid, against the real API and the real database.
 *
 * No mock server, no stubbed fetch. If the API is not running with a migrated,
 * seeded database these fail, which is the correct outcome: a green suite that
 * proves only that components render is worse than a red one.
 */

test.beforeEach(async ({ page }) => {
  await page.goto("/label");
  await expect(page.getByTestId("label-grid")).toBeVisible();
});

test("renders the label queue from real data", async ({ page }) => {
  await expect(page.getByTestId("queue-depth")).toBeVisible();

  const empty = page.getByTestId("empty-queue");
  const tiles = page.getByTestId("queue-tile");

  // Either state is legitimate; what must never happen is neither.
  const hasWork = (await tiles.count()) > 0;
  if (!hasWork) {
    await expect(empty).toBeVisible();
    return;
  }
  await expect(tiles.first()).toBeVisible();
});

test("the label gate never shows a model prediction", async ({ page }) => {
  // The defining property of this screen. A prediction here would turn an
  // independent human label into agreement with the model, making it useless as
  // training signal.
  await expect(page.getByTestId("prediction-shown")).toHaveCount(0);
  await expect(page.getByTestId("confidence-bar")).toHaveCount(0);
  await expect(page.getByText(/confidence/i)).toHaveCount(0);
});

test("wafer maps render from real arrays on canvas", async ({ page }) => {
  const tiles = page.getByTestId("queue-tile");
  test.skip((await tiles.count()) === 0, "queue is empty; run an active learning round");

  const canvas = page.getByTestId("wafer-map").first();
  await expect(canvas).toBeVisible();

  // The canvas must contain actual die, not be a blank placeholder.
  const distinctColours = await canvas.evaluate((element) => {
    const source = element as HTMLCanvasElement;
    const context = source.getContext("2d");
    if (!context) return 0;
    const { data } = context.getImageData(0, 0, source.width, source.height);
    const seen = new Set<string>();
    for (let i = 0; i < data.length; i += 4) {
      seen.add(`${data[i]},${data[i + 1]},${data[i + 2]}`);
    }
    return seen.size;
  });
  expect(distinctColours).toBeGreaterThan(1);
});

test("keyboard shortcuts are visible without opening help", async ({ page }) => {
  const hints = page.getByTestId("keyboard-hints");
  await expect(hints).toBeVisible();
  await expect(hints).toContainText("1-9");
});

test("arrow keys move focus between wafers", async ({ page }) => {
  const tiles = page.getByTestId("queue-tile");
  test.skip((await tiles.count()) < 2, "needs at least two queued wafers");

  const first = await page.getByTestId("focused-wafer-id").textContent();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByTestId("focused-wafer-id")).not.toHaveText(first ?? "");

  await page.keyboard.press("ArrowLeft");
  await expect(page.getByTestId("focused-wafer-id")).toHaveText(first ?? "");
});

test("a digit key labels the focused wafer and advances", async ({ page }) => {
  const tiles = page.getByTestId("queue-tile");
  test.skip((await tiles.count()) === 0, "queue is empty; run an active learning round");

  const before = await tiles.count();
  const labelled = await page.getByTestId("focused-wafer-id").textContent();

  await page.keyboard.press("4"); // edge_ring

  await expect(page.getByTestId("decided-count")).toContainText("1");
  await expect(tiles).toHaveCount(before - 1);
  await expect(page.getByTestId("queue-tile").filter({ hasText: labelled ?? "" })).toHaveCount(0);
});

test("a decision is recorded in under four seconds of interaction time", async ({ page }) => {
  const tiles = page.getByTestId("queue-tile");
  test.skip((await tiles.count()) === 0, "queue is empty; run an active learning round");

  const started = Date.now();
  await page.keyboard.press("9"); // none
  await expect(page.getByTestId("decided-count")).toContainText("1");
  const elapsed = Date.now() - started;

  // The product claim is sub-four-second decisions. This measures the round
  // trip, which is the part the console controls.
  expect(elapsed).toBeLessThan(4000);
  await expect(page.getByTestId("last-decision-ms")).toBeVisible();
});

test("typing in a field does not trigger class shortcuts", async ({ page }) => {
  // A reviewer writing "1 scratch near the notch" must not label nine wafers.
  await page.goto("/hypotheses");
  const input = page.getByTestId("note-input");
  await input.fill("");
  await input.type("1 scratch near the notch");
  await expect(input).toHaveValue("1 scratch near the notch");
});
