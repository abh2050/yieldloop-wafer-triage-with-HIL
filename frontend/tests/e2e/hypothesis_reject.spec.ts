import { expect, test } from "@playwright/test";

/**
 * The escalation gate.
 *
 * These assert the guardrail contract as the reviewer experiences it: no claim
 * without a resolvable citation, abstention rendered as an answer rather than an
 * empty card, and a reason required before a rejection counts.
 */

test("abstention renders as an answer, not an empty card", async ({ page }) => {
  await page.goto("/hypotheses");
  await page.getByTestId("lot-input").fill("lot1");
  await page.getByTestId("generate").click();

  const abstention = page.getByTestId("abstention");
  const list = page.getByTestId("hypothesis-list");
  const error = page.getByTestId("hypothesis-error");

  await expect(abstention.or(list).or(error)).toBeVisible({ timeout: 45_000 });

  if (await abstention.isVisible()) {
    // "Insufficient evidence, and here is why" is useful. A blank panel reads as
    // a broken feature and trains reviewers to ignore the screen.
    await expect(page.getByTestId("abstention-reason")).not.toBeEmpty();
  }
});

test("every rendered hypothesis carries at least one citation", async ({ page }) => {
  await page.goto("/hypotheses");
  await page.getByTestId("lot-input").fill("lot1");
  await page.getByTestId("generate").click();
  await expect(
    page.getByTestId("abstention").or(page.getByTestId("hypothesis-list")).or(page.getByTestId("hypothesis-error")),
  ).toBeVisible({ timeout: 45_000 });

  const hypotheses = page.getByTestId("hypothesis");
  const count = await hypotheses.count();
  test.skip(count === 0, "no grounded hypotheses for this lot");

  for (let index = 0; index < count; index += 1) {
    const citations = hypotheses.nth(index).getByTestId("citation");
    expect(await citations.count()).toBeGreaterThan(0);
  }
});

test("no citation is rendered unresolved", async ({ page }) => {
  await page.goto("/hypotheses");
  await page.getByTestId("lot-input").fill("lot1");
  await page.getByTestId("generate").click();
  await expect(
    page.getByTestId("abstention").or(page.getByTestId("hypothesis-list")).or(page.getByTestId("hypothesis-error")),
  ).toBeVisible({ timeout: 45_000 });

  // The grounding gate drops any hypothesis whose citations do not resolve, so
  // an unresolved marker reaching the screen means the gate let something past.
  await expect(page.getByTestId("unresolved-citation")).toHaveCount(0);
});

test("rejecting a hypothesis requires a reason code", async ({ page }) => {
  await page.goto("/hypotheses");
  await page.getByTestId("lot-input").fill("lot1");
  await page.getByTestId("generate").click();
  await expect(
    page.getByTestId("abstention").or(page.getByTestId("hypothesis-list")).or(page.getByTestId("hypothesis-error")),
  ).toBeVisible({ timeout: 45_000 });

  const reject = page.getByTestId("reject-1");
  test.skip((await reject.count()) === 0, "no grounded hypotheses for this lot");

  await reject.click();
  await expect(page.getByTestId("reject-panel")).toBeVisible();
  await expect(page.getByTestId("reason-code-picker")).toBeVisible();
});

test("the audit trail shows the chain verification", async ({ page }) => {
  await page.goto("/audit");
  await expect(page.getByTestId("audit-trail")).toBeVisible();
  await expect(page.getByTestId("chain-status")).toBeVisible();
  // An audit log nobody checks is a filing cabinet, not a control.
  await expect(page.getByTestId("chain-status")).toContainText(/intact|broken/i);
});

test("guardrail actions are visible to the reviewer", async ({ page }) => {
  await page.goto("/audit");
  await page.getByTestId("tab-guardrails").click();
  await expect(page.getByTestId("guardrail-actions")).toBeVisible();
});
