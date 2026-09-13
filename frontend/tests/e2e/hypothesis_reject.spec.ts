import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

/**
 * The escalation gate.
 *
 * These assert the guardrail contract as the reviewer experiences it: no claim
 * without a resolvable citation, abstention rendered as an answer rather than an
 * empty card, and a reason required before a rejection counts.
 */

const API = process.env.E2E_API_URL ?? "http://localhost:8000";

/**
 * Find a lot that actually has retrievable evidence.
 *
 * Hard-coding a lot name made these tests skip against an abstaining lot, which
 * meant the grounded path -- the one the guardrail contract is about -- was
 * never exercised. Asking the API which lots have evidence costs one cheap call
 * and no tokens, since the summary endpoint does not invoke the agent.
 */
async function lotWithEvidence(request: APIRequestContext): Promise<string | null> {
  for (const candidate of ["lot11055", "lot11514", "lot10362", "lot1"]) {
    const response = await request.get(`${API}/hypothesis/lot/${candidate}`, {
      headers: { "X-Reviewer-Id": "e2e-reviewer" },
    });
    if (!response.ok()) continue;
    const body = await response.json();
    const evidence = body.evidence;
    if (evidence.similar_lots > 0 && evidence.process_events > 0) return candidate;
  }
  return null;
}

async function generate(page: Page, lot: string): Promise<void> {
  await page.goto("/hypotheses");
  await page.getByTestId("lot-input").fill(lot);
  await page.getByTestId("generate").click();
  await expect(
    page
      .getByTestId("abstention")
      .or(page.getByTestId("hypothesis-list"))
      .or(page.getByTestId("hypothesis-error")),
  ).toBeVisible({ timeout: 60_000 });
}

test("abstention renders as an answer, not an empty card", async ({ page }) => {
  // lot1 has similar lots but no process events, which is exactly the thin
  // bundle the agent should decline rather than fill in.
  await generate(page, "lot1");

  const abstention = page.getByTestId("abstention");
  if (await abstention.isVisible()) {
    // "Insufficient evidence, and here is why" is useful. A blank panel reads as
    // a broken feature and trains reviewers to ignore the screen.
    await expect(page.getByTestId("abstention-reason")).not.toBeEmpty();
  }
});

test("every rendered hypothesis carries at least one citation", async ({ page, request }) => {
  const lot = await lotWithEvidence(request);
  test.skip(lot === null, "no lot has retrievable evidence; run seed_from_real_labels.py");
  await generate(page, lot as string);

  const hypotheses = page.getByTestId("hypothesis");
  const count = await hypotheses.count();
  test.skip(count === 0, "the agent abstained on this lot");

  for (let index = 0; index < count; index += 1) {
    const citations = hypotheses.nth(index).getByTestId("citation");
    expect(await citations.count()).toBeGreaterThan(0);
  }
});

test("no citation is rendered unresolved", async ({ page, request }) => {
  const lot = await lotWithEvidence(request);
  test.skip(lot === null, "no lot has retrievable evidence");
  await generate(page, lot as string);

  // The grounding gate drops any hypothesis whose citations do not resolve, so
  // an unresolved marker reaching the screen means the gate let something past.
  await expect(page.getByTestId("unresolved-citation")).toHaveCount(0);
});

test("rejecting a hypothesis requires a reason code", async ({ page, request }) => {
  const lot = await lotWithEvidence(request);
  test.skip(lot === null, "no lot has retrievable evidence");
  await generate(page, lot as string);

  const reject = page.getByTestId("reject-1");
  test.skip((await reject.count()) === 0, "the agent abstained on this lot");

  await reject.click();
  await expect(page.getByTestId("reject-panel")).toBeVisible();
  await expect(page.getByTestId("reason-code-picker")).toBeVisible();
  // The vocabulary must actually load; an empty picker would let a reviewer
  // reject with no reason, which is unusable as signal.
  await expect(page.getByTestId("reason-hypothesis_unsupported")).toBeVisible();
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
