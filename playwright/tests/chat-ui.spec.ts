import { expect, test, type Page } from "@playwright/test";

const mockAnswer = {
  answer: "One person may lift at most 23 kg [manual-handling-2025].",
  citations: ["manual-handling-2025"],
  retrieved_doc_ids: ["manual-handling-2025", "manual-handling-2021"],
};

async function ask(page: Page, question: string) {
  await page.getByLabel("Question").fill(question);
  await page.getByRole("button", { name: "Ask" }).click();
}

const panel = (page: Page) => page.getByTestId("answer-panel");

test.describe("chat UI (deterministic, with a mocked API)", () => {
  test("shows the answer only after streaming finishes and lists the sources", async ({ page }) => {
    await page.route("**/api/ask", (route) => route.fulfill({ json: mockAnswer }));
    await page.goto("/");
    await ask(page, "How much can I lift?");

    // Waiting for the first text would race the stream; wait for the explicit "done" state.
    await expect(panel(page)).toHaveAttribute("data-state", "done");
    await expect(page.getByTestId("answer")).toHaveText(mockAnswer.answer);
    await expect(page.getByTestId("source")).toHaveText(mockAnswer.retrieved_doc_ids);
    await expect(page.locator('[data-testid="source"][data-cited="true"]')).toHaveText(["manual-handling-2025"]);
  });

  test("marks an 'I don't know' answer with a badge", async ({ page }) => {
    await page.route("**/api/ask", (route) =>
      route.fulfill({ json: { answer: "I don't know — it is not in the documents.", citations: [], retrieved_doc_ids: [] } }),
    );
    await page.goto("/");
    await ask(page, "How big is the bonus?");
    await expect(panel(page)).toHaveAttribute("data-state", "done");
    await expect(page.getByTestId("refusal-badge")).toBeVisible();
  });

  test("shows an error and re-enables the button when the LLM backend fails", async ({ page }) => {
    await page.route("**/api/ask", (route) => route.fulfill({ status: 502, json: { error: "[timeout] upstream" } }));
    await page.goto("/");
    await ask(page, "Anything");
    await expect(panel(page)).toHaveAttribute("data-state", "error");
    await expect(page.getByTestId("error")).toContainText("timeout");
    await expect(page.getByRole("button", { name: "Ask" })).toBeEnabled();
  });

  test("does not render HTML from the model's answer (XSS via LLM output)", async ({ page }) => {
    const payload = '<img src=x onerror="window.__pwned=1"> done';
    await page.route("**/api/ask", (route) =>
      route.fulfill({ json: { answer: payload, citations: [], retrieved_doc_ids: [] } }),
    );
    await page.goto("/");
    await ask(page, "test");
    await expect(panel(page)).toHaveAttribute("data-state", "done");
    await expect(page.getByTestId("answer")).toHaveText(payload);
    expect(await page.evaluate(() => (window as unknown as { __pwned?: number }).__pwned)).toBeUndefined();
  });
});

test("smoke against the real backend in fake mode: the cited source is on the source list", async ({ page }) => {
  await page.goto("/");
  await ask(page, "How many kilograms can I lift on my own?");
  await expect(panel(page)).toHaveAttribute("data-state", "done");
  await expect(page.getByTestId("answer")).not.toBeEmpty();
  const cited = await page.locator('[data-testid="source"][data-cited="true"]').count();
  expect(cited).toBeGreaterThan(0);
});
