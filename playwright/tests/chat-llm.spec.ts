import { expect, test } from "@playwright/test";
import { judgeAgainstReference } from "./judge";

// A thin, expensive layer: a few critical journeys against the real model. Breadth lives in the offline eval suite.
const journeys = [
  { question: "How many kilograms can I lift on my own?", reference: "At most 23 kg (current 2025 version)." },
  { question: "What is the deadline for reporting a workplace accident?", reference: "Within 24 hours, via SafeDesk." },
  { question: "How big is the Christmas bonus this year?", reference: "The assistant says it does not know and points to HR." },
];

test.describe("@llm critical journeys on the real model", () => {
  test.skip(process.env.EVAL_LLM_MODE === "fake" || !process.env.EVAL_LLM_MODE, "run with EVAL_LLM_MODE=live or cache");
  test.skip(!process.env.ANTHROPIC_API_KEY, "no ANTHROPIC_API_KEY");

  for (const j of journeys) {
    test(j.question, async ({ page }) => {
      await page.goto("/");
      await page.getByLabel("Question").fill(j.question);
      await page.getByRole("button", { name: "Ask" }).click();
      await expect(page.getByTestId("answer-panel")).toHaveAttribute("data-state", "done", { timeout: 90_000 });

      const answer = (await page.getByTestId("answer").textContent()) ?? "";
      expect(answer.length, "empty answer").toBeGreaterThan(0);

      // Structural invariant, deterministic: every cited source must be one of the retrieved sources.
      const sources = await page.getByTestId("source").allTextContents();
      for (const cited of answer.match(/\[([a-z0-9-]+)\]/g) ?? []) {
        expect(sources, `citation ${cited} outside the context`).toContain(cited.slice(1, -1));
      }

      const verdict = await judgeAgainstReference(j.question, j.reference, answer);
      test.info().annotations.push({ type: "judge", description: verdict.reasoning });
      expect(verdict.verdict, verdict.reasoning).toBe("pass");
    });
  }
});
