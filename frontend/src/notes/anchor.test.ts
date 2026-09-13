// @vitest-environment jsdom
import { expect, it } from "vitest";
import { anchorFor, sectionFor, selectorFor } from "./anchor";

it("anchors a click to a stable selector, its section label and text", () => {
  document.body.innerHTML = `
    <main><section id="evidence">
      <span class="demo-kicker">05 / QUESTION THE RESULT</span>
      <div class="ladder"><article><h3>C0</h3></article><article><h3>C1</h3><p>Envelope</p></article></div>
    </section></main>`;
  const target = document.querySelector("article:nth-of-type(2) p")!;
  expect(selectorFor(target)).toBe(
    "#evidence > div > article:nth-of-type(2) > p",
  );
  expect(document.querySelector(selectorFor(target))).toBe(target);
  expect(sectionFor(target)).toBe("05 / QUESTION THE RESULT");
  expect(anchorFor(target, 0, 0)).toMatchObject({
    element: "p",
    snippet: "Envelope",
    section: "05 / QUESTION THE RESULT",
  });
});
