import { expect, it } from "vitest";
import { createNote, type SiteNote } from "./notesServer";

const anchor = { selector: "#inspect-title", offsetX: 3, offsetY: -1 };

it("validates input and owns id, numbering, status and clamps positions", () => {
  expect(() => createNote({ text: " ", anchor }, [])).toThrow("text");
  expect(() => createNote({ text: "ok", anchor: {} }, [])).toThrow("anchor");
  const first = createNote(
    { text: "  Titre trop petit  ", author: "", anchor, status: "resolved" },
    [],
  );
  expect(first).toMatchObject({
    number: 1,
    author: "anonymous",
    text: "Titre trop petit",
    status: "open",
    anchor: { selector: "#inspect-title", offsetX: 1, offsetY: 0 },
  });
  const next = createNote({ text: "b", anchor }, [
    first,
    { ...first, number: 7 } as SiteNote,
  ]);
  expect(next.number).toBe(8);
  expect(next.id).not.toBe(first.id);
});
