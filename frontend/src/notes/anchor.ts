export type NoteAnchor = {
  selector: string;
  section: string;
  element: string;
  snippet: string;
  offsetX: number;
  offsetY: number;
  viewportWidth: number;
  hash: string;
};

const escapeId = (id: string) =>
  typeof CSS !== "undefined" && "escape" in CSS
    ? CSS.escape(id)
    : id.replace(/[^\w-]/g, (char) => `\\${char}`);

/** Shortest stable path: stops at the nearest ancestor id, nth-of-type otherwise. */
export function selectorFor(target: Element): string {
  const parts: string[] = [];
  let node: Element | null = target;
  while (node && node !== document.body && node !== document.documentElement) {
    if (node.id) {
      parts.unshift(`#${escapeId(node.id)}`);
      break;
    }
    const current: Element = node;
    const parent: Element | null = current.parentElement;
    let part = current.tagName.toLowerCase();
    if (parent) {
      const siblings = [...parent.children].filter(
        (child) => child.tagName === current.tagName,
      );
      if (siblings.length > 1)
        part += `:nth-of-type(${siblings.indexOf(current) + 1})`;
    }
    parts.unshift(part);
    node = parent;
  }
  return parts.join(" > ");
}

/** Human label of the closest block: its kicker ("03 / OBSERVE") or first heading. */
export function sectionFor(target: Element): string {
  for (let node: Element | null = target; node; node = node.parentElement) {
    const label = node.querySelector(
      ".demo-kicker, .building-topline span, h1, h2",
    );
    if (label?.textContent?.trim())
      return label.textContent.trim().replace(/\s+/g, " ").slice(0, 120);
  }
  return "page";
}

export function anchorFor(
  target: Element,
  clientX: number,
  clientY: number,
): NoteAnchor {
  const rect = target.getBoundingClientRect();
  const text = target.getAttribute("aria-label") || target.textContent || "";
  const className =
    typeof target.className === "string"
      ? target.className.trim().split(/\s+/)[0]
      : "";
  return {
    selector: selectorFor(target),
    section: sectionFor(target),
    element: target.tagName.toLowerCase() + (className ? `.${className}` : ""),
    snippet: text.trim().replace(/\s+/g, " ").slice(0, 160),
    offsetX: rect.width ? (clientX - rect.left) / rect.width : 0,
    offsetY: rect.height ? (clientY - rect.top) / rect.height : 0,
    viewportWidth: window.innerWidth,
    hash: location.hash,
  };
}
