import { useCallback, useEffect, useState } from "react";
import { anchorFor, type NoteAnchor } from "./anchor";
import "./notes.css";

type SiteNote = {
  id: string;
  number: number;
  author: string;
  text: string;
  status: "open" | "resolved";
  createdAt: string;
  anchor: NoteAnchor;
};
type Draft = { anchor: NoteAnchor; x: number; y: number };
type Box = { top: number; left: number; width: number; height: number };

const AUTHOR_KEY = "leakless-notes-author";
const isNotesUi = (node: EventTarget | null) =>
  node instanceof Element && !!node.closest("[data-notes-ui]");

function storedAuthor() {
  try {
    return localStorage.getItem(AUTHOR_KEY) ?? "";
  } catch {
    return "";
  }
}

function locate(note: SiteNote) {
  try {
    return document.querySelector(note.anchor.selector);
  } catch {
    return null;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/__notes${path}`, {
    ...init,
    headers: { "Content-Type": "application/json" },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok)
    throw new Error(payload.error ?? `Erreur ${response.status}`);
  return payload as T;
}

/** Internal review tool: pins notes on page elements. Loaded only with ?notes in dev. */
export default function NotesLayer() {
  const [notes, setNotes] = useState<SiteNote[]>([]);
  const [placing, setPlacing] = useState(false);
  const [hover, setHover] = useState<Box | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [text, setText] = useState("");
  const [author, setAuthor] = useState(storedAuthor);
  const [panel, setPanel] = useState(false);
  const [showResolved, setShowResolved] = useState(false);
  const [error, setError] = useState("");
  const [, setTick] = useState(0);

  const load = useCallback(async () => {
    try {
      setNotes((await request<{ notes: SiteNote[] }>("")).notes);
      setError("");
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);
  useEffect(() => {
    void load();
  }, [load]);

  // Pins follow layout changes, scrolling and the 3D panels resizing.
  useEffect(() => {
    const refresh = () => setTick((value) => value + 1);
    window.addEventListener("scroll", refresh, true);
    window.addEventListener("resize", refresh);
    const timer = window.setInterval(refresh, 800);
    return () => {
      window.removeEventListener("scroll", refresh, true);
      window.removeEventListener("resize", refresh);
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!placing) return;
    const move = (event: MouseEvent) => {
      if (isNotesUi(event.target) || !(event.target instanceof Element))
        return setHover(null);
      const rect = event.target.getBoundingClientRect();
      setHover({
        top: rect.top,
        left: rect.left,
        width: rect.width,
        height: rect.height,
      });
    };
    const click = (event: MouseEvent) => {
      if (isNotesUi(event.target) || !(event.target instanceof Element)) return;
      // Capture phase: the site never receives the click while placing a note.
      event.preventDefault();
      event.stopPropagation();
      setDraft({
        anchor: anchorFor(event.target, event.clientX, event.clientY),
        x: event.clientX,
        y: event.clientY,
      });
      setText("");
      setPlacing(false);
      setHover(null);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setPlacing(false);
        setHover(null);
      }
    };
    document.addEventListener("mousemove", move, true);
    document.addEventListener("click", click, true);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("mousemove", move, true);
      document.removeEventListener("click", click, true);
      document.removeEventListener("keydown", escape);
    };
  }, [placing]);

  async function save() {
    if (!draft || !text.trim()) return;
    try {
      localStorage.setItem(AUTHOR_KEY, author);
    } catch {
      // Author name is a convenience only.
    }
    try {
      const note = await request<SiteNote>("", {
        method: "POST",
        body: JSON.stringify({ text, author, anchor: draft.anchor }),
      });
      setNotes((current) => [...current, note]);
      setDraft(null);
      setError("");
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function update(note: SiteNote, method: "PATCH" | "DELETE") {
    try {
      if (method === "DELETE") {
        await request(`/${note.id}`, { method });
        setNotes((current) => current.filter((item) => item.id !== note.id));
      } else {
        const next = await request<SiteNote>(`/${note.id}`, {
          method,
          body: JSON.stringify({
            status: note.status === "open" ? "resolved" : "open",
          }),
        });
        setNotes((current) =>
          current.map((item) => (item.id === note.id ? next : item)),
        );
      }
    } catch (err) {
      setError((err as Error).message);
    }
  }

  function reveal(note: SiteNote) {
    const target = locate(note);
    if (!target) return;
    target.scrollIntoView({ block: "center", behavior: "smooth" });
    target.classList.add("notes-flash");
    window.setTimeout(() => target.classList.remove("notes-flash"), 1600);
  }

  const visible = notes.filter(
    (note) => showResolved || note.status === "open",
  );
  const open = notes.filter((note) => note.status === "open").length;

  return (
    <div
      data-notes-ui
      className={placing ? "notes-root placing" : "notes-root"}
    >
      {hover && (
        <div
          className="notes-hover"
          style={{
            top: hover.top,
            left: hover.left,
            width: hover.width,
            height: hover.height,
          }}
        />
      )}
      {visible.map((note) => {
        const target = locate(note);
        if (!target) return null;
        const rect = target.getBoundingClientRect();
        if (!rect.width && !rect.height) return null;
        return (
          <button
            key={note.id}
            className={`notes-pin ${note.status}`}
            style={{
              top: rect.top + note.anchor.offsetY * rect.height,
              left: rect.left + note.anchor.offsetX * rect.width,
            }}
            title={`${note.author} — ${note.text}`}
            onClick={() => {
              setPanel(true);
              document
                .getElementById(`note-${note.id}`)
                ?.scrollIntoView({ block: "nearest" });
            }}
          >
            {note.number}
          </button>
        );
      })}
      {draft && (
        <div
          className="notes-draft"
          style={{
            top: Math.min(draft.y + 12, window.innerHeight - 250),
            left: Math.min(draft.x + 12, window.innerWidth - 330),
          }}
        >
          <span className="notes-where">
            {draft.anchor.section} · {draft.anchor.element}
          </span>
          {draft.anchor.snippet && (
            <q className="notes-snippet">{draft.anchor.snippet}</q>
          )}
          <textarea
            autoFocus
            value={text}
            placeholder="Ta note sur cet endroit…"
            onChange={(event) => setText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey))
                void save();
              if (event.key === "Escape") setDraft(null);
            }}
          />
          <input
            value={author}
            placeholder="Ton prénom"
            aria-label="Auteur"
            onChange={(event) => setAuthor(event.target.value)}
          />
          <div className="notes-actions">
            <button onClick={() => setDraft(null)}>Annuler</button>
            <button
              className="primary"
              disabled={!text.trim()}
              onClick={() => void save()}
            >
              Enregistrer ⌘↵
            </button>
          </div>
        </div>
      )}
      {panel && (
        <aside className="notes-panel" aria-label="Notes de revue">
          <header>
            <strong>
              Notes · {open} ouverte{open > 1 ? "s" : ""}
            </strong>
            <label>
              <input
                type="checkbox"
                checked={showResolved}
                onChange={(event) => setShowResolved(event.target.checked)}
              />{" "}
              résolues
            </label>
          </header>
          {visible.length === 0 && (
            <p className="notes-empty">
              Aucune note. Clique « + Note » puis sur un élément du site.
            </p>
          )}
          <ol>
            {visible.map((note) => {
              const found = !!locate(note);
              return (
                <li
                  key={note.id}
                  id={`note-${note.id}`}
                  className={note.status}
                >
                  <button className="notes-go" onClick={() => reveal(note)}>
                    <span className="notes-number">{note.number}</span>
                    <span>
                      <span className="notes-where">
                        {note.anchor.section}
                        {!found && " · élément introuvable"}
                      </span>
                      <span className="notes-text">{note.text}</span>
                      <span className="notes-meta">
                        {note.author} ·{" "}
                        {new Date(note.createdAt).toLocaleString("fr-CH")}
                      </span>
                    </span>
                  </button>
                  <div className="notes-actions">
                    <button onClick={() => void update(note, "PATCH")}>
                      {note.status === "open" ? "Résoudre" : "Rouvrir"}
                    </button>
                    <button onClick={() => void update(note, "DELETE")}>
                      Supprimer
                    </button>
                  </div>
                </li>
              );
            })}
          </ol>
        </aside>
      )}
      <div className="notes-toolbar">
        {error && <span className="notes-error">{error}</span>}
        <button
          className={placing ? "primary" : ""}
          onClick={() => {
            setDraft(null);
            setPlacing(!placing);
            setHover(null);
          }}
        >
          {placing ? "Clique un élément · Échap" : "+ Note"}
        </button>
        <button onClick={() => setPanel(!panel)}>Notes ({open})</button>
      </div>
    </div>
  );
}
