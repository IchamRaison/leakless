import { useCallback, useEffect, useRef, useState } from "react";
import { z } from "zod";
import {
  Activity,
  ArrowUpRight,
  AudioLines,
  Check,
  ChevronRight,
  CircleHelp,
  FileAudio,
  FlaskConical,
  Headphones,
  LoaderCircle,
  Plus,
  RefreshCw,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { api, formatTime } from "./api";
import {
  healthSchema,
  sampleSchema,
  visualizationSchema,
  type Health,
  type Sample,
  type Visualization,
} from "./contracts";
import { SignalView } from "./SignalView";

export function App() {
  const [page, setPage] = useState<"studio" | "evaluation">("studio");
  const [health, setHealth] = useState<Health | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selected, setSelected] = useState<Sample | null>(null);
  const [visualization, setVisualization] = useState<Visualization | null>(
    null,
  );
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const [time, setTime] = useState(0);
  const [loop, setLoop] = useState(false);
  const [model, setModel] = useState<"tslm" | "baseline">("tslm");
  const [evaluation, setEvaluation] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const importInFlight = useRef(false);
  const selectionRequest = useRef<AbortController | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [state, clips] = await Promise.all([
        api("/health", healthSchema),
        api("/samples", z.array(sampleSchema)),
      ]);
      setHealth(state);
      setSamples(clips);
      setError("");
      setSelected((current) =>
        current && !clips.some((s) => s.sample_id === current.sample_id)
          ? null
          : current,
      );
    } catch (err) {
      setHealth(null);
      setError((err as Error).message);
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    selectionRequest.current?.abort();
    const controller = new AbortController();
    selectionRequest.current = controller;
    setVisualization(null);
    setTime(0);
    setLoading(false);
    if (!selected) return () => controller.abort();
    setLoading(true);
    api(`/samples/${selected.sample_id}/visualization`, visualizationSchema, {
      signal: controller.signal,
    })
      .then((data) => {
        if (controller.signal.aborted) return;
        if (
          data.input_sha256 !== selected.input_sha256 ||
          data.sample_id !== selected.sample_id
        )
          throw new Error(
            "Le signal reçu ne correspond pas au clip sélectionné.",
          );
        setVisualization(data);
      })
      .catch((err) => {
        if (!controller.signal.aborted) setError(err.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [selected]);

  useEffect(() => {
    if (page !== "evaluation") return;
    const controller = new AbortController();
    setEvaluation("Chargement de l’évaluation…");
    api("/evaluation", z.unknown(), { signal: controller.signal })
      .then(() => {
        if (!controller.signal.aborted)
          setEvaluation(
            "Un artefact est disponible ; son format doit être intégré avec Nevil avant affichage.",
          );
      })
      .catch((err) => {
        if (!controller.signal.aborted) setEvaluation(err.message);
      });
    return () => controller.abort();
  }, [page]);

  async function importFile(file: File | undefined) {
    if (!file || importInFlight.current) return;
    if (!file.name.toLowerCase().endsWith(".wav")) {
      setError("Choisissez un fichier .wav.");
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      setError("Fichier limité à 8 Mio.");
      return;
    }
    importInFlight.current = true;
    setBusy(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const sample = await api("/samples", sampleSchema, {
        method: "POST",
        body,
      });
      setSamples((current) => [
        ...current.filter((s) => s.sample_id !== sample.sample_id),
        sample,
      ]);
      setSelected(sample);
      setPage("studio");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
      importInFlight.current = false;
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function removeSample(sample: Sample) {
    setError("");
    try {
      const response = await fetch(`/api/samples/${sample.sample_id}`, {
        method: "DELETE",
        signal: AbortSignal.timeout(10_000),
      });
      if (!response.ok)
        throw new Error(
          "Impossible de retirer ce clip. Actualisez la connexion.",
        );
      setSamples((current) =>
        current.filter((s) => s.sample_id !== sample.sample_id),
      );
      setSelected((current) =>
        current?.sample_id === sample.sample_id ? null : current,
      );
    } catch (err) {
      setError((err as Error).message);
    }
  }

  const currentModel = health?.models.find((m) => m.name === model);
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(event) => {
            event.preventDefault();
            setPage("studio");
          }}
        >
          <span className="brand-icon">
            <AudioLines size={20} />
          </span>
          PIPE<span className="brand-tag">LAB</span>
        </a>
        <div className="workspace">
          <span className="workspace-avatar">S</span>
          <span>
            Safoan
            <span className="muted small">EHL · Temporal AI Challenge</span>
          </span>
        </div>
        <div className="nav-label">ESPACE DE TRAVAIL</div>
        <nav aria-label="Navigation principale">
          <button
            className={page === "studio" ? "nav-item active" : "nav-item"}
            onClick={() => setPage("studio")}
          >
            <AudioLines size={17} />
            Studio acoustique
          </button>
          <button
            className={page === "evaluation" ? "nav-item active" : "nav-item"}
            onClick={() => setPage("evaluation")}
          >
            <FlaskConical size={17} />
            Comparaison
            <span className="tiny-dot" />
          </button>
        </nav>
        <div className="sidebar-bottom">
          <span className="dev-label">
            <span className="dot amber" />
            En développement
          </span>
          <p>
            Analyse expérimentale.
            <br />
            Validation terrain à établir.
          </p>
          <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">
            Documentation API <ArrowUpRight size={14} />
          </a>
        </div>
      </aside>

      <main>
        <div className="topbar">
          <span>
            Application <ChevronRight size={13} />{" "}
            {page === "studio" ? "Studio acoustique" : "Comparaison"}
          </span>
          <button
            className="connection"
            onClick={() => void refresh()}
            title="Actualiser la connexion"
          >
            <span className={`dot ${health ? "green" : "amber"}`} />
            {health ? "API locale connectée" : "API non connectée"}
            <RefreshCw size={13} />
          </button>
        </div>
        <div className="content">
          <header className="page-head">
            <div>
              <div className="eyebrow">PIPE / SIGNAL LAB</div>
              <h1>
                {page === "studio"
                  ? "Écouter. Observer. Analyser."
                  : "Comparer les modèles."}
              </h1>
              <p>
                {page === "studio"
                  ? "Un enregistrement, ses propriétés acoustiques, une analyse traçable."
                  : "Les résultats réels, leur provenance et leurs limites."}
              </p>
            </div>
            <span className="version">Prototype · v0.1</span>
          </header>
          {error && (
            <div className="error-banner" role="alert">
              <span>{error}</span>
              <button
                aria-label="Fermer le message"
                onClick={() => setError("")}
              >
                <X size={16} />
              </button>
            </div>
          )}

          {page === "evaluation" ? (
            <section className="evaluation-empty">
              <FlaskConical size={34} strokeWidth={1.3} />
              <h2>Pas encore de résultats publiés</h2>
              <p>{evaluation}</p>
              <p className="fine-print">
                Les métriques seront issues du même run d’évaluation : macro-F1,
                rappel fuite, faux positifs, support et versions des modèles.
              </p>
              <button onClick={() => setPage("studio")}>
                Retour au studio
              </button>
            </section>
          ) : (
            <>
              <div className="session-heading">
                <span>
                  <span className="dot amber" />
                  SESSION DE DÉVELOPPEMENT
                </span>
                <span>
                  01 <span className="muted">/ Import et exploration</span>
                </span>
              </div>
              <div className="studio-grid">
                <section className="signal-panel" aria-label="Signal audio">
                  <div className="section-heading">
                    <h2>
                      <Headphones size={18} />
                      Le signal
                    </h2>
                    <button
                      disabled={busy}
                      onClick={() => fileInput.current?.click()}
                    >
                      {busy ? (
                        <LoaderCircle size={15} className="spin" />
                      ) : (
                        <Plus size={15} />
                      )}
                      Importer un WAV
                    </button>
                  </div>
                  <input
                    ref={fileInput}
                    id="audio-upload"
                    type="file"
                    accept=".wav,audio/wav"
                    className="visually-hidden"
                    aria-label="Importer un fichier WAV"
                    onChange={(e) => void importFile(e.target.files?.[0])}
                  />
                  {samples.length > 0 && (
                    <div className="sample-picker">
                      <label htmlFor="sample">Enregistrement</label>
                      <select
                        id="sample"
                        value={selected?.sample_id ?? ""}
                        onChange={(e) => {
                          setError("");
                          setSelected(
                            samples.find(
                              (s) => s.sample_id === e.target.value,
                            ) ?? null,
                          );
                        }}
                      >
                        <option value="" disabled>
                          Choisir un clip
                        </option>
                        {samples.map((s, i) => (
                          <option key={s.sample_id} value={s.sample_id}>
                            Clip {String(i + 1).padStart(2, "0")} ·{" "}
                            {formatTime(s.duration_seconds)}
                          </option>
                        ))}
                      </select>
                      {selected && (
                        <button
                          className="icon-button"
                          aria-label="Retirer le clip sélectionné"
                          onClick={() => void removeSample(selected)}
                        >
                          <Trash2 size={16} />
                        </button>
                      )}
                    </div>
                  )}
                  {!selected ? (
                    <div
                      className="upload-zone"
                      onDragOver={(e) => e.preventDefault()}
                      onDrop={(e) => {
                        e.preventDefault();
                        void importFile(e.dataTransfer.files[0]);
                      }}
                    >
                      <div className="empty-signal-icon">
                        <AudioLines size={40} strokeWidth={1} />
                      </div>
                      <h3>Tout commence par un signal.</h3>
                      <p>
                        Déposez un court enregistrement WAV
                        <br />
                        pour l’écouter et explorer son spectre.
                      </p>
                      <button
                        className="primary"
                        disabled={busy}
                        onClick={() => fileInput.current?.click()}
                      >
                        <Upload size={15} />
                        {busy
                          ? "Import en cours…"
                          : "Choisir un enregistrement"}
                      </button>
                      <span className="upload-hint">
                        WAV · 8 Mio max. · 30 secondes max.
                      </span>
                    </div>
                  ) : (
                    <>
                      <div className="sample-context">
                        <span className="badge">FIXTURE DE DÉVELOPPEMENT</span>
                        <span className="muted small">{selected.source}</span>
                      </div>
                      <div className="sample-metrics">
                        <div>
                          <span>Durée</span>
                          <strong>
                            {formatTime(selected.duration_seconds)}
                          </strong>
                        </div>
                        <div>
                          <span>Échantillonnage</span>
                          <strong>{selected.sample_rate_hz / 1000} kHz</strong>
                        </div>
                        <div>
                          <span>Canaux</span>
                          <strong>
                            {selected.channels === 1 ? "Mono" : "Stéréo"}
                          </strong>
                        </div>
                      </div>
                      <audio
                        key={selected.sample_id}
                        ref={audioRef}
                        controls
                        loop={loop}
                        src={`/api/samples/${selected.sample_id}/audio`}
                        onTimeUpdate={(e) =>
                          setTime(e.currentTarget.currentTime)
                        }
                        onError={() =>
                          setError(
                            "Lecture audio impossible. Réimportez le clip ou vérifiez la connexion.",
                          )
                        }
                        aria-label="Écouter le clip sélectionné"
                      />
                      <div className="playback-options">
                        <span>
                          {formatTime(time)} /{" "}
                          {formatTime(selected.duration_seconds)}
                        </span>
                        <label>
                          <input
                            type="checkbox"
                            checked={loop}
                            onChange={(e) => setLoop(e.target.checked)}
                          />{" "}
                          Lecture en boucle
                        </label>
                      </div>
                      {loop && (
                        <p className="fine-print">
                          Le même clip est répété ; il ne s’agit pas d’un
                          enregistrement continu.
                        </p>
                      )}
                      {selected.warnings.map((w) => (
                        <p className="warning" key={w}>
                          {w}
                        </p>
                      ))}
                      {loading ? (
                        <div className="loading" role="status">
                          <LoaderCircle className="spin" size={20} />
                          Calcul des visuels…
                        </div>
                      ) : visualization ? (
                        <SignalView data={visualization} time={time} />
                      ) : (
                        <p className="fine-print">
                          Visualisation indisponible. Sélectionnez à nouveau le
                          clip pour réessayer.
                        </p>
                      )}
                      <details className="provenance">
                        <summary>Identité du signal et traçabilité</summary>
                        <p>
                          SHA-256 du signal décodé, fréquence et canaux inclus.
                        </p>
                        <code>{selected.input_sha256}</code>
                        <p>
                          Nom d’origine masqué. Aucun label connu pour cet
                          import. Les fichiers restent dans la mémoire du
                          serveur local jusqu’à leur retrait ou son arrêt.
                        </p>
                      </details>
                    </>
                  )}
                  <div className="panel-foot">
                    <FileAudio size={14} />
                    <span>
                      Audio d’origine conservé · aucune donnée envoyée à un
                      modèle externe
                    </span>
                  </div>
                </section>

                <aside
                  className="analysis-panel"
                  aria-label="Analyse du modèle"
                >
                  <div className="section-heading">
                    <h2>
                      <Activity size={18} />
                      L’analyse
                    </h2>
                    <span className="step">02</span>
                  </div>
                  <label className="field-label" htmlFor="model">
                    Modèle
                  </label>
                  <div
                    className="model-switch"
                    id="model"
                    role="group"
                    aria-label="Modèle à utiliser"
                  >
                    <button
                      aria-pressed={model === "tslm"}
                      className={model === "tslm" ? "selected" : ""}
                      onClick={() => setModel("tslm")}
                    >
                      TSLM
                    </button>
                    <button
                      aria-pressed={model === "baseline"}
                      className={model === "baseline" ? "selected" : ""}
                      onClick={() => setModel("baseline")}
                    >
                      Random Forest
                    </button>
                  </div>
                  <p className="model-description">
                    {model === "tslm"
                      ? "Modèle de langage temporel entraîné sur les propriétés acoustiques du signal."
                      : "Baseline de comparaison sur les agrégats du même prétraitement partagé."}
                  </p>
                  <div className="model-status">
                    <span className="dot amber" />
                    {!health ? "Connexion à vérifier" : "Modèle indisponible"}
                  </div>
                  <p className="fine-print">
                    {currentModel?.reason ??
                      "Connectez l’API locale pour vérifier la disponibilité."}
                  </p>
                  <button
                    className="primary analyze"
                    disabled
                    title="L’inférence sera activée après intégration et validation du modèle"
                  >
                    Analyser le signal
                    <ArrowUpRight size={16} />
                  </button>
                  <div className="analysis-placeholder">
                    <span className="quiet-icon">
                      <Activity size={26} strokeWidth={1.2} />
                    </span>
                    <h3>En attente d’une analyse</h3>
                    <p>
                      La prédiction apparaîtra ici après une exécution réelle du
                      modèle.
                    </p>
                    <div className="result-labels">
                      <span>Prédiction</span>
                      <span>—</span>
                      <span>Score</span>
                      <span>—</span>
                      <span>Latence</span>
                      <span>—</span>
                    </div>
                  </div>
                  <div className="reveal">
                    <button disabled>
                      <CircleHelp size={15} />
                      Révéler le label
                    </button>
                    <p className="fine-print">
                      Réservé aux exemples de démonstration autorisés, après
                      l’analyse.
                    </p>
                  </div>
                </aside>
              </div>
              <div className="workflow-note">
                <Check size={15} />
                <span>Écoute et visualisation partagent le même signal.</span>
                <span className="muted">
                  L’intégration des modèles est la prochaine étape.
                </span>
              </div>
              <footer>
                <span>
                  Source expérimentale · aucune promesse de localisation ou de
                  diagnostic terrain
                </span>
                <span>PIPE / EHL 2026</span>
              </footer>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
