import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function ScoreBar({ label, score }) {
  const percentage = Math.round((score || 0) * 100);

  return (
    <div className="score-row">
      <div className="score-label">
        <span>{label}</span>
        <span>{percentage}%</span>
      </div>

      <div className="score-track">
        <div
          className="score-fill"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

function ArgumentCard({ title, argument, grounding }) {
  if (!argument) return null;

  return (
    <div className="argument-card">
      <div className="card-header">
        <h3>{title}</h3>

        <span className="stance-badge">
          {argument.stance}
        </span>
      </div>

      <p className="argument-text">
        {argument.text}
      </p>

      <div className="citation-section">
        <span className="section-label">Citations</span>

        <div className="citation-list">
          {argument.cited_evidence_ids?.length > 0 ? (
            argument.cited_evidence_ids.map((id) => (
              <span className="citation-chip" key={id}>
                {id}
              </span>
            ))
          ) : (
            <span className="muted">
              No citations
            </span>
          )}
        </div>
      </div>

      {grounding && (
        <div className="grounding-box">
          <div>
            <span>Valid citations</span>
            <strong>
              {grounding.valid_citation_count}
            </strong>
          </div>

          <div>
            <span>Invalid citations</span>
            <strong>
              {grounding.invalid_citation_count}
            </strong>
          </div>

          <div>
            <span>Grounding score</span>
            <strong>
              {Math.round(
                grounding.grounding_score * 100
              )}
              %
            </strong>
          </div>
        </div>
      )}
    </div>
  );
}

function RebuttalCard({ title, argument, grounding }) {
  if (!argument) return null;

  return (
    <div className="rebuttal-card">
      <div className="rebuttal-title">
        {title}
      </div>

      <p>{argument.text}</p>

      <div className="rebuttal-footer">
        <span>
          Citations:{" "}
          {argument.cited_evidence_ids?.join(", ") ||
            "None"}
        </span>

        {grounding && (
          <span>
            Grounding:{" "}
            {Math.round(
              grounding.grounding_score * 100
            )}
            %
          </span>
        )}
      </div>
    </div>
  );
}

function App() {
  const [claim, setClaim] = useState(
    "Goldfish have a three second memory."
  );

  const [rebuttalRound, setRebuttalRound] =
    useState(true);

  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  async function runDebate() {
    if (!claim.trim()) {
      setError("Please enter a claim.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        `${API_URL}/debate`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            claim: claim.trim(),
            rebuttal_round: rebuttalRound,
          }),
        }
      );

      if (!response.ok) {
        const message = await response.text();

        throw new Error(
          message || "Backend request failed."
        );
      }

      const data = await response.json();

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
          "Unable to connect to the FastAPI backend."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(event) {
    if (
      (event.ctrlKey || event.metaKey) &&
      event.key === "Enter"
    ) {
      runDebate();
    }
  }

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-badge">
          MULTI-AGENT AI
        </div>

        <h1>
          Adversarial Fact
          <span> Checker</span>
        </h1>

        <p>
          Evidence retrieval, multi-agent debate and
          structured judgment powered by FAISS,
          LangGraph and Gemini.
        </p>
      </header>

      <main className="container">
        {/* INPUT */}
        <section className="panel input-panel">
          <div className="panel-header">
            <div>
              <span className="eyebrow">
                STEP 01
              </span>

              <h2>Enter a claim</h2>
            </div>

            <span className="shortcut">
              Ctrl + Enter
            </span>
          </div>

          <textarea
            value={claim}
            onChange={(e) =>
              setClaim(e.target.value)
            }
            onKeyDown={handleKeyDown}
            placeholder="Enter a factual claim..."
            rows={4}
          />

          <div className="input-footer">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={rebuttalRound}
                onChange={(e) =>
                  setRebuttalRound(
                    e.target.checked
                  )
                }
              />

              <span className="toggle">
                <span />
              </span>

              Enable rebuttal round
            </label>

            <button
              className="debate-button"
              onClick={runDebate}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Running debate...
                </>
              ) : (
                <>
                  Run Debate
                  <span>→</span>
                </>
              )}
            </button>
          </div>
        </section>

        {/* ERROR */}
        {error && (
          <div className="error-box">
            <strong>Request failed</strong>
            <span>{error}</span>
          </div>
        )}

        {/* RESULTS */}
        {result && (
          <>
            {/* VERDICT */}
            <section className="verdict-section">
              <div className="section-top">
                <div>
                  <span className="eyebrow">
                    STEP 02
                  </span>

                  <h2>Final verdict</h2>
                </div>

                <span className="pipeline-tag">
                  LangGraph Pipeline
                </span>
              </div>

              <div className="verdict-card">
                <div className="verdict-main">
                  <span className="verdict-label">
                    VERDICT
                  </span>

                  <div
                    className={`verdict-value ${result.verdict?.verdict?.toLowerCase()}`}
                  >
                    {result.verdict?.verdict}
                  </div>

                  <p>
                    {result.verdict?.reasoning}
                  </p>
                </div>

                <div className="confidence">
                  <span>Confidence</span>

                  <strong>
                    {Math.round(
                      (result.verdict?.confidence ||
                        0) * 100
                    )}
                    %
                  </strong>

                  <div className="confidence-track">
                    <div
                      style={{
                        width: `${
                          (result.verdict
                            ?.confidence || 0) * 100
                        }%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            </section>

            {/* ARGUMENTS */}
            <section className="section-block">
              <div className="section-top">
                <div>
                  <span className="eyebrow">
                    STEP 03
                  </span>

                  <h2>Agent arguments</h2>
                </div>
              </div>

              <div className="arguments-grid">
                <ArgumentCard
                  title="FOR Agent"
                  argument={result.for_argument}
                  grounding={result.for_grounding}
                />

                <ArgumentCard
                  title="AGAINST Agent"
                  argument={result.against_argument}
                  grounding={
                    result.against_grounding
                  }
                />
              </div>
            </section>

            {/* SCORES */}
            <section className="section-block">
              <div className="section-top">
                <div>
                  <span className="eyebrow">
                    STEP 04
                  </span>

                  <h2>Debate scores</h2>
                </div>
              </div>

              <div className="scores-panel">
                <ScoreBar
                  label="FOR"
                  score={
                    result.verdict?.for_score
                  }
                />

                <ScoreBar
                  label="AGAINST"
                  score={
                    result.verdict?.against_score
                  }
                />
              </div>
            </section>

            {/* REBUTTALS */}
            {rebuttalRound &&
              (result.for_rebuttal ||
                result.against_rebuttal) && (
                <section className="section-block">
                  <div className="section-top">
                    <div>
                      <span className="eyebrow">
                        STEP 05
                      </span>

                      <h2>Rebuttal round</h2>
                    </div>
                  </div>

                  <div className="rebuttal-grid">
                    <RebuttalCard
                      title="FOR → AGAINST"
                      argument={
                        result.for_rebuttal
                      }
                      grounding={
                        result.for_rebuttal_grounding
                      }
                    />

                    <RebuttalCard
                      title="AGAINST → FOR"
                      argument={
                        result.against_rebuttal
                      }
                      grounding={
                        result.against_rebuttal_grounding
                      }
                    />
                  </div>
                </section>
              )}

            {/* EVIDENCE */}
            <section className="section-block">
              <div className="section-top">
                <div>
                  <span className="eyebrow">
                    STEP 06
                  </span>

                  <h2>Retrieved evidence</h2>
                </div>

                <span className="evidence-count">
                  {result.evidence?.length || 0}{" "}
                  sources
                </span>
              </div>

              <div className="evidence-list">
                {result.evidence?.map((item) => (
                  <article
                    className="evidence-card"
                    key={item.evidence_id}
                  >
                    <div className="evidence-id">
                      {item.evidence_id}
                    </div>

                    <div className="evidence-content">
                      <p>{item.text}</p>

                      <div className="evidence-score">
                        FAISS similarity
                        <strong>
                          {item.score?.toFixed(3)}
                        </strong>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          </>
        )}
      </main>

      <footer>
        <span>
          Multi-Agent Fact-Checking System
        </span>

        <span>
          FAISS · LangGraph · FastAPI · React
        </span>
      </footer>
    </div>
  );
}

export default App;