import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";


function formatPercent(value) {
  if (value === null || value === undefined) {
    return "0%";
  }

  return `${Math.round(value * 100)}%`;
}


function SourceBadge({ type }) {
  if (!type) return null;

  return (
    <span className={`source-badge ${type.toLowerCase()}`}>
      {type}
    </span>
  );
}


function StatusBadge({ value }) {
  if (!value) return null;

  return (
    <span className="status-badge">
      {value}
    </span>
  );
}


function ScoreBar({ label, score }) {
  const percentage = Math.max(
    0,
    Math.min(100, Math.round((score || 0) * 100))
  );

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


function ArgumentCard({
  title,
  argument,
  grounding,
}) {
  if (!argument) return null;

  return (
    <div className="argument-card">
      <div className="card-header">
        <div>
          <span className="eyebrow">AGENT</span>
          <h3>{title}</h3>
        </div>

        <span className="stance-badge">
          {argument.stance}
        </span>
      </div>

      <p className="argument-text">
        {argument.text}
      </p>

      <div className="citation-section">
        <span className="section-label">
          Citations
        </span>

        <div className="citation-list">
          {argument.cited_evidence_ids?.length > 0 ? (
            argument.cited_evidence_ids.map((id) => (
              <span
                className="citation-chip"
                key={id}
              >
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
            <span>Valid</span>
            <strong>
              {grounding.valid_citation_count}
            </strong>
          </div>

          <div>
            <span>Invalid</span>
            <strong>
              {grounding.invalid_citation_count}
            </strong>
          </div>

          <div>
            <span>Grounding</span>
            <strong>
              {formatPercent(
                grounding.grounding_score
              )}
            </strong>
          </div>
        </div>
      )}
    </div>
  );
}


function RebuttalCard({
  title,
  argument,
  grounding,
}) {
  if (!argument) return null;

  return (
    <div className="rebuttal-card">
      <div className="rebuttal-header">
        <span className="rebuttal-title">
          {title}
        </span>

        <span className="stance-badge">
          {argument.stance}
        </span>
      </div>

      <p>{argument.text}</p>

      <div className="rebuttal-footer">
        <span>
          Citations:{" "}
          {argument.cited_evidence_ids?.length
            ? argument.cited_evidence_ids.join(", ")
            : "None"}
        </span>

        {grounding && (
          <span>
            Grounding:{" "}
            {formatPercent(
              grounding.grounding_score
            )}
          </span>
        )}
      </div>
    </div>
  );
}


function EvidenceCard({ item }) {
  return (
    <article className="evidence-card">
      <div className="evidence-id">
        {item.evidence_id}
      </div>

      <div className="evidence-content">
        <div className="evidence-top">
          <SourceBadge
            type={item.source_type}
          />

          {item.domain && (
            <span className="domain">
              {item.domain}
            </span>
          )}
        </div>

        <h3>
          {item.title ||
            "Evidence from local corpus"}
        </h3>

        <p>{item.text}</p>

        <div className="evidence-metrics">
          <span>
            Retrieval{" "}
            <strong>
              {(
                item.retrieval_score ??
                item.score ??
                0
              ).toFixed(3)}
            </strong>
          </span>

          {item.credibility_score !==
            undefined && (
            <span>
              Credibility{" "}
              <strong>
                {formatPercent(
                  item.credibility_score
                )}
              </strong>
            </span>
          )}

          {item.published_date && (
            <span>
              Published{" "}
              <strong>
                {item.published_date}
              </strong>
            </span>
          )}
        </div>

        {item.url && (
          <a
            href={item.url}
            target="_blank"
            rel="noopener noreferrer"
            className="source-link"
          >
            Open source ↗
          </a>
        )}
      </div>
    </article>
  );
}


function RejectedEvidenceCard({ item }) {
  return (
    <article className="rejected-card">
      <div className="rejected-id">
        {item.evidence_id}
      </div>

      <div>
        <div className="evidence-top">
          <SourceBadge
            type={item.source_type}
          />

          {item.domain && (
            <span className="domain">
              {item.domain}
            </span>
          )}
        </div>

        <h3>
          {item.title ||
            "Rejected evidence"}
        </h3>

        <p>{item.text}</p>

        <div className="rejection-reason">
          Source was excluded by the evidence
          quality filter.
        </div>
      </div>
    </article>
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
        const message =
          await response.text();

        throw new Error(
          message ||
            "Backend request failed."
        );
      }

      const data =
        await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

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


  const verdict =
    result?.verdict?.verdict || "";

  const isSubjective =
    result?.claim_type === "SUBJECTIVE";

  const isInsufficient =
    result &&
    result.evidence_sufficient === false &&
    !isSubjective;


  return (
    <div className="app">

      {/* =====================================================
          HERO
      ===================================================== */}

      <header className="hero">
        <div className="hero-badge">
          MULTI-AGENT AI · V2
        </div>

        <h1>
          Open-World
          <span>Fact Checker</span>
        </h1>

        <p>
          Hybrid local and web retrieval,
          evidence validation, adversarial debate,
          and structured judgment.
        </p>
      </header>


      <main className="container">

        {/* =================================================
            INPUT
        ================================================= */}

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
                  Analyzing...
                </>
              ) : (
                <>
                  Run Analysis
                  <span>→</span>
                </>
              )}
            </button>

          </div>
        </section>


        {/* =================================================
            ERROR
        ================================================= */}

        {error && (
          <div className="error-box">
            <strong>
              Request failed
            </strong>

            <span>{error}</span>
          </div>
        )}


        {/* =================================================
            RESULTS
        ================================================= */}

        {result && (
          <>

            {/* =============================================
                CLAIM ANALYSIS
            ============================================= */}

            <section className="section-block">

              <div className="section-top">
                <div>
                  <span className="eyebrow">
                    STEP 02
                  </span>

                  <h2>
                    Claim analysis
                  </h2>
                </div>
              </div>


              <div className="metadata-grid">

                <div className="metadata-card">
                  <span>
                    CLAIM TYPE
                  </span>

                  <strong>
                    {result.claim_type ||
                      "FACTUAL"}
                  </strong>
                </div>


                <div className="metadata-card">
                  <span>
                    RETRIEVAL MODE
                  </span>

                  <strong>
                    {result.retrieval_mode ||
                      "LOCAL"}
                  </strong>
                </div>


                <div className="metadata-card">
                  <span>
                    EVIDENCE
                  </span>

                  <strong>
                    {result.evidence_sufficient
                      ? "SUFFICIENT"
                      : "INSUFFICIENT"}
                  </strong>
                </div>

              </div>

            </section>


            {/* =============================================
                EVIDENCE STATUS
            ============================================= */}

            {(isSubjective ||
              isInsufficient) && (
              <section
                className={`decision-box ${
                  isSubjective
                    ? "subjective"
                    : "insufficient"
                }`}
              >

                <div className="decision-icon">
                  {isSubjective
                    ? "?"
                    : "!"}
                </div>

                <div>
                  <span className="eyebrow">
                    SYSTEM DECISION
                  </span>

                  <h2>
                    {isSubjective
                      ? "Subjective claim"
                      : "Insufficient evidence"}
                  </h2>

                  <p>
                    {result.evidence_reason ||
                      result.verdict?.reasoning}
                  </p>
                </div>

              </section>
            )}


            {/* =============================================
                VERDICT
            ============================================= */}

            <section className="verdict-section">

              <div className="section-top">
                <div>
                  <span className="eyebrow">
                    STEP 03
                  </span>

                  <h2>
                    Final verdict
                  </h2>
                </div>

                <div className="header-badges">
                  <StatusBadge
                    value={
                      result.claim_type
                    }
                  />

                  <StatusBadge
                    value={
                      result.retrieval_mode
                    }
                  />
                </div>
              </div>


              <div className="verdict-card">

                <div className="verdict-main">

                  <span className="verdict-label">
                    VERDICT
                  </span>

                  <div
                    className={`verdict-value ${verdict.toLowerCase()}`}
                  >
                    {verdict}
                  </div>

                  <p>
                    {result.verdict?.reasoning}
                  </p>

                </div>


                <div className="confidence">

                  <span>
                    Confidence
                  </span>

                  <strong>
                    {formatPercent(
                      result.verdict
                        ?.confidence
                    )}
                  </strong>

                  <div className="confidence-track">
                    <div
                      style={{
                        width: `${Math.round(
                          (result.verdict
                            ?.confidence ||
                            0) * 100
                        )}%`,
                      }}
                    />
                  </div>

                </div>

              </div>
            </section>


            {/* =============================================
                DEBATE SCORES
            ============================================= */}

            {result.verdict?.for_score !==
              undefined &&
              result.verdict?.against_score !==
                undefined && (
                <section className="section-block">

                  <div className="section-top">
                    <div>
                      <span className="eyebrow">
                        SCORE
                      </span>

                      <h2>
                        Debate balance
                      </h2>
                    </div>
                  </div>

                  <div className="scores-panel">

                    <ScoreBar
                      label="FOR"
                      score={
                        result.verdict
                          .for_score
                      }
                    />

                    <ScoreBar
                      label="AGAINST"
                      score={
                        result.verdict
                          .against_score
                      }
                    />

                  </div>

                </section>
              )}


            {/* =============================================
                AGENTS
            ============================================= */}

            {!isSubjective &&
              !isInsufficient && (
                <section className="section-block">

                  <div className="section-top">
                    <div>
                      <span className="eyebrow">
                        STEP 04
                      </span>

                      <h2>
                        Agent arguments
                      </h2>
                    </div>
                  </div>


                  <div className="arguments-grid">

                    <ArgumentCard
                      title="FOR Agent"
                      argument={
                        result.for_argument
                      }
                      grounding={
                        result.for_grounding
                      }
                    />

                    <ArgumentCard
                      title="AGAINST Agent"
                      argument={
                        result.against_argument
                      }
                      grounding={
                        result.against_grounding
                      }
                    />

                  </div>

                </section>
              )}


            {/* =============================================
                REBUTTALS
            ============================================= */}

            {!isSubjective &&
              !isInsufficient &&
              (result.for_rebuttal ||
                result.against_rebuttal) && (
                <section className="section-block">

                  <div className="section-top">
                    <div>
                      <span className="eyebrow">
                        STEP 05
                      </span>

                      <h2>
                        Rebuttal round
                      </h2>
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


            {/* =============================================
                VALIDATED EVIDENCE
            ============================================= */}

            <section className="section-block">

              <div className="section-top">

                <div>
                  <span className="eyebrow">
                    STEP 06
                  </span>

                  <h2>
                    Validated evidence
                  </h2>
                </div>

                <span className="evidence-count">
                  {result.evidence?.length ||
                    0}{" "}
                  sources
                </span>

              </div>


              {result.evidence?.length ? (
                <div className="evidence-list">

                  {result.evidence.map(
                    (item) => (
                      <EvidenceCard
                        key={
                          item.evidence_id
                        }
                        item={item}
                      />
                    )
                  )}

                </div>
              ) : (
                <div className="empty-state">
                  No accepted evidence was
                  available.
                </div>
              )}

            </section>


            {/* =============================================
                REJECTED EVIDENCE
            ============================================= */}

            {result.rejected_evidence
              ?.length > 0 && (
              <section className="section-block">

                <div className="section-top">

                  <div>
                    <span className="eyebrow">
                      FILTER
                    </span>

                    <h2>
                      Rejected evidence
                    </h2>
                  </div>

                  <span className="evidence-count">
                    {
                      result
                        .rejected_evidence
                        .length
                    }{" "}
                    filtered
                  </span>

                </div>


                <div className="rejected-list">

                  {result.rejected_evidence.map(
                    (item) => (
                      <RejectedEvidenceCard
                        key={
                          item.evidence_id
                        }
                        item={item}
                      />
                    )
                  )}

                </div>

              </section>
            )}

          </>
        )}
      </main>


      <footer>
        <span>
          Multi-Agent Open-World Fact Checker
        </span>

        <span>
          FAISS · Tavily · LangGraph · FastAPI · React
        </span>
      </footer>

    </div>
  );
}

export default App;