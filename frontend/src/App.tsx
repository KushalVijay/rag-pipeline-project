import { FormEvent, useState } from 'react'

const tabs = ['Knowledge Base', 'Ask', 'Evals'] as const
type Tab = (typeof tabs)[number]

type IngestResult = {
  url: string
  markdown_preview: string
  character_count: number
  chunk_count: number
  title: string | null
}

type AnswerCitation = {
  label: string
  source_url: string
  chunk_number: number
}

type RetrievedChunk = {
  label: string
  text: string
  source_url: string
  chunk_number: number
  distance: number
}

type AskResult = {
  answer: string
  citations: AnswerCitation[]
  retrieved_chunks: RetrievedChunk[]
}

type EvaluationCaseInput = {
  id: number
  question: string
  expected_answer: string
}

type EvaluationResult = {
  question: string
  expected_answer: string
  generated_answer: string
  retrieval_relevance: number
  groundedness: number
  answer_correctness: number
  citation_validity: number
  overall_score: number
  passed: boolean
}

const defaultEvaluationCases: EvaluationCaseInput[] = [1, 2, 3].map((id) => ({
  id,
  question: '',
  expected_answer: '',
}))

const apiBaseUrl = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

function apiDetail(data: unknown, fallback: string) {
  if (!data || typeof data !== 'object' || !('detail' in data)) return fallback

  const { detail } = data as { detail?: unknown }
  if (typeof detail === 'string') return detail
  if (!Array.isArray(detail)) return fallback

  const messages = detail.flatMap((item) => (
    item && typeof item === 'object' && 'msg' in item
      ? [String(item.msg)]
      : []
  ))
  return messages.join(' ') || fallback
}

async function postJson<T>(path: string, body: unknown, fallback: string) {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data: unknown = await response.json().catch(() => null)

  if (!response.ok) throw new Error(apiDetail(data, fallback))
  if (data === null) throw new Error(fallback)
  return data as T
}

function requestErrorMessage(error: unknown, fallback: string) {
  if (error instanceof TypeError) {
    return 'Cannot reach the backend. Make sure the FastAPI server is running.'
  }
  return error instanceof Error ? error.message : fallback
}

const Icon = ({ name }: { name: Tab }) => {
  if (name === 'Knowledge Base') {
    return <span className="tab-icon tab-icon--database" aria-hidden="true" />
  }

  if (name === 'Ask') {
    return <span className="tab-icon tab-icon--ask" aria-hidden="true">?</span>
  }

  return <span className="tab-icon tab-icon--evals" aria-hidden="true">✓</span>
}

function KnowledgeBase() {
  const [url, setUrl] = useState('')
  const [result, setResult] = useState<IngestResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const ingestWebsite = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError(null)
    setIsLoading(true)

    try {
      const data = await postJson<IngestResult>(
        '/api/ingest',
        { url: url.trim() },
        'The website could not be ingested.',
      )
      setResult(data)
    } catch (error) {
      setError(requestErrorMessage(error, 'The website could not be ingested.'))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <section className="panel" aria-labelledby="knowledge-base-heading">
      <div className="panel-copy">
        <span className="eyebrow">Step 01</span>
        <h2 id="knowledge-base-heading">Add a webpage</h2>
        <p>
          Give the evaluator one public URL. Context.dev will turn its main page
          content into clean, retrieval-ready Markdown.
        </p>
      </div>

      <form className="input-card" onSubmit={ingestWebsite} aria-busy={isLoading}>
        <label htmlFor="source-url">Webpage URL</label>
        <div className="url-row">
          <input
            id="source-url"
            type="url"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            placeholder="https://example.com/article"
            aria-describedby="url-help"
            aria-invalid={Boolean(error)}
            disabled={isLoading}
            required
          />
          <button type="submit" disabled={isLoading || !url.trim()}>
            {isLoading ? 'Ingesting…' : 'Ingest Website'}
          </button>
        </div>
        <p id="url-help" className="field-note">
          Public HTTP and HTTPS webpages only. One page is kept in memory at a time.
        </p>
        {error && <p className="error-message" role="alert">{error}</p>}
      </form>

      <div aria-live="polite">
        {result ? (
          <article className="result-card">
            <div className="result-card__topline">
              <span className="result-status"><i /> Ingestion completed</span>
              <span className="result-source">Context.dev</span>
            </div>
            <h3>{result.title || 'Untitled webpage'}</h3>
            <a href={result.url} target="_blank" rel="noreferrer">{result.url}</a>
            <div className="result-stats">
              <div>
                <span>Markdown size</span>
                <strong>{result.character_count.toLocaleString()}</strong>
                <small>characters</small>
              </div>
              <div>
                <span>Chunks created</span>
                <strong>{result.chunk_count.toLocaleString()}</strong>
                <small>searchable sections</small>
              </div>
            </div>
          </article>
        ) : (
          <div className="empty-state">
            <span className="empty-state__mark" aria-hidden="true">01</span>
            <div>
              <h3>Your knowledge base is empty</h3>
              <p>The page title and Markdown size will appear here after ingestion.</p>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}

function Ask() {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<AskResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const askQuestion = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError(null)
    setResult(null)
    setIsLoading(true)

    try {
      const data = await postJson<AskResult>(
        '/api/ask',
        { question: question.trim() },
        'The question could not be answered.',
      )
      setResult(data)
    } catch (error) {
      setError(requestErrorMessage(error, 'The question could not be answered.'))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <section className="panel" aria-labelledby="ask-heading">
      <div className="panel-copy">
        <span className="eyebrow">Step 02</span>
        <h2 id="ask-heading">Ask a grounded question</h2>
        <p>
          Answers will use only retrieved passages from the webpage, with the
          supporting context kept visible for inspection.
        </p>
      </div>

      <form className="input-card" onSubmit={askQuestion} aria-busy={isLoading}>
        <label htmlFor="question">Question</label>
        <textarea
          id="question"
          rows={4}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="What does this page say about…"
          aria-describedby="question-help"
          aria-invalid={Boolean(error)}
          disabled={isLoading}
          required
        />
        <div className="card-actions">
          <span id="question-help" className="field-note">
            Answers use only the most relevant stored passages.
          </span>
          <button type="submit" disabled={isLoading || !question.trim()}>
            {isLoading ? 'Finding an answer…' : 'Ask question'}
          </button>
        </div>
        {error && <p className="error-message" role="alert">{error}</p>}
      </form>

      <div aria-live="polite">
        {result ? (
          <article className="answer-card">
            <div className="answer-card__topline">
              <span className="result-status"><i /> Grounded answer</span>
              <span className="result-source">OpenAI Responses API</span>
            </div>

            <p className="answer-text">{result.answer}</p>

            <div className="citation-row" aria-label="Answer citations">
              <span className="citation-row__label">Citations</span>
              {result.citations.length > 0 ? result.citations.map((citation) => (
                <a
                  className="citation-chip"
                  href={citation.source_url}
                  target="_blank"
                  rel="noreferrer"
                  key={citation.label}
                >
                  {citation.label}
                  <span>Chunk {citation.chunk_number}</span>
                </a>
              )) : (
                <span className="citation-empty">No source citation was needed.</span>
              )}
            </div>

            <details className="context-details">
              <summary>
                <span>Retrieved Context</span>
                <small>{result.retrieved_chunks.length} passages</small>
              </summary>
              <div className="context-list">
                {result.retrieved_chunks.map((chunk) => (
                  <article className="context-chunk" key={chunk.label}>
                    <div className="context-chunk__meta">
                      <span>{chunk.label} · Chunk {chunk.chunk_number}</span>
                      <span>Distance {chunk.distance.toFixed(4)}</span>
                    </div>
                    <p>
                      {chunk.text.length > 360
                        ? `${chunk.text.slice(0, 360)}…`
                        : chunk.text}
                    </p>
                  </article>
                ))}
                <p className="score-note">Lower distance means a closer retrieval match.</p>
              </div>
            </details>
          </article>
        ) : (
          <div className="empty-state">
            <span className="empty-state__mark" aria-hidden="true">02</span>
            <div>
              <h3>No answer yet</h3>
              <p>Ingest a webpage, then ask a question about its content.</p>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}

function Evals() {
  const [cases, setCases] = useState(defaultEvaluationCases)
  const [results, setResults] = useState<EvaluationResult[] | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const updateCase = (
    id: number,
    field: 'question' | 'expected_answer',
    value: string,
  ) => {
    setCases((currentCases) => currentCases.map((evaluationCase) => (
      evaluationCase.id === id
        ? { ...evaluationCase, [field]: value }
        : evaluationCase
    )))
  }

  const canRun = cases.every(
    (evaluationCase) => evaluationCase.question.trim()
      && evaluationCase.expected_answer.trim(),
  )

  const runEvaluations = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError(null)
    setResults(null)
    setIsLoading(true)

    try {
      const data = await postJson<EvaluationResult[]>(
        '/api/evaluate',
        cases.map((evaluationCase) => ({
          question: evaluationCase.question.trim(),
          expected_answer: evaluationCase.expected_answer.trim(),
        })),
        'The evaluations could not be completed.',
      )
      setResults(data)
    } catch (error) {
      setError(requestErrorMessage(error, 'The evaluations could not be completed.'))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <section className="panel" aria-labelledby="evals-heading">
      <div className="panel-copy">
        <span className="eyebrow">Step 03</span>
        <h2 id="evals-heading">Measure answer quality</h2>
        <p>
          Compare grounded answers with expected answers across retrieval,
          support, correctness, and citation quality.
        </p>
      </div>

      <form className="eval-form" onSubmit={runEvaluations} aria-busy={isLoading}>
        <div className="eval-case-list">
          {cases.map((evaluationCase, index) => (
            <fieldset className="eval-case" key={evaluationCase.id} disabled={isLoading}>
              <legend>Case {String(index + 1).padStart(2, '0')}</legend>
              <div className="eval-case__fields">
                <div className="eval-field">
                  <label htmlFor={`eval-question-${evaluationCase.id}`}>Question</label>
                  <textarea
                    id={`eval-question-${evaluationCase.id}`}
                    rows={2}
                    value={evaluationCase.question}
                    onChange={(event) => updateCase(
                      evaluationCase.id,
                      'question',
                      event.target.value,
                    )}
                    placeholder="What should the page answer?"
                    required
                  />
                </div>
                <div className="eval-field">
                  <label htmlFor={`eval-answer-${evaluationCase.id}`}>Expected answer</label>
                  <textarea
                    id={`eval-answer-${evaluationCase.id}`}
                    rows={2}
                    value={evaluationCase.expected_answer}
                    onChange={(event) => updateCase(
                      evaluationCase.id,
                      'expected_answer',
                      event.target.value,
                    )}
                    placeholder="What should a correct answer mean?"
                    required
                  />
                </div>
              </div>
            </fieldset>
          ))}
        </div>

        <div className="eval-actions">
          <p className="field-note">Each case runs retrieval, answering, and one evaluator call.</p>
          <button type="submit" disabled={isLoading || !canRun}>
            {isLoading ? 'Running evaluations…' : 'Run Evals'}
          </button>
        </div>
        {error && <p className="error-message" role="alert">{error}</p>}
      </form>

      <div className="eval-results" aria-live="polite">
        {results ? (
          <>
            <div className="eval-results__topline">
              <span className="result-status"><i /> Evaluation completed</span>
              <span>{results.filter((result) => result.passed).length}/{results.length} passed</span>
            </div>
            <div className="eval-table-wrap">
              <table className="eval-table">
                <thead>
                  <tr>
                    <th>Question</th>
                    <th>Retrieval</th>
                    <th>Groundedness</th>
                    <th>Correctness</th>
                    <th>Citations</th>
                    <th>Overall</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((result, index) => (
                    <tr key={`${result.question}-${index}`}>
                      <td
                        title={`Expected: ${result.expected_answer}\nGenerated: ${result.generated_answer}`}
                      >
                        {result.question}
                      </td>
                      <td>{result.retrieval_relevance.toFixed(2)}</td>
                      <td>{result.groundedness.toFixed(2)}</td>
                      <td>{result.answer_correctness.toFixed(2)}</td>
                      <td>{result.citation_validity.toFixed(2)}</td>
                      <td className="eval-table__overall">{result.overall_score.toFixed(2)}</td>
                      <td>
                        <span className={result.passed ? 'eval-status eval-status--pass' : 'eval-status eval-status--fail'}>
                          {result.passed ? 'Pass' : 'Fail'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <div className="empty-state">
            <span className="empty-state__mark" aria-hidden="true">03</span>
            <div>
              <h3>No evaluation results yet</h3>
              <p>Complete all three cases to run the current knowledge base.</p>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('Knowledge Base')

  return (
    <main className="app-shell">
      <header className="site-header">
        <div className="brand-mark" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <div>
          <p className="kicker">Tutorial project</p>
          <h1>Web RAG Evaluator</h1>
        </div>
        <div className="status-pill"><span /> Local mode</div>
      </header>

      <p className="intro">
        A small, inspectable pipeline for turning a webpage into grounded answers
        you can actually evaluate.
      </p>

      <nav className="tabs" aria-label="Evaluator sections">
        {tabs.map((tab) => (
          <button
            key={tab}
            type="button"
            className={activeTab === tab ? 'tab tab--active' : 'tab'}
            aria-selected={activeTab === tab}
            role="tab"
            onClick={() => setActiveTab(tab)}
          >
            <Icon name={tab} />
            {tab}
          </button>
        ))}
      </nav>

      <div role="tabpanel">
        {activeTab === 'Knowledge Base' && <KnowledgeBase />}
        {activeTab === 'Ask' && <Ask />}
        {activeTab === 'Evals' && <Evals />}
      </div>

      <footer>
        <span>FastAPI</span>
        <i />
        <span>React + Vite</span>
        <i />
        <span>Chroma</span>
        <i />
        <span>OpenAI</span>
        <i />
        <span>Context.dev</span>
      </footer>
    </main>
  )
}

export default App
