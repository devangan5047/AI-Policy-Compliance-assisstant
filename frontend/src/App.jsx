import { useMemo, useState } from 'react'
import axios from 'axios'
import {
  FiAlertTriangle,
  FiCheckCircle,
  FiDatabase,
  FiFileText,
  FiSearch,
  FiShield,
  FiUpload,
} from 'react-icons/fi'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'

const starterQuestions = [
  'Can I continue working remotely while traveling to another country?',
  'If I accidentally see data that does not belong to my project, should I report it?',
  'Are there restrictions when sharing customer data from different regions with another internal team?',
]

function createStreamingAnswer(text = '') {
  return {
    answer: text,
    compliance_status: 'insufficient_context',
    compliance_score: 0,
    risk_level: 'medium',
    escalation_required: false,
    citations: [],
    conflicts: [],
    recommendations: [],
    agent_trace: [],
    token_usage: {},
    retrieved_context_count: 0,
  }
}

function App() {
  const [question, setQuestion] = useState(starterQuestions[0])
  const [employeeContext, setEmployeeContext] = useState('')
  const [category, setCategory] = useState('')
  const [framework, setFramework] = useState('')
  const [answer, setAnswer] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingStatus, setLoadingStatus] = useState('')
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [uploadMessage, setUploadMessage] = useState('')

  const statusIcon = useMemo(() => {
    if (!answer) return <FiShield />
    if (answer.risk_level === 'low') return <FiCheckCircle />
    return <FiAlertTriangle />
  }, [answer])

  async function submitQuery(event) {
    event.preventDefault()
    setLoading(true)
    setLoadingStatus('Preparing your policy question...')
    setError('')
    setAnswer(createStreamingAnswer(''))
    try {
      const response = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
        question,
        employee_context: employeeContext || null,
        filters: {
          category: category || null,
          framework: framework || null,
        },
        top_k: 10,
        }),
      })
      if (!response.ok || !response.body) {
        const message = await response.text()
        throw new Error(message || 'Unable to get document guidance.')
      }
      setLoadingStatus('Waiting for OpenRouter to finish generating...')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let streamedText = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.trim()) continue
          const event = JSON.parse(line)
          if (event.type === 'heartbeat') {
            const seconds = Math.max(1, Math.round((event.elapsed_ms || 0) / 1000))
            setLoadingStatus(`Still waiting for OpenRouter... ${seconds}s`)
          }
          if (event.type === 'chunk') {
            streamedText += event.text
            setAnswer((current) => ({ ...(current || createStreamingAnswer()), answer: streamedText }))
          }
          if (event.type === 'error') {
            throw new Error(event.detail || 'Unable to get document guidance.')
          }
          if (event.type === 'final') {
            setAnswer(event.answer)
            setLoadingStatus('')
          }
        }
      }
    } catch (err) {
      setAnswer(null)
      setError(err.response?.data?.detail || err.message || 'Unable to get document guidance.')
    } finally {
      setLoading(false)
      setLoadingStatus('')
    }
  }

  async function uploadPolicy(event) {
    const files = Array.from(event.target.files || [])
    if (!files.length) return
    setUploading(true)
    setUploadMessage('')
    setError('')
    const formData = new FormData()
    files.forEach((file) => formData.append('files', file))
    formData.append('category', category || 'internal')
    formData.append('framework', framework || 'internal')
    formData.append('authority', '7')
    try {
      const response = await axios.post(`${API_BASE}/policies/upload-batch`, formData)
      const names = files.map((file) => file.name).join(', ')
      setUploadMessage(
        `Indexed ${response.data.total_chunks_indexed} chunks from ${response.data.total_documents} document${response.data.total_documents === 1 ? '' : 's'}: ${names}.`,
      )
    } catch (err) {
      setError(err.response?.data?.detail || 'Document upload failed.')
    } finally {
      setUploading(false)
      event.target.value = ''
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <FiShield aria-hidden="true" />
          <div>
            <span>PolicyIQ</span>
            <small>Compliance intelligence</small>
          </div>
        </div>

        <label className="upload-control">
          <FiUpload aria-hidden="true" />
          <span>{uploading ? 'Indexing...' : 'Upload documents'}</span>
          <input type="file" accept=".pdf,.txt,.md" onChange={uploadPolicy} disabled={uploading} multiple />
        </label>

        {uploadMessage && <p className="notice">{uploadMessage}</p>}

        <div className="filter-block">
          <label>
            Category
            <select value={category} onChange={(event) => setCategory(event.target.value)}>
              <option value="">All categories</option>
              <option value="hr">HR</option>
              <option value="cybersecurity">Cybersecurity</option>
              <option value="data_governance">Data governance</option>
              <option value="privacy">Privacy</option>
              <option value="remote_work">Remote work</option>
              <option value="incident_response">Incident response</option>
            </select>
          </label>
          <label>
            Framework
            <select value={framework} onChange={(event) => setFramework(event.target.value)}>
              <option value="">All frameworks</option>
              <option value="internal">Internal</option>
              <option value="nist">NIST</option>
              <option value="gdpr">GDPR</option>
              <option value="iso27001">ISO 27001</option>
              <option value="sox">SOX</option>
              <option value="hipaa">HIPAA</option>
            </select>
          </label>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>AI Policy Compliance Assistant</h1>
            <p>Ask natural-language questions across all uploaded documents and get grounded answers with references.</p>
          </div>
          <div className={`risk-badge ${answer?.risk_level || 'neutral'}`}>
            {statusIcon}
            <span>{answer ? answer.risk_level : 'ready'}</span>
          </div>
        </header>

        <form className="query-panel" onSubmit={submitQuery}>
          <div className="question-row">
            <FiSearch aria-hidden="true" />
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows="3" />
          </div>
          <textarea
            className="context-input"
            value={employeeContext}
            onChange={(event) => setEmployeeContext(event.target.value)}
            rows="2"
            placeholder="Optional context: region, role, project, data type, urgency"
          />
          <div className="examples">
            {starterQuestions.map((item) => (
              <button key={item} type="button" onClick={() => setQuestion(item)}>
                {item}
              </button>
            ))}
          </div>
          <button className="primary-action" type="submit" disabled={loading}>
            <FiShield aria-hidden="true" />
            {loading ? 'Answering...' : 'Ask documents'}
          </button>
        </form>

        {error && <div className="error-banner">{error}</div>}

        {loading && !answer?.answer && (
          <section className="loading-panel" aria-live="polite">
            <div className="loading-spinner" aria-hidden="true" />
            <div>
              <strong>Generating answer</strong>
              <p>{loadingStatus || 'Waiting for OpenRouter...'}</p>
            </div>
          </section>
        )}

        {answer && (
          <section className="results-grid">
            <article className="answer-panel">
              <div className="panel-heading">
                <FiFileText aria-hidden="true" />
                <h2>Guidance</h2>
              </div>
              <p>{answer.answer}</p>
              <div className="score-row">
                <span>Compliance score</span>
                <strong>{answer.compliance_score}/100</strong>
              </div>
              <progress value={answer.compliance_score} max="100" />
              <div className="recommendations">
                {answer.recommendations.map((item) => (
                  <p key={item}>{item}</p>
                ))}
              </div>
            </article>

            <article className="facts-panel">
              <div className="fact">
                <span>Status</span>
                <strong>{answer.compliance_status.replaceAll('_', ' ')}</strong>
              </div>
              <div className="fact">
                <span>Escalation</span>
                <strong>{answer.escalation_required ? 'Required' : 'Not required'}</strong>
              </div>
              <div className="fact">
                <span>Context</span>
                <strong>{answer.retrieved_context_count} sections</strong>
              </div>
              <div className="fact">
                <span>Latency</span>
                <strong>{answer.token_usage?.latency_ms || 0} ms</strong>
              </div>
            </article>

            <article className="citations-panel">
              <div className="panel-heading">
                <FiDatabase aria-hidden="true" />
                <h2>Citations</h2>
              </div>
              {answer.citations.length === 0 && <p>No policy sources were retrieved.</p>}
              {answer.citations.map((citation) => (
                <div className="citation" key={`${citation.source}-${citation.clause_id}-${citation.page}`}>
                  <strong>{citation.title || citation.source}</strong>
                  <span>
                    {citation.framework} / {citation.category}
                    {citation.clause_id ? ` / ${citation.clause_id}` : ''}
                  </span>
                  <p>{citation.snippet}</p>
                </div>
              ))}
            </article>

            <article className="trace-panel">
              <h2>Agent Trace</h2>
              {answer.agent_trace.map((trace) => (
                <div className="trace" key={trace.agent}>
                  <strong>{trace.agent}</strong>
                  <p>{trace.summary}</p>
                </div>
              ))}
            </article>
          </section>
        )}
      </section>
    </main>
  )
}

export default App
