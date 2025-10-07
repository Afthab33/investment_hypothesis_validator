import { useState } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [expandedCitation, setExpandedCitation] = useState(null)
  const [highlightedCitation, setHighlightedCitation] = useState(null)
  const [currentStep, setCurrentStep] = useState(0)
  const [showArchitecture, setShowArchitecture] = useState(false)

  const exampleQueries = [
    "Is Tesla's gross margin improving?",
    "Are Tesla's energy storage sales increasing?",
    "Is Tesla facing production challenges?"
  ]

  const processingSteps = [
    { icon: '🔍', text: 'Analyzing your question...' },
    { icon: '📄', text: 'Searching SEC filings...' },
    { icon: '📞', text: 'Reading earnings call transcripts...' },
    { icon: '💬', text: 'Analyzing Bloomberg chats...' },
    { icon: '⚖️', text: 'Evaluating supporting evidence...' },
    { icon: '🔄', text: 'Checking contradicting perspectives...' },
    { icon: '✨', text: 'Synthesizing final verdict...' }
  ]

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)
    setExpandedCitation(null)
    setHighlightedCitation(null)
    setCurrentStep(0)

    // Simulate progress through steps
    const stepInterval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev < processingSteps.length - 1) {
          return prev + 1
        }
        return prev
      })
    }, 3500) // Change step every 3.5 seconds

    try {
      const response = await axios.post(`${API_URL}/validate`, {
        query: query.trim()
      })
      clearInterval(stepInterval)
      setResult(response.data)
    } catch (err) {
      clearInterval(stepInterval)
      setError(err.response?.data?.detail || 'Error processing query. Please try again.')
    } finally {
      setLoading(false)
      setCurrentStep(0)
    }
  }

  const getVerdictColor = (verdict) => {
    switch (verdict) {
      case 'Support':
        return 'bg-green-100 text-green-800 border-green-300'
      case 'Refute':
        return 'bg-red-100 text-red-800 border-red-300'
      case 'Inconclusive':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300'
    }
  }

  const getVerdictIcon = (verdict) => {
    switch (verdict) {
      case 'Support':
        return '✅'
      case 'Refute':
        return '❌'
      case 'Inconclusive':
        return '❔'
      default:
        return '📊'
    }
  }

  const getSourceTypeColor = (sourceType) => {
    switch (sourceType) {
      case 'filing':
        return 'bg-blue-100 text-blue-700 border-blue-300 hover:bg-blue-200'
      case 'call':
        return 'bg-green-100 text-green-700 border-green-300 hover:bg-green-200'
      case 'chat':
        return 'bg-purple-100 text-purple-700 border-purple-300 hover:bg-purple-200'
      default:
        return 'bg-gray-100 text-gray-700 border-gray-300 hover:bg-gray-200'
    }
  }

  const getSourceIcon = (sourceType) => {
    switch (sourceType) {
      case 'filing':
        return '📄'
      case 'call':
        return '📞'
      case 'chat':
        return '💬'
      default:
        return '📌'
    }
  }

  // Process claims to add inline citation superscripts
  const processClaimWithCitations = (claim, citations) => {
    if (!citations || citations.length === 0) return claim

    const parts = []
    let lastIndex = 0

    // Find citation patterns like [filing:TSLA_2024Q3:general]
    const citationPattern = /\[([^\]]+)\]/g
    let match

    while ((match = citationPattern.exec(claim)) !== null) {
      const citationRaw = match[0]
      const citationIndex = citations.findIndex(c => c.raw === citationRaw)

      if (citationIndex !== -1) {
        const citation = citations[citationIndex]

        // Add text before citation
        if (match.index > lastIndex) {
          parts.push({
            type: 'text',
            content: claim.slice(lastIndex, match.index)
          })
        }

        // Add citation reference
        parts.push({
          type: 'citation',
          citation: citation,
          index: citationIndex + 1
        })

        lastIndex = match.index + match[0].length
      }
    }

    // Add remaining text
    if (lastIndex < claim.length) {
      parts.push({
        type: 'text',
        content: claim.slice(lastIndex)
      })
    }

    return parts
  }

  const renderClaimWithCitations = (claim, citations, type) => {
    const parts = processClaimWithCitations(claim, citations)

    if (!Array.isArray(parts)) {
      return <span>{claim}</span>
    }

    return (
      <span>
        {parts.map((part, idx) => {
          if (part.type === 'text') {
            return <span key={idx}>{part.content}</span>
          } else {
            const isHighlighted = highlightedCitation === `${type}-${part.citation.id}`
            return (
              <sup
                key={idx}
                className={`ml-0.5 cursor-pointer transition-all ${
                  isHighlighted
                    ? 'text-lg font-bold text-blue-600'
                    : 'text-blue-500 hover:text-blue-700 hover:font-semibold'
                }`}
                onClick={() => {
                  setHighlightedCitation(`${type}-${part.citation.id}`)
                  setExpandedCitation(`${type}-${part.citation.id}`)
                  // Scroll to reference
                  const element = document.getElementById(`ref-${type}-${part.citation.id}`)
                  if (element) {
                    element.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
                  }
                }}
              >
                [{part.index}]
              </sup>
            )
          }
        })}
      </span>
    )
  }

  // Group citations by fiscal period for timeline
  const groupCitationsByPeriod = (citations) => {
    const grouped = {}
    citations.forEach(citation => {
      const period = citation.fiscal_period || 'Unknown'
      if (!grouped[period]) {
        grouped[period] = []
      }
      grouped[period].push(citation)
    })
    return grouped
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12 relative">
          {/* How This Works Button */}
          <button
            onClick={() => setShowArchitecture(true)}
            className="absolute top-0 right-0 inline-flex items-center px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 hover:border-gray-400 transition-all shadow-sm"
          >
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            How This Works
          </button>

          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Investment Hypothesis Validator
          </h1>
          <p className="text-lg text-gray-600 mb-3">
            AI-powered evidence-based validation using LangGraph + AWS Bedrock
          </p>
          <div className="max-w-3xl mx-auto">
            <p className="text-sm text-gray-700 mb-2">
              Validate investment hypotheses by analyzing multiple sources: SEC 10-Q/10-K filings,
              earnings call transcripts, and Bloomberg chat messages. Get cited, verifiable evidence
              for both supporting and refuting perspectives.
            </p>
            <div className="inline-flex items-center px-4 py-2 bg-yellow-50 border border-yellow-200 rounded-lg">
              <svg className="w-4 h-4 text-yellow-600 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
              <span className="text-xs text-yellow-800 font-medium">
                Prototype: Currently loaded with Tesla (TSLA) data only
              </span>
            </div>
          </div>
        </div>

        {/* Query Form */}
        <div className="bg-white rounded-lg shadow-xl p-8 mb-8">
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
                Enter your investment hypothesis
              </label>
              <textarea
                id="query"
                rows="3"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                placeholder="e.g., Is Tesla's gross margin improving?"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={loading}
              />
            </div>

            {/* Example Queries */}
            <div className="mb-4">
              <p className="text-sm text-gray-600 mb-2">Try these examples:</p>
              <div className="flex flex-wrap gap-2">
                {exampleQueries.map((example, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setQuery(example)}
                    className="text-sm px-3 py-1 bg-blue-50 text-blue-700 rounded-full hover:bg-blue-100 transition"
                    disabled={loading}
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>

            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="w-full bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition duration-200"
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Analyzing... (~25 seconds)
                </span>
              ) : (
                'Validate Hypothesis'
              )}
            </button>
          </form>
        </div>

        {/* Processing Steps Display */}
        {loading && (
          <div className="bg-white rounded-lg shadow-xl p-8 mb-8">
            <div className="text-center">
              <div className="mb-6">
                <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4">
                  <span className="text-3xl animate-pulse">{processingSteps[currentStep].icon}</span>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  {processingSteps[currentStep].text}
                </h3>
                <p className="text-sm text-gray-500">This may take up to 25 seconds</p>
              </div>

              {/* Progress Bar */}
              <div className="w-full bg-gray-200 rounded-full h-2 mb-4">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${((currentStep + 1) / processingSteps.length) * 100}%` }}
                ></div>
              </div>

              {/* Step indicators */}
              <div className="flex justify-between text-xs text-gray-500 px-1">
                {processingSteps.map((step, idx) => (
                  <div
                    key={idx}
                    className={`flex flex-col items-center ${idx <= currentStep ? 'text-blue-600' : 'text-gray-400'}`}
                  >
                    <div className={`w-2 h-2 rounded-full mb-1 ${idx <= currentStep ? 'bg-blue-600' : 'bg-gray-300'}`}></div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-8">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Results Display */}
        {result && (
          <div className="space-y-6">
            {/* Verdict Card */}
            <div className="bg-white rounded-lg shadow-xl p-8">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-4">
                  <span className="text-4xl">{getVerdictIcon(result.verdict)}</span>
                  <div>
                    <span className={`inline-flex items-center px-4 py-2 rounded-full text-lg font-semibold border-2 ${getVerdictColor(result.verdict)}`}>
                      {result.verdict}
                    </span>
                    <p className="text-sm text-gray-500 mt-2">
                      Confidence: {(result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
                <div className="text-right text-sm text-gray-500">
                  <div>Time: {result.execution_time_seconds}s</div>
                </div>
              </div>

              {/* Rationale */}
              {result.rationale && result.rationale.length > 0 && (
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">📝 Key Findings</h3>
                  <ul className="space-y-2">
                    {result.rationale.map((point, idx) => (
                      <li key={idx} className="text-gray-700 flex">
                        <span className="mr-2 text-blue-600">•</span>
                        <span>{point}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Counterpoints */}
              {result.counterpoints && result.counterpoints.length > 0 && (
                <div className="border-t pt-4">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">⚠️ Important Considerations</h3>
                  <ul className="space-y-2">
                    {result.counterpoints.map((point, idx) => (
                      <li key={idx} className="text-gray-700 flex">
                        <span className="mr-2 text-yellow-600">•</span>
                        <span>{point}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Evidence Cards with Citations */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* PRO Evidence */}
              {result.pro_evidence && (
                <div className="bg-white rounded-lg shadow-xl overflow-hidden">
                  <div className="bg-green-50 p-6 border-b border-green-200">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold text-green-700">✅ Supporting Evidence</h3>
                      <span className="text-sm text-gray-600">
                        {(result.pro_evidence.confidence * 100).toFixed(1)}% confidence
                      </span>
                    </div>
                  </div>

                  <div className="p-6">
                    <div className="space-y-4 mb-6">
                      {result.pro_evidence.claims.map((claim, idx) => (
                        <div key={idx} className="border-l-4 border-green-300 pl-4 py-2">
                          <p className="text-sm text-gray-700 leading-relaxed">
                            {renderClaimWithCitations(claim, result.pro_evidence.citations, 'pro')}
                          </p>
                        </div>
                      ))}
                    </div>

                    {/* References */}
                    {result.pro_evidence.citations && result.pro_evidence.citations.length > 0 && (
                      <div className="border-t pt-4">
                        <h4 className="text-sm font-semibold text-gray-700 mb-3">📚 References</h4>
                        <div className="space-y-2">
                          {result.pro_evidence.citations.map((citation, idx) => {
                            const refId = `ref-pro-${citation.id}`
                            const isExpanded = expandedCitation === `pro-${citation.id}`
                            const isHighlighted = highlightedCitation === `pro-${citation.id}`

                            return (
                              <div
                                key={idx}
                                id={refId}
                                className={`text-xs border rounded-lg p-3 transition-all cursor-pointer ${
                                  isHighlighted
                                    ? 'ring-2 ring-blue-500 bg-blue-50'
                                    : getSourceTypeColor(citation.source_type)
                                }`}
                                onClick={() => {
                                  setExpandedCitation(isExpanded ? null : `pro-${citation.id}`)
                                  setHighlightedCitation(`pro-${citation.id}`)
                                }}
                              >
                                <div className="flex items-start justify-between">
                                  <div className="flex items-start space-x-2 flex-1">
                                    <span className="text-sm">{getSourceIcon(citation.source_type)}</span>
                                    <div className="flex-1">
                                      <span className="font-semibold">[{idx + 1}]</span>
                                      <span className="ml-2">{citation.display_text}</span>
                                      {citation.speaker_role && (
                                        <span className="ml-2 px-2 py-0.5 bg-white rounded text-xs font-medium">
                                          {citation.speaker_role}
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                  <svg
                                    className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                  </svg>
                                </div>
                                {isExpanded && citation.full_text && (
                                  <div className="mt-3 pt-3 border-t bg-white rounded p-3">
                                    <p className="text-gray-700 italic text-xs leading-relaxed">
                                      "{citation.full_text}"
                                    </p>
                                  </div>
                                )}
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* CON Evidence */}
              {result.con_evidence && (
                <div className="bg-white rounded-lg shadow-xl overflow-hidden">
                  <div className="bg-red-50 p-6 border-b border-red-200">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold text-red-700">❌ Refuting Evidence</h3>
                      <span className="text-sm text-gray-600">
                        {(result.con_evidence.confidence * 100).toFixed(1)}% confidence
                      </span>
                    </div>
                  </div>

                  <div className="p-6">
                    <div className="space-y-4 mb-6">
                      {result.con_evidence.claims.map((claim, idx) => (
                        <div key={idx} className="border-l-4 border-red-300 pl-4 py-2">
                          <p className="text-sm text-gray-700 leading-relaxed">
                            {renderClaimWithCitations(claim, result.con_evidence.citations, 'con')}
                          </p>
                        </div>
                      ))}
                    </div>

                    {/* References */}
                    {result.con_evidence.citations && result.con_evidence.citations.length > 0 && (
                      <div className="border-t pt-4">
                        <h4 className="text-sm font-semibold text-gray-700 mb-3">📚 References</h4>
                        <div className="space-y-2">
                          {result.con_evidence.citations.map((citation, idx) => {
                            const refId = `ref-con-${citation.id}`
                            const isExpanded = expandedCitation === `con-${citation.id}`
                            const isHighlighted = highlightedCitation === `con-${citation.id}`

                            return (
                              <div
                                key={idx}
                                id={refId}
                                className={`text-xs border rounded-lg p-3 transition-all cursor-pointer ${
                                  isHighlighted
                                    ? 'ring-2 ring-blue-500 bg-blue-50'
                                    : getSourceTypeColor(citation.source_type)
                                }`}
                                onClick={() => {
                                  setExpandedCitation(isExpanded ? null : `con-${citation.id}`)
                                  setHighlightedCitation(`con-${citation.id}`)
                                }}
                              >
                                <div className="flex items-start justify-between">
                                  <div className="flex items-start space-x-2 flex-1">
                                    <span className="text-sm">{getSourceIcon(citation.source_type)}</span>
                                    <div className="flex-1">
                                      <span className="font-semibold">[{idx + 1}]</span>
                                      <span className="ml-2">{citation.display_text}</span>
                                      {citation.speaker_role && (
                                        <span className="ml-2 px-2 py-0.5 bg-white rounded text-xs font-medium">
                                          {citation.speaker_role}
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                  <svg
                                    className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                  </svg>
                                </div>
                                {isExpanded && citation.full_text && (
                                  <div className="mt-3 pt-3 border-t bg-white rounded p-3">
                                    <p className="text-gray-700 italic text-xs leading-relaxed">
                                      "{citation.full_text}"
                                    </p>
                                  </div>
                                )}
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Tone Delta */}
            {result.tone_delta && (
              <div className="bg-white rounded-lg shadow-xl p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">🎭 Sentiment Shift</h3>
                <p className="text-gray-700">{result.tone_delta}</p>
              </div>
            )}
          </div>
        )}

        {/* Architecture Modal */}
        {showArchitecture && (
          <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-lg shadow-2xl max-w-5xl w-full max-h-[90vh] overflow-auto">
              <div className="sticky top-0 bg-white border-b px-6 py-4 flex items-center justify-between">
                <h2 className="text-2xl font-bold text-gray-900">How This Works</h2>
                <button
                  onClick={() => setShowArchitecture(false)}
                  className="text-gray-400 hover:text-gray-600 transition"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="p-6">
                <p className="text-gray-700 mb-6">
                  The Investment Hypothesis Validator uses a sophisticated AI pipeline to analyze your questions
                  against multiple financial data sources and provide evidence-based answers.
                </p>

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
                  <h3 className="font-semibold text-blue-900 mb-2">📊 System Architecture</h3>
                  <p className="text-sm text-blue-800">
                    This diagram shows the complete flow from your question through retrieval, reasoning, and verdict synthesis.
                  </p>
                </div>

                {/* Architecture Diagram */}
                <div className="bg-white p-8 rounded-lg border">
                  <div className="flex flex-col items-center space-y-4">
                    {/* Step 1 */}
                    <div className="w-full max-w-md">
                      <div className="bg-blue-100 border-2 border-blue-400 rounded-lg p-4 text-center">
                        <div className="text-2xl mb-2">💬</div>
                        <div className="font-semibold">Your Question</div>
                      </div>
                    </div>
                    <div className="text-2xl text-gray-400">↓</div>

                    {/* Step 2 */}
                    <div className="w-full max-w-lg">
                      <div className="bg-gray-100 border-2 border-gray-400 rounded-lg p-4 text-center">
                        <div className="text-2xl mb-2">📝</div>
                        <div className="font-semibold">Question Analysis</div>
                        <div className="text-sm text-gray-600 mt-1">Extract ticker, metrics, time period</div>
                      </div>
                    </div>
                    <div className="text-2xl text-gray-400">↓</div>

                    {/* Step 3 */}
                    <div className="w-full max-w-xl">
                      <div className="bg-gray-100 border-2 border-gray-400 rounded-lg p-4 text-center">
                        <div className="text-2xl mb-2">🔍</div>
                        <div className="font-semibold">Multi-Source Retrieval</div>
                        <div className="text-sm text-gray-600 mt-1">SEC Filings • Earnings Calls • Bloomberg Chats</div>
                      </div>
                    </div>
                    <div className="text-2xl text-gray-400">↓</div>

                    {/* Step 4 */}
                    <div className="w-full max-w-lg">
                      <div className="bg-gray-100 border-2 border-gray-400 rounded-lg p-4 text-center">
                        <div className="text-2xl mb-2">📊</div>
                        <div className="font-semibold">Rank & Filter</div>
                        <div className="text-sm text-gray-600 mt-1">Score by relevance & recency</div>
                      </div>
                    </div>
                    <div className="text-2xl text-gray-400">↓</div>

                    {/* Step 5 - Dual Evidence */}
                    <div className="w-full grid grid-cols-2 gap-4 max-w-2xl">
                      <div className="bg-green-100 border-2 border-green-400 rounded-lg p-4 text-center">
                        <div className="text-2xl mb-2">✅</div>
                        <div className="font-semibold">Supporting</div>
                        <div className="text-sm text-gray-600 mt-1">Find PRO evidence</div>
                      </div>
                      <div className="bg-red-100 border-2 border-red-400 rounded-lg p-4 text-center">
                        <div className="text-2xl mb-2">❌</div>
                        <div className="font-semibold">Refuting</div>
                        <div className="text-sm text-gray-600 mt-1">Find CON evidence</div>
                      </div>
                    </div>
                    <div className="text-2xl text-gray-400">↓</div>

                    {/* Step 6 */}
                    <div className="w-full max-w-lg">
                      <div className="bg-yellow-100 border-2 border-yellow-400 rounded-lg p-4 text-center">
                        <div className="text-2xl mb-2">⚖️</div>
                        <div className="font-semibold">Synthesize Verdict</div>
                        <div className="text-sm text-gray-600 mt-1">Compare + Confidence Score</div>
                      </div>
                    </div>
                    <div className="text-2xl text-gray-400">↓</div>

                    {/* Step 7 */}
                    <div className="w-full max-w-md">
                      <div className="bg-green-100 border-2 border-green-400 rounded-lg p-4 text-center">
                        <div className="text-2xl mb-2">📄</div>
                        <div className="font-semibold">Final Report with Citations</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-6 space-y-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h4 className="font-semibold text-gray-900 mb-2">🔍 How It Works:</h4>
                    <ol className="list-decimal list-inside space-y-2 text-sm text-gray-700">
                      <li><strong>Question Analysis:</strong> Extracts company ticker, financial metrics, and time periods from your question</li>
                      <li><strong>Multi-Source Retrieval:</strong> Searches across SEC filings, earnings call transcripts, and Bloomberg chat messages</li>
                      <li><strong>Rank & Filter:</strong> Scores results by relevance and recency, keeping the most useful information</li>
                      <li><strong>Dual Evidence Search:</strong> Claude AI independently finds both supporting AND refuting evidence</li>
                      <li><strong>Verdict Synthesis:</strong> Compares both perspectives and generates a balanced verdict with confidence score</li>
                      <li><strong>Citation Report:</strong> Delivers final answer where every claim links back to source documents</li>
                    </ol>
                  </div>

                  <div className="bg-green-50 rounded-lg p-4">
                    <h4 className="font-semibold text-green-900 mb-2">✨ Why It's Trustworthy:</h4>
                    <ul className="list-disc list-inside space-y-1 text-sm text-green-800">
                      <li><strong>Evidence-Bound:</strong> Every claim must be cited from actual documents - no hallucinations</li>
                      <li><strong>Dual Perspective:</strong> Shows both supporting and refuting evidence - not biased to one side</li>
                      <li><strong>Source Diversity:</strong> Combines official filings, executive statements, and market analysis</li>
                      <li><strong>Transparent Confidence:</strong> Honest about uncertainty when evidence is mixed or limited</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="mt-12 text-center text-sm text-gray-600">
          <p>Built with LangGraph • AWS Bedrock • OpenSearch Serverless</p>
          <p className="mt-2">All claims are cited and verifiable • Evidence-bound AI</p>
        </div>
      </div>
    </div>
  )
}

export default App
