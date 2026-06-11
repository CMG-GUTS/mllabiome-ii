'use client'

import { useState, useEffect, useRef, DragEvent, KeyboardEvent } from 'react'
import Link from 'next/link'
import styles from './inference.module.css'
import ThemeToggle from '../../components/ThemeToggle'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Model {
  model_id: string
  display_name: string
  description: string
  target: string
  n_models: number
  calibrated: boolean
  submission_filename?: string
}

interface Prediction {
  sample_id: string
  prediction: number
  label: string
  probability: number
  confidence: number
}

interface FeatureExplanation {
  feature: string
  importance: number
  std: number
  n_models: number
  abs_importance?: number
  support?: number
  direction?: 'toward' | 'against' | string
  method?: string
  value?: number
  baseline?: number
}

interface InteractionEdge {
  source: string
  target: string
  strength: number
  abs_strength?: number
  support?: number
  direction?: 'synergy' | 'redundancy' | string
  method?: string
}

interface SampleExplanation {
  sample_id: string
  prediction: number
  label: string
  probability: number
  top_features: FeatureExplanation[]
  shap_features?: FeatureExplanation[]
  lime_features?: FeatureExplanation[]
  interactions?: InteractionEdge[]
  interaction_method?: string
  n_models_explained: number
  explanation_method?: string
  explanation_basis?: string
}

interface ExplainabilityStatus {
  method: string
  method_label: string
  basis: string
  dependencies: Record<string, { available: boolean; module: string }>
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

interface ChatConfig {
  provider: string
  ollama_url: string
  default_model: string
  available_models: string[]
  status: string
  detail?: string | null
}

export default function InferencePage() {
  const [models, setModels] = useState<Model[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [dataFile, setDataFile] = useState<File | null>(null)
  const [results, setResults] = useState<Prediction[] | null>(null)
  const [explanations, setExplanations] = useState<SampleExplanation[] | null>(null)
  const [selectedSample, setSelectedSample] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [explaining, setExplaining] = useState(false)
  const [showShap, setShowShap] = useState(true)
  const [showLime, setShowLime] = useState(true)
  const [downloadingSubmission, setDownloadingSubmission] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [explainabilityStatus, setExplainabilityStatus] = useState<ExplainabilityStatus | null>(null)
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [chatModel, setChatModel] = useState('gemma4:31b-cloud')
  const [chatModelOptions, setChatModelOptions] = useState<string[]>(['gemma4:31b-cloud'])
  const [chatConfigStatus, setChatConfigStatus] = useState<string>('unknown')
  const [chatConfigDetail, setChatConfigDetail] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const chatEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchModels()
    fetchChatConfig()
    fetchExplainabilityStatus()
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages])

  const fetchModels = async () => {
    try {
      const response = await fetch(`${API_URL}/api/inference/models`)
      if (response.ok) {
        const data = await response.json()
        setModels(data.models)
        if (data.models.length > 0) {
          setSelectedModel(data.models[0].model_id)
        }
      }
    } catch (err) {
      console.error('Failed to fetch models:', err)
    }
  }

  const fetchChatConfig = async () => {
    try {
      const response = await fetch(`${API_URL}/api/chat/config`)
      if (response.ok) {
        const data: ChatConfig = await response.json()
        setChatModel(data.default_model || 'gemma4:31b-cloud')
        const opts = data.available_models && data.available_models.length > 0
          ? data.available_models
          : [data.default_model || 'gemma4:31b-cloud']
        setChatModelOptions(Array.from(new Set(opts.filter(Boolean))))
        setChatConfigStatus(data.status || 'unknown')
        setChatConfigDetail(data.detail || null)
      } else {
        setChatConfigStatus('unavailable')
        setChatConfigDetail('Could not read Ollama assistant configuration from the backend.')
      }
    } catch (err) {
      console.error('Failed to fetch chat config:', err)
      setChatConfigStatus('unavailable')
      setChatConfigDetail('Could not connect to the backend chat configuration endpoint.')
    }
  }


  const fetchExplainabilityStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/inference/explainability/status`)
      if (response.ok) {
        const data: ExplainabilityStatus = await response.json()
        setExplainabilityStatus(data)
      }
    } catch (err) {
      console.warn('Failed to fetch explainability status:', err)
    }
  }

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    
    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      const file = files[0]
      if (file.name.endsWith('.tsv') || file.name.endsWith('.csv') || file.name.endsWith('.txt')) {
        setDataFile(file)
        setError(null)
      } else {
        setError('Please upload a TSV, CSV, or TXT file')
      }
    }
  }

  const handleRunInference = async () => {
    if (!dataFile || !selectedModel) return
    
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('data', dataFile)
      
      const response = await fetch(
        `${API_URL}/api/inference/predict/batch?model_id=${selectedModel}&sample_id_column=sample_id`,
        {
          method: 'POST',
          body: formData
        }
      )
      
      if (response.ok) {
        const data = await response.json()
        setResults(data.predictions)
        setExplanations(null)
        setSelectedSample(data.predictions?.[0]?.sample_id || null)
      } else {
        const errData = await response.json()
        setError(errData.detail || 'Inference failed')
      }
    } catch (err) {
      setError('Failed to connect to server')
    } finally {
      setLoading(false)
    }
  }

  const handleDownloadSubmission = async () => {
    if (!dataFile || !selectedModel) return

    setDownloadingSubmission(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('data', dataFile)

      const response = await fetch(
        `${API_URL}/api/inference/predict/submission?model_id=${selectedModel}&sample_id_column=sample_id`,
        {
          method: 'POST',
          body: formData
        }
      )

      if (!response.ok) {
        let detail = 'Failed to create benchmark CSV'
        try {
          const errData = await response.json()
          if (errData?.detail) detail = errData.detail
        } catch (_) {
          // Keep fallback detail.
        }
        setError(detail)
        return
      }

      const blob = await response.blob()
      const disposition = response.headers.get('content-disposition') || ''
      const match = disposition.match(/filename="?([^";]+)"?/i)
      const model = models.find(m => m.model_id === selectedModel)
      const filename = match?.[1] || model?.submission_filename || `${selectedModel}.csv`
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      setError('Failed to download benchmark CSV')
    } finally {
      setDownloadingSubmission(false)
    }
  }

  const upsertExplanation = (incoming: SampleExplanation[]) => {
    if (!incoming.length) return
    setExplanations(prev => {
      const bySample = new Map<string, SampleExplanation>()
      ;(prev || []).forEach(exp => bySample.set(exp.sample_id, exp))
      incoming.forEach(exp => bySample.set(exp.sample_id, exp))
      return Array.from(bySample.values())
    })
  }

  const fetchExplanationForSample = async (sampleId?: string | null) => {
    if (!dataFile || !selectedModel) return

    setExplaining(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('data', dataFile)
      const params = new URLSearchParams({
        model_id: selectedModel,
        sample_id_column: 'sample_id',
        num_features: '12',
        num_samples: '128'
      })
      if (sampleId) params.set('sample_id', sampleId)

      const response = await fetch(`${API_URL}/api/inference/explain?${params.toString()}`, {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        const data = await response.json()
        const nextExplanations: SampleExplanation[] = data.explanations || []
        upsertExplanation(nextExplanations)
        if (!results || results.length === 0) {
          setResults(nextExplanations.map((exp: SampleExplanation) => ({
            sample_id: exp.sample_id,
            prediction: exp.prediction,
            label: exp.label,
            probability: exp.probability,
            confidence: Math.abs(exp.probability - 0.5) * 2
          })))
        }
        if (nextExplanations.length > 0) {
          setSelectedSample(nextExplanations[0].sample_id)
        }
      } else {
        const errData = await response.json()
        setError(errData.detail || 'Explanation failed')
      }
    } catch (err) {
      setError('Failed to generate explanations')
    } finally {
      setExplaining(false)
    }
  }

  const handleExplain = async () => {
    const sampleId = selectedSample || results?.[0]?.sample_id || null
    await fetchExplanationForSample(sampleId)
  }

  const handleSampleChange = (sampleId: string) => {
    setSelectedSample(sampleId)
    const cached = explanations?.some(exp => exp.sample_id === sampleId)
    if (!cached) {
      fetchExplanationForSample(sampleId)
    }
  }

  const toggleShap = () => {
    if (showShap && !showLime) return
    setShowShap(prev => !prev)
  }

  const toggleLime = () => {
    if (showLime && !showShap) return
    setShowLime(prev => !prev)
  }

  const buildContext = () => {
    let context = ''
    const model = models.find(m => m.model_id === selectedModel)
    
    if (model) {
      context += `Analysis task: ${model.display_name}\nDescription: ${model.description}\n`
    }
    
    if (results && results.length > 0) {
      context += `\nPrediction results:\n`
      results.forEach(r => {
        context += `- ${r.sample_id}: ${r.label} (${(r.probability * 100).toFixed(1)}% probability)\n`
      })
    }
    
    if (explanations && explanations.length > 0) {
      context += `\nFeature attribution - top contributing taxa:\n`
      explanations.forEach(exp => {
        context += `\nSample ${exp.sample_id} (${exp.label}, ${(exp.probability * 100).toFixed(1)}%):\n`
        context += `  Feature importances (absolute magnitude):\n`
        exp.top_features.slice(0, 5).forEach((f, i) => {
          context += `    ${i + 1}. ${f.feature}: ${Math.abs(f.importance).toFixed(4)}\n`
        })
        context += `  Contribution direction (positive = toward prediction, negative = against):\n`
        exp.top_features.slice(0, 5).forEach((f, i) => {
          const direction = f.importance >= 0 ? 'pushes TOWARD' : 'pushes AGAINST'
          context += `    ${i + 1}. ${f.feature}: ${f.importance >= 0 ? '+' : ''}${f.importance.toFixed(4)} (${direction} ${exp.label})\n`
        })
      })
    }
    
    return context
  }

  const handleSendChat = async () => {
    if (!chatInput.trim() || chatLoading) return
    
    const userMessage = chatInput.trim()
    setChatInput('')
    setChatMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setChatLoading(true)
    
    try {
      const context = buildContext()
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          context: context,
          history: chatMessages.slice(-10),
          model: chatModel.trim() || 'gemma4:31b-cloud'
        })
      })
      
      if (response.ok) {
        const data = await response.json()
        setChatMessages(prev => [...prev, { role: 'assistant', content: data.response }])
      } else {
        let detail = `Assistant request failed with HTTP ${response.status}`
        try {
          const errData = await response.json()
          if (errData?.detail) detail = errData.detail
        } catch (_) {
          // Keep the HTTP status fallback.
        }
        setChatMessages(prev => [...prev, { role: 'assistant', content: `Assistant error: ${detail}` }])
      }
    } catch (err) {
      setChatMessages(prev => [...prev, { role: 'assistant', content: 'Connection error. Please check the backend server and Ollama configuration.' }])
    } finally {
      setChatLoading(false)
    }
  }

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendChat()
    }
  }

  const currentModel = models.find(m => m.model_id === selectedModel)
  const currentExplanation = explanations?.find(e => e.sample_id === selectedSample)
  const taxonLabel = (feature: string, maxLen = 38) => {
    const raw = String(feature || '')
    const last = raw.includes('___') ? raw.split('___').pop() || raw : raw.includes('|') ? raw.split('|').pop() || raw : raw
    const cleaned = last.replace(/^[dkpcofgst]__/, '').replace(/_/g, ' ').trim() || raw
    const rankMatch = last.match(/^([dkpcofgst])__/)
    const rank = rankMatch ? `${rankMatch[1]}. ` : ''
    const label = `${rank}${cleaned}`
    return label.length > maxLen ? `${label.slice(0, maxLen - 1).trim()}…` : label
  }

  const signedValue = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(4)}`

  const renderExplanationList = (features: FeatureExplanation[] = []) => {
    const visible = features.slice(0, 12)
    const maxAbs = Math.max(...visible.map(f => Math.abs(f.importance || 0)), 1e-12)
    return (
      <div className={styles.featureImportanceList}>
        {visible.map((feat, idx) => {
          const magnitude = Math.min(1, Math.abs(feat.importance || 0) / maxAbs)
          const positive = (feat.importance || 0) >= 0
          const width = `${Math.max(1.5, magnitude * 47)}%`
          const left = positive ? '50%' : `${50 - Math.max(1.5, magnitude * 47)}%`
          return (
            <div key={`${feat.feature}-${idx}`} className={styles.featureImportanceItem}>
              <div className={styles.featureImportanceRank}>{idx + 1}</div>
              <div className={styles.featureImportanceName} title={feat.feature}>{taxonLabel(feat.feature)}</div>
              <div className={styles.featureImportanceBarContainer} aria-label={`${taxonLabel(feat.feature)} contribution ${signedValue(feat.importance || 0)}`}>
                <div className={styles.zeroLine} />
                <div
                  className={`${styles.featureImportanceBar} ${positive ? styles.supportToward : styles.supportAgainst}`}
                  style={{ width, left }}
                />
              </div>
              <div className={`${styles.featureImportanceValue} ${positive ? styles.valuePositive : styles.valueNegative}`}>
                {signedValue(feat.importance || 0)}
              </div>
            </div>
          )
        })}
        {visible.length > 0 && (
          <div className={styles.explainAxisRow} aria-hidden="true">
            <div />
            <div />
            <div className={styles.explainAxis}>
              <span className={styles.explainAxisMinus}>−</span>
              <span className={styles.explainAxisZero}>0</span>
              <span className={styles.explainAxisPlus}>+</span>
            </div>
            <div />
          </div>
        )}
      </div>
    )
  }


  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerInner}>
          <Link href="/" className={styles.logo}>
            <h1>mllabiome-ii</h1>
            <span className={styles.subtitle}>interactive inference</span>
          </Link>
          <div className={styles.headerRight}>
            <ThemeToggle />
          </div>
        </div>
      </div>

      <div className={styles.mainLayout}>
        <aside className={styles.sidebar}>
          <div className={styles.sidebarSection}>
            <h3 className={styles.sidebarTitle}>Model</h3>
            <div className={styles.modelSelector}>
              {models.map(model => (
                <button
                  key={model.model_id}
                  className={`${styles.modelOption} ${selectedModel === model.model_id ? styles.modelOptionActive : ''}`}
                  onClick={() => setSelectedModel(model.model_id)}
                >
                  <span className={styles.modelOptionName}>{model.display_name}</span>
                  <span className={styles.modelOptionMeta}>{model.target || 'Prediction model'}</span>
                </button>
              ))}
              {models.length === 0 && (
                <div className={styles.modelInfo}>
                  <p className={styles.modelDescription}>No deployment package found. Place an exported model zip in app/backend/production_models and restart the backend.</p>
                </div>
              )}
            </div>
            {currentModel && (
              <div className={styles.modelInfo}>
                <p className={styles.modelDescription}>Selected model: {currentModel.display_name}</p>
                {currentModel.calibrated && (
                  <span className={styles.calibratedBadge}>Calibrated</span>
                )}
              </div>
            )}
          </div>

          <div className={styles.sidebarSection}>
            <h3 className={styles.sidebarTitle}>Sample Data</h3>
            <div 
              className={`${styles.uploadCardSmall} ${isDragging ? styles.uploadCardDragging : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <label className={styles.dataUploadSmall}>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".tsv,.csv,.txt"
                  onChange={(e) => {
                    setDataFile(e.target.files?.[0] || null)
                    setError(null)
                  }}
                />
                <div className={styles.uploadContentSmall}>
                  <span className={styles.uploadTitleSmall}>
                    {dataFile ? dataFile.name : 'Drop file here'}
                  </span>
                  <span className={styles.uploadHintSmall}>
                    TSV or CSV
                  </span>
                </div>
              </label>
            </div>
            
            {error && (
              <div className={styles.errorMessageSmall}>{error}</div>
            )}

            <div className={styles.buttonGroup}>
              <button
                className={styles.runButtonSmall}
                onClick={handleRunInference}
                disabled={!dataFile || !selectedModel || loading}
              >
                {loading ? 'Running...' : 'Run Inference'}
              </button>
              
              <button
                className={`${styles.runButtonSmall} ${styles.explainButtonSmall}`}
                onClick={handleExplain}
                disabled={!dataFile || !selectedModel || explaining}
              >
                {explaining ? 'Explaining...' : 'Explain'}
              </button>
            </div>
          </div>

          <div className={styles.sidebarSection}>
            <h3 className={styles.sidebarTitle}>Status</h3>
            <div className={styles.statusList}>
              <div className={styles.statusItem}>
                <span className={styles.statusDot} style={{ backgroundColor: models.length > 0 ? '#10b981' : '#ef4444' }} />
                <span>{models.length > 0 ? 'Ready' : 'No model package'}</span>
              </div>
              <div className={styles.statusItem}>
                <span className={styles.statusDot} style={{ backgroundColor: dataFile ? '#10b981' : '#6b7280' }} />
                <span>{dataFile ? 'Data loaded' : 'No data'}</span>
              </div>
              <div className={styles.statusItem}>
                <span className={styles.statusDot} style={{ backgroundColor: results ? '#10b981' : '#6b7280' }} />
                <span>{results ? `${results.length} predictions` : 'No results'}</span>
              </div>
            </div>
          </div>
        </aside>

        <main className={styles.mainContent}>
          <div className={styles.contentGrid}>
            <div className={styles.resultsPanel}>
              {results && results.length > 0 && (
                <section className={styles.section}>
                  <div className={styles.sectionHeader}>
                    <h2>Predictions</h2>
                    <div className={styles.sectionHeaderActions}>
                      <span className={styles.resultCount}>{results.length} samples</span>
                      <button
                        type="button"
                        className={styles.downloadCsvButton}
                        onClick={handleDownloadSubmission}
                        disabled={!dataFile || !selectedModel || downloadingSubmission}
                      >
                        {downloadingSubmission ? 'Preparing...' : 'Download CSV'}
                      </button>
                    </div>
                  </div>
                  <div className={styles.resultsTable}>
                    <div className={styles.tableHeader}>
                      <div className={styles.tableCell}>Sample</div>
                      <div className={styles.tableCell}>Prediction</div>
                      <div className={styles.tableCell}>Probability</div>
                      <div className={styles.tableCell}>Confidence</div>
                    </div>
                    {results.map((pred, idx) => (
                      <div key={idx} className={styles.tableRow}>
                        <div className={`${styles.tableCell} ${styles.sampleIdCell}`} title={pred.sample_id}>
                          <span className={styles.sampleIdText}>{pred.sample_id}</span>
                        </div>
                        <div className={styles.tableCell}>
                          <span className={`${styles.predictionLabel} ${pred.prediction === 1 ? styles.positive : styles.negative}`}>
                            {pred.label}
                          </span>
                        </div>
                        <div className={styles.tableCell}>
                          {(pred.probability * 100).toFixed(1)}%
                        </div>
                        <div className={styles.tableCell}>
                          <span className={styles.confidence}>{(pred.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {explanations && explanations.length > 0 && (
                <section className={`${styles.section} ${styles.explainabilitySection}`}>
                  <div className={styles.sectionHeader}>
                    <h2>Explainability</h2>
                    <div className={styles.explainabilityControls}>
                      <div className={styles.methodToggleGroup} aria-label="Explainability methods">
                        <button
                          type="button"
                          className={`${styles.methodToggle} ${showShap ? styles.methodToggleActive : ''}`}
                          onClick={toggleShap}
                          aria-pressed={showShap}
                        >
                          SHAP
                        </button>
                        <button
                          type="button"
                          className={`${styles.methodToggle} ${showLime ? styles.methodToggleActive : ''}`}
                          onClick={toggleLime}
                          aria-pressed={showLime}
                        >
                          LIME
                        </button>
                      </div>
                      <div className={styles.sampleSelectWrap}>
                        <label className={styles.sampleSelectLabel} htmlFor="explain-sample-select">Sample</label>
                        <select
                          id="explain-sample-select"
                          className={styles.sampleSelect}
                          value={selectedSample || currentExplanation?.sample_id || ''}
                          onChange={(e) => handleSampleChange(e.target.value)}
                          title={selectedSample || undefined}
                        >
                          {(results && results.length > 0 ? results : explanations).map((row) => (
                            <option key={row.sample_id} value={row.sample_id}>
                              {row.sample_id}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>

                  {currentExplanation && (
                    <div className={styles.explanationCard}>
                      <div className={`${styles.explainabilityGrid} ${showShap && showLime ? styles.explainabilityGridTwo : styles.explainabilityGridOne}`}>
                        {showShap && (
                          <div className={styles.supportPanel}>
                            <div className={styles.panelHeadingRow}>
                              <h4 className={styles.subSectionTitle}>SHAP</h4>
                            </div>
                            {renderExplanationList(currentExplanation.shap_features || currentExplanation.top_features || [])}
                          </div>
                        )}

                        {showLime && (
                          <div className={styles.supportPanel}>
                            <div className={styles.panelHeadingRow}>
                              <h4 className={styles.subSectionTitle}>LIME</h4>
                            </div>
                            {renderExplanationList(currentExplanation.lime_features || [])}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </section>
              )}

              {!results && (
                <section className={styles.section}>
                  <div className={styles.emptyState}>
                    <h3>No Results</h3>
                    <p>Upload a microbiome profile and run inference to see predictions. If no model is listed, copy an exported model zip into <code>app/backend/production_models</code> and restart the backend.</p>
                  </div>
                </section>
              )}
            </div>

            <div className={styles.chatPanel}>
              <div className={styles.chatHeader}>
                <h3>Inference Assistant</h3>
                <span className={styles.chatMeta}>Ollama · {chatModel || 'not configured'}</span>
              </div>
              <div className={styles.chatConfigBar}>
                <label className={styles.chatModelLabel} htmlFor="ollama-model">Model</label>
                <input
                  id="ollama-model"
                  list="ollama-model-options"
                  value={chatModel}
                  onChange={(e) => setChatModel(e.target.value)}
                  className={styles.chatModelInput}
                  placeholder="gemma4:31b-cloud"
                  disabled={chatLoading}
                />
                <datalist id="ollama-model-options">
                  {chatModelOptions.map((name) => (
                    <option key={name} value={name} />
                  ))}
                </datalist>
                <button
                  type="button"
                  className={styles.chatRefreshButton}
                  onClick={fetchChatConfig}
                  disabled={chatLoading}
                  title="Refresh Ollama model list"
                >
                  Refresh
                </button>
              </div>
              {chatConfigStatus !== 'ok' && chatConfigDetail && (
                <div className={styles.chatConfigWarning}>{chatConfigDetail}</div>
              )}
              
              <div className={styles.chatMessages}>
                {chatMessages.length === 0 && (
                  <div className={styles.chatWelcome}>
                    {explanations ? (
                      <>
                        <p>Results loaded. Ask about the feature attributions.</p>
                        <p className={styles.chatHint}>Example: Why is Bacteroides important for this prediction?</p>
                      </>
                    ) : results ? (
                      <>
                        <p>Predictions ready. Click Explain for feature attribution, or ask about the prediction summary.</p>
                        <p className={styles.chatHint}>After Explain completes, ask about specific bacterial genera or contribution direction.</p>
                      </>
                    ) : (
                      <>
                        <p>Run inference to get started.</p>
                        <p className={styles.chatHint}>Upload data and click Run Inference.</p>
                      </>
                    )}
                  </div>
                )}
                {chatMessages.map((msg, idx) => (
                  <div key={idx} className={`${styles.chatMessage} ${styles[msg.role]}`}>
                    <div className={styles.messageContent}>{msg.content}</div>
                  </div>
                ))}
                {chatLoading && (
                  <div className={`${styles.chatMessage} ${styles.assistant}`}>
                    <div className={styles.messageContent}>
                      <span className={styles.typingIndicator}>...</span>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
              
              <div className={styles.chatInputContainer}>
                <textarea
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about your results..."
                  className={styles.chatInput}
                  rows={2}
                  disabled={chatLoading}
                />
                <button 
                  onClick={handleSendChat} 
                  className={styles.chatSendButton}
                  disabled={!chatInput.trim() || chatLoading}
                >
                  Send
                </button>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
