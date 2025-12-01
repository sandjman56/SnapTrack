import React, { useEffect, useMemo, useState } from 'react'
import ImageUpload from './components/ImageUpload'
import ReceiptPreview from './components/ReceiptPreview'
import ResultsTable from './components/ResultsTable'
import SummaryCard from './components/SummaryCard'
import BudgetingPage from './components/BudgetingPage'
import { fetchHistory, processReceipt, saveReceipt } from './api'

const emptyResults = {
  store: '',
  date: '',
  subtotal: null,
  total: null,
  taxes: [],
  items: [],
  raw_text: [],
  blocks: [],
  image_base64: '',
}

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [base64Image, setBase64Image] = useState('')
  const [results, setResults] = useState(emptyResults)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState('scan')
  const [saveState, setSaveState] = useState({ status: 'idle', message: '' })
  const [historyData, setHistoryData] = useState(null)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [selectedReceipt, setSelectedReceipt] = useState(null)

  const handleFileSelect = (file) => {
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setError('')
    setResults(emptyResults)
    const reader = new FileReader()
    reader.onload = () => setBase64Image(reader.result?.toString() || '')
    reader.readAsDataURL(file)
  }

  const loadHistory = async () => {
    try {
      setHistoryLoading(true)
      const data = await fetchHistory()
      setHistoryData(data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Unable to load history.')
    } finally {
      setHistoryLoading(false)
    }
  }

  useEffect(() => {
    if (activeTab === 'budget') {
      loadHistory()
    }
  }, [activeTab])

  const handleScan = async () => {
    if (!selectedFile) {
      setError('Please select a receipt image first.')
      return
    }

    try {
      setIsLoading(true)
      setError('')
      const data = await processReceipt(selectedFile)
      setResults(data)
    } catch (err) {
      const message = err.response?.data?.detail || 'Processing failed. Please try again.'
      setError(message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSave = async () => {
    if (!results.total || !base64Image) {
      setError('Scan a receipt first to save it.')
      return
    }
    try {
      setSaveState({ status: 'saving', message: '' })
      const payload = {
        subtotal: results.subtotal,
        total: results.total,
        taxes: results.taxes || [],
        items: results.items,
        raw_text: results.raw_text || [],
        image_base64: base64Image || results.image_base64,
        store: results.store,
        date: results.date,
      }
      const saved = await saveReceipt(payload)
      setSaveState({ status: 'success', message: 'Receipt saved!' })
      setSelectedReceipt(saved)
      if (activeTab === 'budget') {
        loadHistory()
      }
    } catch (err) {
      const message = err.response?.data?.detail || 'Unable to save receipt.'
      setSaveState({ status: 'error', message })
    }
  }

  const shimmer = useMemo(
    () => (
      <div className="animate-pulse text-electric text-sm">Analyzing receipt…</div>
    ),
    []
  )

  return (
    <div className="min-h-screen bg-night text-white p-6">
      <div className="max-w-6xl mx-auto flex flex-col gap-6">
        <header className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-400">SnapTrack</p>
            <h1 className="text-3xl font-bold text-neon">Receipt Budgeting Dashboard</h1>
            <p className="text-gray-400">Upload, scan, and categorize receipts locally.</p>
          </div>
          <div className="flex gap-2">
            <button
              className={`px-3 py-1 rounded-full border text-sm ${
                activeTab === 'scan'
                  ? 'border-electric text-electric'
                  : 'border-gray-600 text-gray-300 hover:border-electric hover:text-electric'
              }`}
              type="button"
              onClick={() => setActiveTab('scan')}
            >
              Scan
            </button>
            <button
              className={`px-3 py-1 rounded-full border text-sm ${
                activeTab === 'budget'
                  ? 'border-electric text-electric'
                  : 'border-gray-600 text-gray-300 hover:border-electric hover:text-electric'
              }`}
              type="button"
              onClick={() => setActiveTab('budget')}
            >
              Budgeting
            </button>
          </div>
        </header>

        {activeTab === 'scan' && (
          <>
            <ImageUpload onFileSelect={handleFileSelect} onScan={handleScan} isLoading={isLoading} />

            {error && <div className="card border-red-600 text-red-400">{error}</div>}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <ReceiptPreview imageUrl={previewUrl} />
              <div className="flex flex-col gap-4">
                <SummaryCard
                  store={results.store}
                  date={results.date}
                  subtotal={results.subtotal}
                  taxes={results.taxes}
                  total={results.total}
                  rawText={results.raw_text}
                  onSave={handleSave}
                  canSave={Boolean((results.total ?? results.subtotal) && base64Image)}
                  saveState={saveState}
                />
                {isLoading ? shimmer : <ResultsTable items={results.items} />}
              </div>
            </div>
          </>
        )}

        {activeTab === 'budget' && (
          <BudgetingPage
            data={historyData}
            loading={historyLoading}
            onRefresh={loadHistory}
            onSelectReceipt={setSelectedReceipt}
          />
        )}

        {selectedReceipt && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
            <div className="bg-[#1a1a1a] rounded-xl p-6 max-w-4xl w-full relative">
              <button
                className="absolute top-3 right-3 text-gray-400 hover:text-white"
                onClick={() => setSelectedReceipt(null)}
                type="button"
              >
                ✕
              </button>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h3 className="text-lg font-semibold text-electric">Receipt Details</h3>
                  <p className="text-sm text-gray-400">{selectedReceipt.store}</p>
                  <p className="text-sm text-gray-400">{selectedReceipt.date}</p>
                  <div className="mt-3 flex gap-4">
                    <div>
                      <p className="text-xs text-gray-400">Subtotal</p>
                      <p className="text-xl font-semibold">${selectedReceipt.subtotal?.toFixed?.(2) || '—'}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400">Taxes</p>
                      <p className="text-lg font-semibold">
                        {selectedReceipt.taxes?.length
                          ? selectedReceipt.taxes.map((t) => t.toFixed?.(2) || t).join(', ')
                          : '—'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400">Total</p>
                      <p className="text-xl font-semibold text-neon">${(selectedReceipt.total ?? selectedReceipt.subtotal)?.toFixed?.(2) || '—'}</p>
                    </div>
                  </div>
                  <div className="mt-4">
                    <p className="text-xs uppercase tracking-wide text-electric">Items</p>
                    <div className="space-y-1 text-sm text-gray-200 max-h-48 overflow-auto">
                      {selectedReceipt.items?.length ? (
                        selectedReceipt.items.map((item, idx) => (
                          <div key={`${item.name}-${idx}`} className="flex justify-between border-b border-[#2a2a2a] pb-1">
                            <span>{item.name}</span>
                            <span className="text-electric">${item.price?.toFixed?.(2) || item.price}</span>
                          </div>
                        ))
                      ) : (
                        <p className="text-gray-500">No items parsed.</p>
                      )}
                    </div>
                  </div>
                  <div className="mt-4">
                    <p className="text-xs uppercase tracking-wide text-electric mb-1">Raw OCR</p>
                    <div className="text-sm text-gray-300 bg-[#0f0f0f] p-3 rounded-lg max-h-40 overflow-auto whitespace-pre-wrap">
                      {(selectedReceipt.raw_text || []).join('\n')}
                    </div>
                  </div>
                </div>
                <div className="border border-[#2a2a2a] rounded-lg bg-[#0f0f0f] flex items-center justify-center p-2">
                  {selectedReceipt.image_path ? (
                    <img
                      src={`http://localhost:5050${selectedReceipt.image_path}`}
                      alt="Receipt thumbnail"
                      className="max-h-[520px] object-contain"
                    />
                  ) : (
                    <p className="text-gray-500">No saved image.</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
