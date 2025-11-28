import React, { useMemo, useState } from 'react'
import ImageUpload from './components/ImageUpload'
import ReceiptPreview from './components/ReceiptPreview'
import ResultsTable from './components/ResultsTable'
import SummaryCard from './components/SummaryCard'
import { processReceipt } from './api'

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [results, setResults] = useState({ store: '', date: '', items: [], total: 0, raw_text: '' })
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const handleFileSelect = (file) => {
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setError('')
  }

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
          <div className="px-3 py-1 rounded-full border border-electric text-electric text-sm">Local mode</div>
        </header>

        <ImageUpload onFileSelect={handleFileSelect} onScan={handleScan} isLoading={isLoading} />

        {error && <div className="card border-red-600 text-red-400">{error}</div>}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ReceiptPreview imageUrl={previewUrl} />
          <div className="flex flex-col gap-4">
            <SummaryCard
              store={results.store}
              date={results.date}
              total={results.total}
              rawText={results.raw_text}
            />
            {isLoading ? shimmer : <ResultsTable items={results.items} />}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
