import React, { useRef } from 'react'

const ImageUpload = ({ onFileSelect, onScan, isLoading }) => {
  const fileInputRef = useRef(null)

  const handleFileChange = (event) => {
    const file = event.target.files?.[0]
    if (file) {
      onFileSelect(file)
    }
  }

  return (
    <div className="card neon-gradient">
      <div className="flex items-center justify-between gap-3">
        <div className="flex flex-col gap-2">
          <h2 className="text-xl font-semibold text-neon">Upload receipt</h2>
          <p className="text-sm text-gray-300">JPG or PNG files, optimized for OCR</p>
          <div className="flex gap-2">
            <button
              className="button-primary"
              onClick={() => fileInputRef.current?.click()}
              type="button"
            >
              Choose file
            </button>
            <button
              type="button"
              className="px-4 py-2 rounded-md border border-electric text-electric hover:bg-electric hover:text-black transition"
              onClick={onScan}
              disabled={isLoading}
            >
              {isLoading ? 'Scanning…' : 'Scan Receipt'}
            </button>
          </div>
        </div>
        <div className="flex items-center gap-2 text-gray-300">
          <span className="w-2 h-2 rounded-full bg-electric animate-pulse" aria-hidden />
          <span className="text-xs">FastAPI running on localhost:5050</span>
        </div>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png, image/jpeg"
        className="hidden"
        onChange={handleFileChange}
      />
    </div>
  )
}

export default ImageUpload
