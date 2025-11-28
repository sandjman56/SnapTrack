import React from 'react'

const ReceiptPreview = ({ imageUrl }) => {
  return (
    <div className="card h-full flex flex-col">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-electric">Receipt Preview</h2>
        <span className="text-xs text-gray-400">Live render</span>
      </div>
      <div className="flex-1 mt-4 rounded-lg overflow-hidden border border-[#2a2a2a] bg-[#0f0f0f] flex items-center justify-center">
        {imageUrl ? (
          <img src={imageUrl} alt="Receipt preview" className="max-h-[480px] object-contain" />
        ) : (
          <p className="text-gray-500">Upload an image to preview</p>
        )}
      </div>
    </div>
  )
}

export default ReceiptPreview
