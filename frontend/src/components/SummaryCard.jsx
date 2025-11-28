import React from 'react'

const SummaryCard = ({ store, date, total, rawText }) => {
  return (
    <div className="card flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-400">Store</p>
          <p className="text-lg font-semibold text-white">{store || '—'}</p>
        </div>
        <div className="text-right">
          <p className="text-sm text-gray-400">Date</p>
          <p className="text-lg font-semibold text-white">{date || '—'}</p>
        </div>
      </div>
      <div className="p-4 rounded-lg border border-[#2a2a2a] bg-[#0f0f0f] flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-400">Total</p>
          <p className="text-3xl font-bold text-neon">${total?.toFixed ? total.toFixed(2) : '0.00'}</p>
        </div>
        <div className="text-xs text-gray-400 text-right max-w-[50%]">
          <p className="uppercase tracking-wide text-electric mb-1">Raw OCR</p>
          <p className="whitespace-pre-wrap leading-relaxed">{rawText || 'Waiting for OCR results…'}</p>
        </div>
      </div>
    </div>
  )
}

export default SummaryCard
