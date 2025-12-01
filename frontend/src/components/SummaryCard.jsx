import React from 'react'

const SummaryCard = ({ store, date, subtotal, taxes = [], total, rawText = [], onSave, canSave, saveState }) => {
  const formattedTaxes = taxes.length ? taxes.map((t) => t.toFixed(2)).join(', ') : '—'
  const rawTextContent = Array.isArray(rawText) ? rawText.join('\n') : rawText
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
      <div className="p-4 rounded-lg border border-[#2a2a2a] bg-[#0f0f0f] flex items-center justify-between gap-4">
        <div className="flex gap-6 items-end">
          <div>
            <p className="text-sm text-gray-400">Subtotal</p>
            <p className="text-2xl font-semibold text-white">${subtotal?.toFixed?.(2) || '—'}</p>
          </div>
          <div>
            <p className="text-sm text-gray-400">Taxes</p>
            <p className="text-lg font-semibold text-white">{formattedTaxes}</p>
          </div>
          <div>
            <p className="text-sm text-gray-400">Total</p>
            <p className="text-3xl font-bold text-neon">${total?.toFixed ? total.toFixed(2) : '0.00'}</p>
          </div>
        </div>
        <div className="flex flex-col items-end gap-2">
          <button
            type="button"
            onClick={onSave}
            disabled={!canSave || saveState.status === 'saving'}
            className="px-4 py-2 rounded-md border border-electric text-electric hover:bg-electric hover:text-black transition disabled:opacity-50"
          >
            {saveState.status === 'saving' ? 'Saving…' : 'Save Receipt'}
          </button>
          {saveState.message && (
            <p
              className={`text-xs ${
                saveState.status === 'error' ? 'text-red-400' : 'text-green-400'
              }`}
            >
              {saveState.message}
            </p>
          )}
        </div>
      </div>
      <div className="text-xs text-gray-400 text-left max-w-full">
        <p className="uppercase tracking-wide text-electric mb-1">Raw OCR</p>
        <p className="whitespace-pre-wrap leading-relaxed bg-[#0f0f0f] border border-[#2a2a2a] rounded-lg p-3 min-h-[120px]">
          {rawTextContent || 'Waiting for OCR results…'}
        </p>
      </div>
    </div>
  )
}

export default SummaryCard
