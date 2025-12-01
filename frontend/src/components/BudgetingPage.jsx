import React from 'react'

const ReceiptCard = ({ receipt, onSelect }) => {
  const imageSrc = receipt.image_path ? `http://localhost:5050${receipt.image_path}` : ''
  return (
    <button
      type="button"
      onClick={() => onSelect(receipt)}
      className="w-full text-left card hover:border-electric transition"
    >
      <div className="flex items-center gap-4">
        <div className="w-16 h-16 rounded-lg overflow-hidden bg-[#0f0f0f] border border-[#2a2a2a] flex items-center justify-center">
          {imageSrc ? (
            <img src={imageSrc} alt="Receipt thumbnail" className="w-full h-full object-cover" />
          ) : (
            <span className="text-gray-500 text-xs">No image</span>
          )}
        </div>
        <div className="flex-1">
          <p className="text-sm text-gray-400">{receipt.date}</p>
          <h3 className="text-lg font-semibold text-white">{receipt.store || 'Receipt'}</h3>
          <p className="text-sm text-gray-400">{receipt.items?.length || 0} items</p>
        </div>
        <div className="text-right">
          <p className="text-xs text-gray-400">Total</p>
          <p className="text-xl font-bold text-neon">${(receipt.total ?? receipt.subtotal)?.toFixed?.(2) || '—'}</p>
          <p className="text-xs text-gray-500">Subtotal: ${receipt.subtotal?.toFixed?.(2) || '—'}</p>
          {receipt.taxes?.length ? (
            <p className="text-xs text-gray-500">Taxes: {receipt.taxes.map((t) => t.toFixed(2)).join(', ')}</p>
          ) : (
            <p className="text-xs text-gray-600">Taxes: —</p>
          )}
        </div>
      </div>
    </button>
  )
}

const BudgetingPage = ({ data, loading, onRefresh, onSelectReceipt }) => {
  const receipts = data?.receipts || []
  const monthlyTotal = data?.monthly_total || 0
  const month = data?.month || ''

  return (
    <div className="flex flex-col gap-4">
      <div className="card flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-400">This month ({month})</p>
          <h2 className="text-2xl font-bold text-neon">You spent ${monthlyTotal.toFixed(2)}</h2>
          <p className="text-gray-400 text-sm">Track receipts and totals over time.</p>
        </div>
        <button
          type="button"
          onClick={onRefresh}
          className="px-4 py-2 rounded-md border border-electric text-electric hover:bg-electric hover:text-black transition"
        >
          Refresh
        </button>
      </div>

      {loading && <div className="card text-gray-300">Loading history…</div>}

      {!loading && !receipts.length && (
        <div className="card text-gray-400">No receipts saved yet. Scan and save a receipt to begin.</div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {receipts.map((receipt) => (
          <ReceiptCard key={receipt.id} receipt={receipt} onSelect={onSelectReceipt} />
        ))}
      </div>
    </div>
  )
}

export default BudgetingPage
