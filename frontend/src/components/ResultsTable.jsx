import React from 'react'

const ResultsTable = ({ items }) => {
  if (!items?.length) {
    return (
      <div className="card">
        <p className="text-gray-400 text-sm">No line items detected yet.</p>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xl font-semibold text-neon">Line Items</h2>
        <span className="text-xs text-gray-400">{items.length} detected</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead className="text-gray-400 text-sm border-b border-[#2a2a2a]">
            <tr>
              <th className="py-2">Name</th>
              <th className="py-2 text-right">Price ($)</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item, idx) => (
              <tr key={`${item.name}-${idx}`} className="border-b border-[#1f1f1f] last:border-0">
                <td className="py-2 font-medium text-white">{item.name || 'Item'}</td>
                <td className="py-2 text-right text-electric">{item.price.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default ResultsTable
