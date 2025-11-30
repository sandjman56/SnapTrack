import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:5050',
})

export const processReceipt = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  const response = await api.post('/process_receipt', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return response.data
}

export const saveReceipt = async (payload) => {
  const response = await api.post('/save_receipt', payload, {
    headers: { 'Content-Type': 'application/json' },
  })
  return response.data
}

export const fetchHistory = async () => {
  const response = await api.get('/history')
  return response.data
}

export default api
