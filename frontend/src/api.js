import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:5050',
  headers: {
    'Content-Type': 'multipart/form-data',
  },
})

export const processReceipt = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  const response = await api.post('/process_receipt', formData)
  return response.data
}

export default api
