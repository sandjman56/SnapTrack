/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        night: '#121212',
        neon: '#39FF14',
        electric: '#00BFFF',
      },
      boxShadow: {
        glow: '0 0 20px rgba(57, 255, 20, 0.3)',
      },
    },
  },
  plugins: [],
}
