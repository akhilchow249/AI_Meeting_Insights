/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: {
          base:    '#07091000',
          DEFAULT: '#080c18',
          surface: '#0e1527',
          elevated:'#141d33',
          border:  '#1a2540',
          hover:   '#1e2b47',
        },
        amber: {
          glow:    '#f0a000',
          DEFAULT: '#e09500',
          dim:     '#7a5000',
          muted:   '#1e1500',
        },
        teal: {
          glow:    '#00d2c8',
          DEFAULT: '#00b8ae',
          dim:     '#005f5a',
          muted:   '#001918',
        },
        ink: {
          DEFAULT: '#f0f4ff',
          muted:   '#8899bb',
          faint:   '#3a4a6a',
        },
        good:  '#00d97e',
        warn:  '#ffb547',
        fail:  '#ff4c6b',
        info:  '#4d9fff',
      },
      fontFamily: {
        display: ['Syne', 'sans-serif'],
        mono:    ['"Space Mono"', 'monospace'],
        body:    ['"DM Sans"', 'sans-serif'],
      },
      animation: {
        'pulse-slow':  'pulse 2.5s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-in':    'slideIn 0.3s ease-out',
        'fade-in':     'fadeIn 0.4s ease-out',
        'shimmer':     'shimmer 1.8s linear infinite',
        'scan':        'scan 2s ease-in-out infinite',
      },
      keyframes: {
        slideIn:  { from: { opacity: 0, transform: 'translateY(8px)' }, to: { opacity: 1, transform: 'translateY(0)' } },
        fadeIn:   { from: { opacity: 0 }, to: { opacity: 1 } },
        shimmer:  { '0%': { backgroundPosition: '-400px 0' }, '100%': { backgroundPosition: '400px 0' } },
        scan:     { '0%,100%': { opacity: 0.4 }, '50%': { opacity: 1 } },
      },
    },
  },
  plugins: [],
}
