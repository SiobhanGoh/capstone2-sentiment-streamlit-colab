/** Pastel kuih palette: pandan (positive), kaya (neutral), jambu (negative), ube (interface). */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: ['class'],
  theme: {
    extend: {
      colors: {
        ink: { DEFAULT: '#2B2540', soft: '#5A5373', faint: '#8A84A0' },
        mist: '#F4F1FB',
        ube: { 50: '#F3EFFC', 100: '#E9E3FA', 300: '#BFB0EC', 500: '#8E7CC3', 600: '#7562B4', 700: '#5C4A96' },
        pandan: { 100: '#DDF3E5', 300: '#A8DDBD', 500: '#5DB585', 700: '#2F7A54' },
        kaya: { 100: '#FBF3D6', 300: '#F2DE97', 500: '#D9B44A', 700: '#8A6A12' },
        jambu: { 100: '#FCE4EA', 300: '#F4AFC0', 500: '#E07892', 700: '#A93556' },
        night: { DEFAULT: '#1C1830', raised: '#262040', line: '#3A3358' },
      },
      fontFamily: {
        display: ['"Bricolage Grotesque Variable"', 'system-ui', 'sans-serif'],
        sans: ['"Plus Jakarta Sans Variable"', 'system-ui', 'sans-serif'],
      },
      // Type scale: hero numbers, secondary numbers, headings, body, small, micro.
      fontSize: {
        hero: ['8.5rem', { lineHeight: '0.85', letterSpacing: '-0.04em' }],
        num: ['3.5rem', { lineHeight: '1', letterSpacing: '-0.03em' }],
        h1: ['2.5rem', { lineHeight: '1.1', letterSpacing: '-0.02em' }],
        h2: ['1.5rem', { lineHeight: '1.25', letterSpacing: '-0.01em' }],
        body: ['0.9375rem', { lineHeight: '1.6' }],
        small: ['0.8125rem', { lineHeight: '1.45' }],
        micro: ['0.6875rem', { lineHeight: '1.4' }],
      },
      // Corners: panels 28px, cards 20px, tiles inside cards 14px.
      borderRadius: { xl2: '1.25rem', xl3: '1.75rem', panel: '1.75rem', card: '1.25rem', tile: '0.875rem' },
      transitionDuration: { hover: '150ms', state: '240ms' },
    },
  },
  plugins: [],
};
