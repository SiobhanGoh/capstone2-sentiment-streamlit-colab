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
      borderRadius: { xl2: '1.25rem', xl3: '1.75rem' },
    },
  },
  plugins: [],
};
