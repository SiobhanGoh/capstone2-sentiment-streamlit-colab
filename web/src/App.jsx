import { useEffect, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { Moon, Sun } from 'lucide-react';
import Analyse from './views/Analyse.jsx';
import Comments from './views/Comments.jsx';
import Compare from './views/Compare.jsx';
import About from './views/About.jsx';
import { loadHistory, saveHistory } from './lib/history.js';

const VIEWS = [
  { path: '/', name: 'Analyse' },
  { path: '/comments', name: 'Comments' },
  { path: '/compare', name: 'Compare' },
  { path: '/about', name: 'About' },
];

function useRoute() {
  const read = () => {
    const p = window.location.hash.replace(/^#/, '') || '/';
    return VIEWS.some((v) => v.path === p) ? p : '/';
  };
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => { setRoute(read()); window.scrollTo({ top: 0 }); };
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return route;
}

function ThemeToggle() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'));
  const toggle = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle('dark', next);
    try { localStorage.setItem('rojak-theme', next ? 'dark' : 'light'); } catch { /* storage unavailable */ }
  };
  return (
    <button onClick={toggle} className="neu-raised grid h-10 w-10 place-items-center rounded-full" aria-label={dark ? 'Use light theme' : 'Use dark theme'}>
      {dark ? <Sun size={17} aria-hidden /> : <Moon size={17} aria-hidden />}
    </button>
  );
}

export default function App() {
  const route = useRoute();
  const reduce = useReducedMotion();
  const [history, setHistory] = useState(loadHistory);
  const [stored, setStored] = useState(true);

  const addPrediction = (item) => setHistory((h) => { const next = [item, ...h]; setStored(saveHistory(next)); return next; });
  const clear = () => { setHistory([]); setStored(saveHistory([])); };

  return (
    <>
      <div aria-hidden>
        <div className="blob left-[-8rem] top-[-6rem] h-[26rem] w-[26rem] bg-ube-300" />
        <div className="blob right-[-6rem] top-[18rem] h-[22rem] w-[22rem] bg-pandan-300" />
        <div className="blob bottom-[-8rem] left-[30%] h-[24rem] w-[24rem] bg-jambu-300" />
      </div>
      <div className="relative z-10 mx-auto max-w-6xl px-4 pb-16 sm:px-6">
        <header className="flex flex-wrap items-center gap-4 py-5">
          <a href="#/" className="mr-auto flex items-center gap-2.5" aria-label="Rojak Sentiment home">
            <span className="flex -space-x-1.5" aria-hidden>
              <span className="h-5 w-5 rounded-full bg-pandan-300" /><span className="h-5 w-5 rounded-full bg-kaya-300" /><span className="h-5 w-5 rounded-full bg-jambu-300" />
            </span>
            <span className="font-display text-lg font-bold">Rojak Sentiment</span>
          </a>
          <nav aria-label="Main" className="glass order-3 flex w-full gap-1 overflow-x-auto rounded-full p-1 sm:order-none sm:w-auto">
            {VIEWS.map((v) => (
              <a key={v.path} href={`#${v.path}`} aria-current={route === v.path ? 'page' : undefined}
                className={`relative whitespace-nowrap rounded-full px-4 py-2 text-sm font-semibold ${route === v.path ? 'text-white' : 'text-ink-soft hover:text-ink dark:text-ube-100'}`}>
                {route === v.path && <motion.span layoutId="nav-pill" className="absolute inset-0 rounded-full bg-ube-600" transition={{ type: 'spring', stiffness: 320, damping: 30 }} />}
                <span className="relative">{v.name}</span>
              </a>
            ))}
          </nav>
          <ThemeToggle />
        </header>
        <AnimatePresence mode="wait">
          <motion.main key={route} initial={reduce ? false : { opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            exit={reduce ? undefined : { opacity: 0 }} transition={{ duration: 0.18 }}>
            {route === '/' && <Analyse onPredicted={addPrediction} history={history} onClearHistory={clear} stored={stored} />}
            {route === '/comments' && <Comments />}
            {route === '/compare' && <Compare />}
            {route === '/about' && <About />}
          </motion.main>
        </AnimatePresence>
        <footer className="mt-12 text-center text-sm text-ink-faint">
          Degree capstone 2025–26 by Siobhan Goh. Labels are suggestions; please check before acting on them.
        </footer>
      </div>
    </>
  );
}
