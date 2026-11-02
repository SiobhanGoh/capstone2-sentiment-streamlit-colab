import React from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource-variable/bricolage-grotesque';
import '@fontsource-variable/plus-jakarta-sans';
import './styles.css';
import App from './App.jsx';
import { ModelProvider } from './lib/useModel.js';

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ModelProvider><App /></ModelProvider>
  </React.StrictMode>,
);
