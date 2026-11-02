// Single-text history kept only in this browser. Storage can be unavailable (private mode),
// so every access is guarded and the app works without it.
const KEY = 'rojak-sentiment-history-v1';
const LIMIT = 200;

export function loadHistory() {
  try {
    const value = JSON.parse(localStorage.getItem(KEY) ?? '[]');
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

export function saveHistory(items) {
  try {
    localStorage.setItem(KEY, JSON.stringify(items.slice(0, LIMIT)));
    return true;
  } catch {
    return false;
  }
}
