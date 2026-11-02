// Shared by the browser and the Cloudflare Pages Function.
const ID = /^[A-Za-z0-9_-]{11}$/;
const HOSTS = new Set(['youtube.com', 'www.youtube.com', 'm.youtube.com', 'music.youtube.com']);

export function extractVideoId(input) {
  let value = String(input ?? '').trim();
  if (ID.test(value)) return value;
  if (!/^https?:\/\//i.test(value)) value = `https://${value}`;
  let url;
  try { url = new URL(value); } catch { url = null; }
  let id = '';
  if (url) {
    const host = url.hostname.toLowerCase();
    const parts = url.pathname.replace(/^\/+|\/+$/g, '').split('/');
    if (host === 'youtu.be' || host === 'www.youtu.be') id = parts[0];
    else if (HOSTS.has(host)) {
      if (url.pathname === '/watch') id = url.searchParams.get('v') ?? '';
      else if (parts.length === 2 && ['shorts', 'embed', 'live'].includes(parts[0])) id = parts[1];
    }
  }
  if (!ID.test(id)) throw new Error('Enter a valid YouTube video URL or its 11-character video ID.');
  return id;
}

export async function fetchComments(videoUrl, max) {
  const id = extractVideoId(videoUrl);
  let response;
  try {
    response = await fetch(`/api/youtube?id=${encodeURIComponent(id)}&max=${max}`);
  } catch {
    throw new Error('YouTube could not be reached. Check your connection and try again.');
  }
  let data = null;
  try { data = await response.json(); } catch { /* handled below */ }
  if (!response.ok || !data) {
    throw new Error(data?.error || 'YouTube retrieval is not available right now. You can still paste comments.');
  }
  return data;
}
