// Cached files of other revisions of this app's model, which the browser no longer needs.
export function staleModelUrls(urls, repoId, revision) {
  const prefix = `/${repoId}/resolve/`;
  return urls.filter((url) => {
    const at = url.indexOf(prefix);
    if (at < 0) return false;
    return !url.slice(at + prefix.length).startsWith(`${revision}/`);
  });
}

// Remove those files after a successful load, so returning visitors keep one copy (about 356 MB).
export async function pruneOldModels(repoId, revision, cacheName = 'transformers-cache') {
  if (typeof caches === 'undefined' || !/^[a-f0-9]{40}$/.test(revision)) return 0;
  const cache = await caches.open(cacheName);
  const requests = await cache.keys();
  const stale = new Set(staleModelUrls(requests.map((r) => r.url), repoId, revision));
  await Promise.all(requests.filter((r) => stale.has(r.url)).map((r) => cache.delete(r)));
  return stale.size;
}
