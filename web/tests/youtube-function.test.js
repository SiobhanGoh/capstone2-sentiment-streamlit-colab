import { describe, expect, it } from 'vitest';
import { getComments, onRequestGet } from '../functions/api/youtube.js';

const ok = (body) => ({ status: 200, json: async () => body });
const fail = (status, reason) => ({ status, json: async () => ({ error: { errors: [{ reason }] } }) });

function fakeYouTube(pages, { video = true } = {}) {
  const calls = [];
  const fetcher = async (url) => {
    calls.push(url);
    const u = new URL(url);
    if (u.pathname.endsWith('/videos')) return ok({ items: video ? [{ snippet: { title: 'T', channelTitle: 'C' } }] : [] });
    const page = pages.shift();
    return typeof page === 'function' ? page() : ok(page);
  };
  return { fetcher, calls };
}
const thread = (id, text) => ({ id, snippet: { topLevelComment: { id, snippet: { textOriginal: text } } } });

describe('YouTube Pages Function', () => {
  it('pages through results, skips duplicates and blanks, and stops at the limit', async () => {
    const { fetcher, calls } = fakeYouTube([
      { items: [thread('a', 'one'), thread('b', '  '), thread('a', 'one again')], nextPageToken: 'p2' },
      { items: [thread('c', 'two'), thread('d', 'three')], nextPageToken: 'p3' },
    ]);
    const { meta, comments } = await getComments('dQw4w9WgXcQ', 2, 'KEY', fetcher);
    expect(comments).toEqual(['one', 'two']);
    expect(meta.title).toBe('T');
    expect(calls).toHaveLength(3);
  });
  it('never returns a partial fetch when a later page fails', async () => {
    const { fetcher } = fakeYouTube([{ items: [thread('a', 'one')], nextPageToken: 'p2' }, () => fail(500, 'backendError')]);
    await expect(getComments('dQw4w9WgXcQ', 50, 'KEY', fetcher)).rejects.toThrow(/could not complete/);
  });
  it('explains disabled comments, quota and key problems without leaking the key', async () => {
    for (const [reason, re] of [['commentsDisabled', /disabled/], ['quotaExceeded', /quota/], ['keyInvalid', /API key/]]) {
      const { fetcher } = fakeYouTube([() => fail(403, reason)]);
      const err = await getComments('dQw4w9WgXcQ', 10, 'SECRET', fetcher).catch((e) => e);
      expect(err.message).toMatch(re);
      expect(err.message).not.toContain('SECRET');
    }
  });
  it('validates input before calling YouTube', async () => {
    await expect(getComments('dQw4w9WgXcQ', 0, 'KEY', async () => { throw new Error('no'); })).rejects.toThrow(/between 1 and 100/);
    await expect(getComments('dQw4w9WgXcQ', 10, '', async () => { throw new Error('no'); })).rejects.toThrow(/not set up/);
  });
  it('answers bad links with a 400 and a readable message', async () => {
    const res = await onRequestGet({ request: new Request('https://x.dev/api/youtube?id=bad&max=10'), env: { YOUTUBE_API_KEY: 'K' }, waitUntil() {} });
    expect(res.status).toBe(400);
    expect((await res.json()).error).toMatch(/valid YouTube/);
  });
});
