// Cloudflare Pages Function: fetch public top-level YouTube comments without exposing the API key.
import { extractVideoId } from '../../src/lib/youtube.js';

const API = 'https://www.googleapis.com/youtube/v3/';
const MAX = 100;

class YouTubeError extends Error {
  constructor(message, status = 400) { super(message); this.status = status; }
}

function json(body, status = 200, extra = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', ...extra },
  });
}

// Messages never include the request URL or the key.
async function youtubeGet(endpoint, params, key, fetcher) {
  const url = new URL(API + endpoint);
  Object.entries({ ...params, key }).forEach(([k, v]) => url.searchParams.set(k, String(v)));
  let response;
  try { response = await fetcher(url.toString()); } catch {
    throw new YouTubeError('YouTube could not be reached. Please try again shortly.', 502);
  }
  let data;
  try { data = await response.json(); } catch {
    throw new YouTubeError('YouTube returned an unreadable response. Please try again.', 502);
  }
  if (response.status !== 200) {
    const reasons = new Set((data?.error?.errors ?? []).map((e) => e?.reason));
    if (reasons.has('commentsDisabled')) throw new YouTubeError('Comments are disabled for this video.');
    if (reasons.has('quotaExceeded') || reasons.has('dailyLimitExceeded')) {
      throw new YouTubeError('The daily YouTube request quota has been reached. Try again tomorrow or paste comments.', 429);
    }
    if (['keyInvalid', 'accessNotConfigured', 'forbidden', 'ipRefererBlocked'].some((r) => reasons.has(r))) {
      throw new YouTubeError('YouTube access is not configured correctly. The site owner needs to check its API key.', 503);
    }
    if (response.status === 404) throw new YouTubeError('This video is unavailable or is not public.', 404);
    throw new YouTubeError('YouTube could not complete this request. Try a public video or paste comments.', 502);
  }
  if (!data || !Array.isArray(data.items ?? [])) throw new YouTubeError('YouTube returned an unexpected response.', 502);
  return data;
}

export async function getComments(input, max, key, fetcher = fetch) {
  const id = extractVideoId(input);
  if (!key) throw new YouTubeError('YouTube retrieval is not set up on this site. You can still paste comments.', 503);
  if (!Number.isInteger(max) || max < 1 || max > MAX) throw new YouTubeError(`Choose between 1 and ${MAX} comments.`);
  const details = await youtubeGet('videos', { part: 'snippet,statistics', id }, key, fetcher);
  if (!details.items?.length) throw new YouTubeError('This video is unavailable or is not public.', 404);
  const snippet = details.items[0].snippet ?? {};
  const stats = details.items[0].statistics ?? {};
  const thumbs = snippet.thumbnails ?? {};
  const count = (v) => (v == null || v === '' ? null : Number(v));
  const meta = {
    source: 'YouTube',
    video_id: id,
    title: snippet.title || 'Untitled video',
    channel: snippet.channelTitle || 'Unknown channel',
    published_at: snippet.publishedAt || '',
    thumbnail_url: (thumbs.maxres || thumbs.high || thumbs.medium || thumbs.default || {}).url || '',
    url: `https://www.youtube.com/watch?v=${id}`,
    views: count(stats.viewCount),
    likes: count(stats.likeCount),
    comment_count: count(stats.commentCount),
    sampling: `Up to ${max} top-level comments, ordered by relevance; replies excluded.`,
  };
  const comments = [];
  const likes = [];
  const seen = new Set();
  const tokens = new Set();
  let pageToken;
  while (comments.length < max) {
    const params = { part: 'snippet', videoId: id, maxResults: Math.min(100, max - comments.length), order: 'relevance', textFormat: 'plainText' };
    if (pageToken) params.pageToken = pageToken;
    const data = await youtubeGet('commentThreads', params, key, fetcher);
    for (const item of data.items ?? []) {
      const top = item?.snippet?.topLevelComment ?? {};
      const cid = top.id || item.id;
      const text = top.snippet?.textOriginal || top.snippet?.textDisplay;
      if (typeof text === 'string' && text.trim() && !seen.has(cid)) {
        comments.push(text.trim());
        likes.push(Number(top.snippet?.likeCount ?? 0) || 0);
        seen.add(cid);
        if (comments.length >= max) break;
      }
    }
    pageToken = data.nextPageToken;
    if (!pageToken || tokens.has(pageToken)) break;
    tokens.add(pageToken);
  }
  // A failed later page throws above, so a partial fetch is never presented as complete.
  return { meta, comments, likes };
}

export async function onRequestGet({ request, env, waitUntil }) {
  const url = new URL(request.url);
  const cache = typeof caches !== 'undefined' ? caches.default : null;
  const cacheKey = new Request(url.toString(), { method: 'GET' });
  if (cache) {
    const hit = await cache.match(cacheKey);
    if (hit) return hit;
  }
  try {
    const max = Number.parseInt(url.searchParams.get('max') ?? '30', 10);
    const body = await getComments(url.searchParams.get('id') ?? '', max, env.YOUTUBE_API_KEY);
    const response = json(body, 200, { 'cache-control': 'public, max-age=600' });
    if (cache) waitUntil(cache.put(cacheKey, response.clone()));
    return response;
  } catch (error) {
    const status = error instanceof YouTubeError ? error.status : 500;
    const message = error instanceof YouTubeError || error?.message?.startsWith('Enter a valid')
      ? error.message : 'Comments could not be retrieved. Please try again.';
    return json({ error: message }, error?.message?.startsWith('Enter a valid') ? 400 : status);
  }
}
