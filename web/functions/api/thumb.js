// Serves a YouTube video thumbnail from this site's own origin, so the browser can embed it
// in the PDF report (images from other sites can't be read back into a PDF).
const ID = /^[A-Za-z0-9_-]{11}$/;

export async function onRequestGet({ request }) {
  const id = new URL(request.url).searchParams.get('id') ?? '';
  if (!ID.test(id)) return new Response('Invalid video ID', { status: 400 });
  for (const name of ['hqdefault', 'mqdefault']) {
    let upstream;
    try {
      upstream = await fetch(`https://i.ytimg.com/vi/${id}/${name}.jpg`, { cf: { cacheTtl: 86400 } });
    } catch {
      continue;
    }
    if (upstream.ok && (upstream.headers.get('content-type') ?? '').startsWith('image/')) {
      return new Response(upstream.body, {
        headers: { 'content-type': upstream.headers.get('content-type'), 'cache-control': 'public, max-age=86400' },
      });
    }
  }
  return new Response('Thumbnail not found', { status: 404 });
}
