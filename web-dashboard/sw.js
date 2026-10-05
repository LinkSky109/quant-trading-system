/* REQ-P1-06 Service Worker：App Shell 预缓存 + 运行时缓存策略 */
const CACHE_VERSION = 'qt-dash-v1.0.0';
const APP_SHELL = [
  '/',
  '/manifest.json',
  '/icons/icon-192.png',
  '/icons/icon-512.png'
];

// install：预缓存 App Shell
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) => {
      // 单个资源失败不阻塞 install（图标等可后续 runtime 补齐）
      return Promise.all(APP_SHELL.map((url) => {
        return cache.add(url).catch(() => console.warn('[SW] precache skip:', url));
      }));
    }).then(() => self.skipWaiting())
  );
});

// activate：清理旧版本缓存
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter((k) => k !== CACHE_VERSION).map((k) => {
          console.log('[SW] delete old cache:', k);
          return caches.delete(k);
        })
      );
    }).then(() => self.clients.claim())
  );
});

// fetch：按策略分流
self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return; // 非 GET 直接放行

  const url = new URL(req.url);
  if (url.origin !== location.origin) return; // 跨源（ECharts CDN 等）放行

  // API：networkFirst，失败回退缓存（保证行情尽量新，离线可看上次数据）
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(req)
        .then((resp) => {
          if (resp && resp.ok) {
            const copy = resp.clone();
            caches.open(CACHE_VERSION).then((c) => c.put(req, copy));
          }
          return resp;
        })
        .catch(() => caches.match(req).then((hit) => hit || Response.error()))
    );
    return;
  }

  // 页面 / 静态资源：cacheFirst，后台异步更新（stale-while-revalidate）
  event.respondWith(
    caches.match(req).then((cached) => {
      const networkFetch = fetch(req)
        .then((resp) => {
          if (resp && resp.ok) {
            const copy = resp.clone();
            caches.open(CACHE_VERSION).then((c) => c.put(req, copy));
          }
          return resp;
        })
        .catch(() => cached);
      return cached || networkFetch;
    })
  );
});
