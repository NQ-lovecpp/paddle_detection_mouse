// Simple log server to receive logs from the React Native app
const http = require('http');

const PORT = 8082;

const server = http.createServer((req, res) => {
  if (req.method === 'POST' && req.url === '/log') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        const prefix = data.type === 'error' ? '❌' : data.type === 'warn' ? '⚠️' : 'ℹ️';
        console.log(`${prefix} [${data.time || ''}] ${data.message}`);
      } catch {
        console.log('RAW:', body);
      }
      res.writeHead(200);
      res.end('ok');
    });
  } else {
    res.writeHead(404);
    res.end();
  }
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`📡 Log server listening on http://0.0.0.0:${PORT}/log`);
  console.log('Waiting for logs from the app...\n');
});
