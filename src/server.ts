import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS for all origins
app.use(cors());

// Serve static files from the root directory (parent of src)
const rootDir = path.join(dirname(__dirname));

// Set proper MIME types for ES6 modules
express.static.mime.define({'application/javascript': ['js']});
express.static.mime.define({'text/css': ['css']});
express.static.mime.define({'text/html': ['html']});
express.static.mime.define({'application/json': ['json']});

// Serve static files
app.use(express.static(rootDir, {
  setHeaders: (res, filePath) => {
    // Set proper content type for JavaScript modules
    if (filePath.endsWith('.js')) {
      res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
    }
    // Allow cross-origin for modules
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  }
}));

// Fallback to index.html for SPA routing
app.get('*', (req, res) => {
  res.sendFile(path.join(rootDir, 'index.html'));
});

// Start the server
app.listen(PORT, () => {
  console.log(`CellFlow server is running at http://localhost:${PORT}`);
  console.log(`Serving files from: ${rootDir}`);
  console.log('\nPress Ctrl+C to stop the server');
});