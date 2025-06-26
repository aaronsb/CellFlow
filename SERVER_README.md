# CellFlow Local Server

This is a minimal TypeScript-based web server to serve the CellFlow application locally, solving CORS issues when loading ES6 modules.

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Build the TypeScript server:
   ```bash
   npm run build
   ```

3. Start the server:
   ```bash
   npm start
   ```

   Or use the development command to build and run in one step:
   ```bash
   npm run dev
   ```

## Available Scripts

- `npm run build` - Compiles TypeScript to JavaScript
- `npm start` - Runs the compiled server
- `npm run dev` - Builds and runs the server
- `npm run watch` - Watches TypeScript files for changes and recompiles automatically

## Server Details

- Default port: 3000
- Serves static files from the project root directory
- Properly handles ES6 modules with correct MIME types
- CORS enabled for all origins
- Includes security headers for cross-origin isolation

## Access the Application

Once the server is running, open your browser and navigate to:
```
http://localhost:3000
```

The server will automatically serve the index.html file and all associated JavaScript modules with proper MIME types.