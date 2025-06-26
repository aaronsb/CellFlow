/**
 * Hybrid GPU Computing Server for CellFlow
 * Provides a WebSocket interface for the web frontend to communicate
 * with a native GPU compute backend
 */

import { WebSocketServer } from 'ws';
import { CellFlowNativeGPU } from './node-webgpu-compute.js';
import express from 'express';
import cors from 'cors';

class CellFlowHybridServer {
    constructor(port = 8080) {
        this.port = port;
        this.simulator = null;
        this.clients = new Set();
        this.frameInterval = null;
        this.isRunning = false;
        
        // Performance metrics
        this.metrics = {
            frameCount: 0,
            totalFrameTime: 0,
            lastFPS: 0
        };
    }

    async initialize() {
        // Initialize GPU simulator
        this.simulator = new CellFlowNativeGPU();
        await this.simulator.initialize();
        
        // Create HTTP server for health checks
        const app = express();
        app.use(cors());
        
        app.get('/health', (req, res) => {
            res.json({
                status: 'healthy',
                particleCount: this.simulator.particleCount,
                isRunning: this.isRunning,
                fps: this.metrics.lastFPS,
                clients: this.clients.size
            });
        });
        
        const server = app.listen(this.port, () => {
            console.log(`HTTP server listening on port ${this.port}`);
        });
        
        // Create WebSocket server
        this.wss = new WebSocketServer({ server });
        
        this.wss.on('connection', (ws) => {
            console.log('New client connected');
            this.handleClient(ws);
        });
        
        console.log(`CellFlow Hybrid Server initialized on port ${this.port}`);
    }

    handleClient(ws) {
        this.clients.add(ws);
        
        // Send initial configuration
        ws.send(JSON.stringify({
            type: 'config',
            data: {
                particleCount: this.simulator.particleCount,
                numParticleTypes: this.simulator.numParticleTypes,
                params: this.simulator.params,
                canRender: true
            }
        }));
        
        ws.on('message', async (message) => {
            try {
                const msg = JSON.parse(message.toString());
                await this.handleMessage(ws, msg);
            } catch (error) {
                console.error('Error handling message:', error);
                ws.send(JSON.stringify({
                    type: 'error',
                    error: error.message
                }));
            }
        });
        
        ws.on('close', () => {
            this.clients.delete(ws);
            console.log('Client disconnected');
            
            // Stop simulation if no clients
            if (this.clients.size === 0 && this.isRunning) {
                this.stopSimulation();
            }
        });
        
        ws.on('error', (error) => {
            console.error('WebSocket error:', error);
            this.clients.delete(ws);
        });
    }

    async handleMessage(ws, msg) {
        switch (msg.type) {
            case 'start':
                this.startSimulation();
                break;
                
            case 'stop':
                this.stopSimulation();
                break;
                
            case 'updateParams':
                this.simulator.updateParameters(msg.data);
                this.broadcastToClients({
                    type: 'paramsUpdated',
                    data: this.simulator.params
                });
                break;
                
            case 'updateParticleCount':
                await this.updateParticleCount(msg.data.count);
                break;
                
            case 'regenerateForces':
                this.simulator.initializeParticles();
                break;
                
            case 'getSnapshot':
                const snapshot = await this.getParticleSnapshot();
                ws.send(JSON.stringify({
                    type: 'snapshot',
                    data: snapshot
                }));
                break;
                
            case 'setRenderMode':
                // Toggle between sending full data or compressed data
                ws.renderMode = msg.data.mode;
                break;
                
            default:
                console.warn('Unknown message type:', msg.type);
        }
    }

    startSimulation() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.metrics.frameCount = 0;
        this.metrics.totalFrameTime = 0;
        
        const targetFPS = 60;
        const frameTime = 1000 / targetFPS;
        let lastFrameTime = performance.now();
        
        const simulationLoop = async () => {
            if (!this.isRunning) return;
            
            const startTime = performance.now();
            
            // Run simulation step
            await this.simulator.simulate(1);
            
            // Get particle data
            const particleData = await this.simulator.getParticleData();
            
            // Prepare frame data
            const frameData = this.prepareFrameData(particleData);
            
            // Broadcast to all clients
            this.broadcastToClients({
                type: 'frame',
                data: frameData,
                timestamp: Date.now()
            });
            
            // Update metrics
            const frameProcessTime = performance.now() - startTime;
            this.metrics.frameCount++;
            this.metrics.totalFrameTime += frameProcessTime;
            
            if (this.metrics.frameCount % 60 === 0) {
                this.metrics.lastFPS = 1000 / (this.metrics.totalFrameTime / this.metrics.frameCount);
                console.log(`FPS: ${this.metrics.lastFPS.toFixed(1)}, Frame time: ${frameProcessTime.toFixed(2)}ms`);
                this.metrics.frameCount = 0;
                this.metrics.totalFrameTime = 0;
            }
            
            // Schedule next frame
            const nextFrameDelay = Math.max(0, frameTime - frameProcessTime);
            setTimeout(simulationLoop, nextFrameDelay);
        };
        
        simulationLoop();
        console.log('Simulation started');
    }

    stopSimulation() {
        this.isRunning = false;
        console.log('Simulation stopped');
    }

    prepareFrameData(particleData) {
        // Convert raw particle data to a more efficient format for transmission
        const particleCount = this.simulator.particleCount;
        const positions = new Float32Array(particleCount * 2);
        const types = new Uint8Array(particleCount);
        const velocities = new Float32Array(particleCount * 2);
        
        // Extract relevant data
        for (let i = 0; i < particleCount; i++) {
            const base = i * 8;
            positions[i * 2] = particleData[base];     // x
            positions[i * 2 + 1] = particleData[base + 1]; // y
            velocities[i * 2] = particleData[base + 2];    // vx
            velocities[i * 2 + 1] = particleData[base + 3]; // vy
            types[i] = particleData[base + 6]; // type
        }
        
        // Option 1: Send as binary data (more efficient)
        // return Buffer.concat([
        //     Buffer.from(positions.buffer),
        //     Buffer.from(types.buffer),
        //     Buffer.from(velocities.buffer)
        // ]);
        
        // Option 2: Send as base64 (easier to handle in browser)
        return {
            positions: Buffer.from(positions.buffer).toString('base64'),
            types: Buffer.from(types.buffer).toString('base64'),
            velocities: Buffer.from(velocities.buffer).toString('base64'),
            count: particleCount
        };
    }

    async getParticleSnapshot() {
        const particleData = await this.simulator.getParticleData();
        return this.prepareFrameData(particleData);
    }

    async updateParticleCount(newCount) {
        // This would require recreating buffers in the native implementation
        console.log(`Updating particle count to ${newCount}`);
        // Implementation would go here
        
        this.broadcastToClients({
            type: 'particleCountUpdated',
            data: { count: newCount }
        });
    }

    broadcastToClients(message) {
        const data = JSON.stringify(message);
        this.clients.forEach(client => {
            if (client.readyState === 1) { // WebSocket.OPEN
                client.send(data);
            }
        });
    }

    async shutdown() {
        this.stopSimulation();
        this.wss.close();
        if (this.simulator) {
            this.simulator.destroy();
        }
        console.log('Server shutdown complete');
    }
}

// Example client code for the web frontend
export const clientExample = `
// Web frontend connection example
class CellFlowClient {
    constructor() {
        this.ws = null;
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.particleData = null;
    }

    connect(url = 'ws://localhost:8080') {
        this.ws = new WebSocket(url);
        
        this.ws.onopen = () => {
            console.log('Connected to CellFlow server');
            this.ws.send(JSON.stringify({ type: 'start' }));
        };
        
        this.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            
            switch (msg.type) {
                case 'config':
                    this.handleConfig(msg.data);
                    break;
                    
                case 'frame':
                    this.handleFrame(msg.data);
                    break;
                    
                case 'error':
                    console.error('Server error:', msg.error);
                    break;
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        this.ws.onclose = () => {
            console.log('Disconnected from server');
        };
    }

    handleConfig(config) {
        console.log('Received config:', config);
        this.particleCount = config.particleCount;
        this.numTypes = config.numParticleTypes;
    }

    handleFrame(frameData) {
        // Decode base64 data
        const positions = new Float32Array(
            Uint8Array.from(atob(frameData.positions), c => c.charCodeAt(0)).buffer
        );
        const types = new Uint8Array(
            Uint8Array.from(atob(frameData.types), c => c.charCodeAt(0)).buffer
        );
        
        // Render particles
        this.renderParticles(positions, types);
    }

    renderParticles(positions, types) {
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        const colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'];
        
        for (let i = 0; i < positions.length / 2; i++) {
            const x = positions[i * 2];
            const y = positions[i * 2 + 1];
            const type = types[i];
            
            this.ctx.fillStyle = colors[type % colors.length];
            this.ctx.beginPath();
            this.ctx.arc(x, y, 2, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    updateParams(params) {
        this.ws.send(JSON.stringify({
            type: 'updateParams',
            data: params
        }));
    }
}
`;

// Run server if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    const server = new CellFlowHybridServer(8080);
    
    server.initialize().catch(error => {
        console.error('Failed to initialize server:', error);
        process.exit(1);
    });
    
    // Handle graceful shutdown
    process.on('SIGINT', async () => {
        console.log('\nShutting down server...');
        await server.shutdown();
        process.exit(0);
    });
}