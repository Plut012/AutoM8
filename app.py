from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import requests
import json
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"

# In-memory storage for workflows
workflows = {}
blocks = {}

class Block:
    def __init__(self, block_type, position, config=None):
        self.id = str(uuid.uuid4())
        self.type = block_type
        self.position = position
        self.config = config or {}
        self.inputs = {}
        self.outputs = {}
        self.connections_in = []
        self.connections_out = []
    
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'position': self.position,
            'config': self.config,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'connections_in': self.connections_in,
            'connections_out': self.connections_out
        }

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Workflow Builder</title>
        <style>
            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: #1a1a1a;
                color: #fff;
                overflow: hidden;
            }
            #container {
                display: flex;
                height: 100vh;
            }
            #sidebar {
                width: 250px;
                background: #2a2a2a;
                padding: 20px;
                overflow-y: auto;
            }
            #canvas-container {
                flex: 1;
                position: relative;
                overflow: hidden;
                background: #0f0f0f;
                background-image: 
                    linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px);
                background-size: 20px 20px;
            }
            #canvas {
                position: absolute;
                cursor: grab;
            }
            #output {
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 200px;
                background: #1a1a1a;
                border-top: 2px solid #3a3a3a;
                padding: 10px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 12px;
            }
            .block-template {
                background: #3a3a3a;
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 5px;
                cursor: grab;
                transition: background 0.2s;
            }
            .block-template:hover {
                background: #4a4a4a;
            }
            .block {
                position: absolute;
                background: #2a2a2a;
                border: 2px solid #4a4a4a;
                border-radius: 8px;
                padding: 10px;
                min-width: 150px;
                cursor: move;
                user-select: none;
            }
            .block.selected {
                border-color: #0066cc;
                box-shadow: 0 0 10px rgba(0,102,204,0.5);
            }
            .block-title {
                font-weight: bold;
                margin-bottom: 5px;
                text-align: center;
            }
            .port {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                position: absolute;
                cursor: crosshair;
            }
            .input-port {
                background: #cc6600;
                left: -6px;
                top: 50%;
                transform: translateY(-50%);
            }
            .output-port {
                background: #00cc66;
                right: -6px;
                top: 50%;
                transform: translateY(-50%);
            }
            .connection {
                stroke: #666;
                stroke-width: 2;
                fill: none;
                pointer-events: none;
            }
            button {
                background: #0066cc;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px;
                width: 100%;
            }
            button:hover {
                background: #0052a3;
            }
            textarea, input, select {
                width: 100%;
                padding: 5px;
                margin: 5px 0;
                background: #1a1a1a;
                border: 1px solid #3a3a3a;
                color: white;
                border-radius: 3px;
            }
            #config-panel {
                margin-top: 20px;
                padding: 10px;
                background: #1a1a1a;
                border-radius: 5px;
            }
            .config-section {
                margin-bottom: 15px;
            }
            .config-label {
                font-size: 12px;
                color: #aaa;
                margin-bottom: 3px;
            }
        </style>
    </head>
    <body>
        <div id="container">
            <div id="sidebar">
                <h3>Blocks</h3>
                <div class="block-template" data-type="input">Text Input</div>
                <div class="block-template" data-type="llm">LLM Query</div>
                <div class="block-template" data-type="output">Output Display</div>
                
                <h3>Controls</h3>
                <button onclick="runWorkflow()">Run Workflow</button>
                <button onclick="clearCanvas()">Clear Canvas</button>
                
                <div id="config-panel">
                    <h4>Block Configuration</h4>
                    <div id="config-content">
                        <p style="color: #666;">Select a block to configure</p>
                    </div>
                </div>
            </div>
            
            <div id="canvas-container">
                <svg id="canvas" width="5000" height="5000">
                    <g id="connections"></g>
                </svg>
                <div id="blocks-container"></div>
                <div id="output">
                    <div style="color: #666;">Output will appear here...</div>
                </div>
            </div>
        </div>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/uuid/8.3.2/uuid.min.js"></script>
        <script>
        let blocks = {};
        let connections = [];
        let selectedBlock = null;
        let draggingBlock = null;
        let connectingFrom = null;
        let panOffset = { x: 0, y: 0 };
        let isPanning = false;
        let lastMousePos = { x: 0, y: 0 };
        
        // Block templates
        document.querySelectorAll('.block-template').forEach(template => {
            template.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('blockType', template.dataset.type);
            });
        });
        
        // Canvas drag and drop
        const canvasContainer = document.getElementById('canvas-container');
        const blocksContainer = document.getElementById('blocks-container');
        const svg = document.getElementById('canvas');
        const connectionsGroup = document.getElementById('connections');
        
        canvasContainer.addEventListener('dragover', (e) => {
            e.preventDefault();
        });
        
        canvasContainer.addEventListener('drop', (e) => {
            e.preventDefault();
            const blockType = e.dataTransfer.getData('blockType');
            if (blockType) {
                const rect = canvasContainer.getBoundingClientRect();
                createBlock(blockType, e.clientX - rect.left - panOffset.x, e.clientY - rect.top - panOffset.y);
            }
        });
        
        // Panning
        canvasContainer.addEventListener('mousedown', (e) => {
            if (e.target === canvasContainer || e.target === svg) {
                isPanning = true;
                lastMousePos = { x: e.clientX, y: e.clientY };
                canvasContainer.style.cursor = 'grabbing';
            }
        });
        
        document.addEventListener('mousemove', (e) => {
            if (isPanning) {
                const dx = e.clientX - lastMousePos.x;
                const dy = e.clientY - lastMousePos.y;
                panOffset.x += dx;
                panOffset.y += dy;
                lastMousePos = { x: e.clientX, y: e.clientY };
                updatePan();
            }
        });
        
        document.addEventListener('mouseup', () => {
            isPanning = false;
            canvasContainer.style.cursor = 'grab';
        });
        
        function updatePan() {
            blocksContainer.style.transform = `translate(${panOffset.x}px, ${panOffset.y}px)`;
            svg.style.transform = `translate(${panOffset.x}px, ${panOffset.y}px)`;
        }
        
        function createBlock(type, x, y) {
            const id = uuid.v4();
            const blockEl = document.createElement('div');
            blockEl.className = 'block';
            blockEl.id = `block-${id}`;
            blockEl.style.left = x + 'px';
            blockEl.style.top = y + 'px';
            
            let title = '';
            switch(type) {
                case 'input': title = 'Text Input'; break;
                case 'llm': title = 'LLM Query'; break;
                case 'output': title = 'Output Display'; break;
            }
            
            blockEl.innerHTML = `
                <div class="block-title">${title}</div>
                ${type !== 'input' ? '<div class="port input-port" data-port="input"></div>' : ''}
                ${type !== 'output' ? '<div class="port output-port" data-port="output"></div>' : ''}
            `;
            
            blocksContainer.appendChild(blockEl);
            
            blocks[id] = {
                id: id,
                type: type,
                position: { x, y },
                config: getDefaultConfig(type),
                element: blockEl
            };
            
            // Block selection
            blockEl.addEventListener('mousedown', (e) => {
                if (!e.target.classList.contains('port')) {
                    selectBlock(id);
                    draggingBlock = id;
                    blockEl.style.zIndex = 1000;
                    const rect = blockEl.getBoundingClientRect();
                    const containerRect = canvasContainer.getBoundingClientRect();
                    blocks[id].dragOffset = {
                        x: e.clientX - rect.left + containerRect.left,
                        y: e.clientY - rect.top + containerRect.top
                    };
                }
            });
            
            // Port connections
            blockEl.querySelectorAll('.port').forEach(port => {
                port.addEventListener('mousedown', (e) => {
                    e.stopPropagation();
                    if (port.dataset.port === 'output') {
                        connectingFrom = { blockId: id, port: 'output' };
                    }
                });
                
                port.addEventListener('mouseup', (e) => {
                    e.stopPropagation();
                    if (connectingFrom && port.dataset.port === 'input' && connectingFrom.blockId !== id) {
                        createConnection(connectingFrom.blockId, id);
                    }
                    connectingFrom = null;
                });
            });
            
            fetch('/api/blocks', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    type: type,
                    position: { x, y },
                    config: blocks[id].config
                })
            });
        }
        
        function getDefaultConfig(type) {
            switch(type) {
                case 'input':
                    return { text: 'Enter your text here...' };
                case 'llm':
                    return { 
                        model: 'llama2',
                        prompt: 'Answer the following: {input}',
                        temperature: 0.7
                    };
                case 'output':
                    return {};
                default:
                    return {};
            }
        }
        
        function selectBlock(id) {
            document.querySelectorAll('.block').forEach(b => b.classList.remove('selected'));
            if (id) {
                blocks[id].element.classList.add('selected');
                selectedBlock = id;
                showBlockConfig(id);
            } else {
                selectedBlock = null;
            }
        }
        
        function showBlockConfig(id) {
            const block = blocks[id];
            const configContent = document.getElementById('config-content');
            
            let html = '';
            switch(block.type) {
                case 'input':
                    html = `
                        <div class="config-section">
                            <div class="config-label">Input Text</div>
                            <textarea id="config-text" rows="4">${block.config.text || ''}</textarea>
                        </div>
                    `;
                    break;
                case 'llm':
                    html = `
                        <div class="config-section">
                            <div class="config-label">Model</div>
                            <select id="config-model">
                                <option value="llama2" ${block.config.model === 'llama2' ? 'selected' : ''}>Llama 2</option>
                                <option value="mistral" ${block.config.model === 'mistral' ? 'selected' : ''}>Mistral</option>
                                <option value="codellama" ${block.config.model === 'codellama' ? 'selected' : ''}>Code Llama</option>
                            </select>
                        </div>
                        <div class="config-section">
                            <div class="config-label">Prompt Template</div>
                            <textarea id="config-prompt" rows="3">${block.config.prompt || ''}</textarea>
                            <small style="color: #666;">Use {input} for input text</small>
                        </div>
                        <div class="config-section">
                            <div class="config-label">Temperature</div>
                            <input type="number" id="config-temperature" min="0" max="2" step="0.1" value="${block.config.temperature || 0.7}">
                        </div>
                    `;
                    break;
            }
            
            if (html) {
                html += '<button onclick="saveBlockConfig()">Save Config</button>';
                configContent.innerHTML = html;
            }
        }
        
        function saveBlockConfig() {
            if (!selectedBlock) return;
            
            const block = blocks[selectedBlock];
            switch(block.type) {
                case 'input':
                    block.config.text = document.getElementById('config-text').value;
                    break;
                case 'llm':
                    block.config.model = document.getElementById('config-model').value;
                    block.config.prompt = document.getElementById('config-prompt').value;
                    block.config.temperature = parseFloat(document.getElementById('config-temperature').value);
                    break;
            }
            
            fetch(`/api/blocks/${selectedBlock}/config`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(block.config)
            });
        }
        
        function createConnection(fromId, toId) {
            const connection = {
                id: uuid.v4(),
                from: fromId,
                to: toId
            };
            
            connections.push(connection);
            drawConnection(connection);
            
            fetch('/api/connections', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(connection)
            });
        }
        
        function drawConnection(connection) {
            const fromBlock = blocks[connection.from].element;
            const toBlock = blocks[connection.to].element;
            const fromPort = fromBlock.querySelector('.output-port');
            const toPort = toBlock.querySelector('.input-port');
            
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('class', 'connection');
            path.setAttribute('id', `conn-${connection.id}`);
            
            updateConnectionPath(path, fromPort, toPort);
            connectionsGroup.appendChild(path);
            
            connection.element = path;
        }
        
        function updateConnectionPath(path, fromPort, toPort) {
            const fromRect = fromPort.getBoundingClientRect();
            const toRect = toPort.getBoundingClientRect();
            const containerRect = canvasContainer.getBoundingClientRect();
            
            const x1 = fromRect.left + fromRect.width/2 - containerRect.left - panOffset.x;
            const y1 = fromRect.top + fromRect.height/2 - containerRect.top - panOffset.y;
            const x2 = toRect.left + toRect.width/2 - containerRect.left - panOffset.x;
            const y2 = toRect.top + toRect.height/2 - containerRect.top - panOffset.y;
            
            const dx = Math.abs(x2 - x1) * 0.5;
            const d = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
            path.setAttribute('d', d);
        }
        
        function updateAllConnections() {
            connections.forEach(conn => {
                const fromBlock = blocks[conn.from].element;
                const toBlock = blocks[conn.to].element;
                const fromPort = fromBlock.querySelector('.output-port');
                const toPort = toBlock.querySelector('.input-port');
                updateConnectionPath(conn.element, fromPort, toPort);
            });
        }
        
        // Block dragging
        document.addEventListener('mousemove', (e) => {
            if (draggingBlock && blocks[draggingBlock]) {
                const block = blocks[draggingBlock];
                const containerRect = canvasContainer.getBoundingClientRect();
                const x = e.clientX - containerRect.left - block.dragOffset.x - panOffset.x;
                const y = e.clientY - containerRect.top - block.dragOffset.y - panOffset.y;
                
                block.element.style.left = x + 'px';
                block.element.style.top = y + 'px';
                block.position = { x, y };
                
                updateAllConnections();
            }
        });
        
        document.addEventListener('mouseup', () => {
            if (draggingBlock) {
                blocks[draggingBlock].element.style.zIndex = '';
                draggingBlock = null;
            }
        });
        
        async function runWorkflow() {
            const output = document.getElementById('output');
            output.innerHTML = '<div style="color: #0066cc;">Running workflow...</div>';
            
            try {
                const response = await fetch('/api/workflow/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        blocks: Object.values(blocks).map(b => ({
                            id: b.id,
                            type: b.type,
                            config: b.config
                        })),
                        connections: connections.map(c => ({
                            from: c.from,
                            to: c.to
                        }))
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    output.innerHTML = '<div style="color: #00cc66;">Workflow completed!</div>';
                    
                    result.outputs.forEach(out => {
                        const div = document.createElement('div');
                        div.style.marginTop = '10px';
                        div.style.padding = '10px';
                        div.style.background = '#2a2a2a';
                        div.style.borderRadius = '5px';
                        div.innerHTML = `<strong>${out.blockId}:</strong><br><pre style="white-space: pre-wrap;">${out.output}</pre>`;
                        output.appendChild(div);
                    });
                } else {
                    output.innerHTML = `<div style="color: #cc0000;">Error: ${result.error}</div>`;
                }
            } catch (error) {
                output.innerHTML = `<div style="color: #cc0000;">Error: ${error.message}</div>`;
            }
        }
        
        function clearCanvas() {
            if (confirm('Clear all blocks and connections?')) {
                blocksContainer.innerHTML = '';
                connectionsGroup.innerHTML = '';
                blocks = {};
                connections = [];
                selectedBlock = null;
                document.getElementById('config-content').innerHTML = '<p style="color: #666;">Select a block to configure</p>';
                document.getElementById('output').innerHTML = '<div style="color: #666;">Output will appear here...</div>';
            }
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Delete' && selectedBlock) {
                // Remove block
                const block = blocks[selectedBlock];
                block.element.remove();
                
                // Remove related connections
                connections = connections.filter(conn => {
                    if (conn.from === selectedBlock || conn.to === selectedBlock) {
                        conn.element.remove();
                        return false;
                    }
                    return true;
                });
                
                delete blocks[selectedBlock];
                selectedBlock = null;
            }
        });
        </script>
    </body>
    </html>
    ''')

@app.route('/api/blocks', methods=['POST'])
def create_block():
    data = request.json
    block = Block(data['type'], data['position'], data.get('config'))
    blocks[block.id] = block
    return jsonify(block.to_dict())

@app.route('/api/blocks/<block_id>/config', methods=['PUT'])
def update_block_config(block_id):
    if block_id in blocks:
        blocks[block_id].config = request.json
        return jsonify({'success': True})
    return jsonify({'error': 'Block not found'}), 404

@app.route('/api/connections', methods=['POST'])
def create_connection():
    data = request.json
    # Store connection info
    return jsonify({'success': True})

@app.route('/api/workflow/run', methods=['POST'])
def run_workflow():
    data = request.json
    workflow_blocks = data['blocks']
    workflow_connections = data['connections']
    
    # Build execution graph
    outputs = []
    block_outputs = {}
    
    try:
        # Find input blocks
        input_blocks = [b for b in workflow_blocks if b['type'] == 'input']
        
        for input_block in input_blocks:
            block_outputs[input_block['id']] = input_block['config'].get('text', '')
        
        # Process LLM blocks
        for connection in workflow_connections:
            from_id = connection['from']
            to_id = connection['to']
            
            to_block = next((b for b in workflow_blocks if b['id'] == to_id), None)
            
            if to_block and to_block['type'] == 'llm':
                # Get input from previous block
                input_text = block_outputs.get(from_id, '')
                
                # Prepare prompt
                prompt = to_block['config']['prompt'].replace('{input}', input_text)
                
                # Call Ollama
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        'model': to_block['config']['model'],
                        'prompt': prompt,
                        'temperature': to_block['config'].get('temperature', 0.7),
                        'stream': False
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    output = result.get('response', '')
                    block_outputs[to_id] = output
                    outputs.append({
                        'blockId': to_id,
                        'output': output
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Ollama error: {response.text}'
                    })
            
            elif to_block and to_block['type'] == 'output':
                # Display output
                input_text = block_outputs.get(from_id, '')
                outputs.append({
                    'blockId': to_id,
                    'output': input_text
                })
        
        return jsonify({
            'success': True,
            'outputs': outputs
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
