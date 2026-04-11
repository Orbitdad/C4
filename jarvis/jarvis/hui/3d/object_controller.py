"""
jarvis/hui/3d/object_controller.py
Frontend structure and logic for the 3D Holographic interface.
Refactored for Multi-Object Workspace natively inside the browser.
"""

HTML_LAYOUT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JARVIS Holographic Workspace</title>
    <style>
        body { margin: 0; overflow: hidden; background: #000; }
        #webcam { position: absolute; top: 0; left: 0; width: 100vw; height: 100vh; object-fit: cover; z-index: -1; transform: scaleX(-1); }
        #scene-container { position: absolute; top: 0; left: 0; width: 100vw; height: 100vh; }
        #debug { position: absolute; top: 10px; left: 10px; color: #00ffcc; font-family: monospace; font-size: 14px; text-shadow: 1px 1px 2px #000; pointer-events: none;}
    </style>
    <!-- Three.js from CDN -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <video id="webcam" autoplay playsinline></video>
    <div id="scene-container"></div>
    <div id="debug">Awaiting Holographic Link...<br><span id="target-id" style="color:#0088ff;">TARGET: NONE</span></div>
    
    <script>
        {MAIN_LOGIC}
    </script>
</body>
</html>
"""

JS_LOGIC = """
// ==========================================
// SceneManager: Handles rendering & webcam
// ==========================================
class SceneManager {
    constructor() {
        this.container = document.getElementById('scene-container');
        this.scene = new THREE.Scene();
        
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.z = 8;
        
        this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.container.appendChild(this.renderer.domElement);
        
        this.setupLights();
        this.startWebcam();
        
        window.addEventListener('resize', () => this.onResize());
    }
    
    setupLights() {
        const dLight = new THREE.DirectionalLight(0xffffff, 1);
        dLight.position.set(0, 10, 10);
        this.scene.add(dLight);
        this.scene.add(new THREE.AmbientLight(0x404040, 1.5));
    }
    
    async startWebcam() {
        const video = document.getElementById('webcam');
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
            video.srcObject = stream;
        } catch (err) {
            document.getElementById('debug').innerHTML += "<br>[!] Webcam Access Denied/Failed.";
        }
    }
    
    onResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
    
    render() {
        this.renderer.render(this.scene, this.camera);
    }
}

// ==========================================
// HologramObject: Represents a single geometry
// ==========================================
class HologramObject {
    constructor(id, scene, initialPos) {
        this.id = id;
        
        const geometry = new THREE.IcosahedronGeometry(1.2, 1);
        this.baseColor = 0x0088ff;
        this.activeColor = 0x00ffcc;
        
        this.material = new THREE.MeshStandardMaterial({ 
            color: this.baseColor,
            wireframe: true,
            emissive: this.baseColor,
            emissiveIntensity: 0.2
        });
        
        this.mesh = new THREE.Mesh(geometry, this.material);
        this.mesh.position.set(initialPos.x, initialPos.y, initialPos.z);
        scene.add(this.mesh);
        
        // Target state for interpolated transformations
        this.state = {
            scale: 1.0,
            rotY: 0.0,
            rotX: 0.0,
            posX: initialPos.x,
            posY: initialPos.y
        };
        
        this.isActive = false;
        this.visualScaleOffset = 0.0; // Extra scale bump when active
    }
    
    setActive(active) {
        this.isActive = active;
        this.material.color.setHex(active ? this.activeColor : this.baseColor);
        this.material.emissive.setHex(active ? this.activeColor : this.baseColor);
        this.material.emissiveIntensity = active ? 0.8 : 0.2;
    }
    
    update(dt) {
        const dtNorm = dt / 16.67;
        const lerpFactor = 0.1 * dtNorm;
        const clampedLerp = Math.min(1.0, lerpFactor);
        
        // Transform interpolation with visual bump for active state
        const visualScale = this.state.scale + (this.isActive ? 0.2 : 0.0);
        this.mesh.scale.lerp(new THREE.Vector3(visualScale, visualScale, visualScale), clampedLerp);
        
        this.mesh.rotation.y += (this.state.rotY - this.mesh.rotation.y) * clampedLerp;
        this.mesh.rotation.x += (this.state.rotX - this.mesh.rotation.x) * clampedLerp;
        
        this.mesh.position.x += (this.state.posX - this.mesh.position.x) * clampedLerp;
        // Ambient hover logic based on global time
        this.mesh.position.y += ((this.state.posY + Math.sin(Date.now() * 0.002) * 0.1) - this.mesh.position.y) * clampedLerp;
    }
}

// ==========================================
// ObjectManager: Handles Spawning & Selection
// ==========================================
class ObjectManager {
    constructor(sceneManager) {
        this.sm = sceneManager;
        this.objects = [];
        this.candidateId = null;
        this.consecutiveFrames = 0;
        this.activeObjectId = null;
        
        // Selection threshold in Normalized Device Coordinates (center is 0,0)
        this.SELECTION_THRESHOLD = 0.4;
        this.DEBOUNCE_FRAMES = 5;
        
        this.spawnObjects();
    }
    
    spawnObjects() {
        this.objects.push(new HologramObject("ALPHA_CUBE", this.sm.scene, {x: -3.5, y: 0, z: 0}));
        this.objects.push(new HologramObject("BETA_CUBE", this.sm.scene, {x: 0, y: 0, z: 0}));
        this.objects.push(new HologramObject("GAMMA_CUBE", this.sm.scene, {x: 3.5, y: 0, z: 0}));
    }
    
    getActiveObject() {
        return this.objects.find(o => o.id === this.activeObjectId);
    }
    
    updateSelection() {
        let closestObj = null;
        let minDistance = Infinity;
        
        // Option A: Distance to screen center (0,0 in NDC)
        for (let obj of this.objects) {
            const pos = obj.mesh.position.clone();
            pos.project(this.sm.camera);
            
            // Calculate distance to origin (0,0) across X and Y
            const dist = Math.sqrt(pos.x * pos.x + pos.y * pos.y);
            
            // Allow selecting if within threshold and z is in front of camera
            if (pos.z < 1.0 && dist < minDistance) {
                minDistance = dist;
                closestObj = obj;
            }
        }
        
        // Determine the best target id considering maximum picking threshold
        const bestId = (minDistance < this.SELECTION_THRESHOLD && closestObj) ? closestObj.id : null;
        
        // Frame-level Debounce Logic prevents flickering on boundary edge
        if (bestId === this.candidateId) {
            this.consecutiveFrames++;
        } else {
            this.candidateId = bestId;
            this.consecutiveFrames = 1;
        }
        
        if (this.consecutiveFrames >= this.DEBOUNCE_FRAMES) {
            if (this.activeObjectId !== this.candidateId) {
                this.activeObjectId = this.candidateId;
                
                // Update visuals for all objects when selection shifts
                this.objects.forEach(o => o.setActive(o.id === this.activeObjectId));
                
                const dbgTarget = document.getElementById("target-id");
                if (dbgTarget) {
                    dbgTarget.innerText = "TARGET: " + (this.activeObjectId || "NONE");
                    dbgTarget.style.color = this.activeObjectId ? "#00ffcc" : "#0088ff";
                }
            }
        }
    }
    
    update(dt) {
        this.updateSelection();
        this.objects.forEach(o => o.update(dt));
    }
}

// ==========================================
// InteractionController: Maps Websocket Data
// ==========================================
class InteractionController {
    constructor(objectManager) {
        this.om = objectManager;
        this.setupWebsocket();
    }
    
    setupWebsocket() {
        const debugEl = document.getElementById('debug');
        const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsHost = window.location.hostname || '127.0.0.1';
        this.ws = new WebSocket(wsProto + '//' + wsHost + ':8765');
        
        this.ws.onopen = () => { 
            debugEl.innerHTML = "HOLOGRAPHIC LINK ESTABLISHED [3D MODE ACTIVE]<br><span id='target-id' style='color:#0088ff;'>TARGET: NONE</span>"; 
        };
        this.ws.onclose = () => { 
            debugEl.innerHTML = "LINK LOST. RECONNECTING...<br><span id='target-id' style='color:#0088ff;'>TARGET: NONE</span>"; 
        };
        
        this.ws.onmessage = (event) => this.handleMessage(event);
    }
    
    handleMessage(event) {
        try {
            const payload = JSON.parse(event.data);
            if (payload.type !== "TRANSFORM") return;
            
            const activeObj = this.om.getActiveObject();
            if (!activeObj) return; // Discard interaction if no object is selected
            
            const action = payload.action;
            const delta = payload.delta;
            
            // Apply bounds and normalization scaling
            if (action === "ZOOM") {
                activeObj.state.scale = Math.max(0.5, Math.min(activeObj.state.scale + delta, 3.0));
            } else if (action === "ROTATE") {
                activeObj.state.rotY += (delta * 0.7); 
            } else if (action === "MOVE") {
                // Future expansion hook
            }
        } catch (e) {
            console.error("Payload parse error: ", e);
        }
    }
}

// ==========================================
// Main Execution Engine
// ==========================================
const sceneManager = new SceneManager();
const objectManager = new ObjectManager(sceneManager);
const interactionController = new InteractionController(objectManager);

let lastTime = performance.now();

function animate() {
    requestAnimationFrame(animate);
    const now = performance.now();
    const dt = now - lastTime;
    lastTime = now;
    
    objectManager.update(dt);
    sceneManager.render();
}
animate();
"""

def generate_frontend():
    return HTML_LAYOUT.replace("{MAIN_LOGIC}", JS_LOGIC)
