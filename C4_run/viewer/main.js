/**
 * main.js — C4 Three.js 3D Model Viewer
 * ======================================
 * ES Module. Loaded by index.html via <script type="module">.
 *
 * Architecture (six self-contained subsystems):
 * ┌──────────────────────────────────────────────────────────┐
 * │  SceneManager   — Renderer, Camera, Lights, OrbitControls │
 * │  ModelLoader    — GLTFLoader, mesh traversal, part store  │
 * │  ExplodedView   — Lerp-based explode / reassemble anim    │
 * │  UIOverlay      — HUD panel updates, command log, toasts  │
 * │  CommandHandler — Routes WS commands → subsystem calls    │
 * │  WebSocketClient— Auto-reconnect WS with back-off        │
 * └──────────────────────────────────────────────────────────┘
 *
 * Communication:
 *   Python backend  ──ws://localhost:8765──►  CommandHandler  ──► subsystems
 *   Toolbar buttons                        ──► subsystems directly
 */

import * as THREE from 'three';
import { OrbitControls }  from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader }     from 'three/addons/loaders/GLTFLoader.js';
import { RGBELoader }     from 'three/addons/loaders/RGBELoader.js';

// ─────────────────────────────────────────────────────────────────────────────
//  CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────

const WS_URL          = 'ws://localhost:8765';  // C4 Python backend
const MODELS_PATH     = './models/';            // Folder served by serve.py
const DEFAULT_MODEL   = 'arc-reactor.glb';
const EXPLODE_FACTOR  = 2.8;                    // How far parts spread (scene-unit multiplier)
const LERP_SPEED      = 0.055;                  // 0 → 1: higher = snappier animation
const LOG_MAX_ENTRIES = 8;                      // Max lines shown in command log panel

// ─────────────────────────────────────────────────────────────────────────────
//  APP STATE
// ─────────────────────────────────────────────────────────────────────────────

const AppState = {
  modelLoaded: false,
  exploded: false,
  selectedPart: null,
  draggedPart: null,
  assemblyMode: true
};

// ─────────────────────────────────────────────────────────────────────────────
//  UI OVERLAY
//  Updates all HUD DOM elements. Pure side-effects, zero state of its own.
// ─────────────────────────────────────────────────────────────────────────────

const UIOverlay = (() => {
  const elStatusDot  = document.getElementById('status-dot');
  const elStatusText = document.getElementById('status-text');
  const elModelName  = document.getElementById('model-name');
  const elMeshCount  = document.getElementById('mesh-count');
  const elViewState  = document.getElementById('view-state');
  const elLog        = document.getElementById('command-log');
  const elLoading    = document.getElementById('loading-overlay');
  const elLoadText   = document.getElementById('loading-text');
  const elToast      = document.getElementById('error-toast');

  let toastTimer = null;

  /** Update WebSocket connection status indicator */
  function setConnectionStatus(state) {
    // state: 'connecting' | 'connected' | 'disconnected'
    const labels = {
      connecting:   'Connecting…',
      connected:    'C4 Backend online',
      disconnected: 'No backend connection',
    };
    elStatusDot.className  = `status-dot ${state}`;
    elStatusText.textContent = labels[state] || state;
  }

  /** Display loaded model filename and mesh count */
  function setModelInfo(filename, meshCount) {
    elModelName.textContent = filename || 'No model loaded';
    elMeshCount.textContent = meshCount != null ? `${meshCount} parts` : '—';
  }

  /** Display current explode/assembled state */
  function setViewState(isExploded) {
    elViewState.textContent = isExploded ? '🔴 Exploded' : '🟢 Assembled';
    elViewState.style.color = isExploded ? '#ff6b6b' : 'var(--c4-teal)';
  }

  /** Add an entry to the scrolling command log */
  function logCommand(text) {
    const now   = new Date();
    const time  = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
    const entry = document.createElement('div');
    entry.className = 'cmd-entry';
    entry.innerHTML = `<span class="cmd-time">${time}</span><span class="cmd-text">» ${text}</span>`;
    elLog.prepend(entry);
    // Trim old entries
    while (elLog.children.length > LOG_MAX_ENTRIES) {
      elLog.removeChild(elLog.lastChild);
    }
  }

  /** Show / hide the fullscreen loading overlay */
  function showLoading(message = 'Loading…') {
    elLoadText.textContent = message;
    elLoading.classList.remove('hidden');
  }

  function hideLoading() {
    elLoading.classList.add('hidden');
  }

  /** Flash a brief error toast at the bottom of the screen */
  function showError(message, durationMs = 4000) {
    elToast.textContent = message;
    elToast.classList.add('visible');
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => elToast.classList.remove('visible'), durationMs);
  }

  return { setConnectionStatus, setModelInfo, setViewState, logCommand, showLoading, hideLoading, showError };
})();

// ─────────────────────────────────────────────────────────────────────────────
//  SCENE MANAGER
//  Owns: renderer, camera, lights, controls, animation loop.
// ─────────────────────────────────────────────────────────────────────────────

const SceneManager = (() => {
  const canvas   = document.getElementById('viewer-canvas');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled  = true;
  renderer.shadowMap.type     = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace   = THREE.SRGBColorSpace;
  renderer.toneMapping        = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;

  const scene  = new THREE.Scene();
  scene.background = new THREE.Color(0x020d18);
  // Subtle cyan fog for depth
  scene.fog = new THREE.FogExp2(0x020d18, 0.025);

  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 1000);
  camera.position.set(3, 2, 5);

  // OrbitControls — lets user drag / zoom the model
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping    = true;
  controls.dampingFactor    = 0.07;
  controls.minDistance      = 0.5;
  controls.maxDistance      = 30;
  controls.autoRotate       = false;
  controls.autoRotateSpeed  = 1.8;
  controls.target.set(0, 0, 0);

  // ── Lighting ───────────────────────────────────────────────────────────────

  // Ambient — fills shadows with a cool tint
  const ambient = new THREE.AmbientLight(0x112244, 1.8);
  scene.add(ambient);

  // Key light — top-right warm
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.5);
  keyLight.position.set(5, 8, 6);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.camera.near = 0.1;
  keyLight.shadow.camera.far  = 50;
  scene.add(keyLight);

  // Fill light — left cool blue
  const fillLight = new THREE.DirectionalLight(0x00c8ff, 0.8);
  fillLight.position.set(-6, 2, -4);
  scene.add(fillLight);

  // Rim light — backlight cyan
  const rimLight = new THREE.PointLight(0x00ffcc, 1.2, 20);
  rimLight.position.set(-3, -1, -5);
  scene.add(rimLight);

  // Ground reflection plane (subtle, receives shadows)
  const groundGeo = new THREE.PlaneGeometry(30, 30);
  const groundMat = new THREE.MeshStandardMaterial({
    color:     0x020d18,
    roughness: 0.9,
    metalness: 0.1,
    transparent: true,
    opacity:   0.6,
  });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.01;
  ground.receiveShadow = true;
  scene.add(ground);

  // Grid helper for the Iron Man holographic look
  const grid = new THREE.GridHelper(20, 40, 0x003344, 0x001a2a);
  grid.position.y = 0;
  scene.add(grid);

  // ── Resize handling ────────────────────────────────────────────────────────

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // ── Raycasting (Selection & Drag) ──────────────────────────────────────────
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  const dragPlane = new THREE.Plane();

  window.addEventListener('mousemove', (event) => {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
  });

  function _performRaycast() {
    if (!AppState.modelLoaded) return;
    
    // Reset previous selection emissive highlight
    if (AppState.selectedPart && AppState.selectedPart.material) {
      if (AppState.selectedPart.material.emissive) {
         AppState.selectedPart.material.emissive.setHex(0x000000);
      }
      AppState.selectedPart = null;
    }

    raycaster.setFromCamera(mouse, camera);
    const parts = ModelLoader.getParts().map(p => p.mesh).filter(m => m.visible);
    const intersects = raycaster.intersectObjects(parts, false);

    if (intersects.length > 0) {
      const mesh = intersects[0].object;
      AppState.selectedPart = mesh;
      // Inject emissive highlight for visual feedback
      if (mesh.material && mesh.material.emissive) {
        mesh.material.emissive.setHex(0x00aaff);
        mesh.material.emissiveIntensity = 0.5;
      }
    }
  }

  function setPointerSync(x, y, gesture) {
      if (!AppState.modelLoaded) return;
      
      const elPointer = document.getElementById('vision-pointer');
      if (elPointer) {
          elPointer.classList.add('active');
          elPointer.style.left = (x * 100) + '%';
          elPointer.style.top = (y * 100) + '%';
          if (gesture === 'PINCH') {
              elPointer.classList.add('pinching');
          } else {
              elPointer.classList.remove('pinching');
          }
      }

      // Override mouse tracker with ML vision coordinates
      mouse.x = (x * 2) - 1;
      mouse.y = -(y * 2) + 1;
      
      raycaster.setFromCamera(mouse, camera);
      
      if (gesture === 'PINCH') {
          if (AppState.draggedPart) {
              const intersectPoint = new THREE.Vector3();
              raycaster.ray.intersectPlane(dragPlane, intersectPoint);
              if (intersectPoint) {
                  // If parented, convert world intersect to local space
                  if (AppState.draggedPart.parent) {
                      AppState.draggedPart.parent.worldToLocal(intersectPoint);
                  }
                  AppState.draggedPart.position.copy(intersectPoint);
              }
          } else {
              // Try to grab
              const parts = ModelLoader.getParts().map(p => p.mesh).filter(m => m.visible);
              const intersects = raycaster.intersectObjects(parts, true); // True fixes nested mesh issues
              
              if (intersects.length > 0) {
                  AppState.draggedPart = intersects[0].object;
                  
                  // Make the dragged part the selected part as well, for UI feedback
                  if (AppState.selectedPart && AppState.selectedPart !== AppState.draggedPart && AppState.selectedPart.material && AppState.selectedPart.material.emissive) {
                      AppState.selectedPart.material.emissive.setHex(0x000000);
                  }
                  AppState.selectedPart = AppState.draggedPart;
                  if (AppState.selectedPart.material && AppState.selectedPart.material.emissive) {
                      AppState.selectedPart.material.emissive.setHex(0x00aaff);
                      AppState.selectedPart.material.emissiveIntensity = 0.5;
                  }
                  
                  // Setup drag plane for this object pointing at camera
                  const targetWorldPos = new THREE.Vector3();
                  AppState.draggedPart.getWorldPosition(targetWorldPos);
                  
                  dragPlane.setFromNormalAndCoplanarPoint(
                      camera.getWorldDirection(new THREE.Vector3()).negate(), 
                      targetWorldPos
                  );
                  controls.enabled = false;
              }
          }
      } else {
          // Not pinching, drop Part
          if (AppState.draggedPart) {
              if (AppState.assemblyMode && typeof AssemblySystem !== 'undefined') {
                  AssemblySystem.checkSnap(AppState.draggedPart);
              }
              AppState.draggedPart = null;
              controls.enabled = true;
          }
      }
  }

  // ── Animation loop ─────────────────────────────────────────────────────────

  const _onTickCallbacks = [];

  function addTickCallback(fn) { _onTickCallbacks.push(fn); }

  function startLoop() {
    renderer.setAnimationLoop(() => {
      controls.update();
      _performRaycast();
      _onTickCallbacks.forEach(fn => fn());
      renderer.render(scene, camera);
    });
  }

  return { scene, camera, controls, renderer, addTickCallback, startLoop, setPointerSync };
})();

// ─────────────────────────────────────────────────────────────────────────────
//  MODEL LOADER
//  Loads a .glb file, traverses its mesh hierarchy, and stores per-part data:
//    { mesh, originalPosition, explodedPosition }
// ─────────────────────────────────────────────────────────────────────────────

const ModelLoader = (() => {
  const loader = new GLTFLoader();

  // Currently loaded model root object (THREE.Group)
  let _modelRoot = null;

  // Array of part descriptors: { mesh, origPos, explodedPos }
  let _parts = [];

  /**
   * Load a GLB model from /models/<filename>.
   * Replaces any previously loaded model.
   * @returns {Promise<void>}
   */
  async function load(filename) {
    UIOverlay.showLoading(`Loading ${filename}…`);
    UIOverlay.logCommand(`load_model → ${filename}`);

    // Discard previous model
    if (_modelRoot) {
      SceneManager.scene.remove(_modelRoot);
      _modelRoot.traverse(obj => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
          else obj.material.dispose();
        }
      });
      _modelRoot = null;
      _parts     = [];
    }

    const url = `${MODELS_PATH}${filename}`;

    try {
      const gltf = await loader.loadAsync(url);
      _modelRoot = gltf.scene;

      // Centre the model at the scene origin
      _centreModel(_modelRoot);

      // Enable shadows on every mesh
      _modelRoot.traverse(obj => {
        if (obj.isMesh) {
          obj.castShadow    = true;
          obj.receiveShadow = true;
        }
      });

      SceneManager.scene.add(_modelRoot);

      // Collect all mesh parts and compute exploded positions
      _collectParts(_modelRoot);

      if (_parts.length === 0) {
        UIOverlay.showError('⚠  No mesh parts found in this model.');
        console.warn('[ModelLoader] Model loaded but contains no meshes.');
      } else {
        console.info(`[ModelLoader] Loaded "${filename}" — ${_parts.length} mesh part(s).`);
      }

      UIOverlay.setModelInfo(filename, _parts.length);
      UIOverlay.hideLoading();
      UIOverlay.logCommand(`loaded ${_parts.length} parts ✓`);
      AppState.modelLoaded = true;
      AppState.selectedPart = null;
    } catch (err) {
      console.error('[ModelLoader] Failed to load model:', err);
      UIOverlay.hideLoading();
      UIOverlay.showError(`Failed to load "${filename}". Check models/ directory.`);
      UIOverlay.logCommand(`ERROR: ${filename} not found`);
      AppState.modelLoaded = false;
    }
  }

  /**
   * Returns a shallow copy of the parts array for ExplodedView.
   */
  function getParts() { return [..._parts]; }

  /**
   * True if a model is currently in the scene.
   */
  function isLoaded() { return _modelRoot !== null; }

  // ── Private helpers ────────────────────────────────────────────────────────

  /** Centre the model's bounding box at the world origin and lift to y=0 */
  function _centreModel(root) {
    const box    = new THREE.Box3().setFromObject(root);
    const centre = new THREE.Vector3();
    box.getCenter(centre);
    // Shift so centre is at origin; floor sits at y=0
    root.position.sub(centre);
    root.position.y += (box.max.y - box.min.y) / 2;
  }

  /**
   * Traverse the loaded GLTF hierarchy.
   * For each Mesh found:
   *   - record its world position as the original position
   *   - compute an exploded position relative to the bounding-sphere centroid
   */
  function _collectParts(root) {
    _parts = [];

    // Bounding sphere of the entire model — used as the explode origin
    const box    = new THREE.Box3().setFromObject(root);
    const centre = new THREE.Vector3();
    box.getCenter(centre);

    root.traverse(obj => {
      if (!obj.isMesh) return;

      // World position of this mesh
      const worldPos = new THREE.Vector3();
      obj.getWorldPosition(worldPos);

      // Direction from model centre to this mesh (for explode spreading)
      const dir = worldPos.clone().sub(centre).normalize();

      // Place exploded position along direction * EXPLODE_FACTOR
      // For meshes at the exact centre, pick a random outward direction
      if (dir.length() < 0.001) {
        dir.set(
          (Math.random() - 0.5),
          (Math.random() - 0.5),
          (Math.random() - 0.5)
        ).normalize();
      }

      const explodedPos = worldPos.clone().add(
        dir.multiplyScalar(EXPLODE_FACTOR + Math.random() * 0.3)
      );

      _parts.push({
        mesh:         obj,
        origPos:      worldPos.clone(),
        explodedPos:  explodedPos,
        // Current lerp target — starts at original position
        targetPos:    worldPos.clone(),
        currentParent: obj.parent,
        isAttached:   false
      });
    });
  }

  return { load, getParts, isLoaded, getRoot: () => _modelRoot };
})();

// ─────────────────────────────────────────────────────────────────────────────
//  EXPLODED VIEW
//  Lerp-animates every mesh between assembled and exploded positions.
// ─────────────────────────────────────────────────────────────────────────────

const ExplodedView = (() => {
  let _isExploded = false;
  let _animating  = false;

  // Register the per-frame lerp update with SceneManager
  SceneManager.addTickCallback(_tick);

  function _tick() {
    if (!_animating) return;
    const parts  = ModelLoader.getParts();
    let   done   = true;

    parts.forEach(part => {
      const mesh = part.mesh;
      if (!mesh.parent) return; // already removed from scene

      // Convert target world position to local space
      const localTarget = new THREE.Vector3();
      if (mesh.parent) {
        mesh.parent.worldToLocal(localTarget.copy(part.targetPos));
      } else {
        localTarget.copy(part.targetPos);
      }

      // Lerp toward target
      mesh.position.lerp(localTarget, LERP_SPEED);

      // Stop animating when close enough
      if (mesh.position.distanceTo(localTarget) > 0.005) done = false;
    });

    if (done) {
      _animating = false;
    }
  }

  /** Toggle between exploded and assembled states. */
  function toggle() {
    if (!ModelLoader.isLoaded()) {
      UIOverlay.showError('No model loaded — say "load model" first.');
      return;
    }
    _isExploded ? _assemble() : _explode();
  }

  function _explode() {
    if (AppState.exploded) return;
    AppState.exploded = true;
    _animating  = true;
    ModelLoader.getParts().forEach(part => {
      part.targetPos.copy(part.explodedPos);
    });
    UIOverlay.setViewState(true);
    UIOverlay.logCommand('explode_model → animating');
  }

  function _assemble() {
    if (!AppState.exploded) return;
    AppState.exploded = false;
    _animating  = true;
    ModelLoader.getParts().forEach(part => {
      part.targetPos.copy(part.origPos);
    });
    UIOverlay.setViewState(false);
    UIOverlay.logCommand('reset_model → animating');
  }

  function explode()  { if (!AppState.exploded) _explode(); }
  function reset()    { if (AppState.exploded)  _assemble(); }
  function isExploded() { return AppState.exploded; }

  // ── Part manipulation ─────────────────────────────────────────────────────
  
  function removeSelectedPart() {
    if (!AppState.selectedPart) {
      UIOverlay.showError("No part selected to remove!");
      return;
    }
    AppState.selectedPart.visible = false;
    UIOverlay.logCommand(`Removed part`);
  }
  
  function addHiddenPartsBack() {
    ModelLoader.getParts().forEach(part => {
      part.mesh.visible = true;
    });
    UIOverlay.logCommand(`Restored hidden parts`);
  }

  return { toggle, explode, reset, isExploded, removeSelectedPart, addHiddenPartsBack };
})();

// ─────────────────────────────────────────────────────────────────────────────
//  ASSEMBLY SYSTEM
//  Handles manual part attachment, detachment, grouping, and snapping logic.
// ─────────────────────────────────────────────────────────────────────────────

const AssemblySystem = (() => {
  const SNAP_THRESHOLD = 0.5;

  function attachPart(child, parent) {
    if (!child || !parent) return;
    parent.attach(child);
    
    const partData = ModelLoader.getParts().find(p => p.mesh === child);
    if (partData) {
      partData.currentParent = parent;
      partData.isAttached = true;
    }
    UIOverlay.logCommand(`Attached part structure`);
  }

  function detachPart(child) {
    if (!child) return;
    SceneManager.scene.attach(child);

    const partData = ModelLoader.getParts().find(p => p.mesh === child);
    if (partData) {
      partData.currentParent = SceneManager.scene;
      partData.isAttached = false;
    }
    UIOverlay.logCommand(`Detached part`);
  }

  function createGroup(partsArr) {
    if (!partsArr || partsArr.length < 1) return;
    const newGroup = new THREE.Group();
    SceneManager.scene.add(newGroup);
    partsArr.forEach(part => {
      newGroup.attach(part);
      const partData = ModelLoader.getParts().find(p => p.mesh === part);
      if (partData) {
        partData.currentParent = newGroup;
        partData.isAttached = true;
      }
    });

    UIOverlay.logCommand(`Created new group`);
    return newGroup;
  }

  function checkSnap(droppedMesh) {
    if (!droppedMesh) return;
    
    let closestDist = Infinity;
    let closestTarget = null;
    
    const worldDroppedPos = new THREE.Vector3();
    droppedMesh.getWorldPosition(worldDroppedPos);

    ModelLoader.getParts().forEach(part => {
      if (part.mesh === droppedMesh) return;
      if (!part.mesh.visible) return;
      
      const worldTargetPos = new THREE.Vector3();
      part.mesh.getWorldPosition(worldTargetPos);

      const dist = worldDroppedPos.distanceTo(worldTargetPos);
      if (dist < SNAP_THRESHOLD && dist < closestDist) {
         closestDist = dist;
         closestTarget = part.mesh;
      }
    });

    if (closestTarget) {
      closestTarget.attach(droppedMesh);
      // Snap closely to center of target for rigid assembly feel
      droppedMesh.position.set(0, 0, 0); 
      droppedMesh.rotation.set(0, 0, 0); 
      
      const partData = ModelLoader.getParts().find(p => p.mesh === droppedMesh);
      if (partData) {
        partData.currentParent = closestTarget;
        partData.isAttached = true;
      }
      UIOverlay.logCommand(`Snapped & Attached part`);
    } else {
      detachPart(droppedMesh);
    }
  }

  return { attachPart, detachPart, createGroup, checkSnap };
})();

// ─────────────────────────────────────────────────────────────────────────────
//  COMMAND HANDLER
//  Maps incoming WebSocket command strings → subsystem calls.
//  Extensible: add new cases here to support voice / gesture inputs.
// ─────────────────────────────────────────────────────────────────────────────

const CommandHandler = (() => {
  /**
   * Execute a command received from C4 backend (or toolbar).
   * @param {string} command  - e.g. "explode", "reset", "load_model"
   * @param {object} payload  - full WS message payload (may contain extra params)
   */
  function execute(command, payload = {}) {
    UIOverlay.logCommand(`cmd ← ${command}`);
    console.info(`[CommandHandler] Received: ${command}`, payload);

    if (!AppState.modelLoaded && command !== "load_model" && command !== "connected") {
      UIOverlay.showError("Action ignored. No model is actively loaded.");
      return;
    }

    switch (command) {
      // ── 3D model commands ─────────────────────────────────────────────────
      case 'explode':
      case 'explode_model':
        ExplodedView.explode();
        break;

      case 'reset':
      case 'reset_model':
        ExplodedView.reset();
        break;

      case 'toggle_explode':
        ExplodedView.toggle();
        break;

      case 'remove_part':
        ExplodedView.removeSelectedPart();
        break;
        
      case 'add_part':
        ExplodedView.addHiddenPartsBack();
        break;

      case 'attach_part': {
        const source = payload.source || 'selected';
        const target = payload.target || 'core';
        
        const sourceMesh = (source === 'selected') ? AppState.selectedPart : null;
        let targetMesh = null;
        if (target === 'core') targetMesh = ModelLoader.getRoot();
        
        if (sourceMesh && targetMesh) {
           AssemblySystem.attachPart(sourceMesh, targetMesh);
        } else if (sourceMesh) {
           AssemblySystem.checkSnap(sourceMesh);
        } else {
           UIOverlay.showError("Needs a selected part to attach");
        }
        break;
      }

      case 'detach_part': {
        if (AppState.selectedPart) {
           AssemblySystem.detachPart(AppState.selectedPart);
        } else {
           UIOverlay.showError("No part selected to detach");
        }
        break;
      }

      case 'create_group': {
        if (AppState.selectedPart) {
           AssemblySystem.createGroup([AppState.selectedPart]);
        } else {
           UIOverlay.showError("Select a part to base a group on");
        }
        break;
      }

      case 'move_part': {
        UIOverlay.logCommand("move_part: Drag directly via gesture");
        break;
      }

      case 'load_model': {
        const params = payload.params || {};
        const model = params.model || DEFAULT_MODEL;
        ModelLoader.load(model);
        break;
      }

      case 'rotate':
      case 'rotate_model':
      case 'rotate_left':
      case 'rotate_right': {
        const increment = (command.includes('left') || (payload.params && payload.params.direction === 'left')) ? -0.15 : 0.15;
        
        // If it's just 'rotate' with no specific direction, it acts as a toggle.
        if (command === 'rotate' && (!payload.params || !payload.params.direction)) {
            const enable = !SceneManager.controls.autoRotate;
            SceneManager.controls.autoRotate = enable;
            UIOverlay.logCommand(`auto-rotate ${enable ? 'ON' : 'OFF'}`);
        } else {
            // Give it an explicit spin
            SceneManager.controls.autoRotate = false; // Turn off auto-rotate to allow manual control
            
            // Adjust the actual camera rotation or controls target
            const azimuth = SceneManager.controls.getAzimuthalAngle();
            SceneManager.controls.setAzimuthalAngle(azimuth + increment);
            SceneManager.controls.update();
            UIOverlay.logCommand(`manual twist ${increment > 0 ? "→" : "←"}`);
        }
        break;
      }

      case 'zoom':
      case 'zoom_model': {
        const params = payload.params || {};
        const dir = params.direction || 'in';
        const factor = dir === 'out' ? 1.3 : 0.77;
        SceneManager.camera.position.multiplyScalar(factor);
        UIOverlay.logCommand(`zoom ${dir}`);
        break;
      }

      // ── Server lifecycle ──────────────────────────────────────────────────
      case 'connected':
        UIOverlay.setConnectionStatus('connected');
        UIOverlay.logCommand('C4 backend handshake ✓');
        break;

      case 'server_shutdown':
        UIOverlay.setConnectionStatus('disconnected');
        UIOverlay.logCommand('Server shut down');
        break;

      default:
        console.warn(`[CommandHandler] Unknown command: "${command}"`);
        UIOverlay.logCommand(`⚠ unknown: ${command}`);
    }
  }

  return { execute };
})();

// ─────────────────────────────────────────────────────────────────────────────
//  WEBSOCKET CLIENT
//  Maintains a live connection to the C4 Python backend.
//  Auto-reconnects with exponential back-off (1s → 2s → 4s … cap 30s).
// ─────────────────────────────────────────────────────────────────────────────

const WebSocketClient = (() => {
  let _ws            = null;
  let _reconnectMs   = 1000;       // Initial delay
  const _maxDelay    = 30000;      // 30-second cap
  let _reconnectTimer = null;
  let _intentionallyClosed = false;

  function connect() {
    if (_ws && (_ws.readyState === WebSocket.CONNECTING || _ws.readyState === WebSocket.OPEN)) {
      return; // Already connected/connecting
    }

    UIOverlay.setConnectionStatus('connecting');
    console.info(`[WebSocketClient] Connecting to ${WS_URL}…`);

    _ws = new WebSocket(WS_URL);

    _ws.onopen = () => {
      _reconnectMs = 1000;  // Reset back-off on successful connect
      UIOverlay.setConnectionStatus('connected');
      console.info('[WebSocketClient] Connected to C4 backend.');
    };

    _ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        
        if (payload.type === "POINTER_SYNC") {
           SceneManager.setPointerSync(payload.x, payload.y, payload.gesture);
           return;
        }
        
        let action = "";
        if (payload.type === "MODEL_ACTION") {
           action = payload.action;
        } else {
           action = payload.command || ''; // legacy fallback
        }
        
        if (action) CommandHandler.execute(action, payload);
      } catch (err) {
        console.warn('[WebSocketClient] Could not parse message:', event.data);
      }
    };

    _ws.onclose = (event) => {
      UIOverlay.setConnectionStatus('disconnected');
      if (_intentionallyClosed) return;
      console.warn(`[WebSocketClient] Disconnected (code ${event.code}). Retry in ${_reconnectMs}ms…`);
      UIOverlay.logCommand(`WS closed — retry in ${(_reconnectMs / 1000).toFixed(0)}s`);
      _scheduleReconnect();
    };

    _ws.onerror = (err) => {
      // Error always fires before close — avoid double-logging
      console.warn('[WebSocketClient] WebSocket error:', err);
    };
  }

  function _scheduleReconnect() {
    if (_reconnectTimer) clearTimeout(_reconnectTimer);
    _reconnectTimer = setTimeout(() => {
      _reconnectMs = Math.min(_reconnectMs * 2, _maxDelay);
      connect();
    }, _reconnectMs);
  }

  function disconnect() {
    _intentionallyClosed = true;
    if (_reconnectTimer) clearTimeout(_reconnectTimer);
    if (_ws) _ws.close();
  }

  // Start immediately on module load
  connect();

  return { connect, disconnect };
})();

// ─────────────────────────────────────────────────────────────────────────────
//  TOOLBAR — Public API
//  Called by HTML onclick="" attributes so they work without a WS connection.
// ─────────────────────────────────────────────────────────────────────────────

/** Called by Explode button */
window.explodeModel = () => {
  UIOverlay.logCommand('btn → explode');
  ExplodedView.explode();
};

/** Called by Reset button */
window.resetModel = () => {
  UIOverlay.logCommand('btn → reset');
  ExplodedView.reset();
};

/** Called by Rotate button — toggles auto-rotate */
window.toggleRotate = () => {
  const next = !SceneManager.controls.autoRotate;
  SceneManager.controls.autoRotate = next;
  UIOverlay.logCommand(`btn → rotate ${next ? 'ON' : 'OFF'}`);
};

/** Called by Load Model button */
window.loadDefaultModel = () => {
  UIOverlay.logCommand(`btn → load_model (${DEFAULT_MODEL})`);
  ModelLoader.load(DEFAULT_MODEL);
};

// ─────────────────────────────────────────────────────────────────────────────
//  BOOTSTRAP
//  Kick everything off: start the render loop, then try to load the default
//  model. The viewer is usable even if no model is present (WS still works).
// ─────────────────────────────────────────────────────────────────────────────

(async function bootstrap() {
  console.info('[C4 Viewer] Initialising…');

  // Start Three.js render loop
  SceneManager.startLoop();

  // Attempt auto-load of the default model
  try {
    await ModelLoader.load(DEFAULT_MODEL);
  } catch (_) {
    // Error already surfaced through UIOverlay.showError
    UIOverlay.hideLoading();
    UIOverlay.showError(
      `Place a .glb file at viewer/models/${DEFAULT_MODEL} to get started.`,
      8000
    );
  }

  console.info('[C4 Viewer] Ready. WebSocket awaiting C4 backend on', WS_URL);
})();
