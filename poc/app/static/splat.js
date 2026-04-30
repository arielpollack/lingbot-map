import * as THREE from "three";
import {
  Viewer,
  RenderMode,
  SceneRevealMode,
  WebXRMode,
  LogLevel,
} from "@mkkellogg/gaussian-splats-3d";

const params = new URLSearchParams(location.search);
const runId = params.get("run");
const root = document.querySelector("#viewer");
const hud = document.querySelector("#hud");
const shDegree = Number.parseInt(params.get("sh") ?? "0", 10);
const alphaThreshold = Number.parseInt(params.get("alpha") ?? "5", 10);

if (!runId) {
  hud.textContent = "Missing ?run=...";
  throw new Error("Missing ?run=...");
}

const splatUrl = `/api/runs/${encodeURIComponent(runId)}/artifact/splat.ply`;
const metadataUrl = `/api/runs/${encodeURIComponent(runId)}/artifact/metadata.json`;

const camera = new THREE.PerspectiveCamera(
  65,
  innerWidth / innerHeight,
  0.01,
  1000,
);
camera.position.set(0, 0, 4);
camera.up.set(0, 1, 0);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.autoClear = true;
renderer.setClearColor(0x111418, 1);
root.appendChild(renderer.domElement);

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

const viewer = new Viewer({
  rootElement: root,
  selfDrivenMode: false,
  renderer,
  camera,
  useBuiltInControls: false,
  ignoreDevicePixelRatio: false,
  gpuAcceleratedSort: false,
  sharedMemoryForWorkers: false,
  integerBasedSort: false,
  halfPrecisionCovariancesOnGPU: false,
  dynamicScene: false,
  webXRMode: WebXRMode.None,
  renderMode: RenderMode.Always,
  antialiased: true,
  focalAdjustment: 1.0,
  sphericalHarmonicsDegree: Number.isFinite(shDegree)
    ? Math.max(0, Math.min(2, shDegree))
    : 0,
  sceneRevealMode: SceneRevealMode.Instant,
  logLevel: LogLevel.Warning,
});

let downloadPct = 0;
let loaded = false;

function setHud(text) {
  hud.textContent = text;
}

viewer
  .addSplatScene(splatUrl, {
    splatAlphaRemovalThreshold: Number.isFinite(alphaThreshold)
      ? Math.max(0, Math.min(255, alphaThreshold))
      : 5,
    showLoadingUI: false,
    progressiveLoad: false,
    onProgress: (pct) => {
      if (typeof pct === "number") {
        downloadPct = Math.round(pct);
        if (!loaded) setHud(`Loading splat… ${downloadPct}%`);
      }
    },
  })
  .then(async () => {
    loaded = true;
    const cameraApplied = await loadViewerCamera();
    setHud(
      `${cameraApplied ? "Started at training camera. " : ""}` +
        "Click to lock pointer. WASD move, mouse look, Space up, Shift down.",
    );
  })
  .catch((err) => {
    console.error("[splat] load error:", err);
    setHud(`Failed to load splat: ${err.message || err}`);
  });

// FPS controls — same shape as viewer.js so muscle memory transfers.
const keys = new Set();
let locked = false;
let yaw = 0;
let pitch = 0;
let last = performance.now();
let moveSpeed = 1.5;

function syncYawPitchFromCamera() {
  camera.rotation.order = "YXZ";
  yaw = camera.rotation.y;
  pitch = camera.rotation.x;
}

function applyViewerCamera(viewerCamera) {
  if (!viewerCamera) return false;
  const {
    position,
    target,
    up,
    fov_degrees: fov,
    near,
    far,
    move_speed: speed,
  } = viewerCamera;
  if (!Array.isArray(position) || !Array.isArray(target)) return false;

  camera.position.set(position[0], position[1], position[2]);
  if (Array.isArray(up)) camera.up.set(up[0], up[1], up[2]);
  if (Number.isFinite(fov)) camera.fov = fov;
  if (Number.isFinite(near)) camera.near = near;
  if (Number.isFinite(far)) camera.far = far;
  if (Number.isFinite(speed)) moveSpeed = speed;
  camera.lookAt(new THREE.Vector3(target[0], target[1], target[2]));
  camera.updateProjectionMatrix();
  syncYawPitchFromCamera();
  return true;
}

async function loadViewerCamera() {
  try {
    const response = await fetch(metadataUrl);
    if (!response.ok) return false;
    const metadata = await response.json();
    return applyViewerCamera(metadata?.splat?.viewer_camera);
  } catch (error) {
    console.warn("[splat] metadata camera unavailable:", error);
    return false;
  }
}

renderer.domElement.addEventListener("click", () =>
  renderer.domElement.requestPointerLock(),
);
document.addEventListener("pointerlockchange", () => {
  locked = document.pointerLockElement === renderer.domElement;
});
document.addEventListener("mousemove", (event) => {
  if (!locked) return;
  yaw -= event.movementX * 0.002;
  pitch -= event.movementY * 0.002;
  pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, pitch));
});
document.addEventListener("keydown", (event) => keys.add(event.code));
document.addEventListener("keyup", (event) => keys.delete(event.code));

function updateCamera(delta) {
  camera.rotation.order = "YXZ";
  camera.rotation.y = yaw;
  camera.rotation.x = pitch;

  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  forward.y = 0;
  forward.normalize();
  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  const movement = new THREE.Vector3();

  if (keys.has("KeyW")) movement.add(forward);
  if (keys.has("KeyS")) movement.sub(forward);
  if (keys.has("KeyD")) movement.sub(right);
  if (keys.has("KeyA")) movement.add(right);
  if (keys.has("Space")) movement.y += 1;
  if (keys.has("ShiftLeft") || keys.has("ShiftRight")) movement.y -= 1;
  if (movement.lengthSq() > 0) {
    movement.normalize().multiplyScalar(moveSpeed * delta);
    camera.position.add(movement);
  }
}

function frame(now) {
  const delta = Math.min((now - last) / 1000, 0.05);
  last = now;
  updateCamera(delta);
  viewer.update();
  viewer.render();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
