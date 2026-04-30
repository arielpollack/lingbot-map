import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
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
const alphaThreshold = Number.parseInt(params.get("alpha") ?? "5", 10);

if (!runId) {
  hud.textContent = "Missing ?run=...";
  throw new Error("Missing ?run=...");
}

// `.splat` is the antimatter15-style binary format (32 B/Gaussian, no SH)
// produced by poc/worker/splat_format.encode_splat_file. mkkellogg's
// Viewer auto-detects format from the URL extension.
const splatUrl = `/api/runs/${encodeURIComponent(runId)}/artifact/splat.splat`;
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
// Keep iOS Safari from hijacking two-finger gestures as page zoom.
renderer.domElement.style.touchAction = "none";

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// Touch-friendly orbiting controls (matches /mesh viewer):
//   • single-finger drag → orbit
//   • pinch              → dolly (zoom)
//   • two-finger drag    → pan
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.6;
controls.panSpeed = 0.8;
controls.zoomSpeed = 1.0;
controls.touches = {
  ONE: THREE.TOUCH.ROTATE,
  TWO: THREE.TOUCH.DOLLY_PAN,
};
controls.minDistance = 0.05;
controls.maxDistance = 200;
controls.target.set(0, 0, 0);

// `.splat` has no spherical harmonics — DC color only — so we initialize
// the Viewer with sphericalHarmonicsDegree = 0. Saves the SH evaluation
// pass per render frame.
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
  sphericalHarmonicsDegree: 0,
  sceneRevealMode: SceneRevealMode.Instant,
  logLevel: LogLevel.Warning,
});

let downloadPct = 0;
let loaded = false;

function setHud(text) {
  hud.textContent = text;
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
  } = viewerCamera;
  if (!Array.isArray(position) || !Array.isArray(target)) return false;

  camera.position.set(position[0], position[1], position[2]);
  if (Array.isArray(up)) camera.up.set(up[0], up[1], up[2]);
  if (Number.isFinite(fov)) camera.fov = fov;
  if (Number.isFinite(near)) camera.near = near;
  if (Number.isFinite(far)) camera.far = far;
  controls.target.set(target[0], target[1], target[2]);
  camera.updateProjectionMatrix();
  controls.update();
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
        "Drag to rotate · pinch to zoom · two-finger drag to pan",
    );
  })
  .catch((err) => {
    console.error("[splat] load error:", err);
    setHud(`Failed to load splat: ${err.message || err}`);
  });

function frame() {
  controls.update();
  viewer.update();
  viewer.render();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
