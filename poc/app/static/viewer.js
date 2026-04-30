import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const params = new URLSearchParams(location.search);
const runId = params.get("run");
const root = document.querySelector("#viewer");
const hud = document.querySelector("#hud");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111418);

const camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.01, 10000);
camera.position.set(0, 0, 2);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
root.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 1.0));
const light = new THREE.DirectionalLight(0xffffff, 1.0);
light.position.set(2, 5, 3);
scene.add(light);

const keys = new Set();
let locked = false;
let yaw = 0;
let pitch = 0;
let last = performance.now();
const speed = 1.5;

renderer.domElement.addEventListener("click", () => renderer.domElement.requestPointerLock());
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

function artifactUrl(name) {
  if (!runId) {
    throw new Error("Missing ?run=...");
  }
  return `/api/runs/${encodeURIComponent(runId)}/artifact/${encodeURIComponent(name)}`;
}

function setCameraFromObject(object) {
  const box = new THREE.Box3().setFromObject(object);
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  const radius = Math.max(size.x, size.y, size.z, 1);
  camera.position.copy(center).add(new THREE.Vector3(0, radius * 0.2, radius * 0.8));
  camera.lookAt(center);
}

const cameraMeshes = [];
let camerasVisible = false;

function tunePointMaterials(root) {
  let pointCount = 0;
  cameraMeshes.length = 0;
  root.traverse((obj) => {
    if (obj.isPoints) {
      pointCount += obj.geometry?.attributes?.position?.count ?? 0;
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const m of mats) {
        if (!m) continue;
        m.size = 3;
        m.sizeAttenuation = false;
        m.vertexColors = true;
        m.needsUpdate = true;
      }
    } else if (obj.isMesh || obj.isLineSegments || obj.isLine) {
      // Camera frustum/cone/line markers exported by trimesh.
      cameraMeshes.push(obj);
      obj.visible = camerasVisible;
    }
  });
  return pointCount;
}

function setCamerasVisible(visible) {
  camerasVisible = visible;
  for (const obj of cameraMeshes) obj.visible = visible;
}

async function loadScene() {
  hud.textContent = "Loading scene...";
  const loader = new GLTFLoader();
  loader.load(
    artifactUrl("scene.glb"),
    (gltf) => {
      const pointCount = tunePointMaterials(gltf.scene);
      scene.add(gltf.scene);
      setCameraFromObject(gltf.scene);
      hud.textContent = `${pointCount.toLocaleString()} points. C: toggle cameras. Click to lock pointer. WASD move, mouse look, Space up, Shift down.`;
    },
    undefined,
    (error) => {
      hud.textContent = `Failed to load GLB: ${error.message}`;
    }
  );
}

document.addEventListener("keydown", (event) => {
  if (event.code === "KeyC") setCamerasVisible(!camerasVisible);
});

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
    movement.normalize().multiplyScalar(speed * delta);
    camera.position.add(movement);
  }
}

function animate(now) {
  const delta = Math.min((now - last) / 1000, 0.05);
  last = now;
  updateCamera(delta);
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

loadScene();
requestAnimationFrame(animate);
