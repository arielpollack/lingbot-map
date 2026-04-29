import * as THREE from "https://unpkg.com/three@0.164.1/build/three.module.js";
import { GLTFLoader } from "https://unpkg.com/three@0.164.1/examples/jsm/loaders/GLTFLoader.js";

const params = new URLSearchParams(location.search);
const runId = params.get("run");
const root = document.querySelector("#viewer");
const hud = document.querySelector("#hud");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf0f0f0);

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

async function loadScene() {
  hud.textContent = "Loading scene...";
  const loader = new GLTFLoader();
  loader.load(
    artifactUrl("scene.glb"),
    (gltf) => {
      scene.add(gltf.scene);
      setCameraFromObject(gltf.scene);
      hud.textContent = "Click to lock pointer. WASD move, mouse look, Space up, Shift down.";
    },
    undefined,
    (error) => {
      hud.textContent = `Failed to load GLB: ${error.message}`;
    }
  );
}

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
