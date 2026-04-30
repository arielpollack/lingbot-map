import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const params = new URLSearchParams(location.search);
const runId = params.get("run");
const root = document.querySelector("#viewer");
const hud = document.querySelector("#hud");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x121417);

const camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.01, 10000);
camera.position.set(0, 0, 2);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
root.appendChild(renderer.domElement);
// Make sure the canvas itself receives touch events without browser
// gesture interference (iOS Safari otherwise treats two-finger gestures as
// page zoom).
renderer.domElement.style.touchAction = "none";

scene.add(new THREE.HemisphereLight(0xffffff, 0x2b3035, 1.4));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
keyLight.position.set(3, 5, 4);
scene.add(keyLight);

// Touch-friendly orbiting controls. Replaces the previous WASD + mouse-look
// pointer-lock setup, which was unusable on mobile.
//   • single-finger drag      → orbit (rotate around target)
//   • pinch                   → dolly (zoom)
//   • two-finger drag         → pan
// Damping makes the manipulation feel less jittery on touch.
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

const vertexColorMaterial = new THREE.MeshStandardMaterial({
  vertexColors: true,
  roughness: 0.95,
  metalness: 0.0,
  side: THREE.DoubleSide,
});
const clayMaterial = new THREE.MeshStandardMaterial({
  color: 0xbebeb8,
  roughness: 0.92,
  metalness: 0.0,
  side: THREE.DoubleSide,
});

function setHud(text) {
  hud.textContent = text;
}

function artifactUrl(name) {
  if (!runId) {
    throw new Error("Missing ?run=...");
  }
  return `/api/runs/${encodeURIComponent(runId)}/artifact/${encodeURIComponent(name)}`;
}

async function fetchMetadata() {
  const response = await fetch(artifactUrl("metadata.json"));
  if (!response.ok) return null;
  return response.json();
}

function applyMeshMaterial(object) {
  let meshCount = 0;
  let hasVertexColors = false;
  let hasTexture = false;
  object.traverse((node) => {
    if (!node.isMesh) return;
    const attrs = node.geometry?.attributes;
    if (attrs?.color && attrs.color.itemSize >= 3) hasVertexColors = true;
    if (node.material?.map) hasTexture = true;
  });
  object.traverse((node) => {
    if (!node.isMesh) return;
    if (hasTexture) {
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      for (const mat of materials) {
        if (!mat) continue;
        mat.side = THREE.DoubleSide;
        mat.roughness = 0.95;
        mat.metalness = 0.0;
        mat.needsUpdate = true;
      }
    } else {
      node.material = hasVertexColors ? vertexColorMaterial : clayMaterial;
    }
    node.castShadow = false;
    node.receiveShadow = true;
    meshCount += 1;
  });
  return { meshCount, hasVertexColors, hasTexture };
}

function frameCamera(object) {
  const box = new THREE.Box3().setFromObject(object);
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  const radius = Math.max(size.x, size.y, size.z, 1);

  // Position the camera at a 3/4 view distance equal to the bounding
  // radius. OrbitControls keeps `target` as the orbit pivot, so set both.
  const offset = new THREE.Vector3(radius * 0.9, radius * 0.6, radius * 0.9);
  camera.position.copy(center).add(offset);
  controls.target.copy(center);
  controls.minDistance = Math.max(radius * 0.05, 0.05);
  controls.maxDistance = radius * 6;
  controls.update();
}

function statsText(metadata, meshCount, hasTexture) {
  const mesh = metadata?.mesh;
  const tier = metadata?.tier || metadata?.manifest?.tier || "";
  const help = "Drag to rotate · pinch to zoom · two-finger drag to pan";
  if (!mesh) {
    return `${meshCount} mesh nodes${tier ? ` · ${tier}` : ""} · ${help}`;
  }
  const faces = Number(mesh.face_count || mesh.faces || 0).toLocaleString();
  const vertices = Number(mesh.vertex_count || mesh.vertices || 0).toLocaleString();
  return `${vertices} verts · ${faces} faces${tier ? ` · ${tier}` : ""} · ${help}`;
}

async function loadMesh() {
  setHud("Loading mesh...");
  let metadata = null;
  try {
    metadata = await fetchMetadata();
  } catch (error) {
    console.warn("[mesh] metadata unavailable:", error);
  }

  const loader = new GLTFLoader();
  loader.load(
    artifactUrl("mesh.glb"),
    (gltf) => {
      const { meshCount, hasVertexColors, hasTexture } = applyMeshMaterial(gltf.scene);
      // Both server tiers (legacy lingbot per-frame depth, new ARKit
      // lidar_mesh) ship the GLB with Y pointing DOWN — the lingbot path
      // because the model outputs OpenCV world coords, the lidar_mesh path
      // because we pre-rotate the ARKit mesh by 180° around X for exactly
      // this purpose. Undo that here so the floor lands at the bottom.
      gltf.scene.rotation.x = Math.PI;
      gltf.scene.updateMatrixWorld(true);
      scene.add(gltf.scene);
      frameCamera(gltf.scene);
      let prefix = "";
      if (hasTexture) prefix = "[textured] ";
      else if (!hasVertexColors) prefix = "[clay] ";
      setHud(prefix + statsText(metadata, meshCount, hasTexture));
    },
    undefined,
    (error) => {
      setHud(`Failed to load mesh: ${error.message || error}`);
    }
  );
}

function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

loadMesh();
requestAnimationFrame(animate);
