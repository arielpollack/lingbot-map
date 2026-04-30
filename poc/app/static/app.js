const form = document.querySelector("#upload-form");
const statusNode = document.querySelector("#status");
const runsNode = document.querySelector("#runs");

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(data.detail || response.statusText);
  }
  return data;
}

function actionLinks(run) {
  if (run.status !== "completed") return [];
  const links = [];

  const points = document.createElement("a");
  points.href = `/viewer?run=${encodeURIComponent(run.id)}`;
  points.textContent = "view points";
  links.push(points);

  const meshKey = run?.output?.mesh_key;
  if (meshKey) {
    const mesh = document.createElement("a");
    mesh.href = `/mesh?run=${encodeURIComponent(run.id)}`;
    mesh.textContent = "view mesh";
    mesh.style.marginLeft = "8px";
    links.push(mesh);
  }

  const splatKey = run?.output?.splat_key;
  if (splatKey) {
    const splat = document.createElement("a");
    splat.href = `/splat?run=${encodeURIComponent(run.id)}`;
    splat.textContent = "view splat";
    splat.style.marginLeft = "8px";
    links.push(splat);
  }
  return links;
}

function renderRuns(runs) {
  runsNode.replaceChildren();
  for (const run of runs) {
    const row = document.createElement("tr");
    for (const value of [run.created_at, run.filename, run.status, run.job_id]) {
      const cell = document.createElement("td");
      cell.textContent = value || "";
      row.appendChild(cell);
    }

    const action = document.createElement("td");
    for (const link of actionLinks(run)) {
      action.appendChild(link);
    }
    row.appendChild(action);
    runsNode.appendChild(row);
  }
}

async function refreshRuns() {
  const data = await apiJson("/api/runs");
  const refreshed = [];
  for (const run of data.runs) {
    if (!["completed", "failed"].includes(run.status) && run.job_id) {
      refreshed.push(await apiJson(`/api/runs/${run.id}`));
    } else {
      refreshed.push(run);
    }
  }
  renderRuns(refreshed);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  statusNode.textContent = "Uploading...";
  const formData = new FormData(form);
  if (!formData.get("first_k")) {
    formData.delete("first_k");
  }
  try {
    const run = await apiJson("/api/runs", { method: "POST", body: formData });
    statusNode.textContent = `Submitted ${run.id}`;
    form.reset();
    await refreshRuns();
  } catch (error) {
    statusNode.textContent = error.message;
  }
});

refreshRuns().catch((error) => {
  statusNode.textContent = error.message;
});
setInterval(() => refreshRuns().catch(() => {}), 5000);
