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

function actionCell(run) {
  if (run.status === "completed") {
    return `<a href="/viewer?run=${encodeURIComponent(run.id)}" target="_blank">view</a>`;
  }
  return "";
}

function renderRuns(runs) {
  runsNode.innerHTML = runs.map((run) => `
    <tr>
      <td>${run.created_at || ""}</td>
      <td>${run.filename || ""}</td>
      <td>${run.status || ""}</td>
      <td>${run.job_id || ""}</td>
      <td>${actionCell(run)}</td>
    </tr>
  `).join("");
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
