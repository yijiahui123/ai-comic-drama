let selectedProject = null;
let eventSource = null;

const $ = (id) => document.getElementById(id);

function api(path, options = {}) {
  const opts = { ...options };
  if (!(opts.body instanceof FormData)) {
    opts.headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  }
  return fetch(path, opts).then(async (res) => {
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    return res.json();
  });
}

function rel(path) {
  if (!path) return "";
  const clean = path.replace(/^.*\/ai-comic-drama\//, "");
  return `/${clean}`;
}

function renderProjects(projects) {
  $("projects").innerHTML = projects.map((p) => `
    <div class="project ${selectedProject === p.project_id ? "active" : ""}" data-id="${p.project_id}">
      <strong>${p.project_id}</strong> ${p.running ? "• running" : ""}
      <small>${p.current_stage} — ${p.user_prompt.slice(0, 64)}</small>
    </div>
  `).join("");
  document.querySelectorAll(".project").forEach((el) => {
    el.onclick = () => selectProject(el.dataset.id);
  });
}

function renderProject(p) {
  selectedProject = p.project_id;
  $("projectTitle").textContent = `项目 ${p.project_id}`;
  $("status").innerHTML = `
    <div>阶段：<strong>${p.current_stage}</strong></div>
    <div>最终视频：<strong>${p.final_video ? "已生成" : "未生成"}</strong></div>
    <div>进度：<strong>${p.progress_current || 0}/${p.progress_total || 0}</strong></div>
    <div>消息：<strong>${p.last_message || "-"}</strong></div>
  `;
  const total = p.progress_total || 0;
  const current = p.progress_current || 0;
  $("progressBar").style.width = total ? `${Math.min(100, (current / total) * 100)}%` : "0";
  $("log").textContent = (p.events || []).slice(-80).map((e) =>
    `[${e.stage}] ${e.message}`
  ).join("\n");
  renderPreview(p.previews || {});
}

function renderPreview(previews) {
  const items = [];
  for (const path of [...(previews.characters || []), ...(previews.scenes || []), ...(previews.shots || [])]) {
    items.push(`<div class="card"><img src="${rel(path)}" loading="lazy"><small>${path}</small></div>`);
  }
  for (const path of [...(previews.videos || []), ...(previews.lipsync || []), ...(previews.final || [])]) {
    items.push(`<div class="card"><video src="${rel(path)}" controls></video><small>${path}</small></div>`);
  }
  $("preview").innerHTML = items.join("") || `<p class="sub">暂无可预览资产。</p>`;
}

async function refreshProjects() {
  const projects = await api("/api/projects");
  renderProjects(projects);
  if (selectedProject) {
    const found = projects.find((p) => p.project_id === selectedProject);
    if (found) renderProject(found);
  }
}

async function selectProject(id) {
  selectedProject = id;
  const project = await api(`/api/projects/${id}`);
  renderProject(project);
  openEvents(id);
  refreshProjects();
}

function openEvents(id) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/api/projects/${id}/events`);
  eventSource.onmessage = async () => {
    const project = await api(`/api/projects/${id}`);
    renderProject(project);
  };
}

$("start").onclick = async () => {
  const prompt = $("prompt").value.trim();
  if (!prompt) return alert("请输入剧情描述");
  const project = await api("/api/projects", {
    method: "POST",
    body: JSON.stringify({ prompt, profile: "default" }),
  });
  await selectProject(project.project_id);
};

$("refresh").onclick = refreshProjects;

$("resume").onclick = async () => {
  if (!selectedProject) return;
  await api(`/api/projects/${selectedProject}/resume`, { method: "POST" });
  openEvents(selectedProject);
};

document.querySelectorAll(".rerun").forEach((btn) => {
  btn.onclick = async () => {
    if (!selectedProject) return;
    await api(`/api/projects/${selectedProject}/rerun`, {
      method: "POST",
      body: JSON.stringify({ stage: btn.dataset.rerun }),
    });
    openEvents(selectedProject);
  };
});

$("validateAssets").onclick = async () => {
  const result = await api("/api/assets/validate", { method: "POST" });
  $("assetsResult").textContent = result.ok
    ? `资源检查通过。Warnings: ${(result.warnings || []).join("; ") || "无"}`
    : `资源检查失败：${(result.errors || []).join("; ")}`;
};

async function loadAssets() {
  const data = await api("/api/assets");
  $("manifestYaml").value = data.manifest_yaml || "";
  $("configYaml").value = data.config_yaml || "";
}

$("loadAssets").onclick = loadAssets;

$("saveManifest").onclick = async () => {
  const result = await api("/api/assets/manifest", {
    method: "PUT",
    body: JSON.stringify({ content: $("manifestYaml").value }),
  });
  $("assetsResult").textContent = result.ok ? "manifest 已保存" : "manifest 保存失败";
  await loadAssets();
};

$("saveConfig").onclick = async () => {
  const result = await api("/api/assets/config", {
    method: "PUT",
    body: JSON.stringify({ content: $("configYaml").value }),
  });
  $("assetsResult").textContent = result.ok ? "资产配置已保存" : "资产配置保存失败";
  await loadAssets();
};

$("uploadForm").onsubmit = async (event) => {
  event.preventDefault();
  const file = $("assetFile").files[0];
  const key = $("assetKey").value.trim();
  if (!file || !key) return alert("请填写 key 并选择图片文件");
  const form = new FormData();
  form.append("asset_type", $("assetType").value);
  form.append("key", key);
  form.append("emotion", $("assetEmotion").value);
  form.append("file", file);
  const result = await api("/api/assets/upload", {
    method: "POST",
    body: form,
  });
  $("assetsResult").textContent = `上传成功：${result.path}`;
  $("assetFile").value = "";
  await loadAssets();
};

$("loraForm").onsubmit = async (event) => {
  event.preventDefault();
  const character = $("loraCharacter").value.trim();
  if (!character) return alert("请填写角色名");
  const result = await api("/api/assets/character-lora", {
    method: "POST",
    body: JSON.stringify({
      character,
      enabled: $("loraEnabled").checked,
      name: $("loraName").value.trim(),
      trigger: $("loraTrigger").value.trim(),
      strength_model: 0.85,
      strength_clip: 0.8,
    }),
  });
  $("assetsResult").textContent = result.ok ? "角色 LoRA 映射已保存" : "保存失败";
  await loadAssets();
};

refreshProjects();
loadAssets();
