let selectedProject = null;
let eventSource = null;
let currentAssetConfig = null;
let selectedShot = null;
let currentProject = null;

const $ = (id) => document.getElementById(id);

// Tab switching
document.querySelectorAll(".tab").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $(`tab-${tab.dataset.tab}`).classList.add("active");
  };
});

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

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function sourceForPath(project, path) {
  const sources = project.asset_sources || {};
  for (const group of Object.values(sources)) {
    if (group && group[path]) return group[path];
  }
  return path.includes("/library/") ? "library" : "unknown";
}

function renderProjects(projects) {
  $("projects").innerHTML = projects.map((p) => `
    <div class="project ${selectedProject === p.project_id ? "active" : ""}" data-id="${p.project_id}">
      <div style="display:flex;justify-content:space-between;align-items:start">
        <div style="flex:1;min-width:0" class="project-select">
          <strong>${p.project_id}</strong> ${p.running ? "• running" : ""}
          <small>${p.current_stage} — ${p.user_prompt.slice(0, 64)}</small>
        </div>
        <button class="ghost project-delete" data-id="${p.project_id}" title="删除项目"
          style="padding:2px 6px;font-size:0.7rem;color:var(--danger);border-color:var(--danger)">✕</button>
      </div>
    </div>
  `).join("");
  document.querySelectorAll(".project-select").forEach((el) => {
    el.onclick = () => selectProject(el.closest(".project").dataset.id);
  });
  document.querySelectorAll(".project-delete").forEach((btn) => {
    btn.onclick = async (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      if (!confirm(`确认删除项目 ${id}？此操作不可恢复。`)) return;
      await api(`/api/projects/${id}`, { method: "DELETE" });
      if (selectedProject === id) selectedProject = null;
      await refreshProjects();
    };
  });
}

function renderProject(p) {
  selectedProject = p.project_id;
  currentProject = p;
  $("projectTitle").textContent = `项目 ${p.project_id}`;
  // Show rerun-failed button only when in ERROR state
  const rerunBtn = $("rerunFailed");
  if (rerunBtn) rerunBtn.style.display = p.current_stage === "ERROR" ? "" : "none";
  // Build stage elapsed info
  let stageInfo = "";
  const stages = p.stages || {};
  for (const [name, result] of Object.entries(stages)) {
    if (result.elapsed_seconds != null) {
      stageInfo += `<div>${name}：<strong>${result.elapsed_seconds.toFixed(1)}s</strong> ${pill(result.status)}</div>`;
    }
  }
  const finals = p.final_videos || (p.final_video ? [p.final_video] : []);
  $("status").innerHTML = `
    <div>阶段：<strong>${p.current_stage}</strong></div>
    <div>最终视频：<strong>${finals.length ? finals.length + " 集已生成" : "未生成"}</strong></div>
    ${finals.map((v, i) => `<div class="sub">EP${String(i+1).padStart(2,"0")}：${escapeHtml(v)}</div>`).join("")}
    <div>进度：<strong>${p.progress_current || 0}/${p.progress_total || 0}</strong></div>
    <div>消息：<strong>${p.last_message || "-"}</strong></div>
    <div>队列：<strong>${p.queue_status || "idle"}</strong></div>
    <div>审核队列：<strong>${(p.review_queue || []).length}</strong></div>
    ${stageInfo}
  `;
  const total = p.progress_total || 0;
  const current = p.progress_current || 0;
  $("progressBar").style.width = total ? `${Math.min(100, (current / total) * 100)}%` : "0";
  $("log").textContent = (p.events || []).slice(-80).map((e) =>
    `[${e.stage}] ${e.message}`
  ).join("\n");
  renderPreview(p);
  renderProduction(p);
}

function renderPreview(project) {
  const previews = project.previews || {};
  const items = [];
  for (const path of [...(previews.characters || []), ...(previews.scenes || []), ...(previews.shots || [])]) {
    const source = sourceForPath(project, path);
    items.push(`<div class="card"><span class="badge ${source}">${source}</span><img src="${rel(path)}" loading="lazy"><small>${escapeHtml(path)}</small></div>`);
  }
  for (const path of [...(previews.videos || []), ...(previews.lipsync || []), ...(previews.final || [])]) {
    items.push(`<div class="card"><video src="${rel(path)}" controls></video><small>${escapeHtml(path)}</small></div>`);
  }
  $("preview").innerHTML = items.join("") || `<p class="sub">暂无可预览资产。</p>`;
}

function pill(value) {
  return `<span class="status-pill ${escapeHtml(value)}">${escapeHtml(value)}</span>`;
}

function renderQualityChecks(checks) {
  if (!checks || !checks.length) return "";
  return `<div class="quality-checks">${checks.map((c) => {
    const icon = c.status === "pass" ? "&#10003;" : c.status === "fail" ? "&#10007;" : c.status === "warn" ? "!" : "-";
    return `<span class="qc-item ${c.status}" title="${escapeHtml(c.message)}">${icon} ${escapeHtml(c.name)}</span>`;
  }).join("")}</div>`;
}

function renderSegmentChecks(report) {
  const segs = report.segment_checks || [];
  if (!segs.length) return "";
  return `<div class="segment-checks"><h4>Segment 校验</h4>${segs.map((s) => `
    <div class="compact-item">${pill(s.status)} <strong>${escapeHtml(s.name)}</strong> <small>${escapeHtml(s.message || "")}</small></div>
  `).join("")}</div>`;
}

function renderQualityReport(report) {
  if (!report || !Object.keys(report).length) return `<p class="sub">暂无质检报告。</p>`;
  let html = `<div class="report-summary">
    <span>总镜头: <strong>${report.total_shots || 0}</strong></span>
    <span class="pass">就绪: <strong>${report.ready || 0}</strong></span>
    <span class="fail">失败: <strong>${report.failed || 0}</strong></span>
    <span class="warn">待审: <strong>${report.needs_review || 0}</strong></span>
  </div>`;
  if (report.final_video) {
    html += `<div class="compact-item">${pill(report.final_video.status || "unknown")} <strong>final_video</strong> <small>${escapeHtml(report.final_video.message || "")}</small></div>`;
  }
  html += renderSegmentChecks(report);
  if (report.review_queue && report.review_queue.length) {
    html += `<div class="report-queue"><h4>审核队列</h4>${report.review_queue.map((id) => `<span class="badge">${escapeHtml(id)}</span>`).join(" ")}</div>`;
  }
  return html;
}

function renderProduction(project) {
  const filter = $("shotFilter") ? $("shotFilter").value : "all";
  const allShots = Object.values(project.shot_states || {});
  const shots = filter === "all" ? allShots : allShots.filter((s) => {
    if (filter === "failed") return s.status === "failed" || s.status === "needs_retry";
    if (filter === "needs_review") return s.review_status === "needs_retry" || s.status === "needs_review";
    return s.status === filter;
  });
  $("shotsList").innerHTML = shots.map((shot) => {
    const checks = shot.quality_checks || [];
    const failedChecks = checks.filter((c) => c.status === "fail").map((c) => c.name);
    const checkSummary = checks.length ? `${checks.filter((c) => c.status === "pass").length}/${checks.length}` : "";
    const pct = shot.progress_pct || 0;
    const eta = shot.eta_seconds || 0;
    const node = shot.progress_node || "";
    const showProgress = pct > 0 && pct < 100;
    const etaStr = eta > 60 ? `${Math.round(eta / 60)}m${Math.round(eta % 60)}s` : eta > 0 ? `${Math.round(eta)}s` : "";
    return `
    <div class="shot-row ${selectedShot === shot.shot_id ? "active" : ""}" data-shot="${escapeHtml(shot.shot_id)}">
      <div class="shot-header">
        <strong>${escapeHtml(shot.shot_id)}</strong>${pill(shot.status || "pending")}${shot.locked ? pill("locked") : ""}
        ${checkSummary ? `<span class="check-summary" title="质检通过">${checkSummary}</span>` : ""}
      </div>
      ${showProgress ? `<div class="shot-progress"><div class="shot-progress-bar" style="width:${pct}%"></div><small>${pct}%${etaStr ? " · ETA " + etaStr : ""}${node ? " · " + escapeHtml(node) : ""}</small></div>` : ""}
      <small>${escapeHtml(shot.scene_id || "")} — review: ${escapeHtml(shot.review_status || "pending")} — retry ${shot.retry_count || 0}/${shot.max_retries || 2}</small>
      ${failedChecks.length ? `<small class="fail">失败项: ${escapeHtml(failedChecks.join(", "))}</small>` : ""}
      ${shot.last_error ? `<small class="fail">${escapeHtml(shot.last_error)}</small>` : ""}
      ${renderQualityChecks(checks)}
    </div>`;
  }).join("") || `<p class="sub">暂无镜头状态。生成脚本后会自动出现。</p>`;
  document.querySelectorAll(".shot-row").forEach((el) => {
    el.onclick = () => selectShot(el.dataset.shot);
  });

  $("reviewQueue").innerHTML = (project.review_queue || []).map((shotId) => {
    const shot = (project.shot_states || {})[shotId] || {};
    return `<div class="compact-item"><strong>${escapeHtml(shotId)}</strong>${pill(shot.status || "unknown")}<br><small>${escapeHtml(shot.last_error || "")}</small></div>`;
  }).join("") || `<p class="sub">暂无待审核镜头。</p>`;
  $("qualityReport").innerHTML = renderQualityReport(project.quality_report || {});
  if (selectedShot && project.shot_states && project.shot_states[selectedShot]) {
    fillShotForm(project.shot_states[selectedShot]);
  }
}

function fillShotForm(shot) {
  const script = shot.script || {};
  $("editShotId").value = shot.shot_id || "";
  $("editVisualPrompt").value = script.visual_prompt || script.description || "";
  $("editMotionPrompt").value = script.motion_prompt || "";
  $("editDialogue").value = script.dialogue || "";
  $("editEmotion").value = script.emotion || script.mood || "";
  $("editDuration").value = script.duration || 4;
}

function renderShotQualityDetail(shot) {
  const el = $("shotQualityDetail");
  if (!el) return;
  const checks = shot.quality_checks || [];
  if (!checks.length) {
    el.innerHTML = `<p class="sub">暂无质检数据。点击"质检该镜头"运行检查。</p>`;
    return;
  }
  el.innerHTML = `<h4>质检详情</h4>` + checks.map((c) => `
    <div class="compact-item">
      ${pill(c.status)} <strong>${escapeHtml(c.name)}</strong>
      <small>${escapeHtml(c.message || "")}</small>
      ${c.details ? `<small class="sub">${escapeHtml(JSON.stringify(c.details))}</small>` : ""}
    </div>
  `).join("");
}

function selectShot(shotId) {
  selectedShot = shotId;
  if (currentProject?.shot_states?.[shotId]) {
    fillShotForm(currentProject.shot_states[shotId]);
    renderShotQualityDetail(currentProject.shot_states[shotId]);
    renderProduction(currentProject);
  }
}

async function refreshQueue() {
  const data = await api("/api/queue");
  $("queueList").innerHTML = (data.tasks || []).slice(-12).reverse().map((task) => `
    <div class="compact-item">
      <strong>${escapeHtml(task.kind)}</strong>${pill(task.status)}
      ${task.queue_position ? `<span style="color:var(--muted);font-size:0.75rem">排队 #${task.queue_position}</span>` : ""}
      <small>${escapeHtml(task.task_id)} ${task.project_id ? `— ${escapeHtml(task.project_id)}` : ""}</small>
      <small>${escapeHtml(task.message || "")}</small>
      ${task.status === "queued" || task.status === "running" ? `<button class="ghost cancel-task" data-task="${escapeHtml(task.task_id)}">取消</button>` : ""}
    </div>
  `).join("") || `<p class="sub">暂无队列任务。</p>`;
  document.querySelectorAll(".cancel-task").forEach((btn) => {
    btn.onclick = async () => {
      await api(`/api/queue/${btn.dataset.task}/cancel`, { method: "POST" });
      refreshQueue();
    };
  });
}

async function refreshHealth() {
  const data = await api("/api/health");
  $("healthList").innerHTML = Object.entries(data.checks || {}).map(([name, check]) => `
    <div class="compact-item">
      <strong>${escapeHtml(name)}</strong>${pill(check.ok ? "pass" : "warn")}
      <small>${escapeHtml(check.path || check.url || check.message || "")}</small>
      ${(check.errors || []).map((x) => `<small>${escapeHtml(x)}</small>`).join("")}
      ${(check.warnings || []).map((x) => `<small>${escapeHtml(x)}</small>`).join("")}
    </div>
  `).join("");
}

async function refreshProductionPanel() {
  if (selectedProject) {
    const project = await api(`/api/projects/${selectedProject}`);
    renderProject(project);
  }
  await refreshQueue();
}

async function refreshProjects() {
  const projects = await api("/api/projects");
  renderProjects(projects);
  if (selectedProject) {
    await selectProject(selectedProject, false);
  }
  refreshQueue();
}

async function selectProject(id, refreshList = true) {
  selectedProject = id;
  const project = await api(`/api/projects/${id}`);
  renderProject(project);
  openEvents(id);
  if (refreshList) refreshProjects();
}

function openEvents(id) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/api/projects/${id}/events`);
  eventSource.onmessage = async () => {
    const project = await api(`/api/projects/${id}`);
    renderProject(project);
    refreshQueue();
  };
}

$("start").onclick = async () => {
  const prompt = $("prompt").value.trim();
  if (!prompt) return alert("请输入剧情描述");
  if ($("batchMode").checked) {
    const prompts = prompt.split("\n").map((l) => l.trim()).filter(Boolean);
    if (!prompts.length) return alert("请输入至少一行 prompt");
    const result = await api("/api/batch", {
      method: "POST",
      body: JSON.stringify({ prompts, profile: "default" }),
    });
    $("assetsResult").textContent = `批量创建 ${result.count} 个项目`;
    await refreshProjects();
  } else {
    const project = await api("/api/projects", {
      method: "POST",
      body: JSON.stringify({ prompt, profile: "default" }),
    });
    await selectProject(project.project_id);
  }
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

$("rerunFailed").onclick = async () => {
  if (!selectedProject || !currentProject) return;
  // Find the last error stage
  const stages = currentProject.stages || {};
  let lastError = null;
  for (const [name, result] of Object.entries(stages)) {
    if (result.status === "error") lastError = name;
  }
  if (!lastError) return alert("未找到失败的 stage");
  await api(`/api/projects/${selectedProject}/rerun`, {
    method: "POST",
    body: JSON.stringify({ stage: lastError, force: true, start: true }),
  });
  $("assetsResult").textContent = `已清理 ${lastError} 并启动重跑`;
  openEvents(selectedProject);
};

$("cleanupStageBtn").onclick = async () => {
  if (!selectedProject) return alert("请先选择项目");
  const result = await api(`/api/projects/${selectedProject}/cleanup`, {
    method: "POST",
    body: JSON.stringify({ stage: $("cleanupStage").value, delete_files: true }),
  });
  $("assetsResult").textContent = `已清理并标记 ${$("cleanupStage").value}，删除文件 ${result.deleted?.filter((x) => x.deleted).length || 0} 个`;
  renderProject(result);
};

$("forceRerunStageBtn").onclick = async () => {
  if (!selectedProject) return alert("请先选择项目");
  const stage = $("cleanupStage").value;
  await api(`/api/projects/${selectedProject}/rerun`, {
    method: "POST",
    body: JSON.stringify({ stage, force: true, start: true }),
  });
  $("assetsResult").textContent = `已清理 ${stage} 并启动后台重跑`;
  openEvents(selectedProject);
};

async function cleanupShot(start) {
  if (!selectedProject) return alert("请先选择项目");
  const shotId = $("shotId").value.trim();
  if (!shotId) return alert("请输入 shot_id");
  const endpoint = start ? "rerun" : "cleanup";
  const result = await api(`/api/projects/${selectedProject}/shots/${encodeURIComponent(shotId)}/${endpoint}`, {
    method: "POST",
    body: JSON.stringify({
      include_asset: $("shotIncludeAsset").checked,
      delete_files: true,
      start,
    }),
  });
  $("assetsResult").textContent = start
    ? `已标记并启动镜头 ${shotId} 重生成`
    : `已清理并标记镜头 ${shotId}，删除文件 ${result.deleted?.filter((x) => x.deleted).length || 0} 个`;
  renderProject(result);
  if (start) openEvents(selectedProject);
}

$("cleanupShotBtn").onclick = () => cleanupShot(false);
$("markShotRerunBtn").onclick = () => cleanupShot(false);

$("refreshProduction").onclick = refreshProductionPanel;
$("runHealth").onclick = refreshHealth;

$("qualityProject").onclick = async () => {
  if (!selectedProject) return alert("请先选择项目");
  const result = await api(`/api/projects/${selectedProject}/quality-check`, { method: "POST" });
  $("qualityReport").innerHTML = renderQualityReport(result);
  await selectProject(selectedProject, false);
};

$("retryFailed").onclick = async () => {
  if (!selectedProject) return alert("请先选择项目");
  const result = await api(`/api/projects/${selectedProject}/retry-failed`, { method: "POST" });
  $("assetsResult").textContent = `已标记 ${result.shot_ids.length} 个镜头重试`;
  await refreshProductionPanel();
};

$("exportProject").onclick = async () => {
  if (!selectedProject) return alert("请先选择项目");
  const task = await api(`/api/projects/${selectedProject}/export`, { method: "POST" });
  $("assetsResult").textContent = `导出任务已入队：${task.task_id}，等待完成...`;
  await refreshQueue();
  // Poll until export completes or fails
  const taskId = task.task_id;
  const poll = setInterval(async () => {
    try {
      const q = await api("/api/queue");
      const found = (q.tasks || []).find((t) => t.task_id === taskId);
      if (!found || found.status === "completed") {
        clearInterval(poll);
        $("assetsResult").textContent = found ? `导出完成：${taskId}` : `导出任务 ${taskId} 已完成`;
        await refreshQueue();
      } else if (found.status === "failed") {
        clearInterval(poll);
        $("assetsResult").textContent = `导出失败：${found.message || taskId}`;
        await refreshQueue();
      }
    } catch (_) { /* ignore transient errors */ }
  }, 2000);
};

$("downloadExport").onclick = () => {
  if (!selectedProject) return alert("请先选择项目");
  window.location.href = `/api/projects/${selectedProject}/export?download=true`;
};

$("shotEditForm").onsubmit = async (event) => {
  event.preventDefault();
  const shotId = $("editShotId").value.trim();
  if (!selectedProject || !shotId) return alert("请先选择镜头");
  await api(`/api/projects/${selectedProject}/shots/${encodeURIComponent(shotId)}`, {
    method: "PUT",
    body: JSON.stringify({
      data: {
        visual_prompt: $("editVisualPrompt").value,
        motion_prompt: $("editMotionPrompt").value,
        dialogue: $("editDialogue").value,
        emotion: $("editEmotion").value,
        duration: Number($("editDuration").value || 4),
      },
    }),
  });
  $("assetsResult").textContent = `镜头 ${shotId} 已保存`;
  await selectProject(selectedProject, false);
};

async function shotAction(action, body = null) {
  const shotId = $("editShotId").value.trim();
  if (!selectedProject || !shotId) return alert("请先选择镜头");
  const options = { method: "POST" };
  if (body) options.body = JSON.stringify(body);
  const result = await api(`/api/projects/${selectedProject}/shots/${encodeURIComponent(shotId)}/${action}`, options);
  selectedShot = shotId;
  $("assetsResult").textContent = `镜头 ${shotId} 操作完成：${action}`;
  await selectProject(selectedProject, false);
  return result;
}

$("lockShot").onclick = () => shotAction("lock");
$("unlockShot").onclick = () => shotAction("unlock");
$("qualityShot").onclick = () => shotAction("quality-check");
$("approveShot").onclick = () => shotAction("review", { status: "approved" });
$("rejectShot").onclick = () => shotAction("review", { status: "rejected", note: "manual rejected" });
$("needsRetryShot").onclick = () => shotAction("review", { status: "needs_retry", note: "manual retry requested" });

$("validateAssets").onclick = async () => {
  const result = await api("/api/assets/validate", { method: "POST" });
  if (result.ok) {
    $("assetsResult").textContent = `资源检查通过。Warnings: ${(result.warnings || []).join("; ") || "无"}`;
  } else {
    const errors = result.errors || [];
    const tips = errors.map((e) => {
      if (e.includes("manifest")) return "→ 请在「资源」tab 上传资源或编辑 manifest.yaml";
      if (e.includes("comfyui") || e.includes("ComfyUI")) return "→ 请确认 ComfyUI 已启动并检查地址配置";
      if (e.includes("config")) return "→ 请在「资源」tab 的资产配置中检查路径";
      return "";
    }).filter(Boolean);
    $("assetsResult").textContent = `资源检查失败：${errors.join("; ")}${tips.length ? "\n" + tips.join("\n") : ""}`;
  }
};

async function loadAssets() {
  const data = await api("/api/assets");
  currentAssetConfig = data.config || {};
  $("manifestYaml").value = data.manifest_yaml || "";
  $("configYaml").value = data.config_yaml || "";
  fillConfigForm(currentAssetConfig);
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

$("deleteBindingForm").onsubmit = async (event) => {
  event.preventDefault();
  const key = $("deleteKey").value.trim();
  if (!key) return alert("请填写 Key");
  const params = new URLSearchParams({
    asset_type: $("deleteAssetType").value,
    key,
    emotion: $("deleteEmotion").value,
    delete_file: $("deletePhysicalFile").checked ? "true" : "false",
  });
  const result = await api(`/api/assets/binding?${params.toString()}`, { method: "DELETE" });
  $("assetsResult").textContent = result.removed_path
    ? `已删除绑定：${result.removed_path}`
    : "已删除绑定或映射";
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
      strength_model: Number($("loraStrengthModel").value || 0.85),
      strength_clip: Number($("loraStrengthClip").value || 0.8),
    }),
  });
  $("assetsResult").textContent = result.ok ? "角色 LoRA 映射已保存" : "保存失败";
  await loadAssets();
};

$("sceneAudioForm").onsubmit = async (event) => {
  event.preventDefault();
  const sceneId = $("sceneAudioId").value.trim();
  if (!sceneId) return alert("请填写 scene_id");
  let segments = [];
  const rawSegments = $("sceneAudioSegments").value.trim();
  if (rawSegments) {
    try {
      segments = JSON.parse(rawSegments);
    } catch (err) {
      return alert(`segments JSON 格式错误：${err.message}`);
    }
    if (!Array.isArray(segments)) return alert("segments 必须是数组");
  }
  const result = await api("/api/assets/scene-audio", {
    method: "PUT",
    body: JSON.stringify({
      scene_id: sceneId,
      audio_path: $("sceneAudioPath").value.trim(),
      segments,
    }),
  });
  $("assetsResult").textContent = result.ok ? "场景音频切分数据已保存" : "保存失败";
  await loadAssets();
};

function fillConfigForm(config) {
  const style = config.style_lora || {};
  const characterLoras = config.character_loras || {};
  const references = config.references || {};
  const continuity = config.continuity || {};
  const voice = config.voice || {};
  $("cfgComfyRoot").value = config.comfyui_models_root || "";
  $("cfgStyleEnabled").checked = Boolean(style.enabled);
  $("cfgStyleName").value = style.name || "";
  $("cfgStyleStrengthModel").value = style.strength_model ?? 0.7;
  $("cfgStyleStrengthClip").value = style.strength_clip ?? 0.7;
  $("cfgCharacterLorasEnabled").checked = Boolean(characterLoras.enabled);
  $("cfgCharacterLorasOptional").checked = characterLoras.optional !== false;
  $("cfgPreferLibrary").checked = references.prefer_library_assets !== false;
  $("cfgGenerateMissing").checked = references.generate_missing_assets !== false;
  $("cfgUseCharRef").checked = references.use_character_reference_for_shots !== false;
  $("cfgUseExprRef").checked = references.use_expression_reference_for_shots !== false;
  $("cfgMaxDuration").value = continuity.max_shot_duration_seconds ?? 5;
  $("cfgSplitLongActions").checked = continuity.split_long_actions !== false;
  $("cfgUseTailFrame").checked = continuity.use_previous_tail_frame !== false;
  $("cfgSceneTts").checked = voice.scene_level_tts !== false;
  $("cfgFallbackShotTts").checked = voice.fallback_per_shot_tts !== false;
}

$("configForm").onsubmit = async (event) => {
  event.preventDefault();
  const next = {
    ...(currentAssetConfig || {}),
    comfyui_models_root: $("cfgComfyRoot").value.trim(),
    style_lora: {
      ...((currentAssetConfig || {}).style_lora || {}),
      enabled: $("cfgStyleEnabled").checked,
      name: $("cfgStyleName").value.trim(),
      strength_model: Number($("cfgStyleStrengthModel").value || 0.7),
      strength_clip: Number($("cfgStyleStrengthClip").value || 0.7),
    },
    character_loras: {
      enabled: $("cfgCharacterLorasEnabled").checked,
      optional: $("cfgCharacterLorasOptional").checked,
    },
    references: {
      prefer_library_assets: $("cfgPreferLibrary").checked,
      generate_missing_assets: $("cfgGenerateMissing").checked,
      use_character_reference_for_shots: $("cfgUseCharRef").checked,
      use_expression_reference_for_shots: $("cfgUseExprRef").checked,
    },
    continuity: {
      max_shot_duration_seconds: Math.min(5, Number($("cfgMaxDuration").value || 5)),
      split_long_actions: $("cfgSplitLongActions").checked,
      use_previous_tail_frame: $("cfgUseTailFrame").checked,
    },
    voice: {
      scene_level_tts: $("cfgSceneTts").checked,
      fallback_per_shot_tts: $("cfgFallbackShotTts").checked,
    },
  };
  const result = await api("/api/assets/config-json", {
    method: "PUT",
    body: JSON.stringify({ data: next }),
  });
  $("assetsResult").textContent = result.ok ? "资产配置表单已保存" : "保存失败";
  await loadAssets();
};

// Shot filter change
const shotFilter = $("shotFilter");
if (shotFilter) shotFilter.onchange = () => { if (currentProject) renderProduction(currentProject); };

refreshProjects();
loadAssets();
refreshQueue();
refreshHealth();
