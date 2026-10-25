// Seamfix front end: one run at a time, fed by the run's server-sent events.
const $ = (id) => document.getElementById(id);
const STATUS = {
  returned: ["Already seamless", "ok"], cut: ["Cut to repeat", "ok"], fixed: ["Repaired", "ok"],
  review: ["Kept original", "warn"], refused: ["Refused", "bad"],
};

// ---------- tabs and segmented controls ----------
document.querySelectorAll(".tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === tab.dataset.tab));
    if (tab.dataset.tab === "results") loadResults();
  })
);

function segment(id, onChange) {
  const el = $(id);
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || b.disabled) return;
    el.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    onChange?.(b.dataset.value ?? b.dataset.view);
  });
  return () => el.querySelector(".on")?.dataset.value;
}
const mode = segment("mode", (v) => {
  $("modeHint").textContent = v === "free"
    ? "Local LaMa only, no API calls. Fine for flat, simple designs."
    : "Nano Banana paints, Gemini judges. Uses API credit.";
});
const attempts = segment("attempts");
segment("views", (v) => show(v));

async function budget() {
  try {
    const b = await (await fetch("/api/budget")).json();
    $("spend").textContent = `API spend $${b.spent.toFixed(2)} of $${b.cap.toFixed(2)}`;
  } catch { /* the meter is not essential */ }
}
budget();

// ---------- choosing a tile ----------
let chosen = null;
function choose(file) {
  if (!file || !file.type.startsWith("image/")) return;
  chosen = file;
  $("preview").src = URL.createObjectURL(file);
  $("preview").hidden = false;
  $("dropText").hidden = true;
  $("filename").textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  $("run").disabled = false;
}
$("file").addEventListener("change", (e) => choose(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => choose(e.dataTransfer.files[0]));

// ---------- one run ----------
let run = null;

function reset() {
  run = { id: null, images: {}, candidates: [], selected: null, windows: 0, accepted: 0 };
  document.querySelectorAll("#steps li").forEach((li) => { li.className = ""; delete li.dataset.seen; });
  $("log").innerHTML = "";
  ["mLr", "mTb", "mKind", "mWin", "mReason"].forEach((id) => $(id).textContent = "—");
  $("status").hidden = true;
  $("download").hidden = true;
  $("strip").hidden = true;
  $("strip").innerHTML = "";
  show("input");
}

function log(text, cls = "") {
  const line = document.createElement("div");
  line.textContent = text;
  if (cls) line.className = cls;
  $("log").append(line);
  $("log").scrollTop = $("log").scrollHeight;
}

function step(name, state = "active") {
  const items = [...document.querySelectorAll("#steps li")];
  const at = items.findIndex((li) => li.dataset.step === name);
  // steps the run went through turn done; steps it never reached are skipped
  items.forEach((li, i) => {
    if (i < at) li.className = li.dataset.seen ? "done" : "skip";
  });
  if (at >= 0) {
    items[at].className = state;
    items[at].dataset.seen = "1";
  }
}

const url = (file) => `/runs/${run.id}/${file}`;
const join = (j) => `${j.overall.toFixed(2)} / ${j.worst.toFixed(2)} ${j.ok ? "pass" : "fail"}`;

function show(view) {
  document.querySelectorAll("#views button").forEach((b) => b.classList.toggle("on", b.dataset.view === view));
  const img = run?.images[view];
  $("compare").hidden = view !== "compare" || !run?.images.after;
  $("strip").hidden = view !== "attempts" || !run?.candidates.length;
  let src = img;
  if (view === "attempts") src = run?.selected?.panel;
  $("shot").hidden = !src || view === "compare";
  $("empty").hidden = !!src || (view === "compare" && run?.images.after);
  if (src && view !== "compare") $("shot").src = src;
  if (view === "compare" && run?.images.after) {
    setCompare("compare", "cmpBefore", "cmpAfter", run.images.original, run.images.after);
  }
}

// a slider when both have the same shape; side by side when a cut changed it
function setCompare(box, before, after, a, b) {
  const x = $(before), y = $(after), el = $(box);
  const fit = () => {
    if (!x.naturalWidth || !y.naturalWidth) return;
    const ra = x.naturalWidth / x.naturalHeight, rb = y.naturalWidth / y.naturalHeight;
    const side = Math.abs(ra / rb - 1) > 0.02;
    el.classList.toggle("side", side);
    el.style.setProperty("--ratio", side ? ra + rb : ra);
  };
  x.onload = y.onload = fit;
  x.src = a;
  y.src = b;
  fit();
}
for (const [range, box] of [["cmpRange", "compare"], ["rRange", "rCompare"]]) {
  $(range).addEventListener("input", (e) => $(box).style.setProperty("--cut", `${e.target.value}%`));
}

function addCandidate(e) {
  const c = { key: `${e.window}-${e.attempt}`, ...e, panel: url(e.images.panel) };
  run.candidates.push(c);
  const card = document.createElement("div");
  card.className = "card";
  card.dataset.key = c.key;
  const gates = e.gates
    .map((g) => `<span class="chip ${g.ok ? "" : "no"}" title="${g.value} (${g.limit})">${g.name.replace(/_(lr|tb)$/, "")}</span>`)
    .join("");
  card.innerHTML = `<div class="head"><span>Window ${e.window + 1} · try ${e.attempt}</span><span>${e.painter.split(":")[0].replace("gemini-", "")}</span></div>
    <div class="gates">${gates}</div><div class="judge">${e.gates_ok ? "waiting for judge…" : "rejected by gates"}</div>`;
  card.addEventListener("click", () => select(c.key));
  $("strip").append(card);
  select(c.key);
}

function select(key) {
  run.selected = run.candidates.find((c) => c.key === key);
  document.querySelectorAll(".card").forEach((c) => c.classList.toggle("sel", c.dataset.key === key));
  if (document.querySelector("#views .on")?.dataset.view === "attempts") show("attempts");
}

function handle(e) {
  switch (e.type) {
    case "run":
      log(`run ${e.run_id}`);
      break;
    case "stage":
      step(e.stage);
      log(`${e.stage}: ${e.message ?? ""}`);
      break;
    case "check": {
      const c = e.check;
      if (e.which === "input" || e.which === "cut") {
        run.images.input = url(e.images["2x2"]);
        run.images.diamond = url(e.images.diamond);
        run.images.marked = url(e.images.marked);
        if (e.which === "input") run.images.original = run.images.input;
        $("mLr").textContent = join(c.lr);
        $("mTb").textContent = join(c.tb);
        if (e.which === "cut") log("cut tile measured");
        show(e.which === "cut" ? "input" : "marked");
      }
      break;
    }
    case "repeat":
      if (!e.found) { log("no smaller repeat inside the image"); step("repeat", "done"); }
      else log(`${e.describe}`);
      break;
    case "diagnosis":
      $("mKind").textContent = `${e.kind.replaceAll("_", " ")}, ${e.difficulty}`;
      if (e.brief?.style) log(`design: ${e.brief.style}`);
      break;
    case "plan":
      run.images.plan = url(e.image);
      run.windows += e.windows.length;
      log(`${e.windows.length} window(s), band ${Object.values(e.band).join("/")} px`);
      show("plan");
      break;
    case "candidate":
      addCandidate(e);
      log(`window ${e.window + 1} try ${e.attempt} (${e.painter}): gates ${e.gates_ok ? "pass" : "fail"}`, e.gates_ok ? "" : "warn");
      show("attempts");
      break;
    case "verdict": {
      const v = e.verdict;
      const card = document.querySelector(`.card[data-key="${e.window}-${e.attempt}"] .judge`);
      const text = `${e.accepted ? "accepted" : "rejected"} ${v.continuity}/${v.fidelity}/${v.finish}${v.fix ? ` · ${v.fix}` : ""}`;
      if (card) { card.textContent = text; card.title = text; }
      log(`judge: ${text}`, e.accepted ? "" : "warn");
      break;
    }
    case "window":
      if (e.accepted) run.accepted += 1;
      $("mWin").textContent = `${run.accepted} of ${run.windows}`;
      log(`window ${e.window + 1}: ${e.accepted ? "accepted" : "kept original"}`);
      break;
    case "assemble":
      $("mLr").textContent = join(e.check.lr);
      $("mTb").textContent = join(e.check.tb);
      break;
    case "warning":
      log(e.message, "warn");
      break;
    case "error":
      log(e.message, "bad");
      finish({ status: "refused", reason: e.message, files: {} });
      break;
    case "result":
      finish(e);
      break;
    case "end":
      run.source?.close();
      $("run").disabled = false;
      budget();
      break;
  }
}

function finish(e) {
  step("finish", "done");
  const [label, tone] = STATUS[e.status] ?? [e.status, ""];
  $("status").textContent = label;
  $("status").className = `pill ${tone}`;
  $("status").hidden = false;
  $("mReason").textContent = e.reason || (e.status === "fixed" ? "every gate and the judge passed" : label);
  $("mReason").title = e.reason || "";
  const files = e.files || {};
  if (files.design) {
    $("download").href = url(files.design);
    $("download").hidden = false;
  }
  if (files["2x2"]) run.images.after = url(files["2x2"]);
  else if (e.status === "cut" || e.status === "returned") run.images.after = run.images.input;
  if (run.images.after) show("compare");
  log(`result: ${label}`);
}

$("run").addEventListener("click", async () => {
  if (!chosen) return;
  reset();
  $("run").disabled = true;
  const form = new FormData();
  form.append("file", chosen);
  form.append("mode", mode());
  form.append("attempts", attempts());
  try {
    const res = await fetch("/api/runs", { method: "POST", body: form });
    if (!res.ok) throw new Error(`upload failed (${res.status})`);
    run.id = (await res.json()).run_id;
    run.source = new EventSource(`/api/runs/${run.id}/events`);
    run.source.onmessage = (m) => handle(JSON.parse(m.data));
    run.source.onerror = () => { run.source.close(); $("run").disabled = false; };
  } catch (err) {
    log(err.message, "bad");
    $("run").disabled = false;
  }
});
reset();

// ---------- results ----------
let results = [];
const filter = segment("filter", () => renderList());

async function loadResults() {
  results = await (await fetch("/api/results")).json();
  renderList();
}

function renderList() {
  const kind = filter();
  const list = $("list");
  list.innerHTML = "";
  results.filter((r) => kind === "all" || r.kind === kind).forEach((r, i) => {
    const li = document.createElement("li");
    const [label, tone] = STATUS[r.status] ?? [r.status, ""];
    const sample = r.name.startsWith("sample: ");
    const name = r.kind === "case" ? (sample ? r.input : r.name.replaceAll("_", " ")) : r.input;
    const when = r.name.slice(0, 15).replace(/(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})\d{2}/, "$1-$2-$3 $4:$5");
    const sub = r.kind !== "case" ? `run · ${when}` : sample ? "sample · free mode" : `test case · ${r.ok ? "pass" : "FAIL"}`;
    li.innerHTML = `<img src="${r.after || r.before || ""}" alt=""><div><div class="name">${name}</div><div class="sub">${sub}</div></div><span class="pill ${tone}">${label}</span>`;
    li.addEventListener("click", () => {
      list.querySelectorAll("li").forEach((x) => x.classList.toggle("sel", x === li));
      openResult(r);
    });
    list.append(li);
    if (i === 0) li.click();
  });
}

function openResult(r) {
  const [label, tone] = STATUS[r.status] ?? [r.status, ""];
  $("rTitle").textContent = r.kind === "case" ? `Test case: ${r.name.replaceAll("_", " ")}` : r.input;
  $("rStatus").textContent = label;
  $("rStatus").className = `pill ${tone}`;
  $("rStatus").hidden = false;
  $("rDownload").hidden = !r.design;
  if (r.design) $("rDownload").href = r.design;
  $("rEmpty").hidden = !!r.before;
  $("rCompare").hidden = !r.before;
  if (r.before) setCompare("rCompare", "rBefore", "rAfter", r.before, r.after || r.before);
  const out = r.size_out ? `${r.size_out[0]}×${r.size_out[1]}` : "—";
  $("rSize").textContent = `${r.size_in[0]}×${r.size_in[1]} → ${out}`;
  $("rJoins").textContent = r.check ? `${r.check.lr.overall.toFixed(2)} / ${r.check.tb.overall.toFixed(2)}` : "—";
  $("rExpect").textContent = r.expect ? r.expect.join(" or ") : "—";
  const text = [r.tests, r.reason].filter(Boolean).join(" — ") || label;
  $("rReason").textContent = text;
  $("rReason").title = text;
}
