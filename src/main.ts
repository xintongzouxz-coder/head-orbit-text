import { FilesetResolver, FaceLandmarker, HandLandmarker } from "@mediapipe/tasks-vision";
import "./style.css";

type Settings = {
  fontSize: number;
  speedMultiplier: number;
};

const settings: Settings = {
  fontSize: 28,
  speedMultiplier: 0.9,
};

const video = document.querySelector<HTMLVideoElement>("#cam")!;
const canvas = document.querySelector<HTMLCanvasElement>("#fx")!;
const ctx = canvas.getContext("2d")!;

const gear = document.querySelector<HTMLButtonElement>("#gear")!;
const panel = document.querySelector<HTMLDivElement>("#panel")!;
const fontSizeSlider = document.querySelector<HTMLInputElement>("#fontSize")!;
const speedSlider = document.querySelector<HTMLInputElement>("#speed")!;
const fontSizeValue = document.querySelector<HTMLSpanElement>("#fontSizeValue")!;
const speedValue = document.querySelector<HTMLSpanElement>("#speedValue")!;
const thoughtInput = document.querySelector<HTMLInputElement>("#thought")!;
const clearBtn = document.querySelector<HTMLButtonElement>("#clear")!;

// --- UI: panel toggle ---
function setPanelOpen(open: boolean) {
  panel.classList.toggle("panel--hidden", !open);
}

let panelOpen = false;
gear.addEventListener("click", () => {
  panelOpen = !panelOpen;
  setPanelOpen(panelOpen);
});

document.addEventListener("click", (e) => {
  if (!panelOpen) return;
  const target = e.target as HTMLElement;
  if (target.closest("#panel") || target.closest("#gear")) return;
  panelOpen = false;
  setPanelOpen(false);
});

// --- UI: sliders (REAL-TIME via input) ---
fontSizeSlider.addEventListener("input", () => {
  settings.fontSize = Number(fontSizeSlider.value);
  fontSizeValue.textContent = String(settings.fontSize);
});

speedSlider.addEventListener("input", () => {
  settings.speedMultiplier = Number(speedSlider.value);
  speedValue.textContent = settings.speedMultiplier.toFixed(2);
});

// --- Camera ---
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    await initVision();
  } catch (err) {
    console.warn("Camera permission denied or unavailable:", err);
  }
}
let faceLandmarker: FaceLandmarker | null = null;
let handLandmarker: HandLandmarker | null = null;

async function initVision() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
  });

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  console.log("Vision ready");
}

function getFaceCenter(faceLandmarks: { x: number; y: number }[]) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of faceLandmarks) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  return { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
}

function drawDot(x: number, y: number, r = 8) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}

type Particle = {
  id: number;
  token: string;

  kind: "orbit" | "veil";
  u: number;
  biasY: number;

  vx: number;   // ✅ 新增
  vy: number;   // ✅ 新增
  vx0: number; // ✅ 原位（初始位置）
  vy0: number;
  theta: number;
  lane: number;
  radiusOffset: number;
  omegaOffset: number;
  omegaBase: number;
  bornAt: number;
};

let nextId = 1;
const particles: Particle[] = [];

function clearAll() {
  particles.length = 0; // 清空数组
  nextId = 1;           // 可选：重置 id（更干净）
}

const ORBIT = {
  laneGap: 18,
  laneCapacity: 16,
  maxParticles: 600,

  // brush
  influenceRadius: 90,
  repelStrength: 140,   // px/s^2-ish (we apply per frame)
  swirlStrength: 3.2,   // rad/s
  // decay (seconds)
  tauRadius: 1.2,
  tauOmega: 0.9,

  // ellipse shape
  ellipseYScale: 0.92,

  veilPush: 2.4,      // 推开强度（越大越“拨开”）
  veilReturnTau: 1.1, // 回流时间（秒，越大回得越慢）
};

function tokenizeGrapheme(text: string) {
  const seg = new Intl.Segmenter("und", { granularity: "grapheme" });
  return Array.from(seg.segment(text)).map(s => s.segment).filter(t => t.trim().length > 0);
}

function enqueueTokens(text: string) {
  const tokens = tokenizeGrapheme(text);
  const now = performance.now();

  for (const t of tokens) {
    const i = particles.length;
    const lane = Math.floor(i / ORBIT.laneCapacity);

    // --- 1) 始终生成 1 个 orbit（保证 3D 环绕存在）---
    const idxInLane = i % ORBIT.laneCapacity;
    const theta0 =
      (idxInLane / ORBIT.laneCapacity) * Math.PI * 2 +
      (Math.random() - 0.5) * 0.08;

    particles.push({
      id: nextId++,
      token: t,
      kind: "orbit",
      u: 0,
      biasY: 0,
      vx: 0,
      vy: 0,
      theta: theta0,
      lane,
      radiusOffset: 0,
      omegaOffset: 0,
      omegaBase: 0.9,
      bornAt: now,
    });

    // --- 2) 额外生成多个 veil（盖脸雾层）---
    // 字越多，veil 越多：让遮挡快速上来
    const density = Math.min(1, particles.length / 120);
    const veilCopies = 2 + Math.floor(6 * density); // 2..8

    for (let k = 0; k < veilCopies; k++) {
      // 椭圆内部均匀采样
      const r = Math.sqrt(Math.random());
      const ang = Math.random() * Math.PI * 2;
      const vx = r * Math.cos(ang);
      const vy = r * Math.sin(ang);

      particles.push({
        id: nextId++,
        token: t,
        kind: "veil",
        u: Math.pow(Math.random(), 0.55),
        biasY: -0.65 + Math.random() * 0.35, // 偏上：更像挡视线
        vx,
        vy,
        vx0: vx,
        vy0: vy,
        theta: Math.random() * Math.PI * 2, // veil 自己也有 theta（用于轻微动态）
        lane: 0,
        radiusOffset: 0,
        omegaOffset: 0,
        omegaBase: 0.15, // veil 慢慢漂
        bornAt: now,
      });
    }
  }

  // cap
  if (particles.length > ORBIT.maxParticles) {
    particles.splice(0, particles.length - ORBIT.maxParticles);
  }
}

function expDecay(value: number, dt: number, tau: number) {
  // dt in seconds
  const k = Math.exp(-dt / Math.max(0.0001, tau));
  return value * k;
}

function mapNormToScreen(nx: number, ny: number) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const sw = window.innerWidth;
  const sh = window.innerHeight;

  // object-fit: cover 的缩放比例
  const scale = Math.max(sw / vw, sh / vh);
  const rw = vw * scale;
  const rh = vh * scale;

  // 居中裁切的偏移
  const ox = (rw - sw) / 2;
  const oy = (rh - sh) / 2;

  // 先把视频像素映射到“被 cover 缩放后”的坐标，再减掉裁切偏移
  let x = nx * vw * scale - ox;
  const y = ny * vh * scale - oy;

  // 因为视频做了 scaleX(-1) 镜像，这里把 x 镜像回来对齐
  x = sw - x;

  return { x, y };
}

// --- Canvas resize ---
function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor(window.innerWidth * dpr);
  const h = Math.floor(window.innerHeight * dpr);
  canvas.width = w;
  canvas.height = h;
  canvas.style.width = `${window.innerWidth}px`;
  canvas.style.height = `${window.innerHeight}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in CSS pixels
}
window.addEventListener("resize", resizeCanvas);

let headX = window.innerWidth * 0.5;
let headY = window.innerHeight * 0.45;

let lastT = performance.now();

// --- Temporary render loop (shows settings are live) ---
function draw() {
  // 用 CSS 像素绘制（因为你前面 setTransform(dpr,...) 了）
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

  // 如果 vision 还没准备好，就先显示一句提示
  if (!faceLandmarker || !handLandmarker || video.readyState < 2) {
    ctx.font = "16px system-ui";
    ctx.fillStyle = "rgba(255,255,255,0.8)";
    ctx.fillText("Loading vision models…", 20, 30);
    requestAnimationFrame(draw);
    return;
  }

  const now = performance.now();

  const faceRes = faceLandmarker.detectForVideo(video, now);
  const handRes = handLandmarker.detectForVideo(video, now);

  // --- get head center + face size (px) ---

let faceWidthPx = 220;
let faceHeightPx = 260;

if (faceRes.faceLandmarks?.length) {
  const lm = faceRes.faceLandmarks[0];
  // ✅ 用脸中心更新 headX/headY（全局变量）
  const center = getFaceCenter(lm);
const c = mapNormToScreen(center.x, center.y);

// ✅ 关键：把锚点从“脸中心”抬到“头部中心”
const headOffsetY = -faceHeightPx * 0.18; // 可调：-0.12 ~ -0.28
const targetX = c.x;
const targetY = c.y + headOffsetY;

const smooth = 0.35;
headX = headX + (targetX - headX) * smooth;
headY = headY + (targetY - headY) * smooth;
  // bbox in normalized space
  let minX = 1, maxX = 0, minY = 1, maxY = 0;
  for (const p of lm) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }

  // cover scale (与 mapNormToScreen 保持一致)
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const sw = window.innerWidth;
  const sh = window.innerHeight;
  const scale = Math.max(sw / vw, sh / vh);

  faceWidthPx  = (maxX - minX) * vw * scale;
  faceHeightPx = (maxY - minY) * vh * scale;

  faceWidthPx  = Math.max(160, Math.min(460, faceWidthPx));
  faceHeightPx = Math.max(200, Math.min(520, faceHeightPx));
}

// --- get index fingertip (screen) ---
let finger: { x: number; y: number } | null = null;
if (handRes.landmarks?.length) {
  const tip = handRes.landmarks[0][8];
  finger = mapNormToScreen(tip.x, tip.y);
}

// --- update + draw orbit particles ---
const dt = Math.min(0.05, (now - lastT) / 1000);
lastT = now;

const baseR = faceWidthPx * 0.55;

// ✅ 先算 headRx/headRy（后面 drawable 要用）
const headRx = faceWidthPx * 0.55;
const headRy = faceHeightPx * 0.62;
const veilRx = headRx * 1.15;  // ✅ 放大 15%
const veilRy = headRy * 1.35;  // ✅ 放大 35%，优先遮住眼睛方向
const drawable = particles.map((p) => {
  let x = headX;
  let y = headY;

  if (p.kind === "orbit") {
    const laneR = baseR + p.lane * ORBIT.laneGap + p.radiusOffset;
    const a = laneR;
    const b = laneR * ORBIT.ellipseYScale;
    x = headX + a * Math.cos(p.theta);
    y = headY + b * Math.sin(p.theta);
  } else {
    // ✅ veil：直接用内部采样点铺满椭圆
const a = veilRx * (0.15 + 0.95 * p.u);
const b = veilRy * (0.15 + 0.95 * p.u);

x = headX + a * p.vx;
y = headY + b * p.vy + (p.biasY * veilRy * 0.25);
  }

  const depth = Math.sin(p.theta);
  return { p, x, y, depth };
});

// Brush interaction (repel + slight swirl)
if (finger) {
  for (const d of drawable) {
    const dx = d.x - finger.x;
    const dy = d.y - finger.y;
    const dist = Math.hypot(dx, dy);

    if (dist < ORBIT.influenceRadius) {
      const t = 1 - dist / ORBIT.influenceRadius;

      if (d.p.kind === "veil") {
        // ✅ veil：推开 vx/vy（更像拨云）
        const nx = dx / (dist + 1e-6);
        const ny = dy / (dist + 1e-6);

        // 这里把屏幕推力映射到 vx/vy 空间（用 face 尺寸归一化）
        const push = ORBIT.veilPush * t * dt;
        d.p.vx += (nx * push) / 0.8;
        d.p.vy += (ny * push) / 0.8;

        // 限制范围，避免飞出椭圆太远
        d.p.vx = Math.max(-1.4, Math.min(1.4, d.p.vx));
        d.p.vy = Math.max(-1.4, Math.min(1.4, d.p.vy));
      } else {
        // ✅ orbit：保持你原来的推开逻辑
        d.p.radiusOffset += ORBIT.repelStrength * t * dt;
        d.p.omegaOffset += ORBIT.swirlStrength * t * dt;
      }
    }
  }
}

// Update angles + decay
for (const d of drawable) {
  const p = d.p;
  p.radiusOffset = expDecay(p.radiusOffset, dt, ORBIT.tauRadius);
  p.omegaOffset = expDecay(p.omegaOffset, dt, ORBIT.tauOmega);

  const omega = (p.omegaBase * settings.speedMultiplier) + p.omegaOffset;
  p.theta += omega * dt;
  if (p.kind === "veil") {
  const k = Math.exp(-dt / ORBIT.veilReturnTau);
  p.vx = p.vx0 + (p.vx - p.vx0) * k;
  p.vy = p.vy0 + (p.vy - p.vy0) * k;
}
}

drawable.sort((a, b) => a.depth - b.depth);

// ========== 1) 先画 BEHIND：depth < 0，但把“头部椭圆区域”挖掉 ==========
// 思路：clip 一个“屏幕矩形 - 头部椭圆”的区域（evenodd），
ctx.save();
ctx.beginPath();
ctx.rect(0, 0, window.innerWidth, window.innerHeight);
ctx.ellipse(headX, headY, headRx, headRy, 0, 0, Math.PI * 2);
ctx.clip("evenodd"); // 只允许画在“头部之外”

for (const d of drawable) {
  if (d.depth >= 0) continue;        // ✅ 只画“在头后面”的
  if (d.p.kind === "veil") continue;  // ✅ veil 不属于“头后面”，跳过

  const t = (d.depth + 1) / 2;

  const isVeil = d.p.kind === "veil";
  const scale = isVeil ? (0.95 + 0.25 * t) : (0.75 + 0.55 * t);
  const alpha = isVeil ? (0.55 + 0.40 * t) : (0.15 + 0.85 * t);

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.font = `${Math.round(settings.fontSize * scale)}px system-ui`;
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(d.p.token, d.x, d.y);
  ctx.restore();
}
ctx.restore();

// ========== 2) 再画 FRONT：depth >= 0，正常画在头前 ==========
for (const d of drawable) {
  const isVeil = d.p.kind === "veil";
  if (!isVeil && d.depth < 0) continue; // orbit 只画前半圈，veil 全画

  const t = (d.depth + 1) / 2;

  const scale = isVeil ? (1.00 + 0.15 * t) : (0.75 + 0.55 * t);
  const alpha = isVeil ? (0.65 + 0.30 * (1 - d.p.u)) : (0.15 + 0.85 * t);
  // 说明：veil 越靠中心(u小)越不透明 -> 更像“挡视线”

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.font = `${Math.round(settings.fontSize * scale)}px system-ui`;
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(d.p.token, d.x, d.y);
  ctx.restore();
}

// 可选：暂时保留调试点（你确认OK后我们再关掉）




  requestAnimationFrame(draw);
}

resizeCanvas();
startCamera();
draw();

// UX: autofocus input
setTimeout(() => thoughtInput.focus(), 300);

function commitThought(raw: string) {
  const text = raw.trim();
  if (!text) return;

  console.log("COMMIT:", text);
enqueueTokens(text);
  // TODO (next step): turn this text into orbit particles
  // e.g. enqueueTokens(text)

  thoughtInput.value = "";
}

thoughtInput.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;

  e.preventDefault();
  commitThought(thoughtInput.value);
});

clearBtn.addEventListener("click", () => {
  clearAll();
  thoughtInput.value = "";
  thoughtInput.focus();
});