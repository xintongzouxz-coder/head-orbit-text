let headMaskPts: { x: number; y: number }[] | null = null;

const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
  361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
  176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
  162, 21, 54, 103, 67, 109,
];
import { FilesetResolver, FaceLandmarker, HandLandmarker } from "@mediapipe/tasks-vision";
import "./style.css";

type Settings = {
  fontSize: number;
  speedMultiplier: number;
  colorMode: "global" | "random";
  globalColor: string;
};

const settings: Settings = {
  fontSize: 28,
  speedMultiplier: 0.9,
  colorMode: "global",
  globalColor: "#ffffff",
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
const modeGlobal = document.querySelector<HTMLInputElement>("#modeGlobal")!;
const modeRandom = document.querySelector<HTMLInputElement>("#modeRandom")!;
const globalColorRow = document.querySelector<HTMLDivElement>("#globalColorRow")!;
const colorPicker = document.querySelector<HTMLInputElement>("#colorPicker")!;

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

function applyColorModeUI() {
  const isGlobal = settings.colorMode === "global";
  globalColorRow.style.display = isGlobal ? "flex" : "none";
}

modeGlobal.addEventListener("change", () => {
  if (!modeGlobal.checked) return;
  settings.colorMode = "global";
  applyColorModeUI();
});

modeRandom.addEventListener("change", () => {
  if (!modeRandom.checked) return;
  settings.colorMode = "random";
  applyColorModeUI();
});

colorPicker.addEventListener("input", () => {
  settings.globalColor = colorPicker.value;
});

applyColorModeUI();

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
  color: string;

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
let orbitCounter = 0;

let lastCommittedText = "";

// ✅ 分开存
const orbitParticles: Particle[] = [];
const veilParticles: Particle[] = [];

function clearAll() {
  orbitParticles.length = 0;
  veilParticles.length = 0;
  nextId = 1;
  orbitCounter = 0;

  lastCommittedText = ""; // ✅ 新增
}
const ORBIT_MAX_LANES = 12; // orbit 最多 8 圈（可调 6~10）
const ORBIT = {
  laneGap: 26,
  laneCapacity: 16,
  maxParticles: 600,
  baseRScale: 0.85,

  // brush
  influenceRadius: 90,
  repelStrength: 140,   // px/s^2-ish (we apply per frame)
  swirlStrength: 3.2,   // rad/s
  // decay (seconds)
  tauRadius: 1.2,
  tauOmega: 0.9,

  // ellipse shape
  ellipseYScale: 0.92,

  veilPush: 220,      // 推开强度（越大越“拨开”）
  veilReturnTau: 2.4, // 回流时间（秒，越大回得越慢）

  veilDriftAmp: 0.09,   // ✅ veil 内部漂移幅度（0.03~0.09 调）
veilDriftFreq: 1.4,   // ✅ 漂移频率倍率（跟 speed 一起生效）

};

function enforceCaps() {
  // 你原来的总上限
  const MAX_TOTAL = ORBIT.maxParticles;

  // orbit 的“预算”：最多保留这么多（避免无限增长）
  const orbitSlots = ORBIT.laneCapacity * ORBIT_MAX_LANES;
  const MAX_ORBITS = orbitSlots;

  // 1) 先限制 orbit 自身不要无限长
  if (orbitParticles.length > MAX_ORBITS) {
    orbitParticles.splice(0, orbitParticles.length - MAX_ORBITS);
  }

  // 2) 再限制总量：超了先砍 veil
  let total = orbitParticles.length + veilParticles.length;
  if (total <= MAX_TOTAL) return;

  let excess = total - MAX_TOTAL;

  // ✅ 先砍 veil（关键）
  if (veilParticles.length >= excess) {
    veilParticles.splice(0, excess);
    return;
  }

  // 如果 veil 不够砍，再砍 orbit（一般不会发生）
  excess -= veilParticles.length;
  veilParticles.length = 0;
  orbitParticles.splice(0, Math.min(excess, orbitParticles.length));
}

function tokenizeMixed(text: string) {
  const wordSeg = new Intl.Segmenter("en", { granularity: "word" });
  const graphemeSeg = new Intl.Segmenter("und", { granularity: "grapheme" });

  const out: string[] = [];

  for (const seg of wordSeg.segment(text)) {
    const s = seg.segment;

    // 过滤纯空白
    if (!s.trim()) continue;

    // 如果是“像单词”的英文片段：整词保留
    const isSimpleEnglishWord =
      (seg as any).isWordLike &&
      /^[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*$/.test(s);

    if (isSimpleEnglishWord) {
      out.push(s);
      continue;
    }

    // 其他情况（中文、emoji、混合、标点等）用 grapheme 保持细粒度
    for (const g of graphemeSeg.segment(s)) {
      const t = g.segment;
      if (t.trim()) out.push(t);
    }
  }

  return out;
}

function randomEntryColor() {
  const h = Math.floor(Math.random() * 360);
  return `hsl(${h} 80% 70%)`;
}

function enqueueTokens(text: string) {
  const tokens = tokenizeMixed(text);
  const now = performance.now();
  const entryColor = randomEntryColor();

  for (const t of tokens) {
    // ✅ 1) 始终生成 1 个 orbit（保持 3D）
    const orbitSlots = ORBIT.laneCapacity * ORBIT_MAX_LANES;
    const iOrbit = orbitCounter++ % orbitSlots; // ✅ wrap，永远不会越打越大
    const lane = Math.floor(iOrbit / ORBIT.laneCapacity);
    const idxInLane = iOrbit % ORBIT.laneCapacity;

    const theta0 =
      (idxInLane / ORBIT.laneCapacity) * Math.PI * 2 +
      (Math.random() - 0.5) * 0.08;

    orbitParticles.push({
      id: nextId++,
      token: t,
      color: entryColor,

      kind: "orbit",
      u: 0,
      biasY: 0,

      vx: 0,
      vy: 0,
      vx0: 0, // ✅ 补齐，避免 TS 报错
      vy0: 0,

      theta: theta0,
      lane,
      radiusOffset: 0,
      omegaOffset: 0,
      omegaBase: 0.9,
      bornAt: now,
    });

    // ✅ 2) veil：数量随“orbitCounter”（而不是 particles.length）变化，避免 veil 反向影响密度判断
    const density = Math.min(1, orbitCounter / 200); // 0..1（可调：200~350）
    const veilCopies = 2 + Math.floor(6 * density);  // 2..8（可调）

    for (let k = 0; k < veilCopies; k++) {
  // ✅ 每个 veil 粒子自己的 y 偏移
  const biasY = -0.65 + Math.random() * 0.35;

  // ✅ 椭圆内部均匀采样（覆盖脸用这个最稳）
  const r = Math.sqrt(Math.random());
  const ang = Math.random() * Math.PI * 2;
  const vx = r * Math.cos(ang);
  const vy = r * Math.sin(ang);

  // ✅ u 建议先直接等于 r（和位置一致，alpha 才自然）
  const u = r;

  veilParticles.push({
    id: nextId++,
    token: t,
    color: entryColor,

    kind: "veil",
    u,
    biasY,

    vx,
    vy,
    vx0: vx,
    vy0: vy,

    theta: Math.random() * Math.PI * 2,
    lane: 0,
    radiusOffset: 0,
    omegaOffset: 0,
    omegaBase: 0.45,
    bornAt: now,
  });
}
  }

  // cap（可保留）
 enforceCaps();
}

function expDecay(value: number, dt: number, tau: number) {
  // dt in seconds
  const k = Math.exp(-dt / Math.max(0.0001, tau));
  return value * k;
}

function buildHeadMaskPts(
  lm: { x: number; y: number }[],
  headX: number,
  headY: number,
  faceHeightPx: number
) {
  const inflateX = 1.18;
  const inflateY = 1.45;
  const hairLift = -faceHeightPx * 0.10;

  const center = getFaceCenter(lm);
  const c = mapNormToScreen(center.x, center.y);

  const pts: { x: number; y: number }[] = [];

  for (const idx of FACE_OVAL) {
    const p = mapNormToScreen(lm[idx].x, lm[idx].y);
    const dx = p.x - c.x;
    const dy = p.y - c.y;

    pts.push({
      x: headX + dx * inflateX,
      y: headY + dy * inflateY + hairLift,
    });
  }

  return pts;
}

function boundsOfPts(pts: { x: number; y: number }[]) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  return { minX, minY, maxX, maxY };
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

function buildHeadMaskPath(
  lm: { x: number; y: number }[],
  headX: number,
  headY: number,
  faceWidthPx: number,
  faceHeightPx: number
) {
  // 让遮罩比“脸”更像“头”（包含头发）
  const inflateX = 1.18;              // 宽一点
  const inflateY = 1.45;              // 高很多（包含头发）
  const hairLift = -faceHeightPx * 0.10; // 整体再往上提一点点

  // 用脸中心当作“局部坐标系原点”
  const center = getFaceCenter(lm);
  const c = mapNormToScreen(center.x, center.y);

  const path = new Path2D();

  for (let i = 0; i < FACE_OVAL.length; i++) {
    const idx = FACE_OVAL[i];
    const p = mapNormToScreen(lm[idx].x, lm[idx].y);

    const dx = p.x - c.x;
    const dy = p.y - c.y;

    // 把 face-oval 点围绕 headX/headY 重新“放大”成头的轮廓
    const x = headX + dx * inflateX;
    const y = headY + dy * inflateY + hairLift;

    if (i === 0) path.moveTo(x, y);
    else path.lineTo(x, y);
  }

  path.closePath();
  return path;
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

function getParticleColor(p: Particle) {
  return settings.colorMode === "global" ? settings.globalColor : p.color;
}

// --- Temporary render loop (shows settings are live) ---
function draw() {
  // 用 CSS 像素绘制（因为你前面 setTransform(dpr,...) 了）
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

  // 如果 vision 还没准备好，就先显示一句提示
 if (!faceLandmarker || !handLandmarker || video.readyState < 2) {
  ctx.font = "16px system-ui";
  ctx.fillStyle = settings.globalColor; // ✅ 用全局色就行
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
const headOffsetY = -faceHeightPx * 0.05; // 可调：-0.12 ~ -0.28
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

let headMaskPts: { x: number; y: number }[] | null = null;
let headBounds: { minX: number; minY: number; maxX: number; maxY: number } | null = null;

if (faceRes.faceLandmarks?.length) {
  const lm = faceRes.faceLandmarks[0];
  headMaskPts = buildHeadMaskPts(lm, headX, headY, faceHeightPx);
  headBounds = boundsOfPts(headMaskPts);
}

let headMaskPath: Path2D | null = null;
if (faceRes.faceLandmarks?.length) {
  const lm = faceRes.faceLandmarks[0];
  headMaskPath = buildHeadMaskPath(lm, headX, headY, faceWidthPx, faceHeightPx);
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

const baseR = faceWidthPx * (ORBIT.baseRScale ?? 0.55);

// ✅ 先算 headRx/headRy（后面 drawable 要用）
const headRx = faceWidthPx * 0.60;  // 原 0.55
const headRy = faceHeightPx * 0.68; // 原 0.62
const veilRx = headBounds
  ? ((headBounds.maxX - headBounds.minX) * 0.5) * 1.01
  : headRx * 1.08;

const veilRy = headBounds
  ? ((headBounds.maxY - headBounds.minY) * 0.5) * 1.01
  : headRy * 1.12;
const allParticles = orbitParticles.concat(veilParticles);

const drawable = allParticles.map((p) => {
  let x = headX;
  let y = headY;

  if (p.kind === "orbit") {
    const laneR = baseR + p.lane * ORBIT.laneGap + p.radiusOffset;
    const a = laneR;
    const b = laneR * ORBIT.ellipseYScale;
    x = headX + a * Math.cos(p.theta);
    y = headY + b * Math.sin(p.theta);
  } else {
  // ✅ veil：顺时针持续旋转（由 p.theta 驱动，speed slider 会影响 p.theta）
  const phi = p.theta; // 方向反了就改成 -p.theta
  const c = Math.cos(phi);
  const s = Math.sin(phi);

  // 用“当前被交互影响后的局部坐标”来旋转（关键）
  const vx = p.vx;
  const vy = Math.max(-0.95, Math.min(0.95, p.vy)); // clamp 生效

  // 前向旋转：局部(vx,vy) -> 旋转后的(rvx,rvy)
  const rvx = vx * c - vy * s;
  const rvy = vx * s + vy * c;

  x = headX + veilRx * rvx;
  y = headY + veilRy * rvy + (p.biasY * veilRy * 0.12);
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
  const nx = dx / (dist + 1e-6);
  const ny = dy / (dist + 1e-6);

  // push in screen space (CSS px)
  const pushPx = ORBIT.veilPush * t * dt;

  // ✅ 关键：把“屏幕推力”转成“veil局部(vx/vy)推力”
  const phi = d.p.theta; // 与你 draw 里用的旋转角保持一致
  const c = Math.cos(phi);
  const s = Math.sin(phi);

  // 屏幕位移 -> 归一化到椭圆坐标
  const lx = (nx * pushPx) / (veilRx + 1e-6);
  const ly = (ny * pushPx) / (veilRy + 1e-6);

  // 逆旋转（rotation matrix transpose）
  const dvx = c * lx + s * ly;
  const dvy = -s * lx + c * ly;

  d.p.vx += dvx;
  d.p.vy += dvy;

  // ✅ 限制范围，避免飞太远（建议更紧一点）
  d.p.vx = Math.max(-1.2, Math.min(1.2, d.p.vx));
  d.p.vy = Math.max(-0.95, Math.min(0.95, d.p.vy));
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
  // ✅ speed 会影响 p.theta（你上面 omega 里已经 * settings.speedMultiplier 了）
  // 所以只要 drift 用 theta，就天然被 speed 控制
  const phase = p.theta * (ORBIT.veilDriftFreq ?? 1);

  // ✅ 越靠外圈(u 越大)漂移稍微更明显一点（你也可以反过来）
  const amp = (ORBIT.veilDriftAmp ?? 0.06) * (0.35 + 0.65 * p.u);

  // ✅ 用 id 做去同步，避免所有字一起晃
  const j = p.id * 0.37;

  // ✅ “移动的原位”目标点（一直在缓慢漂）
  const targetVx = p.vx0 + amp * Math.cos(phase + j);
  const targetVy = p.vy0 + amp * Math.sin(phase * 0.9 + j);

  // ✅ 强回弹：但回弹到 target，而不是静止的 vx0/vy0
  const k = Math.exp(-dt / ORBIT.veilReturnTau);
  p.vx = targetVx + (p.vx - targetVx) * k;
  p.vy = targetVy + (p.vy - targetVy) * k;

  // ✅ 可选：防止漂出太多（建议先留着）
  p.vx = Math.max(-1.4, Math.min(1.4, p.vx));
  p.vy = Math.max(-1.4, Math.min(1.4, p.vy));
}
}

drawable.sort((a, b) => a.depth - b.depth);

// ---------- helpers ----------
function drawToken(d: { p: Particle; x: number; y: number; depth: number }) {
  const isVeil = d.p.kind === "veil";
  const t = (d.depth + 1) / 2;

  const scale = isVeil
    ? (1.00 + 0.10 * t)
    : (0.65 + 0.75 * t);

  const veilFade = Math.pow(1 - d.p.u, 0.9);
  const alpha = isVeil
    ? (0.20 + 0.85 * veilFade)
    : (0.10 + 0.90 * t);

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.font = `${Math.round(settings.fontSize * scale)}px system-ui`;
  ctx.fillStyle = getParticleColor(d.p);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(d.p.token, d.x, d.y);
  ctx.restore();
}

// ---------- Layer A: orbit BEHIND + head occlusion ----------
ctx.save();

const outside = new Path2D();
outside.rect(0, 0, window.innerWidth, window.innerHeight);

if (headMaskPath) {
  outside.addPath(headMaskPath);
} else {
  const fallback = new Path2D();
  fallback.ellipse(headX, headY, headRx, headRy, 0, 0, Math.PI * 2);
  outside.addPath(fallback);
}

// ✅ 只允许画在“屏幕矩形 - 头部遮罩”区域
ctx.clip(outside, "evenodd");

for (const d of drawable) {
  if (d.p.kind !== "orbit") continue;
  if (d.depth >= 0) continue; // 只画后半圈
  drawToken(d);
}

ctx.restore();

// ---------- Layer B: veil ALWAYS FRONT (no clip) ----------
for (const d of drawable) {
  if (d.p.kind !== "veil") continue;
  drawToken(d);
}

// ---------- Layer C: orbit FRONT ----------
for (const d of drawable) {
  if (d.p.kind !== "orbit") continue;
  if (d.depth < 0) continue; // 只画前半圈
  drawToken(d);
}

// 可选：暂时保留调试点（你确认OK后我们再关掉）




  requestAnimationFrame(draw);
}

resizeCanvas();
startCamera();
draw();

// UX: autofocus input
setTimeout(() => thoughtInput.focus(), 300);

function commitThought(raw: string, remember = true) {
  const text = raw.trim();
  if (!text) return;

  if (remember) lastCommittedText = text;

  console.log("COMMIT:", text);

  try {
    enqueueTokens(text);
  } catch (err) {
    console.error("enqueueTokens failed:", err);
  }
}

thoughtInput.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  if (e.repeat) return; // ✅ 防止按住 Enter 连发（你想连发可以删掉这行）

  e.preventDefault();

  const current = thoughtInput.value.trim();

  if (current) {
    commitThought(current, true);
    thoughtInput.value = "";
  } else if (lastCommittedText) {
    commitThought(lastCommittedText, false); // ✅ 不覆盖 lastCommittedText
  }
});

clearBtn.addEventListener("click", () => {
  clearAll();
  thoughtInput.value = "";
  thoughtInput.focus();
});