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
  theta: number;        // angle
  lane: number;         // ring index
  radiusOffset: number; // pushed out by brush, decays back to 0
  omegaOffset: number;  // temporary angular velocity offset, decays back to 0
  omegaBase: number;    // base angular speed (rad/s)
  bornAt: number;
};

let nextId = 1;
const particles: Particle[] = [];

const ORBIT = {
  laneGap: 14,
  laneCapacity: 40,
  maxParticles: 600,

  // brush
  influenceRadius: 90,
  repelStrength: 140,   // px/s^2-ish (we apply per frame)
  swirlStrength: 3.2,   // rad/s
  // decay (seconds)
  tauRadius: 1.2,
  tauOmega: 0.9,

  // ellipse shape
  ellipseYScale: 0.72,
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

    particles.push({
      id: nextId++,
      token: t,
      theta: Math.random() * Math.PI * 2,
      lane,
      radiusOffset: 0,
      omegaOffset: 0,
      omegaBase: 0.9 + lane * 0.10,
      bornAt: now,
    });
  }

  // cap list
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
let headX = window.innerWidth * 0.5;
let headY = window.innerHeight * 0.45;
let faceWidthPx = 220;

if (faceRes.faceLandmarks?.length) {
  const lm = faceRes.faceLandmarks[0];

  // center
  const center = getFaceCenter(lm);
  const c = mapNormToScreen(center.x, center.y);
  headX = c.x;
  headY = c.y;

  // face width from bbox (normalized -> px)
  let minX = 1, maxX = 0;
  for (const p of lm) { minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x); }

  // cover scale (same as mapNormToScreen internal logic)
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const sw = window.innerWidth;
  const sh = window.innerHeight;
  const scale = Math.max(sw / vw, sh / vh);

  faceWidthPx = (maxX - minX) * vw * scale;
  faceWidthPx = Math.max(160, Math.min(420, faceWidthPx)); // clamp
}

// --- get index fingertip (screen) ---
let finger: { x: number; y: number } | null = null;
if (handRes.landmarks?.length) {
  const tip = handRes.landmarks[0][8];
  finger = mapNormToScreen(tip.x, tip.y);
}

// --- update + draw orbit particles ---
const dt = 1 / 60; // 简化：先固定帧步长，后续我们再用真实 dt
const baseR = faceWidthPx * 0.55;

// 先算每个粒子的位置与 depth，再排序绘制
const drawable = particles.map(p => {
  const laneR = baseR + p.lane * ORBIT.laneGap + p.radiusOffset;
  const a = laneR;
  const b = laneR * ORBIT.ellipseYScale;

  const x = headX + a * Math.cos(p.theta);
  const y = headY + b * Math.sin(p.theta);

  const depth = Math.sin(p.theta); // -1 back, +1 front
  return { p, x, y, depth, a, b };
});

// Brush interaction (repel + slight swirl)
if (finger) {
  for (const d of drawable) {
    const dx = d.x - finger.x;
    const dy = d.y - finger.y;
    const dist = Math.hypot(dx, dy);

    if (dist < ORBIT.influenceRadius) {
      const t = 1 - dist / ORBIT.influenceRadius;

      // push outward
      d.p.radiusOffset += ORBIT.repelStrength * t * dt;

      // swirl a bit (always same direction for now)
      d.p.omegaOffset += ORBIT.swirlStrength * t * dt;
    }
  }
}

// Update angles + decay back to stable orbit
for (const d of drawable) {
  const p = d.p;

  // decay offsets
  p.radiusOffset = expDecay(p.radiusOffset, dt, ORBIT.tauRadius);
  p.omegaOffset = expDecay(p.omegaOffset, dt, ORBIT.tauOmega);

  const omega = (p.omegaBase * settings.speedMultiplier) + p.omegaOffset;
  p.theta += omega * dt;
}

// Draw (back -> front)
drawable.sort((a, b) => a.depth - b.depth);

for (const d of drawable) {
  const p = d.p;
  const t = (d.depth + 1) / 2; // 0..1
  const scale = 0.75 + 0.55 * t;
  const alpha = 0.15 + 0.85 * t;

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.font = `${Math.round(settings.fontSize * scale)}px system-ui`;
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(p.token, d.x, d.y);
  ctx.restore();
}

// 可选：暂时保留调试点（你确认OK后我们再关掉）
ctx.fillStyle = "rgba(0,255,0,0.9)";
drawDot(headX, headY, 6);
if (finger) {
  ctx.fillStyle = "rgba(255,0,0,0.9)";
  drawDot(finger.x, finger.y, 5);
}

  // 画头部中心点（绿色）
  if (faceRes.faceLandmarks?.length) {
    const center = getFaceCenter(faceRes.faceLandmarks[0]);

    const { x, y } = mapNormToScreen(center.x, center.y);

    ctx.fillStyle = "rgba(0,255,0,0.9)";
    drawDot(x, y, 10);
  }

  // 画食指指尖（红色，landmark 8）
  if (handRes.landmarks?.length) {
    for (const oneHand of handRes.landmarks) {
      const tip = oneHand[8];
      const { x, y } = mapNormToScreen(tip.x, tip.y);
    

      ctx.fillStyle = "rgba(255,0,0,0.9)";
      drawDot(x, y, 8);
    }
  }

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