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
  } catch (err) {
    console.warn("Camera permission denied or unavailable:", err);
    // Minimal non-intrusive hint in console for MVP; we'll add UI later.
  }
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
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

  // Simple HUD to confirm real-time control works
  ctx.font = `${settings.fontSize}px system-ui`;
  ctx.fillStyle = "rgba(255,255,255,0.85)";
  ctx.fillText(`fontSize: ${settings.fontSize}px`, 20, 40);
  ctx.fillText(`speed: ${settings.speedMultiplier.toFixed(2)}x`, 20, 40 + settings.fontSize);

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
  // TODO (next step): turn this text into orbit particles
  // e.g. enqueueTokens(text)

  thoughtInput.value = "";
}

thoughtInput.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;

  e.preventDefault();
  commitThought(thoughtInput.value);
});