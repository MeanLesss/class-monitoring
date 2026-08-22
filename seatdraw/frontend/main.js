/* seatdraw component - draw / select / move / resize multiple rectangles. */

const lib = window.StreamlitComponentLib;
lib.setComponentReady();

const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");

const FILL = "rgba(255,255,0,0.15)";
const STROKE = "#00FFFF";
const SEL = "#FF00FF";

let image = new Image();
let rects = [];          // [{left,top,width,height}]
let mode = "draw";       // draw | select
let selected = -1;
let drag = null;         // {kind:'draw'|'move'|'resize', ...}
let displayScale = 1;    // natural px -> displayed px
let initialized = false;

function drawBox(r, color, isSel) {
  ctx.fillStyle = FILL;
  ctx.fillRect(r.left, r.top, r.width, r.height);
  ctx.strokeStyle = color;
  ctx.lineWidth = isSel ? 3 : 2;
  ctx.strokeRect(r.left, r.top, r.width, r.height);
  if (isSel) {
    const pts = [[r.left, r.top], [r.left + r.width, r.top],
                 [r.left, r.top + r.height], [r.left + r.width, r.top + r.height]];
    for (const [hx, hy] of pts) {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(hx - 4, hy - 4, 8, 8);
      ctx.strokeStyle = "#0000ff";
      ctx.lineWidth = 1;
      ctx.strokeRect(hx - 4, hy - 4, 8, 8);
    }
  }
}

function redraw() {
  const W = image.naturalWidth, H = image.naturalHeight;
  cv.width = W; cv.height = H;
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(image, 0, 0, W, H);
  rects.forEach((r, i) => drawBox(r, i === selected ? SEL : STROKE, i === selected));
}

function updateDisplayScale() {
  displayScale = image.naturalWidth / cv.getBoundingClientRect().width;
}

function resizeFrame() {
  lib.setFrameHeight(Math.ceil(cv.getBoundingClientRect().height + 60));
}

function onImageLoaded() {
  updateDisplayScale();
  redraw();
  resizeFrame();
}

function pos(e) {
  const r = cv.getBoundingClientRect();
  return { x: (e.clientX - r.left) * displayScale, y: (e.clientY - r.top) * displayScale };
}

function hitTest(p) {
  for (let i = rects.length - 1; i >= 0; i--) {
    const r = rects[i];
    if (p.x >= r.left && p.x <= r.left + r.width &&
        p.y >= r.top && p.y <= r.top + r.height) return i;
  }
  return -1;
}

function cornerAt(p) {
  if (selected < 0) return null;
  const r = rects[selected];
  const corners = [[r.left, r.top, "tl"], [r.left + r.width, r.top, "tr"],
                   [r.left, r.top + r.height, "bl"], [r.left + r.width, r.top + r.height, "br"]];
  for (const [cx, cy, k] of corners) {
    if (Math.abs(p.x - cx) <= 7 && Math.abs(p.y - cy) <= 7) return k;
  }
  return null;
}

function send() {
  lib.setComponentValue({ objects: rects.map(r => ({ left: Math.round(r.left), top: Math.round(r.top),
                                                     width: Math.round(r.width), height: Math.round(r.height) })) });
}

function clampRect(r, W, H) {
  if (r.left < 0) r.left = 0;
  if (r.top < 0) r.top = 0;
  if (r.left + r.width > W) r.width = W - r.left;
  if (r.top + r.height > H) r.height = H - r.top;
  if (r.width < 5) r.width = 5;
  if (r.height < 5) r.height = 5;
}

cv.addEventListener("mousedown", (e) => {
  const p = pos(e);
  if (mode === "draw") {
    drag = { kind: "draw", x: p.x, y: p.y };
    return;
  }
  // select mode
  if (selected >= 0) {
    const k = cornerAt(p);
    if (k) {
      const r = rects[selected];
      drag = { kind: "resize", k, sx: r.left, sy: r.top, sw: r.width, sh: r.height };
      return;
    }
  }
  const i = hitTest(p);
  if (i >= 0) {
    selected = i;
    drag = { kind: "move", idx: i, ox: p.x - rects[i].left, oy: p.y - rects[i].top };
  } else {
    selected = -1;
  }
  redraw();
});

cv.addEventListener("mousemove", (e) => {
  if (!drag) return;
  const p = pos(e);
  const W = image.naturalWidth, H = image.naturalHeight;
  if (drag.kind === "draw") {
    // live preview + remember current corner to commit on mouseup
    drag.cx = p.x; drag.cy = p.y;
    const x = Math.min(drag.x, p.x), y = Math.min(drag.y, p.y);
    const w = Math.abs(p.x - drag.x), h = Math.abs(p.y - drag.y);
    redraw();
    ctx.strokeStyle = STROKE; ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
  } else if (drag.kind === "move") {
    const r = rects[drag.idx];
    r.left = p.x - drag.ox; r.top = p.y - drag.oy;
    clampRect(r, W, H);
    redraw();
  } else if (drag.kind === "resize") {
    const r = rects[selected];
    const k = drag.k;
    if (k.includes("l")) { const nx = Math.min(p.x, drag.sx + drag.sw - 5); r.width = drag.sw + (drag.sx - nx); r.left = nx; }
    if (k.includes("t")) { const ny = Math.min(p.y, drag.sy + drag.sh - 5); r.height = drag.sh + (drag.sy - ny); r.top = ny; }
    if (k.includes("r")) r.width = Math.max(5, p.x - r.left);
    if (k.includes("b")) r.height = Math.max(5, p.y - r.top);
    clampRect(r, W, H);
    redraw();
  }
});

cv.addEventListener("mouseup", (e) => {
  if (!drag) return;
  if (drag.kind === "draw") {
    const p = pos(e);
    const x = Math.min(drag.x, p.x), y = Math.min(drag.y, p.y);
    const w = Math.abs(p.x - drag.x), h = Math.abs(p.y - drag.y);
    if (w > 4 && h > 4) {
      const r = { left: x, top: y, width: w, height: h };
      clampRect(r, image.naturalWidth, image.naturalHeight);
      rects.push(r);
      selected = rects.length - 1;
    }
  }
  drag = null;
  redraw();
  send();
});

cv.addEventListener("mouseleave", () => {
  if (drag) { drag = null; redraw(); }
});

document.getElementById("mode-draw").onclick = () => {
  mode = "draw"; selected = -1;
  document.getElementById("mode-draw").classList.add("active");
  document.getElementById("mode-select").classList.remove("active");
  redraw();
};
document.getElementById("mode-select").onclick = () => {
  mode = "select";
  document.getElementById("mode-draw").classList.remove("active");
  document.getElementById("mode-select").classList.add("active");
  redraw();
};
document.getElementById("del").onclick = () => {
  if (selected >= 0) { rects.splice(selected, 1); selected = -1; redraw(); send(); }
};
document.getElementById("clear").onclick = () => {
  rects = []; selected = -1; redraw(); send();
};

function onRender(event) {
  const props = event.detail.args;
  if (props.image_url && props.image_url !== image.src) {
    image.onload = onImageLoaded;
    image.src = props.image_url;
  }
  if (!initialized && props.rects && props.rects.length) {
    rects = props.rects.map(r => ({ left: r.left, top: r.top, width: r.width, height: r.height }));
    initialized = true;
    redraw();
  }
}
lib.onRenderEvent(onRender);

// keep the component height in sync when the column width changes
if (window.ResizeObserver) {
  new ResizeObserver(() => {
    if (image.naturalWidth) {
      updateDisplayScale();
      resizeFrame();
    }
  }).observe(cv);
}
