const canvas = document.getElementById("stack-scene");
const context = canvas.getContext("2d");
const layers = [];

function resize() {
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.floor(window.innerWidth * scale);
  canvas.height = Math.floor(window.innerHeight * scale);
  canvas.style.width = `${window.innerWidth}px`;
  canvas.style.height = `${window.innerHeight}px`;
  context.setTransform(scale, 0, 0, scale, 0, 0);
}

function buildScene() {
  layers.length = 0;
  const columns = 7;
  const rows = 5;
  const width = window.innerWidth;
  const height = window.innerHeight;
  const startX = width * 0.52;
  const startY = height * 0.16;
  const gapX = width * 0.075;
  const gapY = height * 0.13;

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const jitter = Math.sin(row * 1.7 + column * 0.8) * 18;
      layers.push({
        x: startX + column * gapX + jitter,
        y: startY + row * gapY + Math.cos(column) * 12,
        row,
        column,
        phase: Math.random() * Math.PI * 2,
      });
    }
  }
}

function draw(time) {
  const width = window.innerWidth;
  const height = window.innerHeight;
  context.clearRect(0, 0, width, height);

  const gradient = context.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#0d1320");
  gradient.addColorStop(0.55, "#132036");
  gradient.addColorStop(1, "#07101c");
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);

  context.globalAlpha = 0.16;
  context.strokeStyle = "#7dd3fc";
  context.lineWidth = 1;
  for (let x = 0; x < width; x += 64) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }
  for (let y = 0; y < height; y += 64) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }

  context.globalAlpha = 0.55;
  for (let i = 0; i < layers.length; i += 1) {
    const node = layers[i];
    const pulse = Math.sin(time * 0.0018 + node.phase) * 0.5 + 0.5;
    const x = node.x + Math.sin(time * 0.00045 + node.row) * 8;
    const y = node.y + Math.cos(time * 0.00055 + node.column) * 8;

    if (node.column < 6) {
      const next = layers[i + 1];
      if (next && next.row === node.row) {
        context.strokeStyle = `rgba(125, 211, 252, ${0.18 + pulse * 0.18})`;
        context.lineWidth = 2;
        context.beginPath();
        context.moveTo(x, y);
        context.lineTo(next.x, next.y);
        context.stroke();
      }
    }

    if (node.row < 4) {
      const next = layers[i + 7];
      if (next) {
        context.strokeStyle = `rgba(45, 212, 191, ${0.1 + pulse * 0.12})`;
        context.lineWidth = 1.5;
        context.beginPath();
        context.moveTo(x, y);
        context.lineTo(next.x, next.y);
        context.stroke();
      }
    }

    context.fillStyle = ["#2dd4bf", "#38bdf8", "#f59e0b", "#a78bfa", "#fb7185"][node.row];
    context.globalAlpha = 0.62 + pulse * 0.28;
    context.beginPath();
    context.roundRect(x - 17, y - 17, 34, 34, 8);
    context.fill();
  }

  context.globalAlpha = 1;
  window.requestAnimationFrame(draw);
}

window.addEventListener("resize", () => {
  resize();
  buildScene();
});

resize();
buildScene();
window.requestAnimationFrame(draw);

