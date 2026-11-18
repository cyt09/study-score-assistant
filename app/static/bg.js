/**
 * 背景层：用 Canvas 2D 画一层很淡的"成绩散点图"——
 * 一片稀疏的数据点，配一条虚线的走向趋势，模拟成绩单页脚的统计图。
 * 纯静态装饰：不跟随鼠标、不做视差、不跑动画帧，改完尺寸画一次就结束。
 */

const canvas = document.getElementById("bg-canvas");

if (canvas) {
  let lastWidth = 0;
  let lastHeight = 0;
  let timer = 0;

  paint();

  // 窗口变化时防抖重绘：Canvas 按 CSS 像素重算，避免拉伸模糊
  window.addEventListener("resize", () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(paint, 180);
  });

  function paint() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    if (width === lastWidth && height === lastHeight) return;
    lastWidth = width;
    lastHeight = height;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    drawTrend(ctx, width, height);
    drawPoints(ctx, width, height);
  }

  /* 一条很淡的上升趋势线 + 两侧虚线，像成绩单上的参考带 */
  function drawTrend(ctx, width, height) {
    const bottom = height * 0.86;
    const top = height * 0.30;
    const left = width * 0.04;
    const right = width * 0.96;

    ctx.save();
    ctx.lineWidth = 1.2;
    ctx.setLineDash([7, 9]);
    ctx.strokeStyle = "rgba(54, 96, 146, 0.2)";
    ctx.beginPath();
    ctx.moveTo(left, bottom);
    ctx.lineTo(right, top);
    ctx.stroke();

    // 上下各偏移一点，形成"趋势带"，比单线更像图表
    ctx.strokeStyle = "rgba(232, 162, 92, 0.13)";
    offset(ctx, left, bottom, right, top, -26, height);
    offset(ctx, left, bottom, right, top, 26, height);
    ctx.restore();
  }

  function offset(ctx, x1, y1, x2, y2, delta, height) {
    ctx.beginPath();
    ctx.moveTo(x1, clamp(y1 + delta, 0, height));
    ctx.lineTo(x2, clamp(y2 + delta, 0, height));
    ctx.stroke();
  }

  /* 稀疏散点：位置用固定种子生成，刷新页面不会跳动 */
  function drawPoints(ctx, width, height) {
    const rand = seeded(20261118);
    const count = Math.min(46, Math.round((width * height) / 26000));

    ctx.save();
    for (let i = 0; i < count; i += 1) {
      const x = rand() * width;
      // 点越靠右越高，跟趋势线一起构成"整体在往上走"的观感
      const base = height * (0.78 - (x / width) * 0.44);
      const y = base + (rand() - 0.5) * height * 0.16;
      const radius = 1.2 + rand() * 1.3;
      const warm = i % 5 === 0;

      ctx.beginPath();
      ctx.arc(x, clamp(y, 24, height - 24), radius, 0, Math.PI * 2);
      ctx.fillStyle = warm ? "rgba(232, 162, 92, 0.22)" : "rgba(54, 96, 146, 0.18)";
      ctx.fill();
    }
    ctx.restore();
  }

  /* 固定种子的伪随机，保证每次绘制位置一致 */
  function seeded(seed) {
    let state = seed >>> 0;
    return function next() {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 4294967296;
    };
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }
}
