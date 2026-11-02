/**
 * 背景舞台：Three.js 粒子场 + 漂浮几何体 + 鼠标视差。
 * 仅做氛围，不参与任何业务逻辑；WebGL 不可用时自动退回纯 CSS 渐变。
 */
import * as THREE from "./vendor/three.module.min.js";

const canvas = document.getElementById("bg-canvas");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

if (canvas) {
  try {
    boot();
  } catch (error) {
    canvas.style.display = "none";
  }
}

function dotTexture() {
  const size = 64;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.35, "rgba(255,255,255,.85)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2);
  ctx.fill();
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function boot() {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: true,
    powerPreference: "low-power",
  });
  renderer.setClearAlpha(0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(56, 1, 0.1, 120);
  camera.position.set(0, 0, 8);

  const sprite = dotTexture();
  const world = new THREE.Group();
  scene.add(world);

  /* 粒子场：靠近相机的一层偏大偏亮，远处一层细密铺底 */
  const layers = [
    makeDust({ count: 460, spread: [26, 15, 10], size: 0.06, opacity: 0.55, colors: ["#4f7cff", "#9a7cff"] }),
    makeDust({ count: 700, spread: [34, 20, 16], size: 0.032, opacity: 0.38, colors: ["#6ea8ff", "#3fd6e0"] }),
  ];
  layers.forEach((dots) => world.add(dots));

  /* 漂浮几何体：线框，低透明度，停在画面两侧避免与内容抢视线 */
  const shapes = [
    shape(new THREE.IcosahedronGeometry(1.55, 1), 0x4f7cff, 0.26, [-4.6, 1.15, -1.4], 0.16),
    shape(new THREE.TorusKnotGeometry(0.62, 0.18, 110, 14), 0x8b5cf6, 0.3, [4.5, -1.45, -0.8], -0.12),
    shape(new THREE.OctahedronGeometry(0.92), 0x22c1e0, 0.32, [3.6, 2.15, -3.2], 0.2),
    shape(new THREE.IcosahedronGeometry(0.58, 0), 0xe0a44f, 0.22, [-3.5, -2.3, -2.6], -0.24),
  ];
  shapes.forEach((m) => world.add(m));

  const pointer = new THREE.Vector2(0, 0);
  const target = new THREE.Vector2(0, 0);
  const clock = new THREE.Clock();
  let raf = 0;
  let running = false;

  function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function frame() {
    const t = clock.getElapsedTime();

    target.set(pointer.x * 0.5, pointer.y * 0.34);
    camera.position.x += (target.x - camera.position.x) * 0.045;
    camera.position.y += (target.y - camera.position.y) * 0.045;
    camera.lookAt(0, 0, 0);

    layers.forEach((dots, i) => {
      dots.rotation.y = t * (0.012 + i * 0.006);
      dots.rotation.x = Math.sin(t * 0.14 + i) * 0.05;
      dots.position.y = Math.sin(t * 0.22 + i * 1.6) * 0.22;
    });

    shapes.forEach((mesh, i) => {
      mesh.rotation.x = t * (0.05 + i * 0.015);
      mesh.rotation.y = t * (0.07 + i * 0.02);
      mesh.position.y = mesh.userData.baseY + Math.sin(t * 0.42 + i * 1.4) * 0.32;
      mesh.position.x = mesh.userData.baseX + Math.cos(t * 0.3 + i) * 0.16;
    });

    renderer.render(scene, camera);
    raf = requestAnimationFrame(frame);
  }

  function start() {
    if (running) return;
    running = true;
    raf = requestAnimationFrame(frame);
  }

  function stop() {
    running = false;
    cancelAnimationFrame(raf);
  }

  resize();
  window.addEventListener("resize", resize);
  window.addEventListener("pointermove", (event) => {
    pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
    pointer.y = -((event.clientY / window.innerHeight) * 2 - 1);
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop();
    else start();
  });

  // 首屏直接渲染一帧，避免淡入前出现空白
  renderer.render(scene, camera);

  if (reduceMotion) {
    camera.position.set(0, 0, 8);
    renderer.render(scene, camera);
  } else {
    start();
  }

  function makeDust({ count, spread, size, opacity, colors }) {
    const positions = new Float32Array(count * 3);
    const colorArray = new Float32Array(count * 3);
    const from = new THREE.Color(colors[0]);
    const to = new THREE.Color(colors[1]);
    const mixed = new THREE.Color();

    for (let i = 0; i < count; i += 1) {
      positions[i * 3] = (Math.random() - 0.5) * spread[0];
      positions[i * 3 + 1] = (Math.random() - 0.5) * spread[1];
      positions[i * 3 + 2] = (Math.random() - 0.5) * spread[2];
      mixed.copy(from).lerp(to, Math.random());
      colorArray[i * 3] = mixed.r;
      colorArray[i * 3 + 1] = mixed.g;
      colorArray[i * 3 + 2] = mixed.b;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colorArray, 3));

    const material = new THREE.PointsMaterial({
      size,
      map: sprite,
      transparent: true,
      opacity,
      depthWrite: false,
      vertexColors: true,
      sizeAttenuation: true,
    });

    return new THREE.Points(geometry, material);
  }

  function shape(geometry, color, opacity, position, speed) {
    const material = new THREE.MeshBasicMaterial({
      color,
      wireframe: true,
      transparent: true,
      opacity,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(position[0], position[1], position[2]);
    mesh.userData.baseX = position[0];
    mesh.userData.baseY = position[1];
    mesh.userData.speed = speed;
    return mesh;
  }
}
