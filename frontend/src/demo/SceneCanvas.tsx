import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export type SceneBuilder = (
  scene: THREE.Scene,
  camera: THREE.PerspectiveCamera,
) => {
  frame?: () => void;
  dispose?: () => void;
};

/** One Three.js lifecycle for both views; no second rendering framework. */
export function SceneCanvas({
  build,
  paused,
  label,
}: {
  build: SceneBuilder;
  paused: boolean;
  label: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const pause = useRef(paused);
  const [unavailable, setUnavailable] = useState(false);
  useEffect(() => {
    pause.current = paused;
  }, [paused]);
  useEffect(() => {
    const node = host.current;
    if (!node) return;
    let renderer: THREE.WebGLRenderer;
    // Probe before constructing Three's renderer to avoid an unhandled WebGL error.
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("webgl2", {
      alpha: true,
      antialias: true,
    });
    if (!context) {
      setUnavailable(true);
      return;
    }
    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        context,
        alpha: true,
        antialias: true,
      });
    } catch {
      setUnavailable(true);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    renderer.setClearAlpha(0);
    node.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
    const content = build(scene, camera);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan = false;
    controls.enableZoom = false;
    controls.autoRotateSpeed = 0.18;
    controls.rotateSpeed = 0.45;
    controls.minPolarAngle = Math.PI * 0.24;
    controls.maxPolarAngle = Math.PI * 0.65;
    const motion = matchMedia("(prefers-reduced-motion: reduce)");
    const resize = () => {
      const { width, height } = node.getBoundingClientRect();
      if (!width || !height) return;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(node);
    resize();
    let previous = 0;
    renderer.setAnimationLoop((now) => {
      const delta = Math.min((now - previous) / 1000, 0.05);
      previous = now;
      // Skip frames while the tab or the whole view is hidden (e.g. demo kept mounted behind #monitor).
      if (document.hidden || node.getClientRects().length === 0) return;
      controls.autoRotate = !pause.current && !motion.matches;
      controls.update(delta);
      content.frame?.();
      renderer.render(scene, camera);
    });
    const lost = (event: Event) => {
      event.preventDefault();
      setUnavailable(true);
    };
    canvas.addEventListener("webglcontextlost", lost);
    return () => {
      renderer.setAnimationLoop(null);
      observer.disconnect();
      controls.dispose();
      content.dispose?.();
      scene.traverse((object) => {
        if (
          object instanceof THREE.Mesh ||
          object instanceof THREE.LineSegments ||
          object instanceof THREE.Line
        ) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material)
            ? object.material
            : [object.material];
          materials.forEach((material) => material.dispose());
        }
      });
      canvas.removeEventListener("webglcontextlost", lost);
      renderer.dispose();
      renderer.forceContextLoss();
      canvas.remove();
    };
  }, [build]);
  return (
    <div className="scene-canvas" ref={host} role="img" aria-label={label}>
      {unavailable && (
        <div className="scene-fallback">
          3D unavailable on this device.
          <br />
          Use the measurement buttons and signal plots below.
        </div>
      )}
    </div>
  );
}
