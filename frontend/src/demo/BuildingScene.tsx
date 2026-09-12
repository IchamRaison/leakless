import { useCallback, useEffect, useRef } from "react";
import * as THREE from "three";
import { SceneCanvas, type SceneBuilder } from "./SceneCanvas";

export const measurementPoints = ["N1", "N2", "N3", "N4"] as const;
export type Measurement = (typeof measurementPoints)[number];

export function BuildingScene({
  selected,
  onSelect,
  paused,
}: {
  selected: Measurement | null;
  onSelect: (id: Measurement) => void;
  paused: boolean;
}) {
  const selection = useRef(selected);
  const labels = useRef<(HTMLButtonElement | null)[]>([]);
  useEffect(() => {
    selection.current = selected;
  }, [selected]);
  const build = useCallback<SceneBuilder>((scene, camera) => {
    camera.position.set(6.6, 3.5, 7.9);
    scene.add(new THREE.AmbientLight(0xc9d3dc, 1.8));
    const light = new THREE.DirectionalLight(0xd8e7df, 3);
    light.position.set(4, 7, 6);
    scene.add(light);
    const group = new THREE.Group();
    scene.add(group);
    const frame: number[] = [];
    const windows: number[] = [];
    const hx = 1.45,
      hz = 1.15,
      bottom = -2.4;
    const corners = [
      [-hx, -hz],
      [hx, -hz],
      [hx, hz],
      [-hx, hz],
    ];
    for (const [x, z] of corners) frame.push(x, bottom, z, x, 2.6, z);
    for (let floor = 0; floor <= 5; floor++) {
      const y = bottom + floor;
      corners.forEach(([x, z], i) =>
        frame.push(
          x,
          y,
          z,
          corners[(i + 1) % 4][0],
          y,
          corners[(i + 1) % 4][1],
        ),
      );
      if (floor === 5) continue;
      for (const z of [-hz, hz])
        for (let c = 0; c < 4; c++) {
          const x = -hx + 0.15 + c * 0.7;
          windows.push(
            x,
            y + 0.18,
            z,
            x,
            y + 0.82,
            z,
            x,
            y + 0.82,
            z,
            x + 0.45,
            y + 0.82,
            z,
            x + 0.45,
            y + 0.82,
            z,
            x + 0.45,
            y + 0.18,
            z,
          );
        }
      for (const x of [-hx, hx])
        for (let c = 0; c < 3; c++) {
          const z = -hz + 0.16 + c * 0.75;
          windows.push(
            x,
            y + 0.18,
            z,
            x,
            y + 0.82,
            z,
            x,
            y + 0.82,
            z,
            x,
            y + 0.82,
            z + 0.45,
          );
        }
    }
    function lines(values: number[], color: string, opacity: number) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(values, 3),
      );
      group.add(
        new THREE.LineSegments(
          geometry,
          new THREE.LineBasicMaterial({ color, transparent: true, opacity }),
        ),
      );
    }
    lines(frame, "#aebfc3", 0.7);
    lines(windows, "#819496", 0.28);
    const grid = new THREE.GridHelper(18, 30, "#32483f", "#22352e");
    grid.position.y = bottom;
    scene.add(grid);
    const tube = (start: number[], end: number[]) => {
      const a = new THREE.Vector3(...start),
        b = new THREE.Vector3(...end),
        direction = b.clone().sub(a);
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.035, 0.035, direction.length(), 8),
        new THREE.MeshStandardMaterial({
          color: "#617a70",
          metalness: 0.55,
          roughness: 0.5,
        }),
      );
      mesh.position.copy(a.add(b).multiplyScalar(0.5));
      mesh.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        direction.normalize(),
      );
      group.add(mesh);
    };
    tube([0.22, bottom, 0], [0.22, 2.4, 0]);
    tube([0.22, bottom, 0], [2.5, bottom, 2]);
    const points = measurementPoints.map((id, i) => {
      const y = bottom + i + 0.75;
      const x = i % 2 ? 0.95 : -0.8;
      tube([0.22, y, 0], [x, y, 0.75]);
      const position = new THREE.Vector3(x, y, 0.75);
      const material = new THREE.MeshBasicMaterial({ color: "#81948b" });
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.085, 16, 12),
        material,
      );
      marker.position.copy(position);
      group.add(marker);
      return { id, position, marker, material };
    });
    return {
      frame: () => {
        camera.updateMatrixWorld();
        points.forEach(({ id, position, marker, material }, i) => {
          const active = selection.current === id;
          material.color.set(active ? "#b9f5b2" : "#8dada0");
          marker.scale.setScalar(active ? 1.6 : 1);
          const projected = position.clone().project(camera);
          const button = labels.current[i];
          if (button) {
            button.style.left = `${(projected.x + 1) * 50}%`;
            button.style.top = `${(1 - projected.y) * 50}%`;
          }
        });
      },
    };
  }, []);
  return (
    <div className="building-scene">
      <SceneCanvas
        build={build}
        paused={paused}
        label="Illustrative wireframe building with four selectable measurement points"
      />
      <div className="building-markers">
        {measurementPoints.map((id, i) => (
          <button
            key={id}
            ref={(node) => {
              labels.current[i] = node;
            }}
            className={`measurement-marker ${selected === id ? "is-selected" : ""}`}
            aria-label={`Select measurement ${id}`}
            aria-pressed={selected === id}
            onClick={() => onSelect(id)}
          >
            <span />
            {id}
          </button>
        ))}
      </div>
    </div>
  );
}
