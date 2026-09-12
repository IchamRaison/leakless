import { useCallback, useMemo, useRef } from "react";
import * as THREE from "three";
import type { Visualization } from "../contracts";
import { FullscreenButton } from "./FullscreenButton";
import { SceneCanvas, type SceneBuilder } from "./SceneCanvas";
import { displayEnergy, signalGeometry } from "./signalGeometry";

export function TemporalSignalMap({
  data,
  paused,
}: {
  data: Visualization;
  paused: boolean;
}) {
  const rings = useMemo(() => signalGeometry(data), [data]);
  const scene = useRef<HTMLDivElement>(null);
  const build = useCallback<SceneBuilder>(
    (scene, camera) => {
      camera.position.set(3.6, 1.7, 5.4);
      const group = new THREE.Group();
      scene.add(group);
      for (const ring of rings) {
        for (let band = 0; band < 3; band++) {
          const points: THREE.Vector3[] = [];
          for (let step = 0; step <= 64; step++) {
            const phase = (step / 64) * Math.PI * 2;
            const x = Math.cos(phase) * ring.radius;
            const z = Math.sin(phase) * ring.radius * 0.66;
            points.push(
              new THREE.Vector3(
                x * Math.cos(ring.angle) - z * Math.sin(ring.angle),
                (ring.time / data.duration_seconds) * 3.2 -
                  1.6 +
                  (band - 1) * 0.024,
                x * Math.sin(ring.angle) + z * Math.cos(ring.angle),
              ),
            );
          }
          const curve = new THREE.CatmullRomCurve3(points);
          const geometry = new THREE.TubeGeometry(
            curve,
            64,
            0.0015 + 0.007 * displayEnergy(ring.bandsDb[band]),
            3,
            false,
          );
          // Neutral material. No model output is available; no class color is inferred.
          group.add(
            new THREE.Mesh(
              geometry,
              new THREE.MeshBasicMaterial({
                color: "#b7c8bf",
                transparent: true,
                opacity: 0.54,
              }),
            ),
          );
        }
      }
      const axis = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, -1.65, 0),
        new THREE.Vector3(0, 1.7, 0),
      ]);
      scene.add(
        new THREE.Line(
          axis,
          new THREE.LineBasicMaterial({
            color: "#445b50",
            transparent: true,
            opacity: 0.7,
          }),
        ),
      );
      return {};
    },
    [rings, data.duration_seconds],
  );
  if (!rings.length)
    return (
      <div className="map-pending">
        PENDING — valid spectral measurements required.
      </div>
    );
  return (
    <>
      <div className="temporal-scene" ref={scene}>
        <SceneCanvas
          build={build}
          paused={paused}
          label="Temporal Signal Map derived deterministically from the selected recording's measured spectrum"
        />
        <span className="time-direction">
          TIME ↑ <small>{data.duration_seconds.toFixed(2)} s</small>
        </span>
        <span className="map-measured">{rings.length} measured windows</span>
        <FullscreenButton
          target={scene}
          label="Temporal Signal Map"
          className="quiet-button map-fullscreen"
        />
      </div>
      <p className="map-legend">
        Each ring represents a short temporal window.
        <br />
        Shape changes reflect measured signal properties over time.
      </p>
      <details className="map-method">
        <summary>How measurements become geometry</summary>
        <p>
          Height = time in this clip. Radius = mean spectral power on a fixed
          −120…0 dB scale. Three strand widths = mean power in three equal
          frequency bands. Twist = spectral centroid / maximum frequency × π.
        </p>
        <p>
          The ellipse is a display shape. Rotation is a presentation movement.
          This is the display STFT, not the model’s attention or a water-flow
          simulation. Flatness / irregularity is not mapped.
        </p>
        <p>
          First window: {rings[0].time.toFixed(3)} s ·{" "}
          {rings[0].energyDb.toFixed(1)} dB mean spectral power ·{" "}
          {rings[0].centroidHz.toFixed(0)} Hz centroid.
        </p>
      </details>
    </>
  );
}
