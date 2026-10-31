import { useRef } from "react";
import type { CSSProperties } from "react";
import { seededMotes, usePointerTilt } from "../lib/motion";

const motes = seededMotes(5);

export function Ambient() {
  return (
    <div className="app-canvas" aria-hidden="true">
      <div className="canvas-blob canvas-blob-a" />
      <div className="canvas-blob canvas-blob-b" />
      {motes.map((mote) => (
        <span
          key={mote.id}
          className={`mote${mote.ring ? " is-ring" : ""}`}
          style={
            {
              left: mote.left,
              top: mote.top,
              width: mote.size,
              height: mote.size,
              "--dur": mote.duration,
              "--delay": mote.delay,
            } as CSSProperties
          }
        />
      ))}
    </div>
  );
}

/**
 * DataStack — Interactive 3D DORA Crystal & Data Orbital System.
 * True CSS 3D bipyramid / octahedron matching DORA's brand mark,
 * with luminous crystal facets, inner radiant core, rotating gimbal
 * data rings, and cursor-reactive parallax depth.
 */
export function DataStack() {
  const stage = useRef<HTMLDivElement | null>(null);
  usePointerTilt(stage, 16, 8);

  return (
    <div className="crystal-wrapper" aria-hidden="true">
      <div className="crystal-halo" />
      <div className="crystal-stage" ref={stage}>
        {/* Ground grid / holographic base */}
        <div className="crystal-grid-plane">
          <div className="grid-ring grid-ring-outer" />
          <div className="grid-ring grid-ring-inner" />
          <div className="grid-cross-x" />
          <div className="grid-cross-y" />
        </div>

        {/* 3D Scene */}
        <div className="crystal-scene">
          <div className="crystal-assembly">
            {/* 3D Crystal Octahedron (8 Facets) */}
            <div className="crystal-polyhedron">
              {/* Top 4 pyramid facets */}
              <div className="crystal-facet facet-top facet-t1" />
              <div className="crystal-facet facet-top facet-t2" />
              <div className="crystal-facet facet-top facet-t3" />
              <div className="crystal-facet facet-top facet-t4" />
              {/* Bottom 4 pyramid facets */}
              <div className="crystal-facet facet-bottom facet-b1" />
              <div className="crystal-facet facet-bottom facet-b2" />
              <div className="crystal-facet facet-bottom facet-b3" />
              <div className="crystal-facet facet-bottom facet-b4" />
              {/* Inner glowing energy core */}
              <div className="crystal-core" />
            </div>

            {/* Equatorial Data Orbit */}
            <div className="crystal-orbit orbit-equator">
              <div className="orbit-track" />
              <div className="orbit-node node-csv" title="Dataset CSV">
                <span>CSV</span>
              </div>
              <div className="orbit-node node-ml" title="Machine Learning">
                <span>ML</span>
              </div>
              <div className="orbit-node node-research" title="Research Papers">
                <span>RESEARCH</span>
              </div>
              <div className="orbit-node node-quality" title="Quality Diagnosis">
                <span>QUALITY</span>
              </div>
            </div>

            {/* Polar Gimbal Ring */}
            <div className="crystal-orbit orbit-polar">
              <div className="orbit-track polar-track" />
              <div className="orbit-pulse-dot dot-1" />
              <div className="orbit-pulse-dot dot-2" />
            </div>
          </div>
        </div>

        {/* Floating Contextual Badges */}
        <div className="badge-float badge-top-left">
          <span className="badge-pill">
            <i className="badge-dot dot-blue" />
            <b>PROFILING</b>
            <span>4 Pillars</span>
          </span>
        </div>
        <div className="badge-float badge-top-right">
          <span className="badge-pill">
            <i className="badge-dot dot-cyan" />
            <b>ML ESTIMATION</b>
            <span>Auto-detect</span>
          </span>
        </div>
        <div className="badge-float badge-bottom-center">
          <span className="badge-pill">
            <i className="badge-dot dot-emerald" />
            <b>5 ACADEMIC SOURCES</b>
            <span>OpenAlex · Crossref</span>
          </span>
        </div>
      </div>
    </div>
  );
}
