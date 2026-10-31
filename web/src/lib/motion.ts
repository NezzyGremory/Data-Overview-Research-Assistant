import { useEffect, useState } from "react";
import type { RefObject } from "react";

const EASE_OUT_CUBIC = (t: number) => 1 - Math.pow(1 - t, 3);

export function prefersReducedMotion(): boolean {
  if (typeof window === "undefined" || !window.matchMedia) return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Counts from 0 to target with a cubic ease-out, cancelled on unmount. */
export function useCountUp(target: number, duration = 900): number {
  const [value, setValue] = useState(() => (prefersReducedMotion() ? target : 0));

  useEffect(() => {
    if (!Number.isFinite(target)) {
      setValue(0);
      return;
    }
    if (prefersReducedMotion() || target === 0) {
      setValue(target);
      return;
    }
    let frame = 0;
    const started = performance.now();
    const tick = (now: number) => {
      const progress = Math.min(1, (now - started) / duration);
      setValue(target * EASE_OUT_CUBIC(progress));
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, duration]);

  return value;
}

/**
 * Tilts one element toward the pointer. The rAF loop parks itself as soon as
 * the motion settles, so an idle page costs nothing.
 */
export function usePointerTilt(ref: RefObject<HTMLElement | null>, depth = 22, rotation = 6) {
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    if (prefersReducedMotion()) return;
    if (!window.matchMedia("(pointer: fine)").matches) return;

    let targetX = 0;
    let targetY = 0;
    let currentX = 0;
    let currentY = 0;
    let frame = 0;
    let running = false;

    const apply = () => {
      element.style.transform =
        `translate3d(${(currentX * depth).toFixed(2)}px, ${(currentY * depth).toFixed(2)}px, 0)` +
        ` rotateX(${(-currentY * rotation).toFixed(2)}deg) rotateY(${(currentX * rotation).toFixed(2)}deg)`;
    };

    const loop = () => {
      currentX += (targetX - currentX) * 0.09;
      currentY += (targetY - currentY) * 0.09;
      apply();
      if (Math.abs(targetX - currentX) < 0.0015 && Math.abs(targetY - currentY) < 0.0015) {
        currentX = targetX;
        currentY = targetY;
        apply();
        running = false;
        return;
      }
      frame = requestAnimationFrame(loop);
    };

    const start = () => {
      if (running) return;
      running = true;
      frame = requestAnimationFrame(loop);
    };

    const onMove = (event: PointerEvent) => {
      targetX = (event.clientX / window.innerWidth - 0.5) * 2;
      targetY = (event.clientY / window.innerHeight - 0.5) * 2;
      start();
    };

    window.addEventListener("pointermove", onMove, { passive: true });
    return () => {
      window.removeEventListener("pointermove", onMove);
      cancelAnimationFrame(frame);
    };
  }, [ref, depth, rotation]);
}

/** Small deterministic PRNG so ambient motes keep their layout between renders. */
export function seededMotes(count: number, seed = 7) {
  let state = seed;
  const next = () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
  return Array.from({ length: count }, (_, index) => {
    const size = 8 + next() * 18;
    return {
      id: index,
      left: `${next() * 90 + 4}%`,
      top: `${next() * 80 + 8}%`,
      size,
      ring: false,
      duration: `${12 + next() * 14}s`,
      delay: `${next() * -16}s`,
    };
  });
}

/** Reveals an SVG ring once the element has mounted. */
export function useRingSweep(score: number, radius: number) {
  const circumference = 2 * Math.PI * radius;
  const [offset, setOffset] = useState(circumference);

  useEffect(() => {
    const target = circumference - (Math.max(0, Math.min(100, score)) / 100) * circumference;
    if (prefersReducedMotion()) {
      setOffset(target);
      return;
    }
    const frame = requestAnimationFrame(() => setOffset(target));
    return () => cancelAnimationFrame(frame);
  }, [circumference, score]);

  return { circumference, offset };
}

/** Fades in `[data-reveal]` sections as they enter the viewport. */
export function useScrollReveal(dependency?: unknown) {
  useEffect(() => {
    const nodes = Array.from(document.querySelectorAll<HTMLElement>("[data-reveal]"));
    if (!nodes.length) return;
    if (prefersReducedMotion() || typeof IntersectionObserver === "undefined") {
      nodes.forEach((node) => node.classList.add("is-visible"));
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      },
      { rootMargin: "0px 0px -6% 0px", threshold: 0.06 },
    );
    for (const node of nodes) {
      if (node.getBoundingClientRect().top < window.innerHeight * 0.95) {
        node.classList.add("is-visible");
      } else {
        observer.observe(node);
      }
    }
    return () => observer.disconnect();
  }, [dependency]);
}
