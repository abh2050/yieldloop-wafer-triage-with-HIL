import { useEffect, useRef } from "react";

/** Matches tailwind.config.js `die` colours, and the WM811K encoding. */
export const DIE_COLOURS = {
  0: "#0f172a", // outside the wafer
  1: "#1e5f8c", // passing die
  2: "#f0663f", // failing die
} as const;

interface WaferMapProps {
  /** Flat row-major array of 0/1/2, exactly as stored and as the model saw it. */
  grid: number[] | Uint8Array;
  height: number;
  width: number;
  /** Rendered size in CSS pixels. */
  size?: number;
  className?: string;
  label?: string;
}

/**
 * Draws a wafer map from the real array on a canvas.
 *
 * Canvas rather than SVG or an <img>: a 64x64 grid is 4,096 elements, and the
 * label grid shows dozens at once. SVG would put a quarter of a million DOM
 * nodes on the page and destroy the sub-four-second decision target the whole
 * screen is built around.
 *
 * Nothing is interpolated. Each die is drawn as a rectangle of exact colour, so
 * what the reviewer judges is what the classifier consumed, not a smoothed
 * approximation of it.
 */
export function WaferMap({
  grid,
  height,
  width,
  size = 180,
  className = "",
  label,
}: WaferMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    // Render at device resolution so the die grid stays crisp on retina
    // displays; a blurred map is a map a reviewer cannot judge.
    const ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.imageSmoothingEnabled = false;

    const image = context.createImageData(width, height);
    for (let index = 0; index < width * height; index += 1) {
      const value = (grid[index] ?? 0) as 0 | 1 | 2;
      const colour = DIE_COLOURS[value] ?? DIE_COLOURS[0];
      const r = parseInt(colour.slice(1, 3), 16);
      const g = parseInt(colour.slice(3, 5), 16);
      const b = parseInt(colour.slice(5, 7), 16);
      const offset = index * 4;
      image.data[offset] = r;
      image.data[offset + 1] = g;
      image.data[offset + 2] = b;
      image.data[offset + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }, [grid, height, width]);

  return (
    <canvas
      ref={canvasRef}
      data-testid="wafer-map"
      data-wafer-label={label}
      aria-label={label ? `Wafer map for ${label}` : "Wafer map"}
      role="img"
      className={`rounded border border-slate-700 [image-rendering:pixelated] ${className}`}
      style={{ width: size, height: size }}
    />
  );
}

export function WaferMapLegend() {
  return (
    <div className="flex gap-4 text-xs text-slate-400">
      {[
        ["Outside wafer", DIE_COLOURS[0]],
        ["Passing die", DIE_COLOURS[1]],
        ["Failing die", DIE_COLOURS[2]],
      ].map(([name, colour]) => (
        <span key={name} className="flex items-center gap-1.5">
          <span
            className="inline-block h-3 w-3 rounded-sm border border-slate-600"
            style={{ background: colour }}
          />
          {name}
        </span>
      ))}
    </div>
  );
}
