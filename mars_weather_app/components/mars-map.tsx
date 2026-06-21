"use client"

import Image from "next/image"
import { useCallback, useEffect, useRef, useState } from "react"
import {
  layerValueAt,
  type ForecastFrame,
  type ForecastManifest,
} from "@/lib/forecast-file"
import {
  LAYERS,
  layerValue,
  normalize,
  sampleColor,
  type Layer,
} from "@/lib/mars-data"

// Low-resolution overlay buffer; the browser smoothly upscales it via CSS.
const GRID_W = 360
const GRID_H = 180
const MAP_ASPECT = 2
const CONTOUR_LEVELS = [0.18, 0.34, 0.5, 0.66, 0.82]

export interface GeoPoint {
  lon: number // -180..180
  lat: number // -90..90
}

interface MarsMapProps {
  layer: Layer
  leadHours: number
  selected: GeoPoint
  forecast?: {
    manifest: ForecastManifest
    frame: ForecastFrame
  } | null
  onSelect: (point: GeoPoint) => void
}

function contourIntersections(
  x: number,
  y: number,
  v00: number,
  v10: number,
  v11: number,
  v01: number,
  level: number,
): [number, number][] {
  const points: [number, number][] = []
  const add = (a: number, b: number, ax: number, ay: number, bx: number, by: number) => {
    if ((a < level && b >= level) || (b < level && a >= level)) {
      const t = (level - a) / (b - a || 1)
      points.push([ax + (bx - ax) * t, ay + (by - ay) * t])
    }
  }

  add(v00, v10, x, y, x + 1, y)
  add(v10, v11, x + 1, y, x + 1, y + 1)
  add(v01, v11, x, y + 1, x + 1, y + 1)
  add(v00, v01, x, y, x, y + 1)
  return points
}

function drawContours(
  ctx: CanvasRenderingContext2D,
  values: Float32Array,
  width: number,
  height: number,
) {
  const sx = width / (GRID_W - 1)
  const sy = height / (GRID_H - 1)

  ctx.save()
  ctx.globalCompositeOperation = "source-over"
  ctx.lineCap = "round"
  ctx.lineJoin = "round"

  for (const level of CONTOUR_LEVELS) {
    ctx.beginPath()
    for (let y = 0; y < GRID_H - 1; y++) {
      for (let x = 0; x < GRID_W - 1; x++) {
        const i = y * GRID_W + x
        const v00 = values[i]
        const v10 = values[i + 1]
        const v01 = values[i + GRID_W]
        const v11 = values[i + GRID_W + 1]
        const points = contourIntersections(x, y, v00, v10, v11, v01, level)

        if (points.length === 2) {
          ctx.moveTo(points[0][0] * sx, points[0][1] * sy)
          ctx.lineTo(points[1][0] * sx, points[1][1] * sy)
        } else if (points.length === 4) {
          ctx.moveTo(points[0][0] * sx, points[0][1] * sy)
          ctx.lineTo(points[1][0] * sx, points[1][1] * sy)
          ctx.moveTo(points[2][0] * sx, points[2][1] * sy)
          ctx.lineTo(points[3][0] * sx, points[3][1] * sy)
        }
      }
    }

    ctx.lineWidth = 2.6
    ctx.strokeStyle = "rgba(0, 0, 0, 0.38)"
    ctx.stroke()
    ctx.lineWidth = 1.15
    ctx.strokeStyle = "rgba(255, 255, 255, 0.74)"
    ctx.stroke()
  }

  ctx.restore()
}

export function MarsMap({ layer, leadHours, selected, forecast, onSelect }: MarsMapProps) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const fillCanvasRef = useRef<HTMLCanvasElement>(null)
  const contourCanvasRef = useRef<HTMLCanvasElement>(null)
  const hasInitialPanRef = useRef(false)
  const dragRef = useRef<{
    pointerId: number
    startX: number
    startPan: number
    moved: boolean
  } | null>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  const [panX, setPanX] = useState(0)

  const mapWidth = size.height * MAP_ASPECT
  const maxPan = 0
  const minPan = Math.min(0, size.width - mapWidth)

  const clampPan = useCallback(
    (value: number) => Math.max(minPan, Math.min(maxPan, value)),
    [minPan, maxPan],
  )

  useEffect(() => {
    const node = viewportRef.current
    if (!node) return

    const updateSize = () => {
      const rect = node.getBoundingClientRect()
      const nextSize = { width: rect.width, height: rect.height }
      setSize(nextSize)
      if (!hasInitialPanRef.current) {
        hasInitialPanRef.current = true
        setPanX(Math.min(0, (nextSize.width - nextSize.height * MAP_ASPECT) / 2))
      }
    }
    updateSize()

    const observer = new ResizeObserver(updateSize)
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const fillCanvas = fillCanvasRef.current
    const contourCanvas = contourCanvasRef.current
    if (!fillCanvas || !contourCanvas || !mapWidth || !size.height) return

    const fillCtx = fillCanvas.getContext("2d")
    const contourCtx = contourCanvas.getContext("2d")
    if (!fillCtx || !contourCtx) return

    const meta = LAYERS[layer]
    const img = fillCtx.createImageData(GRID_W, GRID_H)
    const data = img.data
    const normalizedValues = new Float32Array(GRID_W * GRID_H)

    for (let y = 0; y < GRID_H; y++) {
      const lat = 90 - (y / (GRID_H - 1)) * 180
      for (let x = 0; x < GRID_W; x++) {
        const lon = (x / (GRID_W - 1)) * 360 - 180
        const value = forecast
          ? layerValueAt(forecast.manifest, forecast.frame, layer, lon, lat)
          : layerValue(layer, lon, lat, leadHours)
        const n = normalize(meta, value)
        normalizedValues[y * GRID_W + x] = n
        const [r, g, b] = sampleColor(meta, n)
        const i = (y * GRID_W + x) * 4
        data[i] = r
        data[i + 1] = g
        data[i + 2] = b
        // Dust fades out where there is little of it; others use a flat wash.
        data[i + 3] = meta.alphaByValue ? Math.round(95 + n * 160) : 200
      }
    }
    fillCtx.putImageData(img, 0, 0)

    const ratio = window.devicePixelRatio || 1
    const contourWidth = Math.max(1, Math.round(mapWidth * ratio))
    const contourHeight = Math.max(1, Math.round(size.height * ratio))
    if (contourCanvas.width !== contourWidth) contourCanvas.width = contourWidth
    if (contourCanvas.height !== contourHeight) contourCanvas.height = contourHeight
    contourCtx.setTransform(ratio, 0, 0, ratio, 0, 0)
    contourCtx.clearRect(0, 0, mapWidth, size.height)
    drawContours(contourCtx, normalizedValues, mapWidth, size.height)
  }, [forecast, layer, leadHours, mapWidth, size.height])

  function selectAt(clientX: number, clientY: number) {
    const rect = viewportRef.current?.getBoundingClientRect()
    if (!rect || !mapWidth || !size.height) return
    const fx = (clientX - rect.left - clampPan(panX)) / mapWidth
    const fy = (clientY - rect.top) / size.height
    const boundedFx = Math.max(0, Math.min(1, fx))
    const boundedFy = Math.max(0, Math.min(1, fy))
    onSelect({
      lon: Math.round((boundedFx * 360 - 180) * 10) / 10,
      lat: Math.round((90 - boundedFy * 180) * 10) / 10,
    })
  }

  function handlePointerDown(e: React.PointerEvent<HTMLDivElement>) {
    e.currentTarget.setPointerCapture(e.pointerId)
    dragRef.current = {
      pointerId: e.pointerId,
      startX: e.clientX,
      startPan: panX,
      moved: false,
    }
  }

  function handlePointerMove(e: React.PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current
    if (!drag || drag.pointerId !== e.pointerId) return
    const dx = e.clientX - drag.startX
    if (Math.abs(dx) > 4) drag.moved = true
    setPanX(clampPan(drag.startPan + dx))
  }

  function handlePointerUp(e: React.PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current
    if (!drag || drag.pointerId !== e.pointerId) return
    dragRef.current = null
    e.currentTarget.releasePointerCapture(e.pointerId)
    if (!drag.moved) selectAt(e.clientX, e.clientY)
  }

  const markerLeft = clampPan(panX) + ((selected.lon + 180) / 360) * mapWidth
  const markerTop = ((90 - selected.lat) / 180) * size.height

  return (
    <div
      ref={viewportRef}
      className="absolute inset-0 touch-none cursor-grab overflow-hidden active:cursor-grabbing"
      role="button"
      tabIndex={0}
      aria-label="Mars map. Drag horizontally to pan. Click to select a forecast location."
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={() => {
        dragRef.current = null
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") onSelect({ lon: 0, lat: 0 })
      }}
    >
      <div
        className="absolute top-0 h-full"
        style={{
          width: `${mapWidth || 0}px`,
          transform: `translate3d(${clampPan(panX)}px, 0, 0)`,
        }}
      >
        <Image
          src="/mars-map.webp"
          alt="Global surface map of Mars"
          fill
          priority
          sizes="200vh"
          className="absolute inset-0 h-full w-full object-fill select-none"
          draggable={false}
        />

        <canvas
          ref={fillCanvasRef}
          width={GRID_W}
          height={GRID_H}
          className="pointer-events-none absolute inset-0 h-full w-full object-fill mix-blend-screen"
          style={{ opacity: layer === "dustcol" ? 0.82 : 0.68 }}
          aria-hidden="true"
        />

        <canvas
          ref={contourCanvasRef}
          className="pointer-events-none absolute inset-0 h-full w-full"
          aria-hidden="true"
        />
      </div>

      {/* Readability vignette */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(120% 80% at 50% 40%, transparent 40%, rgba(10,6,4,0.55) 100%), linear-gradient(to bottom, rgba(10,6,4,0.45) 0%, transparent 22%, transparent 60%, rgba(10,6,4,0.78) 100%)",
        }}
        aria-hidden="true"
      />

      {/* Selected location marker */}
      <div
        className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2"
        style={{ left: `${markerLeft}px`, top: `${markerTop}px` }}
        aria-hidden="true"
      >
        <span className="block h-3 w-3 rounded-full border-2 border-white bg-white/35 shadow-[0_0_12px_3px] shadow-white/80" />
        <span className="absolute left-1/2 top-1/2 h-7 w-7 -translate-x-1/2 -translate-y-1/2 animate-ping rounded-full border border-white/70" />
      </div>
    </div>
  )
}
