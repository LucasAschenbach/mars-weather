// Mock OpenMARS Aurora output.
//
// The trained model emits OpenMARS fields every 2 Martian hours:
// surface variables ps, tsurf, co2ice, dustcol and atmospheric variables
// u, v, temp on 35 sigma levels. These deterministic functions keep the UI
// interactive until trained rollouts are available.

export type SurfaceVariable = "ps" | "tsurf" | "co2ice" | "dustcol"
export type AtmosVariable = "u" | "v" | "temp"
export type ModelVariable = SurfaceVariable | AtmosVariable
export type Layer = "tsurf" | "ps" | "dustcol"

export const SURFACE_VARIABLES: SurfaceVariable[] = ["ps", "tsurf", "co2ice", "dustcol"]
export const ATMOS_VARIABLES: AtmosVariable[] = ["u", "v", "temp"]
export const MODEL_VARIABLES: ModelVariable[] = [...SURFACE_VARIABLES, ...ATMOS_VARIABLES]
export const ATMOS_LEVEL_COUNT = 35

// One Martian sol is ~24.66 hours.
const SOL_HOURS = 24.66

// Smooth, seamless-in-longitude pseudo-random field in roughly [-1, 1].
function noise(lon: number, lat: number, t: number, seed: number): number {
  const lo = (lon * Math.PI) / 180
  const la = (lat * Math.PI) / 180
  let v = Math.sin(lo * 2 + seed) * Math.cos(la * 2 - t * 0.025)
  v += 0.5 * Math.sin(lo * 4 - t * 0.05 + seed * 1.3) * Math.cos(la * 3 + seed)
  v += 0.25 * Math.sin(lo * 8 + seed * 2.1) * Math.sin(la * 6 - t * 0.1)
  return v / 1.75
}

function subsolarLon(t: number): number {
  return (270 - (t / SOL_HOURS) * 360) % 360
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

/** OpenMARS tsurf in Kelvin. */
export function tsurf(lon: number, lat: number, t: number): number {
  const la = (lat * Math.PI) / 180
  const sun = subsolarLon(t)
  const dayNight = Math.cos(((lon - sun) * Math.PI) / 180)
  const equatorWarmth = Math.cos(la)
  const insolation = equatorWarmth * (0.45 + 0.55 * Math.max(0, dayNight))
  const celsius = -123 + 128 * insolation + 7 * noise(lon, lat, t, 1) - dustcol(lon, lat, t) * 6
  return clamp(celsius + 273.15, 143, 285)
}

/** OpenMARS ps in Pascals. */
export function ps(lon: number, lat: number, t: number): number {
  const la = (lat * Math.PI) / 180
  return clamp(610 + 110 * noise(lon, lat, t, 5) - 70 * (1 - Math.cos(la)), 400, 900)
}

/** OpenMARS dustcol column opacity. */
export function dustcol(lon: number, lat: number, t: number): number {
  const raw = 0.35 + 1.15 * noise(lon, lat, t, 9) + 0.45 * noise(lon, lat, t * 1.4, 3)
  return clamp(raw, 0, 2.6)
}

/** OpenMARS co2ice in kg/m^2. */
export function co2Ice(lon: number, lat: number, t: number): number {
  const polar = Math.pow(Math.abs(lat) / 90, 5)
  const hemisphere = lat < 0 ? 1 : 0.25
  const ice = polar * 950 * hemisphere + 120 * polar * noise(lon, lat, t, 7)
  return clamp(ice, 0, 1100)
}

function levelFactor(level: number): number {
  return clamp(level / (ATMOS_LEVEL_COUNT - 1), 0, 1)
}

/** OpenMARS atmospheric temp in Kelvin at a model sigma level index [0..34]. */
export function temp(lon: number, lat: number, t: number, level: number): number {
  const z = levelFactor(level)
  const v = tsurf(lon, lat, t) - z * 92 + 5 * noise(lon, lat, t, 11 + level * 0.1)
  return clamp(v, 115, 285)
}

/** OpenMARS u in m/s (+ is eastward) at a model sigma level. */
export function windU(lon: number, lat: number, t: number, level: number): number {
  const la = (lat * Math.PI) / 180
  const z = levelFactor(level)
  return (18 + z * 112) * Math.sin(la * 2) + (8 + z * 32) * noise(lon, lat, t, 13)
}

/** OpenMARS v in m/s (+ is northward) at a model sigma level. */
export function windV(lon: number, lat: number, t: number, level: number): number {
  const la = (lat * Math.PI) / 180
  const z = levelFactor(level)
  return (6 + z * 56) * Math.cos(la * 3) + (6 + z * 32) * noise(lon, lat, t, 17)
}

export function layerValue(layer: Layer, lon: number, lat: number, t: number): number {
  switch (layer) {
    case "tsurf":
      return tsurf(lon, lat, t)
    case "ps":
      return ps(lon, lat, t)
    case "dustcol":
      return dustcol(lon, lat, t)
  }
}

export interface LayerMeta {
  key: Layer
  label: string
  unit: string
  domain: [number, number]
  stops: [number, [number, number, number]][]
  alphaByValue?: boolean
}

export const LAYERS: Record<Layer, LayerMeta> = {
  tsurf: {
    key: "tsurf",
    label: "tsurf",
    unit: "K",
    domain: [145, 285],
    stops: [
      [0, [12, 24, 78]],
      [0.3, [22, 96, 168]],
      [0.5, [40, 168, 158]],
      [0.7, [232, 178, 64]],
      [0.88, [222, 96, 42]],
      [1, [176, 32, 30]],
    ],
  },
  ps: {
    key: "ps",
    label: "ps",
    unit: "Pa",
    domain: [450, 800],
    stops: [
      [0, [26, 54, 64]],
      [0.4, [34, 122, 124]],
      [0.7, [120, 176, 96]],
      [1, [228, 206, 96]],
    ],
  },
  dustcol: {
    key: "dustcol",
    label: "dustcol",
    unit: "opacity",
    domain: [0.02, 1.1],
    alphaByValue: true,
    stops: [
      [0, [92, 54, 32]],
      [0.28, [172, 86, 34]],
      [0.6, [232, 154, 54]],
      [1, [255, 226, 142]],
    ],
  },
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

export function sampleColor(meta: LayerMeta, n: number): [number, number, number] {
  const v = clamp(n, 0, 1)
  const { stops } = meta
  for (let i = 0; i < stops.length - 1; i++) {
    const [p0, c0] = stops[i]
    const [p1, c1] = stops[i + 1]
    if (v >= p0 && v <= p1) {
      const f = (v - p0) / (p1 - p0 || 1)
      return [
        Math.round(lerp(c0[0], c1[0], f)),
        Math.round(lerp(c0[1], c1[1], f)),
        Math.round(lerp(c0[2], c1[2], f)),
      ]
    }
  }
  return stops[stops.length - 1][1]
}

export function normalize(meta: LayerMeta, value: number): number {
  const [lo, hi] = meta.domain
  return clamp((value - lo) / (hi - lo), 0, 1)
}

export const FORECAST_BASE = new Date("2026-06-01T00:00:00Z")
export const MAX_LEAD_HOURS = 168
export const LEAD_STEP_HOURS = 2

export function forecastDate(leadHours: number): Date {
  return new Date(FORECAST_BASE.getTime() + leadHours * 3600 * 1000)
}

export function formatLead(leadHours: number): string {
  const days = Math.floor(leadHours / 24)
  const hrs = leadHours % 24
  if (days > 0 && hrs > 0) return `+${days}D ${hrs}H`
  if (days > 0) return `+${days} DAY${days > 1 ? "S" : ""}`
  return `+${hrs} HRS`
}
