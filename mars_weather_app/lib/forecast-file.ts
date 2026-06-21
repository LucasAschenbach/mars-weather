import {
  ATMOS_LEVEL_COUNT,
  type AtmosVariable,
  type Layer,
  type SurfaceVariable,
} from "@/lib/mars-data"

export const FORECAST_MANIFEST_PATH = "/forecasts/latest/manifest.json"

const SURFACE_ORDER: SurfaceVariable[] = ["ps", "tsurf", "co2ice", "dustcol"]
const ATMOS_ORDER: AtmosVariable[] = ["u", "v", "temp"]

export interface ForecastManifest {
  schema: "mars-weather-forecast-v1"
  source: string
  generatedAt?: string
  baseTime?: string
  stepHours: number
  grid: {
    lat: number[]
    lon: number[]
    lev: number[]
  }
  variables: Record<string, { unit: string; dims: string[]; encoding: ForecastEncoding }>
  frames: ForecastFrameRef[]
}

export interface ForecastEncoding {
  dtype: "int16"
  min: number
  scale: number
}

export interface ForecastFrameRef {
  leadHours: number
  path: string
}

export interface ForecastFrame {
  ref: ForecastFrameRef
  values: Int16Array
}

export interface ForecastPoint {
  ps: number
  tsurf: number
  co2ice: number
  dustcol: number
  u: number
  v: number
  temp: number
}

function surfaceSize(manifest: ForecastManifest): number {
  return manifest.grid.lat.length * manifest.grid.lon.length
}

function atmosSize(manifest: ForecastManifest): number {
  return ATMOS_LEVEL_COUNT * surfaceSize(manifest)
}

function nearestIndex(values: number[], target: number): number {
  let best = 0
  let bestDistance = Number.POSITIVE_INFINITY
  for (let i = 0; i < values.length; i++) {
    const distance = Math.abs(values[i] - target)
    if (distance < bestDistance) {
      best = i
      bestDistance = distance
    }
  }
  return best
}

function bracketIndex(values: number[], target: number): [number, number, number] {
  if (values.length <= 1) return [0, 0, 0]

  const ascending = values[0] < values[values.length - 1]
  const first = values[0]
  const last = values[values.length - 1]

  if ((ascending && target <= first) || (!ascending && target >= first)) return [0, 0, 0]
  if ((ascending && target >= last) || (!ascending && target <= last)) {
    const i = values.length - 1
    return [i, i, 0]
  }

  for (let i = 0; i < values.length - 1; i++) {
    const a = values[i]
    const b = values[i + 1]
    const contains = ascending
      ? target >= a && target <= b
      : target <= a && target >= b
    if (contains) {
      return [i, i + 1, (target - a) / (b - a || 1)]
    }
  }

  const best = nearestIndex(values, target)
  return [best, best, 0]
}

function normalizeLonForGrid(lon: number, gridLon: number[]): number {
  if (gridLon.length === 0) return lon
  const minLon = Math.min(...gridLon)
  const maxLon = Math.max(...gridLon)
  if (minLon >= 0 && lon < 0) return lon + 360
  if (maxLon <= 180 && lon > 180) return lon - 360
  return lon
}

function surfaceOffset(
  manifest: ForecastManifest,
  variable: SurfaceVariable,
  latIndex: number,
  lonIndex: number,
): number {
  const latCount = manifest.grid.lat.length
  const lonCount = manifest.grid.lon.length
  const variableIndex = SURFACE_ORDER.indexOf(variable)
  return variableIndex * latCount * lonCount + latIndex * lonCount + lonIndex
}

function atmosOffset(
  manifest: ForecastManifest,
  variable: AtmosVariable,
  level: number,
  latIndex: number,
  lonIndex: number,
): number {
  const latCount = manifest.grid.lat.length
  const lonCount = manifest.grid.lon.length
  const surfaceBlock = SURFACE_ORDER.length * surfaceSize(manifest)
  const variableIndex = ATMOS_ORDER.indexOf(variable)
  return (
    surfaceBlock +
    variableIndex * atmosSize(manifest) +
    level * latCount * lonCount +
    latIndex * lonCount +
    lonIndex
  )
}

function decodeValue(manifest: ForecastManifest, variable: SurfaceVariable | AtmosVariable, raw: number) {
  const encoding = manifest.variables[variable].encoding
  return (raw + 32767) * encoding.scale + encoding.min
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function bilerp(v00: number, v10: number, v01: number, v11: number, tx: number, ty: number): number {
  return lerp(lerp(v00, v10, tx), lerp(v01, v11, tx), ty)
}

function surfaceInterpolated(
  manifest: ForecastManifest,
  frame: ForecastFrame,
  variable: SurfaceVariable,
  lon: number,
  lat: number,
): number {
  const [lat0, lat1, ty] = bracketIndex(manifest.grid.lat, lat)
  const [lon0, lon1, tx] = bracketIndex(manifest.grid.lon, normalizeLonForGrid(lon, manifest.grid.lon))

  const v00 = decodeValue(manifest, variable, frame.values[surfaceOffset(manifest, variable, lat0, lon0)])
  const v10 = decodeValue(manifest, variable, frame.values[surfaceOffset(manifest, variable, lat0, lon1)])
  const v01 = decodeValue(manifest, variable, frame.values[surfaceOffset(manifest, variable, lat1, lon0)])
  const v11 = decodeValue(manifest, variable, frame.values[surfaceOffset(manifest, variable, lat1, lon1)])

  return bilerp(v00, v10, v01, v11, tx, ty)
}

function atmosInterpolated(
  manifest: ForecastManifest,
  frame: ForecastFrame,
  variable: AtmosVariable,
  lon: number,
  lat: number,
  level: number,
): number {
  const boundedLevel = Math.max(0, Math.min(ATMOS_LEVEL_COUNT - 1, level))
  const [lat0, lat1, ty] = bracketIndex(manifest.grid.lat, lat)
  const [lon0, lon1, tx] = bracketIndex(manifest.grid.lon, normalizeLonForGrid(lon, manifest.grid.lon))

  const v00 = decodeValue(
    manifest,
    variable,
    frame.values[atmosOffset(manifest, variable, boundedLevel, lat0, lon0)],
  )
  const v10 = decodeValue(
    manifest,
    variable,
    frame.values[atmosOffset(manifest, variable, boundedLevel, lat0, lon1)],
  )
  const v01 = decodeValue(
    manifest,
    variable,
    frame.values[atmosOffset(manifest, variable, boundedLevel, lat1, lon0)],
  )
  const v11 = decodeValue(
    manifest,
    variable,
    frame.values[atmosOffset(manifest, variable, boundedLevel, lat1, lon1)],
  )

  return bilerp(v00, v10, v01, v11, tx, ty)
}

export async function loadForecastManifest(): Promise<ForecastManifest | null> {
  const response = await fetch(FORECAST_MANIFEST_PATH, { cache: "no-store" })
  if (!response.ok) return null
  return response.json()
}

export function frameForLead(
  manifest: ForecastManifest,
  leadHours: number,
): ForecastFrameRef {
  return manifest.frames.reduce((best, frame) =>
    Math.abs(frame.leadHours - leadHours) < Math.abs(best.leadHours - leadHours)
      ? frame
      : best,
  )
}

export async function loadForecastFrame(
  manifest: ForecastManifest,
  ref: ForecastFrameRef,
): Promise<ForecastFrame> {
  const response = await fetch(`/forecasts/latest/${ref.path}`, { cache: "force-cache" })
  if (!response.ok) {
    throw new Error(`Unable to load forecast frame ${ref.path}`)
  }
  return {
    ref,
    values: new Int16Array(await response.arrayBuffer()),
  }
}

export function valueAt(
  manifest: ForecastManifest,
  frame: ForecastFrame,
  variable: SurfaceVariable,
  lon: number,
  lat: number,
): number {
  return surfaceInterpolated(manifest, frame, variable, lon, lat)
}

export function layerValueAt(
  manifest: ForecastManifest,
  frame: ForecastFrame,
  layer: Layer,
  lon: number,
  lat: number,
): number {
  return valueAt(manifest, frame, layer, lon, lat)
}

export function pointAt(
  manifest: ForecastManifest,
  frame: ForecastFrame,
  lon: number,
  lat: number,
  level: number,
): ForecastPoint {
  return {
    ps: surfaceInterpolated(manifest, frame, "ps", lon, lat),
    tsurf: surfaceInterpolated(manifest, frame, "tsurf", lon, lat),
    co2ice: surfaceInterpolated(manifest, frame, "co2ice", lon, lat),
    dustcol: surfaceInterpolated(manifest, frame, "dustcol", lon, lat),
    u: atmosInterpolated(manifest, frame, "u", lon, lat, level),
    v: atmosInterpolated(manifest, frame, "v", lon, lat, level),
    temp: atmosInterpolated(manifest, frame, "temp", lon, lat, level),
  }
}
