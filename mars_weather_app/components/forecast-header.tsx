interface ForecastHeaderProps {
  date: Date
  coord: string
}

export function ForecastHeader({ date, coord }: ForecastHeaderProps) {
  const formatted = date.toLocaleString("en-US", {
    month: "short",
    day: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC",
  })

  return (
    <header className="flex items-center gap-2.5 rounded-lg border border-border/60 bg-card/55 px-3 py-2 backdrop-blur-md sm:gap-3 sm:px-4 sm:py-3">
      <span
        className="h-2.5 w-2.5 shrink-0 rounded-full bg-mars shadow-[0_0_12px_2px] shadow-mars/70"
        aria-hidden="true"
      />
      <div className="flex flex-col leading-tight">
        <h1 className="font-heading text-sm font-semibold uppercase tracking-[0.18em] text-card-foreground sm:text-base sm:tracking-[0.2em]">
          Martian Weather
        </h1>
        <p className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground sm:text-xs">
          <span>{formatted} UTC</span>
          <span className="text-mars/90 sm:hidden"> · {coord}</span>
        </p>
        <p className="hidden font-mono text-[11px] uppercase tracking-wider text-mars/90 sm:block">
          {coord}
        </p>
      </div>
    </header>
  )
}
