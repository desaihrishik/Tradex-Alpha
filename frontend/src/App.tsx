// src/App.tsx

import { useEffect, useMemo, useState, type KeyboardEvent } from "react";
import ReactApexChart from "react-apexcharts";
import type { ApexOptions } from "apexcharts";

import {
  adminLogin,
  fetchAdminStatus,
  fetchAnalyze,
  fetchAgenticSignal,
  fetchCurrentTrend,
  fetchSentimentDetails,
  llmEvaluate,
} from "./api";

import type {
  AdminStatusResponse,
  AnalyzeResponse,
  Candle,
  LatestSignalResponse,
  RiskProfile,
  AgenticSignalResponse,
  CurrentTrendResponse,
  SentimentDetailsResponse,
} from "./api";

import "./App.css";
import tradexAlphaLogo from "./assets/tradex-alpha-logo.svg";
import alphaLogo from "./assets/alpha-logo.svg";

function formatPercent(p: number) {
  return (p * 100).toFixed(1) + "%";
}

function formatSignedCurrency(value: number) {
  const sign = value > 0 ? "+" : "";
  return `${sign}$${value.toFixed(2)}`;
}

function formatSignedPercent(value: number) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(2)}%`;
}

function formatCurrency(value: number) {
  return `$${value.toFixed(2)}`;
}

function formatSignedPct(value: number) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(2)}%`;
}

function formatDateTimeWithZone(value?: string | null) {
  if (!value) return "n/a";
  if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    return `${value} 00:00:00`;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "n/a";
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZoneName: "short",
  });
}

function readStatusString(
  status: Record<string, string | boolean | null> | null | undefined,
  key: string,
) {
  const value = status?.[key];
  return typeof value === "string" && value.trim() ? value : null;
}

type RangeKey = "1M" | "3M" | "6M" | "1Y";

const RANGE_LIMITS: Record<RangeKey, number> = {
  "1M": 22,
  "3M": 66,
  "6M": 132,
  "1Y": 252,
};

const REFRESH_INTERVAL_MS = 30_000;
const INITIAL_RETRY_DELAYS_MS = [500, 1200];

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function getUsMarketStatus(now = new Date()) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).formatToParts(now);

  const weekday = parts.find((part) => part.type === "weekday")?.value ?? "Sun";
  const hour = Number(parts.find((part) => part.type === "hour")?.value ?? "0");
  const minute = Number(parts.find((part) => part.type === "minute")?.value ?? "0");
  const totalMinutes = hour * 60 + minute;
  const isWeekday = ["Mon", "Tue", "Wed", "Thu", "Fri"].includes(weekday);
  const isOpen = isWeekday && totalMinutes >= 9 * 60 + 30 && totalMinutes < 16 * 60;

  return {
    isOpen,
    label: isOpen ? "Live" : "Closed",
  };
}

function buildWeightedTrendSeries(candles: Candle[], window = 8) {
  if (candles.length === 0) return [];
  return candles.map((_, index) => {
    const start = Math.max(0, index - window + 1);
    const slice = candles.slice(start, index + 1);
    const weights = slice.map((__, i) => Math.pow(1.35, i));
    const weightedPoints = slice.map((candle, i) => ({
      x: i,
      y: candle.close,
      w: weights[i],
    }));
    const totalWeight = weightedPoints.reduce((sum, point) => sum + point.w, 0);
    const xBar = weightedPoints.reduce((sum, point) => sum + point.x * point.w, 0) / totalWeight;
    const yBar = weightedPoints.reduce((sum, point) => sum + point.y * point.w, 0) / totalWeight;
    const numerator = weightedPoints.reduce(
      (sum, point) => sum + point.w * (point.x - xBar) * (point.y - yBar),
      0,
    );
    const denominator = weightedPoints.reduce(
      (sum, point) => sum + point.w * Math.pow(point.x - xBar, 2),
      0,
    );
    const slope = denominator !== 0 ? numerator / denominator : 0;
    const intercept = yBar - slope * xBar;
    const fittedClose = intercept + slope * (slice.length - 1);
    return {
      x: new Date(candles[index].date),
      y: Number(fittedClose.toFixed(2)),
    };
  });
}

function buildForecastLineSeries(
  forecast: AgenticSignalResponse["trend_forecast"] | null | undefined,
  latestCandle: Candle | undefined,
  horizonDays = 5,
) {
  if (!forecast || !latestCandle) return [];

  const latestDateTs = new Date(latestCandle.date).getTime();

  return forecast.dates
    .map((date, index) => ({
      date,
      value: forecast.values[index],
      ts: new Date(date).getTime(),
    }))
    .filter((point) => Number.isFinite(point.ts) && Number.isFinite(point.value) && point.ts > latestDateTs)
    .sort((a, b) => a.ts - b.ts)
    .slice(0, horizonDays)
    .map((point) => ({
      x: new Date(point.date),
      y: Number(point.value.toFixed(2)),
    }));
}

function useDebouncedValue<T>(value: T, delayMs: number) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedValue(value), delayMs);
    return () => window.clearTimeout(timer);
  }, [value, delayMs]);

  return debouncedValue;
}

function App() {
  const [candles, setCandles] = useState<Candle[]>([]);
  const [signal, setSignal] = useState<LatestSignalResponse | null>(null);
  const [agentic, setAgentic] = useState<AgenticSignalResponse | null>(null);
  const [currentTrend, setCurrentTrend] = useState<CurrentTrendResponse | null>(null);
  const [sentimentDetails, setSentimentDetails] = useState<SentimentDetailsResponse | null>(null);
  const [analysisValidation, setAnalysisValidation] = useState<AnalyzeResponse["validation"] | null>(null);
  const [quantInsights, setQuantInsights] = useState<AnalyzeResponse["quant_insights"] | null>(null);

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [trendRefreshing, setTrendRefreshing] = useState(false);
  const [marketStatus, setMarketStatus] = useState(() => getUsMarketStatus());
  const [agenticLoading, setAgenticLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [agenticError, setAgenticError] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState(true);

  const [budget, setBudget] = useState(1000);
  const [budgetInput, setBudgetInput] = useState("1000");
  const [risk, setRisk] = useState<RiskProfile>("medium");
  const [entryPrice, setEntryPrice] = useState<number | undefined>(undefined);
  const [patternPage, setPatternPage] = useState(0);
  const [marketPanel, setMarketPanel] = useState<"trend" | "sentiment">("trend");
  const [insightPanel, setInsightPanel] = useState<"momentum" | "volatility" | "risk" | "levels" | "relative">("momentum");

  const [range] = useState<RangeKey>("3M");
  const debouncedBudget = useDebouncedValue(budget, 450);
  const debouncedEntryPrice = useDebouncedValue(entryPrice, 450);

  useEffect(() => {
    const trimmed = budgetInput.trim();
    if (!trimmed) {
      return;
    }
    const nextValue = Number(trimmed);
    if (Number.isFinite(nextValue) && nextValue >= 0) {
      setBudget(nextValue);
    }
  }, [budgetInput]);

  // ----------- LLM State ---------------
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmError, setLlmError] = useState<string | null>(null);
  const [llmOutput, setLlmOutput] = useState<string | null>(null);
  const [llmQuestion, setLlmQuestion] = useState("");
  const [showEdgeOracle, setShowEdgeOracle] = useState(false);
  const [showAdminPanel, setShowAdminPanel] = useState(false);
  const [showAdminLogin, setShowAdminLogin] = useState(false);
  const [adminPasswordInput, setAdminPasswordInput] = useState("");
  const [adminPassword, setAdminPassword] = useState<string | null>(null);
  const [adminStatus, setAdminStatus] = useState<AdminStatusResponse | null>(null);
  const [adminStatusLoading, setAdminStatusLoading] = useState(false);
  const [adminLoginError, setAdminLoginError] = useState<string | null>(null);
  const alphaPromptChips = [
    "Why now?",
    "Top risk?",
    "Wait or enter?",
    "Invalidation?",
  ];

  const loadDashboard = async (persist = true) => {
    const hasExistingResults = signal !== null;

    if (hasExistingResults) {
      setRefreshing(true);
    } else {
      setLoading(true);
      setAgenticLoading(true);
    }
    setError(null);
    setAgenticError(null);
    if (!hasExistingResults) {
      setIsConnecting(true);
    }

    const limit = RANGE_LIMITS[range];

    for (let attempt = 0; attempt <= INITIAL_RETRY_DELAYS_MS.length; attempt += 1) {
      try {
        const [analysis, trend, sentiment] = await Promise.all([
          fetchAnalyze(debouncedBudget, risk, limit, debouncedEntryPrice, persist),
          fetchCurrentTrend(),
          fetchSentimentDetails(),
        ]);

        setCandles(analysis.candles);
        setSignal(analysis.latest_signal);
        setAgentic(analysis.agentic_signal);
        setCurrentTrend(trend);
        setSentimentDetails(sentiment);
        setAnalysisValidation(analysis.validation);
        setQuantInsights(analysis.quant_insights ?? null);
        setIsConnecting(false);
        setLoading(false);
        setRefreshing(false);
        setAgenticLoading(false);
        return;
      } catch (err) {
        console.error(err);

        if (attempt < INITIAL_RETRY_DELAYS_MS.length) {
          await sleep(INITIAL_RETRY_DELAYS_MS[attempt]);
          continue;
        }

        setError("Failed to load data from backend.");
        setAgenticError("Failed to load agentic signal.");
        setIsConnecting(false);
        setLoading(false);
        setRefreshing(false);
        setAgenticLoading(false);
      }
    }
  };

  const refreshLivePanels = async () => {
    try {
      setTrendRefreshing(true);
      const [trend, sentiment] = await Promise.all([
        fetchCurrentTrend(),
        fetchSentimentDetails(),
      ]);
      setCurrentTrend(trend);
      setSentimentDetails(sentiment);
    } catch (err) {
      console.error(err);
    } finally {
      setTrendRefreshing(false);
    }
  };

  // ---------------------------------------------------------
  // Load candles + latest signal
  // ---------------------------------------------------------
  useEffect(() => {
    void loadDashboard(true);
  }, [debouncedBudget, risk, range, debouncedEntryPrice]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void refreshLivePanels();
    }, REFRESH_INTERVAL_MS);

    return () => window.clearInterval(timer);
  }, []);

  const loadAdminStatus = async (password: string) => {
    try {
      setAdminStatusLoading(true);
      const status = await fetchAdminStatus(password, 8);
      setAdminStatus(status);
    } catch (err) {
      console.error(err);
    } finally {
      setAdminStatusLoading(false);
    }
  };

  useEffect(() => {
    if (!showAdminPanel || !adminPassword) {
      return;
    }
    void loadAdminStatus(adminPassword);
    const timer = window.setInterval(() => {
      void loadAdminStatus(adminPassword);
    }, REFRESH_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [showAdminPanel, adminPassword]);

  useEffect(() => {
    const syncMarketStatus = () => setMarketStatus(getUsMarketStatus());
    syncMarketStatus();
    const timer = window.setInterval(syncMarketStatus, 60_000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const onGlobalKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        if (showAdminLogin) {
          setShowAdminLogin(false);
          return;
        }
        if (showAdminPanel) {
          setShowAdminPanel(false);
          return;
        }
        if (showEdgeOracle) {
          setShowEdgeOracle(false);
        }
      }
    };

    window.addEventListener("keydown", onGlobalKeyDown);
    return () => window.removeEventListener("keydown", onGlobalKeyDown);
  }, [showAdminLogin, showAdminPanel, showEdgeOracle]);

  // ---------------------------------------------------------
  // Load agentic signal
  // ---------------------------------------------------------
  useEffect(() => {
    if (debouncedEntryPrice === undefined) {
      return;
    }

    async function loadAgentic() {
      try {
        setAgenticLoading(true);
        setAgenticError(null);

        const res = await fetchAgenticSignal(debouncedBudget, risk, debouncedEntryPrice);
        setAgentic(res);
      } catch (err) {
        console.error(err);
        setAgenticError("Failed to load agentic signal.");
      } finally {
        setAgenticLoading(false);
      }
    }

    loadAgentic();
  }, [debouncedBudget, risk, debouncedEntryPrice]);

  const latestCandle = candles[candles.length - 1];
  const recentPatternCandles = [...candles].reverse();
  const PATTERN_PAGE_SIZE = 4;
  const maxPatternPage = Math.max(
    0,
    Math.ceil(recentPatternCandles.length / PATTERN_PAGE_SIZE) - 1,
  );
  const visiblePatternCandles = recentPatternCandles.slice(
    patternPage * PATTERN_PAGE_SIZE,
    (patternPage + 1) * PATTERN_PAGE_SIZE,
  );

  useEffect(() => {
    setPatternPage(0);
  }, [candles]);

  useEffect(() => {
    if (patternPage > maxPatternPage) {
      setPatternPage(maxPatternPage);
    }
  }, [patternPage, maxPatternPage]);

  const getDailyMoveForCandle = (candle: Candle) => {
    const candleIndex = candles.findIndex((item) => item.date === candle.date);
    const previousCandle = candleIndex > 0 ? candles[candleIndex - 1] : null;
    const basePrice = previousCandle?.close ?? candle.open;
    const change = candle.close - basePrice;
    const changePct = basePrice ? change / basePrice : 0;
    return { change, changePct };
  };

  const trendLineSeries = buildWeightedTrendSeries(candles, 8);
  const forecastHorizonDays = agentic ? Math.min(agentic.horizon_days, 5) : 5;
  const forecastLineSeries = buildForecastLineSeries(agentic?.trend_forecast, latestCandle, forecastHorizonDays);
  const forecastSeriesIndex = forecastLineSeries.length > 0 ? 2 : -1;
  const chartMin = candles[0] ? new Date(candles[0].date).getTime() : undefined;
  const chartMax =
    forecastLineSeries.length > 0
      ? forecastLineSeries[forecastLineSeries.length - 1].x.getTime()
      : latestCandle
        ? new Date(latestCandle.date).getTime()
        : undefined;
  const allChartLows = [
    ...candles.map((c) => c.low),
  ];
  const allChartHighs = [
    ...candles.map((c) => c.high),
  ];
  const chartLow = allChartLows.length > 0 ? Math.min(...allChartLows) : undefined;
  const chartHigh = allChartHighs.length > 0 ? Math.max(...allChartHighs) : undefined;
  const chartPadding =
    chartLow !== undefined && chartHigh !== undefined ? Math.max((chartHigh - chartLow) * 0.08, 2.5) : 0;
  const chartYMin = chartLow !== undefined ? Number(Math.max(0, chartLow - chartPadding).toFixed(2)) : undefined;
  const chartYMax = chartHigh !== undefined ? Number((chartHigh + chartPadding).toFixed(2)) : undefined;
  const liveAdminStatus = adminStatus?.live_status;
  const schedulerRunning = liveAdminStatus?.scheduler_running === true;
  const marketRefreshAt = readStatusString(liveAdminStatus, "market_last_run");
  const sentimentRefreshAt = readStatusString(liveAdminStatus, "sentiment_last_run");
  const modelRetrainAt = readStatusString(liveAdminStatus, "model_last_run");
  const modelVersion = readStatusString(liveAdminStatus, "model_version");
  const marketLastError = readStatusString(liveAdminStatus, "market_last_error");
  const lastHeartbeatAt = adminStatus?.recent_events?.[0]?.created_at ?? null;

  // ---------------------------------------------------------
  // Candlestick chart
  // ---------------------------------------------------------
  const chartOptions: ApexOptions = {
    chart: {
      type: "candlestick",
      height: "100%",
      background: "transparent",
      foreColor: "#c9d6c0",
      toolbar: {
        tools: {
          pan: true,
          zoom: true,
          zoomin: true,
          zoomout: true,
          reset: true,
        },
      },
    },
    grid: {
      borderColor: "rgba(118, 136, 103, 0.28)",
      strokeDashArray: 4,
    },
    xaxis: {
      type: "datetime",
      min: chartMin,
      max: chartMax,
      labels: { style: { colors: "#9eb595" } },
      axisBorder: { color: "rgba(118, 136, 103, 0.28)" },
      axisTicks: { color: "rgba(118, 136, 103, 0.28)" },
    },
    yaxis: {
      tooltip: { enabled: true },
      min: chartYMin,
      max: chartYMax,
      labels: {
        formatter: (val) => `$${val.toFixed(0)}`,
        style: { colors: "#9eb595" },
      },
    },
    stroke: {
      curve: "smooth",
      width: [0, 3, 4],
      dashArray: [0, 0, 6],
    },
    legend: {
      show: false,
    },
    fill: {
      type: ["solid", "solid", "solid"],
      opacity: [1, 0.12, 0.2],
    },
    colors: ["#94bd52", "#9fb290", "#facc15"],
    markers: {
      size: [0, 0, 4],
      strokeWidth: [0, 0, 1],
      colors: ["#94bd52", "#9fb290", "#facc15"],
      hover: { size: 6 },
    },
    plotOptions: {
      candlestick: {
        colors: { upward: "#94bd52", downward: "#ff6b6b" },
        wick: { useFillColor: true },
      },
    },
    tooltip: {
      theme: "dark",
      shared: true,
      custom: ({ seriesIndex, dataPointIndex }) => {
        if (seriesIndex === 0) {
          const candle = candles[dataPointIndex];
          if (!candle) return "";
          const patternText =
            candle.patterns?.length > 0
              ? `<div style="margin-top:4px;font-size:11px;color:#facc15">
                   Patterns: ${candle.patterns.join(", ")}
                 </div>`
              : "";

          return `
            <div style="padding:8px;font-size:12px;color:#edf7e5;background:#131a12;border:1px solid rgba(118,185,0,0.3);">
              <div><strong>${candle.date}</strong></div>
              <div>O: ${candle.open.toFixed(2)}  
                   H: ${candle.high.toFixed(2)}  
                   L: ${candle.low.toFixed(2)}  
                   C: ${candle.close.toFixed(2)}</div>
              <div>Vol: ${candle.volume.toLocaleString()}</div>
              ${patternText}
            </div>
          `;
        }

        if (seriesIndex === forecastSeriesIndex) {
          const forecastPoint = forecastLineSeries[dataPointIndex];
          if (!forecastPoint) return "";
          const predictedPrice = forecastPoint.y.toFixed(2);
          const forecastDate = new Date(forecastPoint.x).toISOString().slice(0, 10);
          const priorDate = dataPointIndex > 0 ? new Date(forecastLineSeries[dataPointIndex - 1].x).toISOString().slice(0, 10) : "latest candle";
          return `
            <div style="padding:8px;font-size:12px;color:#edf7e5;background:#131a12;border:1px solid rgba(118,185,0,0.3);">
              <div><strong>${forecastDate}</strong> Forecast</div>
              <div>Predicted close: $${predictedPrice}</div>
              <div style="margin-top:4px;color:#facc15;font-size:11px;">
                Next step after ${priorDate}
              </div>
            </div>
          `;
        }

        return "";
      },
    },
  };

  const chartSeries = [
    {
      name: "Candles",
      type: "candlestick" as const,
      data: candles.map((c) => ({
        x: new Date(c.date),
        y: [c.open, c.high, c.low, c.close],
      })),
    },
    {
      name: "Trendline",
      type: "line" as const,
      data: trendLineSeries,
    },
    ...(forecastLineSeries.length > 0
      ? [
          {
            name: "Forecast Trend",
            type: "line" as const,
            data: forecastLineSeries,
          },
        ]
      : []),
  ];

  // ---------------------------------------------------------
  // LLM Evaluate
  // ---------------------------------------------------------
  const handleEvaluateClick = async () => {
    if (!latestCandle || !signal || !llmQuestion.trim()) return;

    try {
      setLlmLoading(true);
      setLlmError(null);
      setLlmOutput(null);

      const res = await llmEvaluate(
        llmQuestion.trim(),
        debouncedBudget,
        risk,
        debouncedEntryPrice,
      );
      setLlmOutput(res.answer);
    } catch (err) {
      console.error(err);
      setLlmError("Failed to contact LLM.");
    } finally {
      setLlmLoading(false);
    }
  };

  const handleAdminClick = () => {
    if (!adminPassword) {
      setShowAdminLogin(true);
      return;
    }
    setShowAdminPanel((value) => !value);
  };

  const handleAdminLogin = async () => {
    try {
      setAdminLoginError(null);
      await adminLogin(adminPasswordInput);
      setAdminPassword(adminPasswordInput);
      setShowAdminLogin(false);
      setShowAdminPanel(true);
      void loadAdminStatus(adminPasswordInput);
    } catch (err) {
      console.error(err);
      setAdminLoginError("Invalid admin password.");
    }
  };

  const handleLlmQuestionKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      setShowEdgeOracle(false);
      return;
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleEvaluateClick();
    }
  };

  const handleAlphaPromptClick = (prompt: string) => {
    setLlmQuestion(prompt);
  };

  const showMarketLoader = loading || (refreshing && signal !== null);
  const loaderSubtitle = loading
    ? "Booting models, pricing feed, and sentiment intelligence."
    : "Pulling latest candles, recommendation engine output, and quant factors.";
  const loaderTapeItems = useMemo(() => {
    const liveValues = [
      currentTrend ? `NVDA ${formatSignedCurrency(currentTrend.change)} (${formatSignedPercent(currentTrend.change_pct)})` : null,
      latestCandle ? `Last ${formatCurrency(latestCandle.close)}` : null,
      signal ? `Signal ${signal.action} ${formatPercent(signal.confidence)}` : null,
      agentic ? `Horizon ${agentic.horizon_days}D` : null,
      quantInsights?.momentum ? `Momentum ${quantInsights.momentum.score.toFixed(1)}` : null,
      quantInsights?.market_regime ? `Regime ${quantInsights.market_regime.volatility_regime}` : null,
      quantInsights?.risk_metrics ? `Sharpe ${quantInsights.risk_metrics.sharpe.toFixed(2)}` : null,
      quantInsights?.volume_intelligence ? `RVOL ${quantInsights.volume_intelligence.rvol_20.toFixed(2)}` : null,
      "Alpha engine online",
      "Syncing historical snapshots",
      "Rebuilding signal stack",
    ].filter((value): value is string => Boolean(value));

    const syntheticValues = Array.from({ length: 42 }, (_, index) => {
      const mode = index % 6;
      if (mode === 0) return `NVDA ${180 + (index % 11) * 2}.${(index * 7) % 10}${(index * 3) % 10}`;
      if (mode === 1) return `ROC ${(Math.sin(index) * 4.8).toFixed(2)}%`;
      if (mode === 2) return `ATR ${(2.1 + ((index * 13) % 31) / 10).toFixed(2)}`;
      if (mode === 3) return `MFI ${48 + ((index * 9) % 43)}`;
      if (mode === 4) return `ADX ${(16 + ((index * 11) % 23)).toFixed(1)}`;
      return `Spread ${(0.08 + ((index * 5) % 17) / 100).toFixed(2)}%`;
    });

    return [...liveValues, ...syntheticValues];
  }, [agentic, currentTrend, latestCandle, quantInsights, signal]);
  // ---------------------------------------------------------
  // UI
  // ---------------------------------------------------------
  return (
    <div className="app-root">
      {showMarketLoader && (
        <div className="market-loader-overlay" role="status" aria-live="polite" aria-label="Loading latest market data">
          <div className="market-loader-copy">
            <div className="market-loader-loading-line market-loader-loading-line-top">
              <span>Loading Signal</span>
              <span className="market-loader-loading-bar" aria-hidden="true" />
            </div>
            <p className="market-loader-subtitle">{loaderSubtitle}</p>
          </div>

          <div className="market-loader-chart" aria-hidden="true">
            <div className="market-loader-grid" />
            <svg viewBox="0 0 1000 420" className="market-loader-svg">
              <polyline
                className="market-loader-line-glow"
                points="0,368 120,390 220,338 330,282 450,312 560,286 690,210 820,166 930,126 1000,112"
              />
              <polyline
                className="market-loader-line"
                points="0,368 120,390 220,338 330,282 450,312 560,286 690,210 820,166 930,126 1000,112"
              />
              <circle className="market-loader-dot" cx="1000" cy="112" r="4" />
            </svg>
            <div className="market-loader-scan" />
          </div>

          <div className="market-loader-bottom">
            <div className="market-loader-tape">
              <div className="market-loader-tape-track">
                {[...loaderTapeItems, ...loaderTapeItems].map((item, index) => (
                  <span key={`${item}-${index}`} className="market-loader-tape-item">
                    {item}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      <header className="top-bar">
        <div className="site-title">
          <div className="brand-lockup">
            <div className="brand-logo-wrap">
              <img className="brand-logo" src={tradexAlphaLogo} alt="Tradex Alpha logo" />
            </div>
            <div className="brand-copy">
              <h1>
                <span className="brand-wordmark-main">Tradex</span>{" "}
                <span className="brand-wordmark-accent">Alpha</span>
              </h1>
              <p className="app-subtitle">Intelligent Trading Starts with Alpha</p>
            </div>
          </div>
        </div>

        <div className="top-actions">
          <button className="btn-secondary btn-tiny" onClick={handleAdminClick}>
            Admin
          </button>

          <button className="btn-secondary btn-tiny" onClick={() => window.location.reload()}>
            Refresh
          </button>
        </div>
      </header>

      <main className="app-main">
        <section className="card chart-card">
          <div className="card-header">
            <div className="card-header-main">
              <h2>Trendline Projection</h2>
              <div className="chart-legend">
                <span className="chart-legend-item">
                  <span className="chart-legend-dot chart-legend-dot-candles" />
                  Candles
                </span>
                <span className="chart-legend-item">
                  <span className="chart-legend-dot chart-legend-dot-trend" />
                  Trendline
                </span>
                <span className="chart-legend-item">
                  <span className="chart-legend-dot chart-legend-dot-forecast" />
                  Forecast
                </span>
              </div>
            </div>
            {latestCandle && (
              <div className="chart-header-meta">
                <span className="tag">Last close: ${latestCandle.close.toFixed(2)}</span>
                <span className="tag tag-soft">{latestCandle.date}</span>
                <span
                  className={`live-badge ${marketStatus.isOpen ? "live-badge-open" : "live-badge-closed"} ${trendRefreshing ? "live-badge-updating" : ""}`.trim()}
                >
                  <span className="live-dot" />
                  {marketStatus.label}
                </span>
              </div>
            )}
          </div>

          <div className="chart-fill-wrap">
            <ReactApexChart options={chartOptions} series={chartSeries} type="candlestick" height="100%" />
          </div>
        </section>

        <section className="card market-card">
          <div className="card-header">
            <h2>Market Pulse</h2>
            <div className="trend-header-meta">
              <div className="panel-switch">
                <button
                  className={`panel-switch-btn ${marketPanel === "trend" ? "panel-switch-btn-active" : ""}`.trim()}
                  onClick={() => setMarketPanel("trend")}
                  type="button"
                >
                  Trend
                </button>
                <button
                  className={`panel-switch-btn ${marketPanel === "sentiment" ? "panel-switch-btn-active" : ""}`.trim()}
                  onClick={() => setMarketPanel("sentiment")}
                  type="button"
                >
                  Sentiment
                </button>
              </div>
            </div>
          </div>

          {marketPanel === "trend" ? (
            currentTrend ? (
              <>
                <div className="signal-row">
                  <div>
                    <div className="signal-label">Trend</div>
                    <div className={`signal-action trend-${currentTrend.trend_label}`}>{currentTrend.trend_label}</div>
                    <div className="muted-small">As of: {formatDateTimeWithZone(currentTrend.as_of)}</div>
                  </div>
                  <div>
                    <div className="signal-label">Last Close</div>
                    <div className="signal-value">${currentTrend.latest_close.toFixed(2)}</div>
                  </div>
                </div>

                <div className="signal-row">
                  <div>
                    <div className="signal-label">Daily Change</div>
                    <div className="signal-value">{formatSignedCurrency(currentTrend.change)}</div>
                    <div className="muted-small">{formatSignedPercent(currentTrend.change_pct)}</div>
                  </div>

                  <div>
                    <div className="signal-label">Volume</div>
                    <div className="signal-value">{Math.round(currentTrend.volume).toLocaleString()}</div>
                  </div>
                </div>

                <div className="trend-snapshot-grid">
                  <div className="trend-snapshot-item">
                    <div className="signal-label">Pattern Snapshot</div>
                    <div className="signal-value-small">
                      {currentTrend.patterns?.length ? currentTrend.patterns.join(", ") : "No active patterns"}
                    </div>
                  </div>

                  <div className="trend-snapshot-item">
                    <div className="signal-label">Forecast Range</div>
                    <div className="signal-value-small">
                      {agentic ? `${formatCurrency(agentic.forecast.p10)} - ${formatCurrency(agentic.forecast.p90)}` : "n/a"}
                    </div>
                    <div className="muted-small">
                      {agentic ? `Median ${formatCurrency(agentic.forecast.p50)}` : "Waiting for projection"}
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <p className="muted">Loading current trend...</p>
            )
          ) : sentimentDetails ? (
            <>
              <div className="signal-row">
                <div>
                  <div className="signal-label">Score</div>
                  <div className="signal-value">{formatPercent(sentimentDetails.score)}</div>
                  <div className="muted-small">As of: {formatDateTimeWithZone(sentimentDetails.as_of)}</div>
                </div>
              </div>

              <div className="card-subsection">
                <div className="signal-label">Interpretation</div>
                <p className="explanation-text">{sentimentDetails.interpretation}</p>
              </div>
            </>
          ) : (
            <p className="muted">Loading sentiment details...</p>
          )}
        </section>

        <section className="card recommendation-card">
          <div className="card-header">
            <h2>Decision Engine</h2>
          </div>

          <div className="combined-controls">
            <div className="field">
              <label>Budget (USD)</label>
              <input
                type="number"
                value={budgetInput}
                min={100}
                step={100}
                onChange={(e) => setBudgetInput(e.target.value)}
                onBlur={() => {
                  const trimmed = budgetInput.trim();
                  if (!trimmed) {
                    setBudgetInput(String(budget));
                    return;
                  }
                  const nextValue = Number(trimmed);
                  if (Number.isFinite(nextValue) && nextValue >= 0) {
                    setBudgetInput(String(nextValue));
                  } else {
                    setBudgetInput(String(budget));
                  }
                }}
              />
            </div>
            <div className="field">
              <label>Risk Profile</label>
              <select value={risk} onChange={(e) => setRisk(e.target.value as RiskProfile)}>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
            <div className="field">
              <label>Entry Price (Optional)</label>
              <input
                type="number"
                value={entryPrice ?? ""}
                onChange={(e) => setEntryPrice(e.target.value === "" ? undefined : Number(e.target.value))}
              />
            </div>
          </div>

          <div className="combined-reco-grid">
            <div className="card-subsection combined-reco-panel">
              <div className="signal-label">Current NVDA Recommendation</div>
              {loading && !signal && <p className="muted">Loading signal...</p>}
              {refreshing && signal && <p className="muted">Refreshing...</p>}
              {error && !loading && !isConnecting && <p className="error">{error}</p>}

              {signal && !error && (
                <>
                  <div className="signal-row">
                    <div>
                      <div className={`signal-action action-${signal.action}`}>{signal.action}</div>
                    </div>

                    <div>
                      <div className="signal-label">Confidence</div>
                      <div className="confidence-row">
                        <span className="confidence-value">{formatPercent(signal.confidence)}</span>
                        <div className="confidence-bar">
                          <div className="confidence-fill" style={{ width: `${signal.confidence * 100}%` }} />
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="signal-row">
                    <div>
                      <div className="signal-label">Suggested Quantity</div>
                      <div className="signal-value">
                        {signal.suggested_shares} share{signal.suggested_shares !== 1 && "s"}
                      </div>
                      <div className="muted-small">Capital used: ${signal.capital_used.toFixed(2)}</div>
                    </div>

                    <div>
                      <div className="signal-label">Model Price</div>
                      <div className="signal-value">${signal.latest_close.toFixed(2)}</div>
                      <div className="muted-small">As of: {signal.date}</div>
                    </div>
                  </div>

                  <p className="explanation-text">{signal.explanation}</p>
                </>
              )}
            </div>

            <div className="card-subsection combined-reco-panel">
              <div className="signal-label">Agentic Horizon</div>
              {agenticLoading && <p className="muted">Computing...</p>}
              {agenticError && <p className="error">{agenticError}</p>}

              {agentic && latestCandle && (
                <>
                  <div className="signal-row">
                    <div>
                      <div className={`signal-action action-${agentic.action}`}>{agentic.action}</div>
                      <div className="muted-small">Horizon: ~{agentic.horizon_days} trading days</div>
                    </div>

                    <div>
                      <div className="signal-label">Sentiment</div>
                      <div className="signal-value">
                        {agentic.sentiment_label} ({formatPercent(agentic.sentiment_score)})
                      </div>
                    </div>
                  </div>

                  <div className="trend-snapshot-grid">
                    <div className="trend-snapshot-item">
                      <div className="signal-label">Forecast</div>
                      <div className="signal-value-small">
                        {formatCurrency(agentic.forecast.p10)} - {formatCurrency(agentic.forecast.p90)}
                      </div>
                    </div>
                    <div className="trend-snapshot-item">
                      <div className="signal-label">Median</div>
                      <div className="signal-value-small">{formatCurrency(agentic.forecast.p50)}</div>
                    </div>
                  </div>

                  <p className="explanation-text">{agentic.explanation}</p>
                </>
              )}
            </div>
          </div>

          {analysisValidation && (
            <div
              className={`card-subsection validation-panel ${
                analysisValidation.is_fresh ? "validation-pass" : "validation-warn"
              }`}
            >
              <div className="card-header validation-header">
                <div className="signal-label">Calculation Check</div>
                <span className="tag">
                  {analysisValidation.is_fresh ? "Synced to latest market data" : "Needs refresh review"}
                </span>
              </div>

              <div className="validation-grid">
                <div>
                  <div className="signal-label">Model Price</div>
                  <div className="signal-value-small">{formatCurrency(analysisValidation.model_price)}</div>
                </div>
                <div>
                  <div className="signal-label">Latest Market Price</div>
                  <div className="signal-value-small">{formatCurrency(analysisValidation.market_price)}</div>
                </div>
                <div>
                  <div className="signal-label">Shown Quantity</div>
                  <div className="signal-value-small">{analysisValidation.displayed_suggested_shares} shares</div>
                </div>
                <div>
                  <div className="signal-label">Rechecked at Market Price</div>
                  <div className="signal-value-small">{analysisValidation.suggested_shares_at_market_price} shares</div>
                </div>
              </div>

              <p className="muted-small">
                Price gap: {formatSignedPercent(analysisValidation.price_diff_pct)}. Capital shown {" "}
                {formatCurrency(analysisValidation.displayed_capital_used)} vs latest price{" "}
                {formatCurrency(analysisValidation.capital_used_at_market_price)}.
              </p>
            </div>
          )}
        </section>

        <section className="card insights-card">
          <div className="card-header">
            <h2>Alpha Quant Lens</h2>
            <div className="trend-header-meta">
              <div className="panel-switch">
                <button
                  className={`panel-switch-btn ${insightPanel === "momentum" ? "panel-switch-btn-active" : ""}`.trim()}
                  onClick={() => setInsightPanel("momentum")}
                  type="button"
                >
                  Momentum
                </button>
                <button
                  className={`panel-switch-btn ${insightPanel === "volatility" ? "panel-switch-btn-active" : ""}`.trim()}
                  onClick={() => setInsightPanel("volatility")}
                  type="button"
                >
                  Volatility
                </button>
                <button
                  className={`panel-switch-btn ${insightPanel === "risk" ? "panel-switch-btn-active" : ""}`.trim()}
                  onClick={() => setInsightPanel("risk")}
                  type="button"
                >
                  Risk
                </button>
                <button
                  className={`panel-switch-btn ${insightPanel === "levels" ? "panel-switch-btn-active" : ""}`.trim()}
                  onClick={() => setInsightPanel("levels")}
                  type="button"
                >
                  Levels
                </button>
                <button
                  className={`panel-switch-btn ${insightPanel === "relative" ? "panel-switch-btn-active" : ""}`.trim()}
                  onClick={() => setInsightPanel("relative")}
                  type="button"
                >
                  Relative
                </button>
              </div>
              <span className="tag">
                {quantInsights?.as_of ? `As of ${quantInsights.as_of}` : "Live"}
              </span>
            </div>
          </div>

          {!quantInsights?.available ? (
            <p className="muted">Preparing quant insights...</p>
          ) : (
            <>
              <div className="insight-topline">
                <span className="tag">{quantInsights.market_regime?.label ?? "Regime n/a"}</span>
                <span className="tag tag-soft">{quantInsights.market_regime?.volatility_regime ?? "Vol n/a"}</span>
                <span className="tag tag-soft">
                  Momentum {quantInsights.momentum?.score?.toFixed(1) ?? "n/a"}
                </span>
              </div>

              {insightPanel === "momentum" && (
                <div className="insight-grid">
                  <div className="trend-snapshot-item">
                    <div className="signal-label">Composite Momentum</div>
                    <div className="signal-value">{quantInsights.momentum?.score?.toFixed(1) ?? "n/a"}</div>
                  </div>
                  <div className="trend-snapshot-item">
                    <div className="signal-label">RSI / Stoch RSI</div>
                    <div className="signal-value-small">
                      {(quantInsights.momentum?.rsi_14 ?? 0).toFixed(1)} / {(quantInsights.momentum?.stoch_rsi ?? 0).toFixed(1)}
                    </div>
                  </div>
                  <div className="trend-snapshot-item">
                    <div className="signal-label">MACD Hist / ADX</div>
                    <div className="signal-value-small">
                      {(quantInsights.momentum?.macd_hist ?? 0).toFixed(4)} / {(quantInsights.momentum?.adx_14 ?? 0).toFixed(1)}
                    </div>
                  </div>
                  <div className="trend-snapshot-item">
                    <div className="signal-label">EMA / SMA Alignment</div>
                    <div className="signal-value-small">
                      {quantInsights.momentum?.ema_alignment ?? "n/a"} / {quantInsights.momentum?.sma_alignment ?? "n/a"}
                    </div>
                  </div>
                </div>
              )}

              {insightPanel === "volatility" && (
                <div className="insight-grid">
                  <div className="trend-snapshot-item">
                    <div className="signal-label">ATR (14)</div>
                    <div className="signal-value-small">
                      {(quantInsights.volatility?.atr_14 ?? 0).toFixed(2)} ({formatSignedPct(quantInsights.volatility?.atr_pct ?? 0)})
                    </div>
                  </div>
                  <div className="trend-snapshot-item">
                    <div className="signal-label">Hist / Realized Vol</div>
                    <div className="signal-value-small">
                      {formatSignedPct(quantInsights.volatility?.historical_vol_20 ?? 0)} / {formatSignedPct(quantInsights.volatility?.realized_vol_20 ?? 0)}
                    </div>
                  </div>
                  <div className="trend-snapshot-item">
                    <div className="signal-label">Bollinger Range</div>
                    <div className="signal-value-small">
                      {formatCurrency(quantInsights.volatility?.bollinger.lower ?? 0)} - {formatCurrency(quantInsights.volatility?.bollinger.upper ?? 0)}
                    </div>
                  </div>
                  <div className="trend-snapshot-item">
                    <div className="signal-label">Execution Hint</div>
                    <div className="muted-small">{quantInsights.volatility?.regime_hint ?? "n/a"}</div>
                  </div>
                </div>
              )}

              {insightPanel === "risk" && (
                <div className="insight-grid">
                  <div className="trend-snapshot-item">
                    <div className="signal-label">Sharpe / Sortino</div>
                    <div className="signal-value-small">
                      {(quantInsights.risk_metrics?.sharpe ?? 0).toFixed(2)} / {(quantInsights.risk_metrics?.sortino ?? 0).toFixed(2)}
                    </div>
                  </div>
                  <div className="trend-snapshot-item">
                    <div className="signal-label">Max DD / Win Rate</div>
                    <div className="signal-value-small">
                      {formatSignedPct(quantInsights.risk_metrics?.max_drawdown ?? 0)} / {formatSignedPct(quantInsights.risk_metrics?.win_rate ?? 0)}
                    </div>
                  </div>
                  <div className="trend-snapshot-item">
                    <div className="signal-label">Suggested Shares</div>
                    <div className="signal-value-small">
                      {quantInsights.position_sizing?.suggested_shares ?? 0} shares
                    </div>
                    <div className="muted-small">
                      Stop {formatCurrency(quantInsights.position_sizing?.stop_loss ?? 0)} • Target {formatCurrency(quantInsights.position_sizing?.profit_target ?? 0)}
                    </div>
                  </div>
                  <div className="trend-snapshot-item">
                    <div className="signal-label">Target / Stop Probability</div>
                    <div className="signal-value-small">
                      {formatSignedPct(quantInsights.position_sizing?.probability_target_hit_12d ?? 0)} / {formatSignedPct(quantInsights.position_sizing?.probability_stop_hit_12d ?? 0)}
                    </div>
                  </div>
                </div>
              )}

              {insightPanel === "levels" && (
                <div className="insight-grid">
                  <div className="trend-snapshot-item">
                    <div className="signal-label">Nearest Support</div>
                    <div className="signal-value-small">{formatCurrency(quantInsights.support_resistance?.nearest_support ?? 0)}</div>
                  </div>
                  <div className="trend-snapshot-item">
                    <div className="signal-label">Nearest Resistance</div>
                    <div className="signal-value-small">{formatCurrency(quantInsights.support_resistance?.nearest_resistance ?? 0)}</div>
                  </div>
                  <div className="trend-snapshot-item">
                    <div className="signal-label">Pivot / S1 / R1</div>
                    <div className="signal-value-small">
                      {formatCurrency(quantInsights.support_resistance?.pivot ?? 0)} / {formatCurrency(quantInsights.support_resistance?.support_1 ?? 0)} / {formatCurrency(quantInsights.support_resistance?.resistance_1 ?? 0)}
                    </div>
                  </div>
                  <div className="trend-snapshot-item">
                    <div className="signal-label">Breakout State</div>
                    <div className="signal-value-small">{quantInsights.support_resistance?.breakout_signal ?? "n/a"}</div>
                  </div>
                </div>
              )}

              {insightPanel === "relative" && (
                <>
                  <div className="insight-grid">
                    <div className="trend-snapshot-item">
                      <div className="signal-label">Relative Composite</div>
                      <div className="signal-value-small">
                        {quantInsights.relative_strength?.composite_score?.toFixed(1) ?? "n/a"}
                      </div>
                    </div>
                    <div className="trend-snapshot-item">
                      <div className="signal-label">Volume Intelligence</div>
                      <div className="signal-value-small">
                        RVOL {(quantInsights.volume_intelligence?.rvol_20 ?? 0).toFixed(2)} • {quantInsights.volume_intelligence?.volume_spike ? "Spike" : "Normal"}
                      </div>
                    </div>
                  </div>
                  <div className="card-subsection">
                    <div className="signal-label">NVDA Outperformance</div>
                    <div className="insight-row-list">
                      {(quantInsights.relative_strength?.benchmarks ?? []).map((item) => (
                        <div key={item.ticker} className="insight-row-item">
                          <span>{item.ticker}</span>
                          <span className={item.outperforming ? "success-text" : "error"}>
                            {formatSignedPct(item.relative_alpha_20d)}
                          </span>
                        </div>
                      ))}
                    </div>
                    {!(quantInsights.relative_strength?.benchmarks?.length) && (
                      <p className="muted-small">Benchmark feed unavailable at this moment.</p>
                    )}
                  </div>
                </>
              )}
            </>
          )}
        </section>

      </main>

      <section className="bottom-strip">
        <div className="card compact-pattern-card">
          <div className="card-header">
            <h2>Recent Pattern Signals</h2>
            {recentPatternCandles.length > PATTERN_PAGE_SIZE && (
              <div className="pattern-pager">
                <button
                  className="pattern-nav-btn"
                  type="button"
                  onClick={() => setPatternPage((page) => Math.max(0, page - 1))}
                  disabled={patternPage === 0}
                  aria-label="Show newer pattern signals"
                >
                  {"<"}
                </button>
                <span className="tag">
                  {patternPage + 1} / {maxPatternPage + 1}
                </span>
                <button
                  className="pattern-nav-btn"
                  type="button"
                  onClick={() => setPatternPage((page) => Math.min(maxPatternPage, page + 1))}
                  disabled={patternPage === maxPatternPage}
                  aria-label="Show older pattern signals"
                >
                  {">"}
                </button>
              </div>
            )}
          </div>

          {recentPatternCandles.length > 0 ? (
            <div className="pattern-summary-list">
              {visiblePatternCandles.map((c, index) => (
                <div
                  key={`${c.date}-${index}`}
                  className="pattern-summary-item"
                >
                  {(() => {
                    const { change, changePct } = getDailyMoveForCandle(c);
                    return (
                      <>
                        <div className="pattern-summary-meta">
                          <span className="pattern-date">{c.date}</span>
                          <span className="pattern-price">${c.close.toFixed(2)}</span>
                        </div>
                        <div className="pattern-summary-change">
                          <span className="pattern-change-value">
                            {formatSignedCurrency(change)} ({formatSignedPercent(changePct)})
                          </span>
                        </div>
                        <div className="pattern-tags">
                          {c.patterns.slice(0, 1).map((p) => (
                            <span key={p} className="pattern-tag">{p}</span>
                          ))}
                          {c.patterns.length > 1 && (
                            <span className="pattern-tag pattern-tag-muted">+{c.patterns.length - 1}</span>
                          )}
                          {c.patterns.length === 0 && (
                            <span className="pattern-tag pattern-tag-muted">no_pattern</span>
                          )}
                        </div>
                      </>
                    );
                  })()}
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">No recent pattern signals detected.</p>
          )}
        </div>
      </section>

      <div className="signature-line">Developed by Hrishik Desai</div>

      <button
        className="edge-oracle-launcher"
        type="button"
        onClick={() => setShowEdgeOracle(true)}
      >
        <img className="edge-oracle-launcher-logo" src={alphaLogo} alt="" aria-hidden="true" />
        Alpha
      </button>

      {showEdgeOracle && (
        <>
          <div className="edge-oracle-overlay" onClick={() => setShowEdgeOracle(false)} />
          <section className="edge-oracle-panel" aria-label="Alpha copilot chat assistant">
          <div className="edge-oracle-header">
            <div className="edge-oracle-brand">
              <img className="edge-oracle-avatar" src={alphaLogo} alt="Alpha logo" />
              <div>
                <h3>Alpha</h3>
                <p>AI Copilot</p>
              </div>
            </div>
            <button
              className="edge-oracle-close"
              type="button"
              onClick={() => setShowEdgeOracle(false)}
              aria-label="Close Alpha"
            >
              x
            </button>
          </div>

          <div className="edge-oracle-feed">
            {!llmOutput && !llmLoading && (
              <p className="edge-oracle-hint">
                Ask about timing, downside risk, confidence, or invalidation.
              </p>
            )}

            {llmLoading && <p className="muted">Alpha is analyzing your setup...</p>}
            {llmError && <p className="error">{llmError}</p>}

            {llmOutput && (
              <>
                <div className="chat-bubble chat-bubble-user">{llmQuestion}</div>
                <div className="chat-bubble chat-bubble-assistant" style={{ whiteSpace: "pre-line" }}>
                  {llmOutput}
                </div>
              </>
            )}
          </div>

          <div className="edge-oracle-prompt-row">
            {alphaPromptChips.map((prompt) => (
              <button
                key={prompt}
                type="button"
                className="edge-oracle-prompt-chip"
                onClick={() => handleAlphaPromptClick(prompt)}
              >
                {prompt}
              </button>
            ))}
          </div>

          <div className="edge-oracle-input-wrap">
            <textarea
              className="llm-question-input"
              value={llmQuestion}
              onChange={(e) => setLlmQuestion(e.target.value)}
              onFocus={() => {
                if (llmQuestion.trim().toLowerCase().startsWith("why is the current recommendation")) {
                  setLlmQuestion("");
                }
              }}
              onKeyDown={handleLlmQuestionKeyDown}
              placeholder="Type your question..."
              rows={3}
            />
            <button
              className="evaluate-btn edge-oracle-send"
              onClick={handleEvaluateClick}
              disabled={!latestCandle || !signal || llmLoading || !llmQuestion.trim()}
            >
              {llmLoading ? "Thinking..." : "Ask Alpha"}
            </button>
          </div>
          </section>
        </>
      )}
      {showAdminLogin && (
        <div className="modal-backdrop">
          <div className="modal-card">
            <h3>Admin Login</h3>
            <p className="muted">Enter the admin panel password to view system status.</p>
            <input
              className="modal-input"
              type="password"
              value={adminPasswordInput}
              onChange={(e) => setAdminPasswordInput(e.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Escape") {
                  setShowAdminLogin(false);
                  return;
                }
                if (event.key === "Enter") {
                  event.preventDefault();
                  void handleAdminLogin();
                }
              }}
            />
            {adminLoginError && <p className="error">{adminLoginError}</p>}
            <div className="modal-actions">
              <button className="btn-secondary" onClick={() => setShowAdminLogin(false)}>
                Cancel
              </button>
              <button className="evaluate-btn" onClick={handleAdminLogin}>
                Login
              </button>
            </div>
          </div>
        </div>
      )}

      {showAdminPanel && (
        <div className="modal-backdrop">
          <div className="modal-card admin-modal-card">
            <div className="card-header">
              <h3>Admin & System Status</h3>
              <span className="tag">Protected</span>
            </div>

            {adminStatusLoading && <p className="muted">Refreshing admin status…</p>}

            {adminStatus && (
              <>
                <div className="status-grid">
                  <div>
                    <div className="signal-label">Scheduler</div>
                    <div
                      className={`status-badge ${schedulerRunning ? "status-badge-on" : "status-badge-off"}`}
                    >
                      <span className="status-badge-dot" />
                      {schedulerRunning ? "Running" : "Stopped"}
                    </div>
                  </div>
                  <div>
                    <div className="signal-label">Model Version</div>
                    <div className="signal-value-small">
                      {modelVersion ?? "n/a"}
                    </div>
                  </div>
                  <div>
                    <div className="signal-label">Market Refresh</div>
                    <div className="signal-value-small">
                      {formatDateTimeWithZone(marketRefreshAt)}
                    </div>
                  </div>
                  <div>
                    <div className="signal-label">Sentiment Refresh</div>
                    <div className="signal-value-small">
                      {formatDateTimeWithZone(sentimentRefreshAt)}
                    </div>
                  </div>
                  <div>
                    <div className="signal-label">Model Retrain</div>
                    <div className="signal-value-small">
                      {formatDateTimeWithZone(modelRetrainAt)}
                    </div>
                  </div>
                  <div>
                    <div className="signal-label">Last Heartbeat</div>
                    <div className="signal-value-small">
                      {formatDateTimeWithZone(lastHeartbeatAt)}
                    </div>
                  </div>
                  <div>
                    <div className="signal-label">Persisted Snapshot</div>
                    <div className="signal-value-small">
                      {adminStatus.persisted_status
                        ? formatDateTimeWithZone(adminStatus.persisted_status.updated_at)
                        : "n/a"}
                    </div>
                  </div>
                </div>
                {marketLastError && (
                  <p className="admin-warning">
                    Market refresh issue: {marketLastError}
                  </p>
                )}

                <div className="card-subsection">
                  <div className="signal-label">Recent Admin Events</div>
                  <div className="admin-event-list">
                    {adminStatus.recent_events.map((event) => (
                      <div key={`${event.event_type}-${event.created_at}`} className="admin-event-item">
                        <span>{event.event_type}</span>
                        <span className="muted-small">{formatDateTimeWithZone(event.created_at)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            <div className="modal-actions">
              <button className="btn-secondary" onClick={() => setShowAdminPanel(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;


