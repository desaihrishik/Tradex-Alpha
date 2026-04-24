// src/api.ts
import axios from "axios";

const BASE_URL = "https://tradex-alpha-backend.onrender.com";

export type RiskProfile = "low" | "medium" | "high";

export interface Candle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  patterns: string[];
  signal: number;
}

export interface LatestSignalResponse {
  date: string;
  latest_close: number;
  action: "BUY" | "SELL" | "HOLD";
  signal_value: -1 | 0 | 1;
  confidence: number;
  probas: Record<string, number>;
  suggested_shares: number;
  capital_used: number;
  patterns: string[];
  risk_profile: RiskProfile;
  budget: number;
  explanation: string;
}

export interface AgenticForecast {
  p10: number;
  p50: number;
  p90: number;
}

export interface TrendForecastPoint {
  date: string;
  value: number;
}

export interface SellAnalysis {
  current_return: number;
  potential_downside_if_hold: number;
  expected_future_ret: number;
  best_case_ret: number;
}

export interface AgenticSignalResponse {
  date: string;
  latest_close: number;
  action: "BUY" | "SELL" | "HOLD";
  signal_value: -1 | 0 | 1;
  confidence: number;
  probas: Record<string, number>;
  patterns: string[];
  pattern_strength: number;
  trend_strength: number;
  sentiment_label: string;
  sentiment_score: number;
  sentiment_strength: number;
  horizon_days: number;
  forecast: AgenticForecast;
  trend_forecast: {
    dates: string[];
    values: number[];
    direction_probability: number;
    direction: "bullish" | "bearish" | "sideways";
  };
  suggested_shares: number;
  capital_used: number;
  sell_analysis: SellAnalysis | null;
  entry_price_used: number | null;
  risk_profile: RiskProfile;
  budget: number;
  explanation: string;
}

export interface AnalyzeResponse {
  symbol: string;
  candles: Candle[];
  latest_signal: LatestSignalResponse;
  agentic_signal: AgenticSignalResponse;
  decision: {
    symbol: string;
    action: "BUY" | "SELL" | "HOLD";
    confidence: number;
    risk_profile: RiskProfile;
    budget: number;
  };
  explanation: string;
  validation: {
    is_fresh: boolean;
    model_date: string;
    market_date: string;
    date_gap_days: number | null;
    model_price: number;
    market_price: number;
    price_diff: number;
    price_diff_pct: number;
    displayed_suggested_shares: number;
    suggested_shares_at_market_price: number;
    displayed_capital_used: number;
    capital_used_at_market_price: number;
    checks: string[];
  };
  quant_insights?: {
    available: boolean;
    as_of?: string;
    market_regime?: {
      label: string;
      volatility_regime: string;
    };
    momentum?: {
      score: number;
      ema_alignment: string;
      sma_alignment: string;
      rsi_14: number;
      stoch_rsi: number;
      macd_hist: number;
      adx_14: number;
      roc_10: number;
    };
    volatility?: {
      atr_14: number;
      atr_pct: number;
      bollinger: { upper: number; mid: number; lower: number };
      historical_vol_20: number;
      realized_vol_20: number;
      implied_vol: number | null;
      regime_hint: string;
    };
    volume_intelligence?: {
      rvol_20: number;
      volume_spike: boolean;
      obv_trend: string;
      vwap_20: number;
      money_flow_index: number;
      acc_dist_slope_10: number;
      volume_last: number;
    };
    risk_metrics?: {
      sharpe: number;
      sortino: number;
      max_drawdown: number;
      win_rate: number;
      risk_reward_ratio: number;
      expected_return: number;
      expected_loss: number;
      kelly_fraction: number;
      var_95: number;
      cvar_95: number;
    };
    position_sizing?: {
      risk_pct: number;
      suggested_shares: number;
      capital_at_risk: number;
      stop_loss: number;
      trailing_stop_pct: number;
      profit_target: number;
      stop_loss_distance: number;
      confidence_scaling: number;
      entry_price_used: number | null;
      probability_target_hit_12d: number;
      probability_stop_hit_12d: number;
    };
    support_resistance?: {
      pivot: number;
      support_1: number;
      resistance_1: number;
      previous_high_20: number;
      previous_low_20: number;
      nearest_support: number;
      nearest_resistance: number;
      breakout_signal: string;
      gap_pct: number;
    };
    relative_strength?: {
      available: boolean;
      composite_score: number | null;
      benchmarks: Array<{
        ticker: string;
        symbol_return_20d: number;
        benchmark_return_20d: number;
        relative_alpha_20d: number;
        outperforming: boolean;
        score: number;
      }>;
    };
    summary?: {
      momentum_state: string;
      risk_state: string;
      last_daily_move: number;
    };
  };
  persistence?: {
    forecast_run_id: string | null;
    recommendation_id: string | null;
  };
}

export interface CurrentTrendResponse {
  symbol: string;
  as_of: string;
  market_refresh_at?: string;
  latest_close: number;
  previous_close: number;
  change: number;
  change_pct: number;
  trend_label: string;
  trend_strength: number;
  patterns: string[];
  volume: number;
  sentiment?: {
    label: string;
    score: number;
    article_count: number;
    ts: string;
  };
  recommendation: {
    action: "BUY" | "SELL" | "HOLD";
    confidence: number;
    suggested_amount: number | null;
    suggested_shares: number | null;
    suggested_duration_days: number | null;
    sentiment_label: string | null;
    sentiment_score: number | null;
    explanation: string;
    ts: string;
  } | null;
}

export interface SentimentDetailsResponse {
  symbol: string;
  as_of: string;
  label: string;
  score: number;
  article_count: number;
  interpretation: string;
  previous_score: number | null;
  previous_label: string | null;
  previous_as_of: string | null;
  score_change: number | null;
  score_change_pct: number | null;
  usefulness: string[];
}

export interface AdminStatusResponse {
  live_status: Record<string, string | boolean | null>;
  persisted_status: {
    service_name: string;
    symbol: string;
    updated_at: string;
    status: Record<string, string | boolean | null>;
  } | null;
  recent_events: Array<{
    event_type: string;
    status: Record<string, string | boolean | null>;
    created_at: string;
  }>;
}

export async function fetchCandles(limit = 120): Promise<Candle[]> {
  const res = await axios.get<{ candles: Candle[] }>(
    `${BASE_URL}/api/nvda/candles`,
    { params: { limit } }
  );
  return res.data.candles;
}

export async function fetchLatestSignal(
  budget: number,
  risk: RiskProfile
): Promise<LatestSignalResponse> {
  const res = await axios.get<LatestSignalResponse>(
    `${BASE_URL}/api/nvda/latest_signal`,
    { params: { budget, risk } }
  );
  return res.data;
}

export async function fetchAgenticSignal(
  budget: number,
  risk: RiskProfile,
  entryPrice?: number
): Promise<AgenticSignalResponse> {
  const res = await axios.get<AgenticSignalResponse>(
    `${BASE_URL}/api/nvda/agentic_signal`,
    {
      params: { budget, risk, entry_price: entryPrice ?? null },
    }
  );
  return res.data;
}

export async function fetchAnalyze(
  budget: number,
  risk: RiskProfile,
  limit = 120,
  entryPrice?: number,
  persist = true
): Promise<AnalyzeResponse> {
  const res = await axios.get<AnalyzeResponse>(
    `${BASE_URL}/api/nvda/analyze`,
    {
      params: {
        budget,
        risk,
        limit,
        entry_price: entryPrice ?? null,
        persist,
      },
    }
  );
  return res.data;
}

export async function fetchCurrentTrend(): Promise<CurrentTrendResponse> {
  const res = await axios.get<CurrentTrendResponse>(
    `${BASE_URL}/api/nvda/current_trend`
  );
  return res.data;
}

export async function fetchSentimentDetails(): Promise<SentimentDetailsResponse> {
  const res = await axios.get<SentimentDetailsResponse>(
    `${BASE_URL}/api/nvda/sentiment`
  );
  return res.data;
}

export async function adminLogin(password: string): Promise<{ ok: true }> {
  const res = await axios.post<{ ok: true }>(`${BASE_URL}/api/admin/login`, {
    password,
  });
  return res.data;
}

export async function fetchAdminStatus(
  password: string,
  eventLimit = 10
): Promise<AdminStatusResponse> {
  const res = await axios.get<AdminStatusResponse>(
    `${BASE_URL}/api/admin/status`,
    {
      params: { symbol: "NVDA", event_limit: eventLimit },
      headers: { "x-admin-password": password },
    }
  );
  return res.data;
}

// LLM → observe/inference only
export async function llmEvaluate(
  question: string,
  budget: number,
  risk: RiskProfile,
  entryPrice?: number,
  mode: "fast" | "deep" = "fast",
): Promise<{ answer: string }> {
  const res = await axios.post(`${BASE_URL}/api/llm/trade_question`, {
    question,
    symbol: "NVDA",
    budget,
    risk,
    entry_price: entryPrice ?? null,
    mode,
  });
  return res.data;
}
