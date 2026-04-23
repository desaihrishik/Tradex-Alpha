create extension if not exists pgcrypto;

create table if not exists symbols (
  id bigserial primary key,
  ticker text unique not null,
  name text,
  is_active boolean not null default true,
  created_at timestamptz not null default now()
);

create table if not exists market_candles (
  id bigserial primary key,
  symbol_id bigint not null references symbols(id) on delete cascade,
  timeframe text not null,
  ts timestamptz not null,
  open numeric not null,
  high numeric not null,
  low numeric not null,
  close numeric not null,
  volume numeric not null,
  source text not null default 'alpaca',
  unique(symbol_id, timeframe, ts)
);

create index if not exists market_candles_symbol_time_idx
  on market_candles(symbol_id, timeframe, ts desc);

create table if not exists sentiment_snapshots (
  id bigserial primary key,
  symbol_id bigint not null references symbols(id) on delete cascade,
  ts timestamptz not null default now(),
  sentiment_label text not null,
  sentiment_score numeric not null,
  article_count int not null default 0,
  raw jsonb not null default '{}'::jsonb
);

create table if not exists technical_snapshots (
  id bigserial primary key,
  symbol_id bigint not null references symbols(id) on delete cascade,
  timeframe text not null,
  ts timestamptz not null,
  indicators jsonb not null default '{}'::jsonb,
  patterns jsonb not null default '[]'::jsonb,
  trend_label text,
  trend_strength numeric,
  unique(symbol_id, timeframe, ts)
);

create table if not exists model_registry (
  id uuid primary key default gen_random_uuid(),
  model_name text not null,
  version text not null,
  storage_path text not null,
  metadata_path text,
  framework text not null default 'sklearn',
  status text not null default 'active',
  trained_at timestamptz,
  metrics jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique(model_name, version)
);

create table if not exists forecast_runs (
  id uuid primary key default gen_random_uuid(),
  symbol_id bigint not null references symbols(id) on delete cascade,
  model_id uuid references model_registry(id),
  ts timestamptz not null default now(),
  horizon_days int not null,
  p10 numeric,
  p50 numeric,
  p90 numeric,
  raw jsonb not null default '{}'::jsonb
);

create table if not exists recommendations (
  id uuid primary key default gen_random_uuid(),
  symbol_id bigint not null references symbols(id) on delete cascade,
  model_id uuid references model_registry(id),
  forecast_run_id uuid references forecast_runs(id),
  ts timestamptz not null default now(),
  risk_profile text not null,
  budget numeric not null,
  action text not null,
  confidence numeric not null,
  suggested_amount numeric,
  suggested_shares numeric,
  suggested_duration_days int,
  current_trend text,
  historical_trend text,
  sentiment_label text,
  sentiment_score numeric,
  explanation text,
  payload jsonb not null default '{}'::jsonb
);
