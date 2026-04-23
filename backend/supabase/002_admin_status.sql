create table if not exists admin_runtime_status (
  id uuid primary key default gen_random_uuid(),
  service_name text not null,
  symbol text not null default 'NVDA',
  status jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now(),
  unique(service_name, symbol)
);

create index if not exists admin_runtime_status_updated_idx
  on admin_runtime_status(updated_at desc);

create table if not exists admin_status_events (
  id uuid primary key default gen_random_uuid(),
  service_name text not null,
  symbol text not null default 'NVDA',
  event_type text not null,
  status jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists admin_status_events_created_idx
  on admin_status_events(created_at desc);

create index if not exists admin_status_events_service_symbol_idx
  on admin_status_events(service_name, symbol, created_at desc);
